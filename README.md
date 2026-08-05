# PPO with memory: an encoder ablation on MiniGrid

Does a **Transformer** remember better than an **LSTM** or **GRU**?

This project trains PPO on a partially-observable MiniGrid maze where the agent
must carry one bit of information across the episode, and swaps out only the
feature extractor — MLP, LSTM, GRU, Transformer — leaving every other part of
the pipeline byte-identical. Hyperparameters are tuned per encoder with Optuna
so no encoder is judged on another's settings.

The MLP is the control: it has no memory at all, so whatever it scores is the
ceiling for a policy that only reacts to the current frame.

---

## The task

Default env: **`MiniGrid-MemoryS11-v0`** (11×11).

The agent starts in a corridor. In a small room at the west end sits a **cue**
object — a key or a ball. At the east end the corridor forks into a T, with a
key at one prong and a ball at the other. The episode succeeds only if the
agent walks to the prong holding **the same object as the cue**.

```
    cue room             corridor                    fork
   ┌───────┐                                       ┌── ⬤  key
   │   ⬤   │══════════════  ▶  ═══════════════════─┤
   └───────┘              agent spawns             └── ⬥  ball
    key or ball           anywhere here,
                          facing east →
```

Two properties make this a genuine memory problem rather than a maze:

- **The cue is out of sight when the decision is made.** The agent sees a 7×7
  egocentric patch. By the time it reaches the junction the cue is long gone
  from view, so the junction observation is *identical* whether the answer is
  "up" or "down" — classic state aliasing. A memoryless policy therefore cannot
  do better than chance, no matter how well it is trained.
- **The information must be sought out.** The agent spawns at a random spot in
  the corridor, facing east — away from the cue. In ~90% of episodes the cue is
  behind it at t=0, so the agent has to *choose* to turn around, walk back, look,
  and turn again before running to the junction. That detour costs a little
  reward and pays nothing until the memory also works.

Reward is MiniGrid's standard `1 - 0.9 * step_count / max_steps` on success, 0
otherwise. Chance is ~0.5.

`config.force_cue_visible = True` swaps in a wrapper (`StartInCueView`) that
pins the spawn to the one tile where the cue is in view at t=0. That removes
the *exploration* half of the problem while leaving the *memory* half intact —
useful for isolating which of the two an encoder is failing at.

---

## Results

GRU, seed 0, on the unmodified task (`force_cue_visible = False`):

| | success | return | episode length | `value_loss` |
|---|---|---|---|---|
| memoryless policy (iteration 100) | 0.58 | 0.566 | 7.1 | 0.162 |
| **final, iteration 1999** | **1.00** | **0.957** | 14.5 | **0.0000** |

Solved in 35 minutes on CPU, evaluated on 50 fixed mazes. The three numbers
tell one story:

- `success 1.00` — the agent is using the cue, not guessing.
- **`length` doubled**, 7.1 → 14.5. The agent *chose* to spend ~7 extra steps
  walking back to look at the cue. The detour costs ~0.02 of reward and buys
  ~+0.4 success, and nothing in the reward function told it to do this.
- `value_loss` fell from 0.162 to 0.0000. 0.162 is the irreducible variance of
  an outcome the critic cannot predict; reaching ~0 means the network's hidden
  state actually encodes which prong is correct. **This is the leading indicator
  — it moves several hundred iterations before the success rate does.**

Settings for that run (see *Reproducing* below — the committed config has since
drifted off two of them):

```
n_workers 32   worker_steps 302   n_iterations 2000   hidden_size 64
lr 1e-3 (annealed to 0)   n_epochs 3   target_kl 0.02   entropy_coef 0.005
gamma 0.99   gae_lambda 0.95   clip_eps 0.2   value_coef 0.1   max_grad_norm 0.5
```

**Caveats, stated plainly:** this is one seed, and the curve has two abrupt
behavioural transitions (~iteration 300 and ~900) of exactly the kind that vary
a lot between seeds. The MLP/LSTM/Transformer arms have **not** been run at
these settings, so the ablation this project exists to answer is not yet
answered. `worker_steps` doubles as the env's `max_steps` and therefore rescales
the reward, so these numbers are not comparable to runs at a different `T`.

---

## Install

Python 3.11.

```bash
conda create -n rl_project python=3.11 && conda activate rl_project
pip install -r requirements.txt          # requirements_windows.txt on Windows
```

`requirements.txt` is the authoritative list. `environment.yml` is stale and
gitignored — ignore it. Core deps: `torch`, `gymnasium`, `minigrid`, `optuna`,
`pygame`.

A fresh shell starts in conda `base`, which has no torch. Activate the env, or
prefix commands with `conda run -n rl_project`.

---

## Quickstart

```bash
# 1. train one encoder with hand-picked hyperparameters (no search)
python main_no_hpo.py GRU

# 2. watch what it learned, in a pygame window
python watch.py GRU 1.5

# 3. tune that encoder with Optuna, then report the winner
python main.py GRU --trials 30
```

### All commands

```bash
# hyperparameter search — resumable, rerunning continues an interrupted study
python main.py MLP|LSTM|GRU|TRANSFORMER [--trials N] [--search-only] [--final-only]

# hand-picked run, values from config/config_no_hpo.py
python main_no_hpo.py [MLP|LSTM|GRU|TRANSFORMER] [--report-only]

# replay a saved checkpoint; --seed is an INDEX into seed_list, not a seed value
python watch.py [MODEL] [steps_per_sec] [--hpo] [--seed INDEX] [--trial best|N]

# self-contained shape / forward-pass / memory / causality checks (a few seconds)
python models/feature_extractor.py
python models/model.py
```

There is no pytest suite. The `__main__` blocks in `models/` are the fastest
way to check an encoder change.

Every invocation writes a timestamped log to `logs/`, mirroring the terminal
and dumping every hyperparameter at the top.

### The viewer

`watch.py` opens a pygame window showing the full maze, the agent's 7×7
observation, its action distribution and its value estimate, replayed over the
fixed eval mazes. It is the quickest way to see *whether* a policy detours to
the cue rather than inferring it from numbers.

```
SPACE pause    ← → step within an episode    P/N previous/next maze
R replay       A auto-advance the whole eval set    Q quit
```

---

## Layout

```
config/       hyperparameters + the project's toolbox
  config.py         Config: every shared PPO/env/eval/HPO knob. Abstract.
  config_{mlp,lstm,gru,transformer}.py
                    one per encoder; sets feature_extractor + its architecture
                    knobs and appends them to the shared search space
  config_no_hpo.py  ConfigNoHPO: everything hand-picked, search_space = []
  helper.py         Helper (~2100 lines): env builders, checkpoint I/O, batch-size
                    probing, Optuna persistence, plotting, the pygame viewer,
                    Timing, StartInCueView, SequenceDataset
models/
  feature_extractor.py   MLP / LSTM / GRU / Transformer — all map
                         (batch, seq_len, 7, 7, 3) -> (batch, seq_len, hidden_size)
  model.py               Network: one encoder + linear actor head + linear critic head
agents/
  ppo.py            PPOAgent: rollout collection, GAE, sequence batching, the
                    clipped loss, evaluation, the training loop
  hpo_ppo.py        HPOPPO: wraps PPOAgent in an Optuna study
main.py             tuned entry point
main_no_hpo.py      hand-picked entry point
watch.py            viewer (loads a checkpoint, trains nothing)
```

### Checkpoints

```
agents/pretrained_model_<ENCODER>/
    hpo/trial_<n>/     one Optuna trial
    hpo/best_trial/    the winner, copied out so trial_*/ can be deleted
    no_hpo/            hand-picked runs
```

`config.select_run()` is the single place that resolves *(mode, trial, seed
index)* to a path — `HPOPPO`, `main_no_hpo.py` and `watch.py` all go through it,
so they cannot drift apart. Each `.pth` carries its own architecture, its
`eval_history` (which is what the learning-curve plots are redrawn from) and the
`force_cue_visible` it was trained under; `load_model` refuses a checkpoint whose
env, encoder, widths or cue setting disagree with the live config.

---

## How it fits together

**Config layer.** `Config` holds every shared hyperparameter and inherits the
whole toolbox from `Helper`. Each encoder subclass implements only
`_configure_model()`. The PPO half of the Optuna search space lives in `Config`
so all four encoders search identical ranges — the widths a study settles on are
then comparable across the ablation. `make_config("GRU")` returns the *tuned*
config; `ConfigNoHPO("GRU")` is the hand-picked one, which is why `watch.py`
branches between them.

**Encoders.** All four take the same input and return the same shape, so
`Network` and `PPOAgent` never branch on which one is in use. `flatten_obs`
one-hots the observation's 3 channels into 980 features first. All four act one
step at a time during rollout collection and see whole sequences during the PPO
update.

**Agent.** `PPOAgent.sample()` collects a rollout across `n_workers` parallel
envs; `gae()` computes advantages; `split_pad_mask()` cuts each worker's stream
at episode boundaries and pads to rectangles for truncated-BPTT; `learn()`
runs the clipped-surrogate + value + entropy loss over minibatches of whole
sequences, stopping early on `target_kl`.

---

## Design notes worth knowing before editing

- **Rollout length is the env's time limit, in both directions.** `worker_steps`
  (T) defaults to the env's own `max_steps`, and `build_env` passes it back into
  `gym.make`. An episode can therefore never outlive one rollout — but MiniGrid's
  success reward moves with `max_steps`, so **changing T silently rescales the
  reward and every number the study compares.**
- **`eval_deterministic` is one setting for a whole run.** It decides whether the
  learning curve, the checkpoint's stored eval and the HPO score are *all*
  sampled or *all* argmax, so switching it can never make two numbers
  incomparable. The reporting paths read the mode off the checkpoint, measure
  the other one fresh, and print them in separate blocks.
- **Minibatch size is resolved once, then defended.** `mini_batch_size` is a list
  of candidates, largest first. A probe runs a worst-case forward/backward before
  iteration 0 (early rollouts pack fewer sequences than later ones, so probing on
  real data would underestimate), and an OOM-retry wrapper guards each real
  update. The resolved size is logged and written into the checkpoint.
- **A tuned checkpoint carries its own architecture.** `save_model` stores the
  searched params, so every reload path must call
  `config.apply_params(config.checkpoint_params(path))` *before* building the
  agent. A new `search_space` entry requires the attribute to already exist on
  the config — `apply_params` raises rather than inventing one.
- **`hpo_objective` is one string**, `<metric>_<center>_<spread>` (e.g.
  `return_mean_minus-std`). `center` and `spread` are always aggregated **across
  seeds**, never across the eval episodes of one run — within-run spread on a
  bimodal return is a function of the success rate itself and would penalise
  partial success.
- **`p_drop` must stay `0.0`** on the Transformer. PPO compares rollout
  log-probs against log-probs recomputed during the update; dropout makes those
  differ on identical inputs and corrupts the ratio. Same reason `n_heads` must
  divide `d_model`.
- **`async_envs`** picks `AsyncVectorEnv` (separate processes, default) vs
  `SyncVectorEnv`. It must be `False` in notebooks and any script without an
  `__main__` guard, because macOS spawns subprocesses.
- **Timing is first-class.** `PPOAgent` only names phases
  (`with self.timing.phase("sample"):`); all arithmetic lives in `Timing`.
  Adding a phase means adding a key to `Timing.ROWS` — an untimed-but-unlisted
  phase accumulates silently and just isn't printed.
- **`config.tbptt_length` is dead** — set, read nowhere. `split_pad_mask` cuts on
  episode boundaries only; there is no fixed-length chunking.

---

## Reproducing the solved GRU run

The committed config has drifted off two of the settings that produced the
result above. To repeat it, in `config/config.py`:

```python
self.force_cue_visible = False       # the real task, no spawn wrapper
self.n_workers = 32
self.worker_steps = 302              # currently None -> 605; must be set explicitly
self.n_iterations = 2000
```

then `python main_no_hpo.py GRU`. Note this **overwrites**
`agents/pretrained_model_GRU/no_hpo/` — the directory does not separate runs by
setting, so move a checkpoint you want to keep before rerunning.

Why these values matter: at `n_workers = 8` and `T = 605` the same
hyperparameters plateau at chance for 800 iterations and never learn the memory.
Widening the rollout buffer collects more of the rare episodes in which the agent
happens to see the cue, and halving `T` raises the number of gradient updates per
environment step; together they get the memory to form before the policy commits
to the free ~0.5 available from ignoring the cue.

## Roadmap

- [ ] Run seeds `[0, 26, 98]` — the current result is single-seed
- [ ] Run MLP, LSTM and Transformer at the same settings (the actual ablation)
- [ ] Fix `T` across encoders before comparing, since it rescales the reward
- [ ] Optuna study per encoder once the hand-picked baseline is stable
