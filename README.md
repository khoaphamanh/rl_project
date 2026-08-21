# PPO with memory: an encoder ablation on MiniGrid

Does a **GRU** recover a task a memoryless policy cannot solve — and how far
back does its gradient actually need to reach?

This project trains PPO on a partially-observable MiniGrid maze where the agent
must carry one bit of information across the episode, and swaps out only the
feature extractor — **MLP** or **GRU** — leaving every other part of the pipeline
byte-identical. Hyperparameters are tuned per encoder with Optuna so no encoder
is judged on another's settings. A second axis tunes the GRU once per
truncated-BPTT length (`--tbptt L`), each length its own independent study.

The MLP is the control: it has no memory at all, so whatever it scores is the
ceiling for a policy that only reacts to the current frame.

(LSTM and Transformer encoders were part of an earlier version and have been
removed; `git log` still has them.)

---

## The task

The current default env is **`MiniGrid-MemoryS17Random-v0`** (17×17, random cue
placement). The walkthrough below was written for **`MiniGrid-MemoryS11-v0`**
(11×11); the layout and the reason it is a memory problem are identical, only
the size and the numbers differ.

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

> **Stale section.** Everything below — this table, the settings under it, and
> *Reproducing the solved GRU run* — was measured on `MiniGrid-MemoryS11-v0`
> with a single seed, before the encoder set was cut to MLP/GRU. The current
> default env is `MiniGrid-MemoryS17Random-v0` and the current five-seed results
> are the table in *Running the code* above. `worker_steps` rescales the reward,
> so the two sets of numbers are not comparable.

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

`requirements.txt` is the authoritative list, and is Linux/macOS — there PyPI's
default `torch` wheel is already the CUDA build. On Windows use
`requirements_windows.txt`, which is **not** a subset: PyPI's Windows wheel is
CPU-only, so it pins `torch==2.11.0+cu128` against the PyTorch index (the
`+cu128` is load-bearing — without it pip may satisfy the version from PyPI and
silently install the CPU build), uses `pygame-ce`, and adds `kaleido`, without
which every `.svg` figure silently goes missing. `environment.yml` is stale and
gitignored — ignore it. Core deps: `torch`, `gymnasium`, `minigrid`, `optuna`,
`pygame`.

A fresh shell starts in conda `base`, which has no torch. Activate the env, or
prefix commands with `conda run -n rl_project`.

---

## Running the code

Three entry points; nothing else in the repo is meant to be run directly.

| | what it does | writes to |
|---|---|---|
| `main.py` | the Optuna search: 50 trials × 5 seeds, then a report on the winner | `hpo/` |
| `main_no_hpo.py` | trains at the hand-picked values in `config/config_no_hpo.py` — one run per seed in *its* `seed_list`, currently just seed 0, for 2000 iterations | `no_hpo/` |
| `watch.py` | loads one checkpoint and plays it in a pygame window — trains nothing, writes nothing | — |

`MLP` and `GRU` are the only encoders. Run every command **from the repo root**:
checkpoint paths and the Optuna sqlite URL are relative to it.

### What already ships in the repo

Four finished studies are committed, so the reporting and viewing commands work
on a fresh clone without training anything:

| directory | encoder | score¹ | success | winning trial |
|---|---|---|---|---|
| `agents/pretrained_model_GRU/` | GRU, full BPTT | **0.956** | **1.00** | 36 (`hidden_size` 280) |
| `agents/pretrained_model_GRU_tbptt8/` | GRU, `--tbptt 8` | **0.955** | **1.00** | 25 (`hidden_size` 480) |
| `agents/pretrained_model_GRU_tbptt1/` | GRU, `--tbptt 1` | 0.509 | 0.59 | 47 (`hidden_size` 216) |
| `agents/pretrained_model_MLP/` | MLP (the memoryless control) | 0.527 | 0.59 | 3 (`hidden_size` 360) |

¹ `mean_minus_1std(return_mean)` across the five seeds — the single number a
trial is ranked on. Success is `success_rate`, meaning "walked to the prong
matching the cue", over 50 fixed eval mazes per seed, averaged over the seeds.
Both encoders that can remember solve the task outright; the MLP and the GRU
whose gradient reaches back exactly one step sit at chance, which is the result
the ablation exists to produce.

Two things are deliberately *not* committed, and both are ordinary, not bugs:

- `hpo/trial_*/` is gitignored (~5 GB of per-trial checkpoints). Only
  `hpo/best_trial/` — a copy of the winner — ships, so `watch.py --trial 12`
  works only on the machine that ran the study.
- There is no `no_hpo/` anywhere. `main_no_hpo.py --report-only` has nothing to
  read until you train one.

### Quickstart

```bash
# 1. report the tuned GRU: reloads best_trial/ and prints both eval modes.
#    Trains nothing, takes seconds.
python main.py GRU --final-only

# 2. watch what it learned, at 1.5 agent-steps per second
python watch.py GRU 1.5 --hpo

# 3. compare it against the memoryless control
python main.py MLP --final-only

# 4. train something yourself: one seed, hand-picked hyperparameters, no search
python main_no_hpo.py GRU
```

### The commands in full

```bash
# ---- hyperparameter search (resumable; see "Retraining" below) -------------
python main.py MLP|GRU [--tbptt L] [--trials N] [--search-only] [--final-only]

#   --tbptt L      GRU only. Fixes the gradient's backward reach at L steps for
#                  the WHOLE study and writes to pretrained_model_GRU_tbptt<L>/.
#                  Omit for full BPTT. Each length is its own study on purpose —
#                  see "The tbptt ablation" in the design notes.
#   --trials N     override config.n_trials (50). It is a TOTAL, not "N more".
#   --search-only  search, skip the closing report
#   --final-only   skip the search, just report the winning trial's saved runs
#                  (it reloads best_trial/; it does not retrain)

# ---- hand-picked run, values from config/config_no_hpo.py -----------------
python main_no_hpo.py [MLP|GRU] [--tbptt L] [--report-only]

#   MODEL          optional, defaults to MLP
#   --report-only  load and report what is in no_hpo/ instead of training

# ---- replay a saved checkpoint in a pygame window ------------------------
python watch.py [MODEL] [steps_per_sec] [--hpo] [--tbptt L] [--seed INDEX] \
                [--trial best|N|final]

#   MODEL          optional, defaults to MLP; steps_per_sec defaults to 2.5
#                  (both positional, so the number comes before the flags)
#   --hpo          read the tuned run (hpo/) instead of the hand-picked one
#   --seed INDEX   an INDEX into seed_list, not a seed value. With --hpo the list
#                  is [0, 15, 12, 97, 98], so index 3 means seed 97; without it,
#                  ConfigNoHPO's list is just [0] and 0 is the only valid index.
#   --trial WHICH  with --hpo only: 'best' (default), 'final' (an alias for it),
#                  or a trial number. Without --hpo it is a usage error, because
#                  a hand-picked run has no trials.

# ---- self-contained shape / forward-pass / memory / causality checks ------
python models/feature_extractor.py
python models/model.py
```

Worked examples:

```bash
python watch.py GRU --hpo                        # the winner, seed 0
python watch.py MLP 1 --hpo --seed 3             # MLP winner, seed 97, one step/sec
python watch.py GRU --hpo --tbptt 8 --seed 1     # the tbptt8 tree, seed 15
python main.py GRU --tbptt 4                     # start a NEW study at L=4
python main.py GRU --trials 70                   # 20 more trials on the finished GRU study
python main_no_hpo.py GRU --tbptt 20             # hand-picked run, writes .../GRU_tbptt20/no_hpo/
```

`--tbptt` on an MLP is rejected rather than silently making a directory that
means nothing: an MLP has no recurrence to truncate.

There is no pytest suite. The `__main__` blocks in `models/` are the fastest way
to check an encoder change (a few seconds each).

### Where the output goes

```
logs/log_<timestamp>.log        one file per invocation, mirroring the terminal,
                                with every hyperparameter dumped at the top
agents/pretrained_model_<ENC>[_tbptt<L>]/
  hpo/hpo_db_<name>.db          the Optuna study — this is what makes a run resumable
  hpo/hpo_sampler_<name>.pkl    the pickled TPE sampler, saved after every trial
  hpo/hpo_csv_<name>.csv        every trial's params, value and clocks, as a table
  hpo/trial_<n>/                one checkpoint per seed, plus that trial's curves
  hpo/best_trial/               the winner copied out, plus best_params.json and
                                final_<name>.json (the closing report)
  no_hpo/                       the same shape for a hand-picked run
```

Curves are written as `.html` (interactive) and `.svg` next to the checkpoints
they were redrawn from, one mean±std pair per metric. They are regenerated from
the `eval_history` stored inside each `.pth`, so plots are never the only copy of
a result.

### Retraining: what to delete, and when

**Adding trials to an existing study — delete nothing.** The search is
resume-aware: it counts the trials already in the `.db` and runs only
`n_trials - done`. Re-running `python main.py GRU` on a finished 50-trial study
prints *"50 of 50 trials already done, nothing to run"* and falls straight
through to the report. `--trials 70` runs 20 more, numbered 50–69, keeping what
the sampler already learned. An interrupted study is resumed the same way, and a
trial that was killed mid-run has its parameters re-queued.

**Starting a genuinely fresh search — delete the whole `hpo/` directory**, not
just the database:

```bash
rm -rf agents/pretrained_model_GRU/hpo          # full BPTT
rm -rf agents/pretrained_model_GRU_tbptt8/hpo   # or one truncated length
```

Deleting only the `.db` is the tempting half-measure, and it leaves three traps:
trial numbering restarts at 0, so the new `trial_0/` overwrites the old one and
the tree silently mixes two studies; `hpo_sampler_*.pkl` survives, so the "fresh"
search resumes the old sampler's state instead of rebuilding it from `seed_hpo`
and is no longer reproducible; and `best_trial/` survives, so every report and
`watch.py --hpo` keeps serving the previous winner until some new trial completes
and replaces it.

**You must also start fresh after changing `seed_list`, `n_iterations`,
`worker_steps` or the env.** The study records the first two and only *warns*
that its older trials were pruned against different units — it will not stop you.
`worker_steps` doubles as the env's `max_steps` and therefore rescales the
reward, so every score already in the database means something else.

**Hand-picked runs have nothing to delete.** `main_no_hpo.py` overwrites
`no_hpo/` in place; the directory does not separate runs by setting, so move
aside a checkpoint you want to keep before re-running.

### The viewer

`watch.py` opens a pygame window showing the full maze, the agent's 7×7
observation, its action distribution and its value estimate, replayed over the
fixed eval mazes. It is the quickest way to see *whether* a policy detours to
the cue rather than inferring it from numbers.

```
SPACE pause    ← → step within an episode    P/N previous/next maze
R replay       A auto-advance the whole eval set    Q quit
```

Naming a run that was never trained is the ordinary way to mistype these
commands, so a missing checkpoint prints the path it looked for rather than a
traceback.

---

## Layout

```
config/       hyperparameters + the project's toolbox
  config.py         Config: every shared PPO/env/eval/HPO knob. Abstract.
  config_{mlp,gru}.py
                    one per encoder; sets feature_extractor + its architecture
                    knobs and appends them to the shared search space
  config_no_hpo.py  ConfigNoHPO: everything hand-picked, search_space = []
  helper.py         Helper: env builders, checkpoint I/O, batch-size probing,
                    Optuna persistence, plotting, the pygame viewer, Timing,
                    StartInCueView, SequenceDataset
models/
  feature_extractor.py   MLP / GRU — both map
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
agents/pretrained_model_<ENCODER>[_tbptt<L>]/
    hpo/trial_<n>/     one Optuna trial (gitignored)
    hpo/best_trial/    the winner, copied out so trial_*/ can be deleted
    no_hpo/            hand-picked runs
```

The `_tbptt<L>` suffix covers **both** `hpo/` and `no_hpo/`, so full BPTT keeps
the plain `pretrained_model_GRU/` it always had and each truncated length gets a
complete sibling tree that cannot overwrite the baseline it exists to be
compared against.

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
so both encoders search identical ranges — the widths a study settles on are
then comparable across the ablation. `make_config("GRU", tbptt_length=L)` returns
the *tuned* config; `ConfigNoHPO("GRU")` is the hand-picked one, which is why
`watch.py` branches between them.

**Encoders.** Both take the same input and return the same shape, so `Network`
and `PPOAgent` never branch on which one is in use. `flatten_obs` one-hots the
observation's 3 channels into 980 features first. Both act one step at a time
during rollout collection and see whole sequences during the PPO update.

**Agent.** `PPOAgent.sample()` collects a rollout across `n_workers` parallel
envs; `gae()` computes advantages; `split_pad_mask()` cuts each worker's stream
at episode boundaries — and at `tbptt_length`, when one is set — then pads to
rectangles; `learn()`
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
- **No dropout, anywhere.** PPO compares rollout log-probs against log-probs
  recomputed during the update; dropout makes those differ on identical inputs
  for reasons unrelated to learning, and corrupts the ratio. Neither encoder has
  any, and none should be added.
- **`async_envs`** picks `AsyncVectorEnv` (separate processes, default) vs
  `SyncVectorEnv`. It must be `False` in notebooks and any script without an
  `__main__` guard, because macOS spawns subprocesses.
- **Timing is first-class.** `PPOAgent` only names phases
  (`with self.timing.phase("sample"):`); all arithmetic lives in `Timing`.
  Adding a phase means adding a key to `Timing.ROWS` — an untimed-but-unlisted
  phase accumulates silently and just isn't printed.
- **The tbptt ablation truncates only the *backward* pass.**
  `config.tbptt_length` (`"max"`, or an int from `--tbptt L`) is read by
  `split_pad_mask`, which cuts each worker's stream at episode boundaries *and*
  every `L` steps. A chunk starting mid-episode is seeded from the hidden state
  the rollout recorded at that timestep, so the encoder still sees the whole
  history going forward — only the gradient's reach shrinks. It is never
  searched: each length is a separate study in its own directory, so no sampler
  or pruner ever ranks one length against another, and the lengths are compared
  afterwards by reading the studies' `final_*.json`. Shorter is not automatically
  cheaper — `L=1` measured *slower* than full BPTT, because the update then runs
  many more minibatches of length-1 sequences.

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

- [x] Run every arm over the five seeds in `seed_list` — no more single-seed claims
- [x] Run MLP and GRU at the same settings (the encoder ablation)
- [x] Optuna study per encoder, one per truncated-BPTT length
- [ ] Fill in the middle of the tbptt sweep (`L = 4`, and something between 8 and full)
- [ ] Rewrite *The task* / *Results* above against `MiniGrid-MemoryS17Random-v0`
