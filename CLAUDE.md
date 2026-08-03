# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An ablation study of PPO with four interchangeable feature extractors (MLP, LSTM, GRU, Transformer) on MiniGrid's partially-observable memory tasks (default env: `MiniGrid-DoorKey-8x8-v0`, originally targeting `MiniGrid-MemoryS*-v0`). The recurrent/attention encoders exist to recover performance under state aliasing (the agent must remember something seen earlier); the question the project asks is whether a Transformer encoder does that better than LSTM/GRU. Hyperparameters are tuned per encoder with an Optuna study (`main.py`), and there's also a "hand-picked" no-search path (`main_no_hpo.py`) for baselines/smoke tests.

## Environment setup

- Python 3.11, conda env named `rl_project`. **A fresh shell here starts in conda `base`, which has no torch** — activate first (`conda activate rl_project`) or prefix commands with `conda run -n rl_project`. Every command below assumes that env.
- `requirements.txt` is the authoritative dependency list. `environment.yml` is gitignored and stale (missing `optuna`, `torchinfo`, `pandas`, `plotly`) — don't treat it as the spec.
- `optuna`/`joblib` are imported *inside* the `Helper` methods that need them, and `torchinfo`/`pandas`/`plotly` are imported lazily with a graceful skip, so a checkout missing them still trains, watches and scores — only the study, model summary, CSV export and plots drop out.

## Common commands

```bash
# Hyperparameter search (Optuna study, resumable — reruns continue an interrupted study)
python main.py MLP|LSTM|GRU|TRANSFORMER [--trials N] [--search-only] [--final-only]

# Hand-picked run (no Optuna), values from config/config_no_hpo.py
python main_no_hpo.py [MLP|LSTM|GRU|TRANSFORMER] [--report-only]

# Watch a saved checkpoint play in a pygame window
python watch.py [MODEL] [steps_per_sec] [--hpo] [--seed INDEX] [--trial best|N|final]

# Self-contained shape/forward-pass/memory/causality checks for the encoders
python models/feature_extractor.py
python models/model.py
```

There is no pytest/unittest suite and no linter or formatter configured in this repo. The closest things to tests are the `if __name__ == "__main__":` demo blocks in `models/feature_extractor.py` and `models/model.py` (both run in a few seconds and are the fastest way to check an encoder change), plus the throwaway exploration scripts under `test_enviroment/` and `no_need/` (both gitignored — not part of the shipped code, not meant to be maintained).

## Architecture

### Config layer (`config/`)

- `Config` (`config/config.py`) — abstract base holding every shared PPO/env/eval/HPO hyperparameter. Subclasses only implement `_configure_model()`.
- `ConfigMLP` / `ConfigLSTM` / `ConfigGRU` / `ConfigTransformer` (`config/config_*.py`) — one per encoder; each sets `feature_extractor` and appends its own architecture knobs to the shared `search_space` (same `low`/`high`/`step` ranges across all four, so the widths a study settles on are comparable).
- `ConfigNoHPO` (`config/config_no_hpo.py`) — every hyperparameter hand-picked instead of searched (`search_space = []`); used by `main_no_hpo.py`; writes checkpoints to `no_hpo/` instead of `hpo/`.
- `make_config(name)` in `config/__init__.py` turns a CLI model name into the right *tuned* `Config` subclass. It never returns `ConfigNoHPO` — that one takes the encoder as a constructor argument instead (`ConfigNoHPO("GRU")`), which is why `watch.py` branches `make_config(model) if args.hpo else ConfigNoHPO(model)`.
- `Helper` (`config/helper.py`, ~2100 lines) — large mixin `Config` inherits from; it's the rest of the project's toolbox, not more hyperparameters: env/vector-env/encoder builders, checkpoint save/load (refuses to load a checkpoint whose env/encoder/widths/`force_cue_visible` don't match the live config), `probe_batch_size`/`run_with_batch_size_fallback`, Optuna study/sampler persistence and trial-directory bookkeeping, HPO and learning-curve plotting, and the pygame `watch_agent` viewer used by `watch.py`. Also defines `Timing` (below), `StartInCueView` (an env wrapper) and `SequenceDataset` (feeds `PPOAgent`'s `DataLoader`).

### Models (`models/`)

- `feature_extractor.py` — four interchangeable encoders (`MLP`, `LSTM`, `GRU`, `Transformer`), all mapping `(batch, seq_len, 7, 7, 3)` MiniGrid observations to `(batch, seq_len, hidden_size)`. `flatten_obs` one-hots the observation's 3 channels into 980 features before anything else runs. The Transformer acts one step at a time (`seq_len=1`) just like the recurrent encoders, so during rollout collection it currently has no history to attend over and behaves like an MLP — it only sees real sequences during the PPO update.
- `model.py` — `Network` wraps one encoder plus a linear actor head and a linear critic head: shared encoder, two heads, trained jointly by one optimizer.

### Agents (`agents/`)

- `ppo.py` — `PPOAgent`, the whole PPO pipeline: `sample()` (vectorized rollout collection across `n_workers` envs), `gae()`, `split_pad_mask()` (cuts each worker's `worker_steps` at episode boundaries and pads to rectangles for truncated-BPTT), `clip_loss()`/`minibatch_loss()` (the clipped-surrogate + value + entropy loss), `learn()`/`train()` (one iteration), `evaluate()` (sampled or argmax, on private envs), `train_agent()` (the full loop: minibatch-size probe, logging, periodic eval, timing tables, final checkpoint).
- `hpo_ppo.py` — `HPOPPO`, wraps `PPOAgent` in an Optuna study. `hpo()` runs/resumes the search: one trial draws hyperparameters and trains a fresh `PPOAgent` per seed in `config.seed_list`, pruned *between* seeds (not mid-training) via `MedianPruner`. `final()` does **not** retrain — it reloads the winning trial's already-saved checkpoints from `best_trial/` and reports both eval modes side by side. (`main.py`'s `--final-only` help text still says "retrain"; it doesn't.)

### Entry points

- `main.py` — HPO entry point.
- `main_no_hpo.py` — hand-picked entry point; trains (or, with `--report-only`, just reports) every seed in `ConfigNoHPO.seed_list`.
- `watch.py` — loads one saved checkpoint (no training) and plays it in a pygame window. `--hpo` switches between the tuned run (`hpo/best_trial/` by default, or `--trial N`/`final`) and the hand-picked one (`no_hpo/`); `--seed` is an **index** into `seed_list`, not a raw seed value — the same index means a different actual seed in HPO vs. hand-picked mode.

## Cross-cutting things worth knowing before touching code

- **Rollout length is the env's time limit, in both directions.** `worker_steps` (T) defaults to `Helper.env_max_steps` (read off a throwaway env, not hardcoded), and `build_env` then passes `max_steps=worker_steps` back into `gym.make`. So an episode can never outlive one rollout, and MiniGrid's success reward (`1 - 0.9 * step_count / max_steps`) moves with `worker_steps` — changing T silently rescales the reward and therefore every number the study compares.
- **`eval_deterministic` is one setting for an entire run**: it decides whether the learning curve, the checkpoint's stored eval, and the HPO score are *all* measured by sampling or *all* by argmax. It's read once and threaded through everywhere rather than passed per-call, specifically so switching it can't silently make two numbers incomparable. The reporting paths (`HPOPPO.score_saved`, `main_no_hpo.report_saved`) read the mode *off the checkpoint*, then measure the other mode fresh, and print the two in separate blocks — they are not comparable across.
- **Minibatch size is resolved once per run, then defended.** `mini_batch_size` in `Config` is a list of fallback candidates, largest first. `train_agent` calls `Helper.probe_batch_size` *before* iteration 0, which runs a forward/backward/step on an all-zero worst-case batch against a deepcopy of the model (early rollouts pack fewer sequences than later ones, so probing on real iteration-0 data would underestimate). `Helper.run_with_batch_size_fallback` then wraps each real update as a retry-on-OOM safety net. The resolved size can differ machine to machine, so it's logged and written into the checkpoint.
- **Timing is first-class** (`Timing` in `config/helper.py`). `PPOAgent` only names phases (`with self.timing.phase("sample"):`); all arithmetic/formatting lives in `Timing`. Two sets of totals are kept — `window` (since the last `report()`) and `run` (whole seed) — and `report()` folds one into the other so they never double-count. Hot inner loops pass `sync=False` because a `cuda.synchronize()` would cost more than what it measures; `evaluate()` is timed *outside* the `"iteration"` clock, since it's the cost of measuring the policy, not of training it. Adding a phase means adding a key to `Timing.ROWS`/`ROWS_EVAL` — an untimed-but-unlisted phase accumulates silently and just isn't printed.
- **`hpo_objective` is one string**, `<metric>_<center>_<spread>` (e.g. `return_mean_minus-std`, `success-rate_median_minus-iqr`), parsed by `helper.parse_hpo_objective`, which raises on an unrecognized field rather than defaulting silently. `center` and `spread` are always aggregated **across seeds**, never across eval episodes within one run — the within-run spread of a bimodal return is a function of the success rate itself and would penalize partial success. (`return_std_episodes` in reports is the within-run one; `std_across_seeds` is the scored one.)
- **A tuned checkpoint carries its own architecture.** `save_model` writes `searched_params()` into the file, so reloading a trial drawn at `hidden_size=384` means `config.apply_params(config.checkpoint_params(path))` *before* building `PPOAgent` — the agent reads sizes into the model at construction and never consults the config again. Every reload path does this; a new one must too. `apply_params` raises on a name that isn't already an attribute of the config, so a new `search_space` entry requires the attribute to exist on the config first.
- **Checkpoint layout**: `agents/pretrained_model_<ENCODER>/`, split into `hpo/trial_<n>/`, `hpo/best_trial/` (the winner, copied out so `trial_*/` can be deleted), and `no_hpo/` (hand-picked runs). `config.select_run()` is the single place that resolves a seed index to a checkpoint path across all three; `HPOPPO`, `main_no_hpo.py`, and `watch.py` all go through it so they can't drift apart. Each `.pth` also carries its `eval_history`, which is what `plot_eval_curves` redraws from — a directory of checkpoints is a set of curves.
- **`p_drop` must stay `0.0`** on the Transformer. PPO compares rollout `log_probs` against `log_probs` recomputed during the update; dropout makes those differ on identical inputs for reasons unrelated to learning, corrupting the ratio. Same reason `n_heads` must divide `d_model` (`config_transformer.py` picks `step`/`choices` so every drawn combination does).
- **`async_envs`** picks `AsyncVectorEnv` (separate processes, the default) vs `SyncVectorEnv`. Async must be `False` in notebooks and any script without an `__main__` guard — macOS spawns subprocesses.
- **`config.tbptt_length` is dead** — set in `config/config.py`, read nowhere. `split_pad_mask` cuts on episode boundaries only; there is no fixed-length chunking. Don't assume the knob works.
- `logs/` holds one timestamped log file per command invocation (`build_logger`), mirroring everything printed to the terminal, with every hyperparameter dumped at the top.
