# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An ablation study of PPO with four interchangeable feature extractors (MLP, LSTM, GRU, Transformer) on MiniGrid's partially-observable memory tasks (default env: `MiniGrid-DoorKey-8x8-v0`, originally targeting `MiniGrid-MemoryS*-v0`). The recurrent/attention encoders exist to recover performance under state aliasing (the agent must remember something seen earlier); the question the project asks is whether a Transformer encoder does that better than LSTM/GRU. Hyperparameters are tuned per encoder with an Optuna study (`main.py`), and there's also a "hand-picked" no-search path (`main_no_hpo.py`) for baselines/smoke tests.

## Environment setup

- Python 3.11, conda env named `rl_project`.
- `pip install -r requirements.txt` is the authoritative dependency list (includes `optuna`, which `agents/hpo_ppo.py` imports unconditionally and `main.py` requires — `environment.yml` is missing it). `environment.yml` also lacks `torchinfo`, `pandas`, `plotly`, but those are imported lazily inside `config/helper.py` with a graceful skip if absent (model-summary/plotting features just no-op with a log message).
- Key deps: `torch`, `gymnasium`, `minigrid`, `optuna`, `pygame` (for `watch.py`'s viewer), `matplotlib`/`plotly` (HPO plots), `torchinfo` (optional model summary), `pandas` (optional CSV/table helpers).

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

There is no pytest/unittest suite and no linter or formatter configured in this repo. The closest things to tests are the `if __name__ == "__main__":` demo blocks in `models/feature_extractor.py` and `models/model.py`, plus the throwaway exploration scripts under `test_enviroment/` and `no_need/` (both gitignored — not part of the shipped code, not meant to be maintained).

## Architecture

### Config layer (`config/`)

- `Config` (`config/config.py`) — abstract base holding every shared PPO/env/eval/HPO hyperparameter. Subclasses only implement `_configure_model()`.
- `ConfigMLP` / `ConfigLSTM` / `ConfigGRU` / `ConfigTransformer` (`config/config_*.py`) — one per encoder; each sets `feature_extractor` and appends its own architecture knobs to the shared `search_space` (same `low`/`high`/`step` ranges across all four, so the widths a study settles on are comparable).
- `ConfigNoHPO` (`config/config_no_hpo.py`) — every hyperparameter hand-picked instead of searched (`search_space = []`); used by `main_no_hpo.py`; writes checkpoints to `no_hpo/` instead of `hpo/`.
- `make_config(name)` in `config/__init__.py` is how every entry point turns a CLI model name into the right `Config` subclass.
- `Helper` (`config/helper.py`) — large mixin `Config` inherits from; it's the rest of the project's toolbox, not more hyperparameters: env/vector-env/encoder builders, checkpoint save/load (refuses to load a checkpoint whose env/encoder/widths/`force_cue_visible` don't match the live config), the OOM-safe `run_with_batch_size_fallback`, Optuna study/sampler persistence and trial-directory bookkeeping, HPO plotting, and the pygame `watch_agent` viewer used by `watch.py`. Also defines `StartInCueView` (an env wrapper) and `SequenceDataset` (feeds `PPOAgent`'s `DataLoader`).

### Models (`models/`)

- `feature_extractor.py` — four interchangeable encoders (`MLP`, `LSTM`, `GRU`, `Transformer`), all mapping `(batch, seq_len, 7, 7, 3)` MiniGrid observations to `(batch, seq_len, hidden_size)`. `flatten_obs` one-hots the observation's 3 channels into 980 features before anything else runs. The Transformer acts one step at a time (`seq_len=1`) just like the recurrent encoders, so during rollout collection it currently has no history to attend over and behaves like an MLP — it only sees real sequences during the PPO update.
- `model.py` — `Network` wraps one encoder plus a linear actor head and a linear critic head: shared encoder, two heads, trained jointly by one optimizer.

### Agents (`agents/`)

- `ppo.py` — `PPOAgent`, the whole PPO pipeline: `sample()` (vectorized rollout collection across `n_workers` envs), `gae()`, `split_pad_mask()` (cuts each worker's `worker_steps` at episode boundaries and pads to rectangles for truncated-BPTT), `clip_loss()`/`minibatch_loss()` (the clipped-surrogate + value + entropy loss), `learn()`/`train()` (one iteration, with OOM fallback on minibatch size), `evaluate()` (sampled or argmax, on private envs), `train_agent()` (the full loop: logging, periodic eval, final checkpoint).
- `hpo_ppo.py` — `HPOPPO`, wraps `PPOAgent` in an Optuna study. `hpo()` runs/resumes the search: one trial draws hyperparameters and trains a fresh `PPOAgent` per seed in `config.seed_list`, pruned *between* seeds (not mid-training) via `MedianPruner`. `final()` does **not** retrain — it reloads the winning trial's already-saved checkpoints from `best_trial/` and reports both eval modes side by side.

### Entry points

- `main.py` — HPO entry point.
- `main_no_hpo.py` — hand-picked entry point; trains (or, with `--report-only`, just reports) every seed in `ConfigNoHPO.seed_list`.
- `watch.py` — loads one saved checkpoint (no training) and plays it in a pygame window. `--hpo` switches between the tuned run (`hpo/best_trial/` by default, or `--trial N`/`final`) and the hand-picked one (`no_hpo/`); `--seed` is an **index** into `seed_list`, not a raw seed value — the same index means a different actual seed in HPO vs. hand-picked mode.

### Cross-cutting things worth knowing before touching code

- `eval_deterministic` is one setting for an entire run: it decides whether the learning curve, the checkpoint's stored eval, and the HPO score are *all* measured by sampling or *all* by argmax. It's read once and threaded through everywhere rather than passed per-call, specifically so switching it can't silently make two numbers incomparable.
- `mini_batch_size` in `Config` is a list of fallback candidates, largest first; `Helper.run_with_batch_size_fallback` tries them in order and keeps the largest that doesn't OOM, once per run — the resolved size (which can differ machine to machine) is what gets logged and checkpointed.
- `hpo_objective` is one string, `<metric>_<center>_<spread>` (e.g. `return_mean_minus-std`, `success-rate_median_minus-iqr`), parsed by `helper.parse_hpo_objective`. `center` and `spread` are always aggregated **across seeds**, never across eval episodes within one run — the within-run spread of a bimodal return is a function of the success rate itself and would penalize partial success.
- Checkpoints live under `agents/pretrained_model_<ENCODER>/`, split into `hpo/trial_<n>/`, `hpo/best_trial/` (the winner, copied out so `trial_*/` can be deleted), and `no_hpo/` (hand-picked runs). `config.select_run()` is the single place that resolves a seed index to a checkpoint path across all three; `HPOPPO`, `main_no_hpo.py`, and `watch.py` all go through it so they can't drift apart.
- `logs/` holds one timestamped log file per command invocation (`build_logger`), mirroring everything printed to the terminal.
