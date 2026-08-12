"""
Hyperparameter search: one Optuna study per encoder, resumable and pruned.
main.py builds a Config and calls HPOPPO.hpo() then .final(); this class
trains each trial's runs via PPOAgent and scores them via config.hpo_objective.
"""

import os

import numpy as np
import optuna

from agents.ppo import PPOAgent
from config import make_config


class HPOPPO:
    """An Optuna study over one encoder: draws params, trains a PPOAgent per seed, scores them."""

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger if logger is not None else config.build_logger()

        self.feature_extractor = config.feature_extractor
        self.name_env = config.name_env
        self.seed_list = config.seed_list
        self.n_trials = config.n_trials

        # every make_config() below has to repeat it: the length decides the
        # directory this study writes to, and the run it trains. Config stores
        # "max" where the constructor wants None, so keep both forms -- one to
        # report, one to pass on.
        self.tbptt_length = config.tbptt_length
        self._length_arg = None if self.tbptt_length == "max" else self.tbptt_length

        # all three parsed from the one config string, so they cannot drift
        self.hpo_objective = config.hpo_objective
        self.hpo_metric = config.hpo_metric
        self.score_name = config.score_name

        # hpo/ must exist before optuna gets a sqlite path inside it -- it
        # won't create the parent directory itself
        config.build_hpo_dir()

        # resuming without the pickled sampler would re-explore instead of
        # exploiting what the finished trials already showed
        self.sampler = config.load_sampler()
        if self.sampler is None:
            self.sampler = optuna.samplers.TPESampler(seed=config.seed_hpo)
            self.logger.info(f"new TPE sampler, seed_hpo={config.seed_hpo}")
        else:
            self.logger.info(f"resumed the sampler from {config.path_hpo_sampler}")

        # what the knobs mean and why none is left at optuna's default lives in
        # helper.build_pruner. What this file owns is the step they are counted
        # in: one seed, i.e. the index into seed_list (see objective()). A bad
        # draw is killed BETWEEN two runs, never inside one -- so the finest
        # granularity is one whole seed, and PPOAgent needs no hook for optuna.
        self.pruner = config.build_pruner()

        self.study = optuna.create_study(
            study_name=config.name_study,
            storage=config.path_hpo_db,
            direction=config.hpo_direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True,  # the whole resume mechanism
        )

        # A step is a seed index and the value at it is one finished run, so
        # three things have to hold for the median at step s to mean anything:
        # every trial counted its steps in seeds, ran the same seeds in the
        # same order, and trained each of them equally long. Break any one and
        # step s compares things that are not the same measurement -- optuna
        # cannot notice, so they are recorded on the study and checked here.
        #
        # pruner_step_unit is what catches a study from before this scheme:
        # those trials reported once per training ITERATION, at steps up to
        # n_iterations * len(seed_list), so their step 0 is an UNTRAINED network
        # and sits in the median every new trial is judged against. A missing
        # marker on a study that already has trials means exactly that; on a
        # study with no trials yet it just means nobody has written it.
        existing = len(self.study.get_trials(deepcopy=False)) > 0

        for name, current in (
            ("pruner_step_unit", "seed"),
            ("seed_list", list(self.seed_list)),
            ("n_iterations", int(config.n_iterations)),
        ):
            recorded = self.study.user_attrs.get(name)
            if recorded is None and not existing:
                self.study.set_user_attr(name, current)
            elif recorded is None:
                self.logger.info(
                    f"WARNING: this study already holds trials but never "
                    f"recorded {name}, so it predates the current pruning "
                    f"scheme. Its trials' intermediate values are not in the "
                    f"units this run reports, and the pruner will compare them "
                    f"anyway. Start a fresh study."
                )
                # the three are written together, so one missing means all of
                # them are: saying it once is enough
                break
            elif recorded != current:
                self.logger.info(
                    f"WARNING: this study's earlier trials were pruned against "
                    f"{name}={recorded}, but config says {current}. Their "
                    f"intermediate values mean something different from this "
                    f"run's, so the pruner will compare trials that are not "
                    f"comparable. Start a fresh study, or put {name} back to "
                    f"{recorded}."
                )

    # ---- one training run ----
    def run_split(self, seed, params, trial_number):
        """Train a fresh PPOAgent at these params/seed, checkpoint it, return
        its last-iteration scores. The run is never cut short: pruning happens
        in objective(), between two calls to this method."""
        config = make_config(self.feature_extractor, tbptt_length=self._length_arg)

        # before PPOAgent is built: the agent reads lr/wd and the architecture
        # sizes into the model at construction and never consults config again
        config.apply_params(params)

        config.dir_pretrained_model = config.build_hpo_trial_dir(trial_number)

        # one eval mode for the whole run, so switching hpo_objective cannot
        # silently switch the measuring instrument too
        deterministic = config.eval_deterministic

        agent = PPOAgent(config, seed=seed)
        try:
            agent.train_agent(logger=self.logger)
            curve = agent.eval_history
            last = curve[-1]

            return {
                "seed": seed,
                "iteration": int(last["iteration"]),
                "success_rate": float(last["success_rate"]),
                "return_mean": float(last["return_mean"]),
                # spread over eval episodes within this run -- NOT what
                # mean_minus_std subtracts, see helper.aggregate_scores
                "return_std_episodes": float(last["return_std"]),
                "deterministic": bool(deterministic),
                "eval_history": curve,
            }
        finally:
            # released even on raise/interrupt -- leaked envs across many
            # trials can exhaust file descriptors
            agent.close()

    def plot_trial(self, trial_number):
        """Draw one trial's curves from the checkpoints it just wrote. Called
        per trial rather than only at the end, so a study that is pruned,
        crashed or ctrl-c'd still leaves a figure for every trial it ran."""
        return self.config.plot_eval_curves(
            self.config.dir_hpo_trial(trial_number), logger=self.logger
        )

    # ---- the search ----
    def hpo(self):
        """Run (or resume) the study. Returns the best trial, or None."""

        def objective(trial):
            # exported first so an interrupted study still leaves a csv of
            # how far it got
            self.config.csv_study_export(self.study, self.config.path_hpo_csv)

            params = self.config.suggest_from_search_space(trial)

            # saved right after the draw, not after the trial, so a crash
            # mid-trial doesn't also lose the sampler state
            self.config.save_sampler(self.sampler)

            self.logger.info("")
            self.logger.info(f"TRIAL {trial.number}  {params}")

            scores = []
            per_seed = {metric: [] for metric in ("return_mean", "success_rate")}

            for index_split, seed in enumerate(self.seed_list):
                self.logger.info(
                    f"trial {trial.number}, seed {seed} "
                    f"({index_split + 1}/{len(self.seed_list)})"
                )

                result = self.run_split(seed, params, trial_number=trial.number)
                scores.append(result[self.hpo_metric])

                # four numbers, rewritten after every seed so a pruned trial
                # carries them too. These are the only user attrs, and so the
                # only non-param columns csv_study_export writes -- the per-seed
                # values stay in the log, the curves in the checkpoints.
                for metric, values in per_seed.items():
                    values.append(result[metric])
                    trial.set_user_attr(metric, float(np.mean(values)))
                    trial.set_user_attr(f"{metric}_std", float(np.std(values)))

                # between seeds, not inside train_agent, so optuna stays out of
                # the agent. The running cross-seed aggregate, not this seed's
                # raw metric: it is the same quantity the trial's final value is
                # built from, over the same number of seeds for every trial at
                # this step, so the pruner ranks on exactly what the study
                # ranks on. At step 0 the two coincide anyway -- the std of one
                # value is 0, so mean-minus-std of one seed IS that seed.
                running = self.config.aggregate_scores(scores)
                trial.report(running, step=index_split)

                if trial.should_prune():
                    self.logger.info(
                        f"trial {trial.number} pruned after seed {seed} "
                        f"({index_split + 1}/{len(self.seed_list)} seeds run, "
                        f"{self.hpo_metric} {result[self.hpo_metric]:.3f}, "
                        f"running {self.score_name} {running:.3f})"
                    )
                    # before the raise: the seeds that did finish still wrote
                    # checkpoints, and a pruned trial's curve is why it lost
                    self.plot_trial(trial.number)
                    raise optuna.TrialPruned()

            self.plot_trial(trial.number)

            value = self.config.aggregate_scores(scores)
            self.logger.info(
                f"TRIAL {trial.number} done: {self.score_name} {value:.3f}   "
                f"per-seed {self.hpo_metric} over {self.seed_list} "
                f"-> {[round(s, 3) for s in scores]}"
            )
            return value

        self.config.hpo_optimize(
            study=self.study,
            n_trials=self.n_trials,
            objective=objective,
            logger=self.logger,
            kind_training=self.feature_extractor,
        )

        best = self.config.summary_hpo(
            self.logger, self.study, self.config.path_hpo_csv
        )

        # copied out of trial_<n>/ so it survives deleting the other trials
        self.config.copy_best_trial(self.study, self.logger)

        # a refresh, not the only pass: plot_trial already drew each trial as
        # it finished. This still redraws them, so trials from a study resumed
        # from before per-trial plotting also get a figure, plus best_trial/.
        self.config.plot_hpo(logger=self.logger)

        return best

    # ---- the number to report ----
    def score_saved(self, seed_index, seed):
        """Load one winning-trial checkpoint (no training) and score it in both eval modes, or None if missing."""
        # fresh config, same reason as run_split: nothing leaks to the next seed
        config = make_config(self.feature_extractor, tbptt_length=self._length_arg)

        # the same two things watch.py sets to find this path
        path = config.select_run(trial="best", seed_index=seed_index)
        if not os.path.exists(path):
            self.logger.info(f"seed {seed}: no checkpoint at {path}, skipped")
            return None

        # the architecture the file was trained with -- a fresh make_config()
        # describes a different network. Read from the checkpoint, so this is
        # right even if optuna no longer has a record of the trial.
        config.apply_params(config.checkpoint_params(path))

        agent = PPOAgent(config, seed=seed)
        try:
            # validates encoder/widths/env/force_cue_visible against this
            # config, raises rather than silently loading a mismatched file
            checkpoint = config.load_model(agent.model, path)

            curve = list(checkpoint.get("eval_history") or [])
            if not curve:
                self.logger.info(f"seed {seed}: {path} holds no eval_history, skipped")
                return None

            last = curve[-1]

            # read from the file, not this config: config.eval_deterministic
            # may have changed since the study ran
            trained_deterministic = bool(
                checkpoint.get("eval_deterministic", config.eval_deterministic)
            )

            # the one mode not already in the curve
            fresh = agent.evaluate(deterministic=not trained_deterministic)

            if trained_deterministic:
                argmax, sampled = last, fresh
            else:
                sampled, argmax = last, fresh

            return {
                "seed": seed,
                "iteration": int(last["iteration"]),
                # same keys/numbers the trial was ranked on, so recomputing
                # the score from these reproduces study.best_value exactly
                "success_rate": float(sampled["success_rate"]),
                "return_mean": float(sampled["return_mean"]),
                "return_std_episodes": float(sampled["return_std"]),
                "timeout_rate": float(sampled["timeout_rate"]),
                "length_mean": float(sampled["length_mean"]),
                # prefixed argmax_*, reported only, never scored
                "argmax_success_rate": float(argmax["success_rate"]),
                "argmax_return_mean": float(argmax["return_mean"]),
                "argmax_return_std_episodes": float(argmax["return_std"]),
                "argmax_timeout_rate": float(argmax["timeout_rate"]),
                "argmax_length_mean": float(argmax["length_mean"]),
                "curve_deterministic": trained_deterministic,
                "path": path,
                "eval_history": curve,
            }
        finally:
            agent.close()

    def final(self):
        """Report the winning trial's saved runs, training nothing. Writes the final JSON and curve plots."""
        try:
            best = self.study.best_trial
        except ValueError:
            self.logger.info("no completed trial -- nothing to report")
            return None

        # re-copied, not assumed: --final-only skips hpo(), and a resumed study
        # can crown a new winner over an old best_trial/.
        directory = self.config.copy_best_trial(self.study, self.logger)
        if directory is None:
            self.logger.info("nothing to report")
            return None

        self.logger.info("")
        self.config.print_separate_lines(self.logger)
        self.logger.info(
            f"FINAL: reporting {self.feature_extractor} trial {best.number} "
            f"over seeds {self.seed_list} -- loaded from {directory}, not retrained"
        )
        self.logger.info(f"  {best.params}")

        results = [
            result
            for index, seed in enumerate(self.seed_list)
            if (result := self.score_saved(seed_index=index, seed=seed)) is not None
        ]
        if not results:
            self.logger.info("no checkpoint could be scored -- nothing to report")
            return None

        # every metric summarised, in both eval modes -- which are NOT
        # comparable to each other, see score_saved. `skip` holds the labels
        # and lists that don't belong in a mean; they stay raw in "results".
        summary = {}
        skip = {"seed", "path", "curve_deterministic", "iteration", "eval_history"}
        for key in sorted(set(results[0]) - skip):
            values = [r[key] for r in results]
            summary[key] = {
                "mean": float(np.mean(values)),
                # spread across seeds -- not to be confused with
                # return_std_episodes (spread across episodes in one run)
                "std_across_seeds": float(np.std(values)),
                "per_seed": values,
            }

        # recomputed from the loaded runs; should equal best.value exactly
        # since these are the same runs the study ranked (checked below).
        score = self.config.aggregate_scores([r[self.hpo_metric] for r in results])

        report = {
            "feature_extractor": self.feature_extractor,
            "name_env": self.name_env,
            "objective": self.hpo_objective,
            "metric": self.hpo_metric,
            "aggregation": self.config.hpo_aggregation,
            "hpo_lambda": getattr(self.config, "hpo_lambda", 1.0),
            "tbptt_length": self.tbptt_length,
            "score_name": self.score_name,
            "score": score,
            "best_value_in_study": best.value,
            "retrained": False,
            "best_trial": best.number,
            "best_params": best.params,
            "seed_list": self.seed_list,
            "seeds_scored": [r["seed"] for r in results],
            "eval_seed": self.config.eval_seed,
            "n_eval_episodes": self.config.n_eval_episodes,
            # so the json is readable without knowing what config.py said that day
            "n_iterations": self.config.n_iterations,
            "n_iterations_report": self.config.n_iterations_report,
            "scored_at_iteration": results[0]["iteration"],
            # read off the checkpoint; the other mode is measured fresh
            "curve_deterministic": results[0]["curve_deterministic"],
            "trials_deterministic": self.config.eval_deterministic,
            "results": results,
            "summary": summary,
        }

        path = os.path.join(directory, f"final_{self.config.name_hpo}.json")
        self.config.save_json(path, report)

        self.config.plot_eval_curves(directory, name="best_trial", logger=self.logger)

        curve_mode = "argmax" if results[0]["curve_deterministic"] else "sampled"
        other_mode = "sampled" if results[0]["curve_deterministic"] else "argmax"

        self.logger.info("")
        self.logger.info(
            f"FINAL  {self.feature_extractor} on {self.name_env}   "
            f"(trial {best.number}, {self.config.n_eval_episodes} episodes "
            f"from eval_seed {self.config.eval_seed})"
        )
        self.logger.info(
            f"       scored at iteration {results[0]['iteration']} "
            f"of {self.config.n_iterations}, "
            f"{len(results[0]['eval_history'])} evaluations kept per seed"
        )

        # two separate blocks, not one six-column table: different
        # measurements of the same weights, must not be read across
        self.logger.info("")
        self.logger.info(
            f"FINAL ({curve_mode})   the run's own last evaluation "
            f"-- what the study ranked on"
        )
        self.logger.info(
            f"{'seed':>8} {'return':>8} {'success':>9} {'timeout':>9} {'length':>8}"
        )
        for r in results:
            self.logger.info(
                f"{r['seed']:>8} {r['return_mean']:>8.3f} "
                f"{r['success_rate']:>9.3f} {r['timeout_rate']:>9.3f} "
                f"{r['length_mean']:>8.1f}"
            )
        for key in ("return_mean", "success_rate"):
            s = summary[key]
            self.logger.info(
                f"{key:>14}  mean {s['mean']:.3f}  "
                f"std_across_seeds {s['std_across_seeds']:.3f}"
            )

        self.logger.info("")
        self.logger.info(
            f"FINAL ({other_mode})   the same weights, re-evaluated -- "
            f"reported only, never scored"
        )
        self.logger.info(
            f"{'seed':>8} {'return':>8} {'success':>9} {'timeout':>9} {'length':>8}"
        )
        for r in results:
            self.logger.info(
                f"{r['seed']:>8} {r['argmax_return_mean']:>8.3f} "
                f"{r['argmax_success_rate']:>9.3f} "
                f"{r['argmax_timeout_rate']:>9.3f} "
                f"{r['argmax_length_mean']:>8.1f}"
            )
        for key in ("argmax_return_mean", "argmax_success_rate"):
            s = summary[key]
            self.logger.info(
                f"{key:>14}  mean {s['mean']:.3f}  "
                f"std_across_seeds {s['std_across_seeds']:.3f}"
            )

        # the same closing table main_no_hpo.py ends with, so a tuned run and a
        # hand-picked one are read the same way. The two blocks above stay --
        # they carry timeout/length and keep the eval modes visibly apart.
        self.config.log_seed_summary(
            self.logger,
            [
                self.config.seed_result_row(
                    r["seed"],
                    sampled={
                        "return_mean": r["return_mean"],
                        "success_rate": r["success_rate"],
                    },
                    argmax={
                        "return_mean": r["argmax_return_mean"],
                        "success_rate": r["argmax_success_rate"],
                    },
                )
                for r in results
            ],
            header=(
                f"{self.feature_extractor}  trial {best.number}  "
                f"over {len(results)} seed(s)"
            ),
        )

        self.logger.info("")
        self.logger.info(f"  SCORE  {self.score_name} = {score:.4f}")
        # these are the winning trial's own runs, so this carries the
        # winner's curse -- not an independent estimate
        if abs(score - best.value) > 1e-9:
            self.logger.info(
                f"  WARNING: the study recorded {best.value:.4f} for trial "
                f"{best.number}, but its saved runs score {score:.4f}. "
                f"best_trial/ may not hold what the database describes."
            )
        else:
            self.logger.info(
                f"  same as the study's best_value ({best.value:.4f}) by "
                f"construction -- these ARE the winning trial's runs, so the "
                f"number is selection-biased upward and is not an independent "
                f"estimate of what these params are worth."
            )
        self.logger.info(f"written to {path}")

        return report
