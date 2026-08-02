"""
The hyperparameter search. One study per encoder, resumable, pruned.

NOT AN ENTRY POINT -- main.py is. This file holds the HPOPPO class; main.py
builds a config, hands it over and calls hpo() then final():

    python main.py GRU               run (or resume) the study, then retrain
    python main.py GRU --final-only  retrain at the best params only
    python main.py GRU --trials 10   a shorter study

The split is the same one the rest of the project uses: config/ decides what
to search and where it goes, agents/ppo.py knows how to train, and this knows
how to turn the two into a study. None of the three imports the others'
command line.

WHAT ONE TRIAL IS. A draw of hyperparameters from config.search_space, trained
once per seed in config.seed_list, and scored by config.hpo_objective -- one
string that names both the per-run metric and how the len(seed_list) of those
become one number, e.g. "return_mean_minus-std" or
"success-rate_median_minus-iqr". See helper.parse_hpo_objective. Three full
training runs per trial -- that is the expense, and it is the point: a single
PPO run on a sparse-reward MiniGrid task can land anywhere, so a trial scored
on one seed would mostly be measuring the seed.

WHY EVERY TRIAL USES THE SAME SEEDS. Deliberately not a seed derived from the
trial number or its params. Two trials then differ ONLY in hyperparameters,
so the difference between their scores is the thing being searched over and
not also the difference between two draws of mazes. The paired comparison is
what makes 30 trials worth anything at this noise level.

RESUMING IS THE DEFAULT, not a flag. The study lives in a sqlite file and the
sampler is pickled beside it, so re-running the same command continues where
it stopped:

    - budget is counted in COMPLETE + PRUNED trials, so a crash does not
      spend one
    - the interrupted trial's exact params are put back on the queue, so the
      run that was lost is the run that is retried
    - the pickled sampler carries what TPE has LEARNED; without it a resumed
      study re-draws its random startup trials and throws that away

See helper.hpo_optimize, which is where those three live.

WHAT IT WRITES, all under agents/pretrained_model_<ENC>/hpo/ :

    hpo_csv_<ENC>_<env>.csv      every trial, rewritten at each trial's start
    hpo_db_<ENC>_<env>.db        the study, so it can resume
    hpo_sampler_<ENC>_<env>.pkl  the TPE state, likewise

    trial_0/ trial_1/ ...        one directory per trial
        ppo_<seed>_<ENC>_<env>.pth       ONE PER SEED, curve included
        curve_<metric>_mean_std.html     + .svg
        curve_<metric>_median_iqr.html   + .svg

    best_trial/                  a byte copy of whichever trial won, and THE
                                 RESULT -- final() reports these very runs
        (the same files, plus best_params.json)
        final_<ENC>_<env>.json           every metric, per seed and summarised,
                                         in BOTH eval modes

TWO DIRECTORIES, ONE SHAPE. trial_<n>/ and best_trial/ each hold one set of
runs and nothing else, so the same reader covers both -- which is what lets
helper.plot_hpo plot them without being told which it has. <metric> is
config.hpo_metric, the quantity the study was ranked on.

THERE IS NO final/ AND NO RETRAIN. final() used to train the winning params
again from scratch, three more runs, into its own directory. Those runs used
the same encoder, env, seeds and hyperparameters as the winning trial, so they
produced another sample of a run that had already been made -- see final() for
what the second sample was worth and why it is gone.

The plots come from the CHECKPOINTS, not from the log: train_agent stores its
eval_history in the .pth, so a directory of them is a set of learning curves
over a shared x axis. config.plot_hpo() rebuilds every figure from files
already on disk, which is the fix for a study interrupted before it plotted.

Nothing is written outside hpo/. The sibling no_hpo/ belongs to
main_no_hpo.py, and the two never collide.
"""

import os

import numpy as np
import optuna

from agents.ppo import PPOAgent
from config import make_config


class HPOPPO:
    """A study over one encoder on one env.

    config : a Config subclass instance -- the encoder is read off it, and its
             search_space is what gets searched. It is used as a TEMPLATE only:
             each training run builds its own fresh config, because PPOAgent
             copies values out of one and train() writes the resolved
             mini_batch_size back into it.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger if logger is not None else config.build_logger()

        self.feature_extractor = config.feature_extractor
        self.name_env = config.name_env
        self.seed_list = config.seed_list
        self.n_trials = config.n_trials

        # THE OBJECTIVE, in its three readings. All derived from the one config
        # string, so they cannot drift apart:
        #
        #   hpo_objective  "success-rate_median_minus-iqr", what was asked for
        #   hpo_metric     "success_rate", the key looked up in a run's result
        #   score_name     "median_minus_1iqr(success_rate)", what to print
        self.hpo_objective = config.hpo_objective
        self.hpo_metric = config.hpo_metric
        self.score_name = config.score_name

        # hpo/ has to exist before optuna is handed a sqlite path inside it --
        # it will not create the parent directory for its own database
        config.build_hpo_dir()

        # THE SAMPLER IS LOADED IF THERE IS ONE. A fresh TPESampler(seed_hpo)
        # proposes the same first n_startup_trials random points every time, so
        # resuming without the pickle means re-exploring instead of exploiting
        # what the finished trials already showed. See config.path_hpo_sampler.
        self.sampler = config.load_sampler()
        if self.sampler is None:
            self.sampler = optuna.samplers.TPESampler(seed=config.seed_hpo)
            self.logger.info(f"new TPE sampler, seed_hpo={config.seed_hpo}")
        else:
            self.logger.info(f"resumed the sampler from {config.path_hpo_sampler}")

        # MedianPruner: stop a trial whose running mean is worse than the
        # median of the finished trials at the same step. "Step" here is a
        # SEED, not an iteration -- see objective() -- so the coarsest thing it
        # can do is kill a hopeless draw after its first seed instead of its
        # third, which is already a third of the study's cost.
        self.pruner = optuna.pruners.MedianPruner()

        self.study = optuna.create_study(
            study_name=config.name_study,
            storage=config.path_hpo_db,
            direction=config.hpo_direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True,  # this is the whole resume mechanism
        )

    # ------------------------------------------------------------------
    # one training run
    # ------------------------------------------------------------------
    def run_split(self, seed, params, trial_number):
        """Train once, at these params, from this seed. Returns its scores.

        A FRESH CONFIG EVERY CALL, never self.config. Two reasons, both of
        which have bitten this file: PPOAgent copies every value it needs out
        of the config in __init__, and train() writes the resolved
        mini_batch_size BACK into it -- so a shared config would carry one
        run's state into the next, and a trial that fell back to 32 would
        silently hold every later trial at 32.

        trial_number puts the checkpoint in hpo/trial_<n>/, and is REQUIRED:
        every run this class starts belongs to a trial, now that final() no
        longer trains anything. The number cannot go in the FILENAME instead:
        watch.py and load_model rebuild that from the config alone and have no
        trial number to put in it. The SEED is in the filename, which is why
        one trial directory holds one checkpoint per entry in seed_list rather
        than one file overwritten three times.

        ONE EVAL MODE FOR THE WHOLE RUN -- config.eval_deterministic, read by
        PPOAgent before training starts, so the learning curve and the run's
        closing number are the same kind of measurement. See below.

        THE SCORE IS THE LAST ITERATION. train_agent() evaluates on every
        report iteration and keeps them in agent.eval_history -- iterations
        0, 100, 200, 300, 400, 499 at the current n_iterations /
        n_iterations_report -- and this reads the LAST of them. Not the best
        one the run passed through: that is the maximum of six noisy
        measurements and biased upward the same way study.best_value is, and
        it would score a policy that no longer exists by the end of the run.
        The last entry is also the policy that was checkpointed, so the number
        the study ranks on and the weights on disk are the same thing.

        There is deliberately NO second evaluate() call here. Iteration
        n_iterations-1 already evaluated exactly these weights; with
        eval_deterministic=False a repeat would return a different number for
        the same policy, and the trial would be scored on something that
        appears nowhere in its own log.

        Returns {"seed", "iteration", "success_rate", "return_mean",
        "return_std_episodes", "deterministic", "eval_history"}. Everything
        but the last two is a scalar, which is what final()'s summary and
        objective()'s user attributes assume.
        """
        config = make_config(self.feature_extractor)

        # BEFORE PPOAgent, not after. The agent reads lr and wd into the
        # optimizer and hidden_size / d_model into the encoder it builds, and
        # never looks at the config again -- applied afterwards, the trial
        # would train the defaults and report them as the drawn params.
        config.apply_params(params)

        config.dir_pretrained_model = config.build_hpo_trial_dir(trial_number)

        # ONE EVAL MODE FOR THE WHOLE RUN. PPOAgent copies eval_deterministic
        # out of the config in __init__ and train_agent()'s periodic
        # evaluations read it, so the LEARNING CURVE and the run's CLOSING
        # NUMBER are the same kind of measurement.
        #
        # That is the whole point: success_rate and return_mean are then never
        # measured differently from one another. Previously the curve was
        # sampled while the endpoint was argmax, so switching hpo_objective
        # silently switched the measuring instrument too.
        deterministic = config.eval_deterministic

        agent = PPOAgent(config, seed=seed)
        try:
            agent.train_agent(logger=self.logger)

            # one entry per report iteration, in order. train_agent() measured
            # every one of them under the single mode set above, so nothing in
            # here was taken with a different instrument.
            curve = agent.eval_history

            # THE TRIAL'S SCORE. The policy the run ended with, which is the
            # policy that was saved -- see the docstring.
            last = curve[-1]

            return {
                "seed": seed,
                # which iteration the scalars below were measured at, recorded
                # so the json cannot be misread as "somewhere in the run"
                "iteration": int(last["iteration"]),
                # both measured under the SAME mode -- see above
                "success_rate": float(last["success_rate"]),
                "return_mean": float(last["return_mean"]),
                # the spread over EVAL EPISODES within this one run. Recorded
                # because it describes the run, but deliberately NOT what
                # mean_minus_std subtracts -- see helper.aggregate_scores.
                "return_std_episodes": float(last["return_std"]),
                # so the csv and the json say how these numbers were taken
                "deterministic": bool(deterministic),
                # the whole curve, carried out of the run so final()'s json
                # holds it. The only non-scalar here, which is why both
                # objective() and final() name it explicitly below.
                "eval_history": curve,
            }
        finally:
            # the W envs are released even if this raises or is interrupted --
            # thirty trials leaking sixteen envs each is the difference between
            # a study that finishes and one that runs the machine out of file
            # descriptors
            agent.close()

    # ------------------------------------------------------------------
    # the search
    # ------------------------------------------------------------------
    def hpo(self):
        """Run (or resume) the study. Returns the best trial, or None."""

        def objective(trial):
            # exported FIRST, at the start of every trial, so an interrupted
            # study still leaves a csv of how far it got
            self.config.csv_study_export(self.study, self.config.path_hpo_csv)

            params = self.config.suggest_from_search_space(trial)

            # pickled after the draw, not after the trial: a trial takes three
            # training runs and a crash inside one of them should not also cost
            # the sampler state the draw already updated
            self.config.save_sampler(self.sampler)

            self.logger.info("")
            self.logger.info(f"TRIAL {trial.number}  {params}")

            scores = []
            for index_split, seed in enumerate(self.seed_list):
                self.logger.info(
                    f"trial {trial.number}, seed {seed} "
                    f"({index_split + 1}/{len(self.seed_list)})"
                )

                result = self.run_split(seed, params, trial_number=trial.number)
                scores.append(result[self.hpo_metric])

                # EVERY metric is kept per seed, not only the one being
                # optimized: the csv then answers "what would the other
                # objective have chosen?" without re-running the study, and
                # the per-seed spread is what says whether a trial's score
                # means anything at all. All of them are the LAST ITERATION's
                # numbers -- the iteration whose weights were checkpointed.
                #
                # eval_history is skipped HERE and re-added below: it is a list
                # of dicts, and one of those per seed would put an unreadable
                # blob in every row of the csv.
                for key, value in result.items():
                    if key not in ("seed", "eval_history"):
                        trial.set_user_attr(f"{key}_seed_{seed}", value)

                # THE CURVE, flattened to one list per seed per metric. This
                # is what the ablation figure plots, and keeping it in the
                # study means a finished search can be re-plotted without
                # re-reading thirty trials' worth of checkpoints. Optuna
                # serialises user attributes as json, so lists of floats are
                # fine; lists of dicts would not be worth reading back.
                history = result["eval_history"]
                trial.set_user_attr(
                    "curve_iterations", [int(c["iteration"]) for c in history]
                )
                trial.set_user_attr(
                    f"curve_success_rate_seed_{seed}",
                    [float(c["success_rate"]) for c in history],
                )
                trial.set_user_attr(
                    f"curve_return_mean_seed_{seed}",
                    [float(c["return_mean"]) for c in history],
                )

                # PRUNING HAPPENS BETWEEN SEEDS. The running score is reported
                # with the seed index as the step, so a draw that is clearly
                # worse than the finished trials can be abandoned after one
                # training run instead of three. Doing it any finer would mean
                # threading the trial down into train_agent, which would put
                # optuna inside the agent -- the agent does not know what a
                # study is, and that is worth keeping.
                #
                # THE SAME aggregate_scores AS THE FINAL VALUE, so the number a
                # trial is pruned on and the number it is judged on are the
                # same kind of thing. At step 0 the std of one value is 0, so
                # every trial is equally optimistic there -- fair, because
                # MedianPruner only ever compares trials at the same step.
                running = self.config.aggregate_scores(scores)
                trial.report(running, step=index_split)
                if trial.should_prune():
                    self.logger.info(
                        f"trial {trial.number} pruned after seed {seed} "
                        f"(running {self.score_name} {running:.3f})"
                    )
                    raise optuna.TrialPruned()

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

        # the winner's checkpoints copied out of trial_<n>/, so they survive
        # deleting the other trial directories
        self.config.copy_best_trial(self.study, self.logger)

        # ONE PAIR OF FIGURES PER TRIAL, plus the winner's. Drawn from the
        # checkpoints, so this runs after copy_best_trial -- best_trial/ has
        # no .pth in it until then and there would be nothing to plot.
        #
        # AFTER the search rather than inside objective(), for two reasons: a
        # trial's plot is only complete once all its seeds have run, and a
        # pruned trial would otherwise be plotted from one seed and then never
        # updated. This also means a study interrupted with ctrl-c leaves no
        # plots -- re-run with --final-only, or call config.plot_hpo()
        # directly, and they are rebuilt from the files already on disk.
        self.config.plot_hpo(logger=self.logger)

        return best


    # ------------------------------------------------------------------
    # the number to report
    # ------------------------------------------------------------------
    def score_saved(self, seed_index, seed):
        """Load ONE saved run of the winning trial and score it in BOTH eval modes.

        Trains nothing. The weights are already on disk -- best_trial/ holds
        one checkpoint per seed, curve included -- so this rebuilds the network
        they belong to, loads them, and evaluates.

        TWO NUMBERS FOR ONE POLICY, and only one of them costs anything:

            SAMPLED   actions drawn from pi, the way training and every trial
                      measured. Already in the file: it is the last entry of
                      the run's own eval_history, so it is not re-measured --
                      re-running a sampled evaluation would give a DIFFERENT
                      number for the same weights, and then the score in the
                      report would be one that appears nowhere in the study.
            ARGMAX    the greedy policy. Fully reproducible (fixed mazes AND
                      no action sampling), which is what a results table
                      wants, and the one measurement made here.

        WHICH OF THE TWO CAME FROM THE CURVE DEPENDS ON HOW THE TRIAL WAS RUN.
        The checkpoint records its own eval_deterministic, so whichever mode
        the curve was NOT measured in is the one evaluated fresh. With the
        default (eval_deterministic=False) that means the curve gives sampled
        and this call measures argmax.

        THE TWO ROUTINELY DISAGREE, and not by a little: a policy can solve
        every eval maze when sampled and none of them under argmax, because
        greedy action selection deadlocks -- one action repeated into the time
        limit. That is a real property of the policy, not a measurement error,
        and it is the reason both are reported instead of one being chosen.

        Returns the same keys run_split does, plus the argmax block, or None if
        this seed has no checkpoint (an interrupted study, or a seed whose run
        crashed). The caller reports what it got.
        """
        # a FRESH config, for the same reason run_split builds one: PPOAgent
        # copies values out of it and nothing here should be able to leak into
        # the next seed
        config = make_config(self.feature_extractor)

        # WHERE, and under WHICH NAME. select_run sets dir_pretrained_model to
        # best_trial/ and config.seed to seed_list[seed_index], which together
        # spell the path -- the same two things watch.py sets. Nothing is read
        # yet; the file may not exist.
        path = config.select_run(trial="best", seed_index=seed_index)
        if not os.path.exists(path):
            self.logger.info(f"seed {seed}: no checkpoint at {path}, skipped")
            return None

        # THE ARCHITECTURE THE FILE WAS TRAINED WITH, before anything is built.
        # search_space tunes hidden_size for every encoder and n_layers_mlp /
        # d_model / n_heads / n_layers_transformer for some of them, so a
        # config fresh from make_config() describes a DIFFERENT network than
        # the winning trial's. Read out of the checkpoint rather than off
        # best.params, so this is right even for a file whose trial optuna no
        # longer has a record of. See helper.searched_params.
        config.apply_params(config.checkpoint_params(path))

        # builds the network at those values and seeds the process. The seed
        # matters here only for the sampled half of any evaluation -- nothing
        # is trained -- but it costs nothing to make the numbers repeatable.
        agent = PPOAgent(config, seed=seed)
        try:
            # guards the encoder, the widths, the env AND force_cue_visible
            # against this config, and raises rather than loading a file that
            # does not belong to it
            checkpoint = config.load_model(agent.model, path)

            curve = list(checkpoint.get("eval_history") or [])
            if not curve:
                self.logger.info(f"seed {seed}: {path} holds no eval_history, skipped")
                return None

            last = curve[-1]

            # how the CURVE was measured, taken from the file rather than from
            # this config -- config.eval_deterministic could have been edited
            # since the study ran, and then the two blocks below would be
            # labelled by what the file says today instead of what happened
            trained_deterministic = bool(
                checkpoint.get("eval_deterministic", config.eval_deterministic)
            )

            # the one measurement: whichever mode the curve does NOT hold
            fresh = agent.evaluate(deterministic=not trained_deterministic)

            if trained_deterministic:
                argmax, sampled = last, fresh
            else:
                sampled, argmax = last, fresh

            return {
                "seed": seed,
                "iteration": int(last["iteration"]),
                # THE SCORED BLOCK -- the same keys and the same numbers the
                # trial was ranked on, so recomputing the score from these
                # reproduces study.best_value exactly rather than approximately
                "success_rate": float(sampled["success_rate"]),
                "return_mean": float(sampled["return_mean"]),
                "return_std_episodes": float(sampled["return_std"]),
                "timeout_rate": float(sampled["timeout_rate"]),
                "length_mean": float(sampled["length_mean"]),
                # THE ARGMAX BLOCK, prefixed so nothing can read one for the
                # other. Not part of any score -- reported only.
                "argmax_success_rate": float(argmax["success_rate"]),
                "argmax_return_mean": float(argmax["return_mean"]),
                "argmax_return_std_episodes": float(argmax["return_std"]),
                "argmax_timeout_rate": float(argmax["timeout_rate"]),
                "argmax_length_mean": float(argmax["length_mean"]),
                # which of the two was re-measured here rather than read off
                # the curve, so a reader can tell what was reproducible
                "curve_deterministic": trained_deterministic,
                "path": path,
                "eval_history": curve,
            }
        finally:
            # 16 envs per seed, released even if this raises
            agent.close()

    def final(self):
        """Report the winning trial's saved runs. TRAINS NOTHING.

        WHAT THIS USED TO DO, AND WHY IT STOPPED. It retrained at the winning
        params, once per seed, into hpo/final/. The argument for that is the
        winner's curse: study.best_value is the MAXIMUM of n_trials noisy
        measurements, so it is biased upward by the selection itself, and a
        fresh run at those params is an unbiased sample of what they are worth.

        THAT ARGUMENT IS STILL TRUE, and dropping the retrain gives it up: the
        number this reports IS study.best_value, selection bias included. What
        it bought was expensive and thin -- three more full training runs, at
        the same params, on the same env, from the same seeds, differing from
        the trial only in that they happened later. One extra sample does not
        measure the curse, it just adds noise of its own; the honest way to
        quantify it is more seeds, not a second draw of the same three. So the
        cost went and the caveat stayed, and it is written into the log line
        this prints rather than left implicit.

        WHAT IT DOES INSTEAD. Re-copies the winner into best_trial/ (a resumed
        study can have found a new one), loads each seed's checkpoint, and
        reports both eval modes side by side -- see score_saved. Writes
        final_<ENC>_<env>.json beside the checkpoints it describes and redraws
        their two curve figures, so --final-only on a finished study is a
        cheap way to rebuild the report.

        Returns the report dict, or None if no trial ever completed.
        """
        try:
            best = self.study.best_trial
        except ValueError:
            self.logger.info("no completed trial -- nothing to report")
            return None

        # RE-COPIED, not assumed. hpo() copies the winner at the end of the
        # search, but --final-only skips hpo() entirely, and a study resumed
        # for more trials can crown a new winner over an old best_trial/. The
        # copy empties the target first, so this is also what stops the
        # directory being half one trial and half another.
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

        # EVERY metric summarised, not only the searched one, so the json can
        # be read against a different objective later without re-running
        # anything. Both eval modes are in here, and they are NOT comparable to
        # each other -- see score_saved.
        #
        # FIVE KEYS ARE NOT MEASUREMENTS and are dropped: seed, path and
        # curve_deterministic are labels, iteration is the same number for
        # every seed (a mean of it says nothing), and eval_history is a list --
        # the curves stay in "results" below, where they can be read as curves
        # rather than averaged into one.
        summary = {}
        skip = {"seed", "path", "curve_deterministic", "iteration", "eval_history"}
        for key in sorted(set(results[0]) - skip):
            values = [r[key] for r in results]
            summary[key] = {
                "mean": float(np.mean(values)),
                # the spread ACROSS SEEDS -- the run-to-run reproducibility.
                # Not to be confused with return_std_episodes, which is the
                # spread across eval episodes inside a single run.
                "std_across_seeds": float(np.std(values)),
                "median": float(np.median(values)),
                "iqr_across_seeds": float(
                    np.percentile(values, 75) - np.percentile(values, 25)
                ),
                "per_seed": values,
            }

        # the study's own score, recomputed from the loaded runs. It should
        # equal best.value to the last digit, because these ARE the runs the
        # study ranked -- the check below says so out loud, and a mismatch
        # means best_trial/ does not hold what the db thinks it does.
        score = self.config.aggregate_scores([r[self.hpo_metric] for r in results])

        report = {
            "feature_extractor": self.feature_extractor,
            "name_env": self.name_env,
            "objective": self.hpo_objective,
            "metric": self.hpo_metric,
            "aggregation": self.config.hpo_aggregation,
            "hpo_lambda": getattr(self.config, "hpo_lambda", 1.0),
            "score_name": self.score_name,
            # the same number twice, unless something is wrong -- see above
            "score": score,
            "best_value_in_study": best.value,
            "retrained": False,
            "best_trial": best.number,
            "best_params": best.params,
            "seed_list": self.seed_list,
            "seeds_scored": [r["seed"] for r in results],
            "eval_seed": self.config.eval_seed,
            "n_eval_episodes": self.config.n_eval_episodes,
            # WHAT "the score" IS MEASURED AT. Every scalar above comes from
            # the last report iteration, and these three say which one that
            # was -- so the json is readable without also knowing what
            # config.py said on the day it was written.
            "n_iterations": self.config.n_iterations,
            "n_iterations_report": self.config.n_iterations_report,
            "scored_at_iteration": results[0]["iteration"],
            # how the two blocks were taken. The curve's mode is read off the
            # checkpoints; the other mode is the one measured by score_saved.
            "curve_deterministic": results[0]["curve_deterministic"],
            "trials_deterministic": self.config.eval_deterministic,
            "results": results,
            "summary": summary,
        }

        path = os.path.join(directory, f"final_{self.config.name_hpo}.json")
        self.config.save_json(path, report)

        # the two curve figures, redrawn from the checkpoints just re-copied.
        # Same seeds and the same directory as the table below, so the plot and
        # the numbers cannot disagree.
        self.config.plot_eval_curves(directory, name="best_trial", logger=self.logger)

        curve_mode = "argmax" if results[0]["curve_deterministic"] else "sampled"
        other_mode = "sampled" if results[0]["curve_deterministic"] else "argmax"

        self.logger.info("")
        self.logger.info(
            f"FINAL  {self.feature_extractor} on {self.name_env}   "
            f"(trial {best.number}, {self.config.n_eval_episodes} episodes "
            f"from eval_seed {self.config.eval_seed})"
        )
        # says which iteration the table below is, so a row cannot be mistaken
        # for the best the run ever reached
        self.logger.info(
            f"       scored at iteration {results[0]['iteration']} "
            f"of {self.config.n_iterations}, "
            f"{len(results[0]['eval_history'])} evaluations kept per seed"
        )

        # TWO BLOCKS, NOT ONE TABLE WITH SIX COLUMNS. They are different
        # measurements of the same weights and must not be read across.
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
                f"std_across_seeds {s['std_across_seeds']:.3f}  "
                f"median {s['median']:.3f}  iqr {s['iqr_across_seeds']:.3f}"
            )

        self.logger.info("")
        self.logger.info(
            f"FINAL ({other_mode})   the same weights, re-evaluated -- "
            f"reported only, never scored"
        )
        self.logger.info(f"{'seed':>8} {'return':>8} {'success':>9} {'timeout':>9} {'length':>8}")
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
                f"std_across_seeds {s['std_across_seeds']:.3f}  "
                f"median {s['median']:.3f}  iqr {s['iqr_across_seeds']:.3f}"
            )

        self.logger.info("")
        self.logger.info(f"  SCORE  {self.score_name} = {score:.4f}")
        # NOT an independent estimate, and the log says so. These are the very
        # runs the study picked, so this number carries the winner's curse --
        # see the docstring.
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
