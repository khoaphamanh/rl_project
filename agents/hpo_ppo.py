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
once per seed in config.seed_list, each run scored by config.hpo_objective
("return_mean" by default), and those per-seed numbers combined by
config.hpo_aggregation ("mean_minus_std", i.e. mean - hpo_lambda * std across
seeds). Three full training runs per trial -- that is the expense, and it is
the point: a single PPO run on a sparse-reward MiniGrid task can land
anywhere, so a trial scored on one seed would mostly be measuring the seed.

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
    trial_0/ trial_1/ ...        that trial's checkpoints, ONE PER SEED
    best_trial/                  a copy of the winner's, + best_params.json
    ppo_<seed>_<ENC>_<env>.pth   final()'s retrained models
    final_<ENC>_<env>.json       what that retrain scored

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
        self.hpo_objective = config.hpo_objective  # the PER-SEED metric
        self.final_deterministic = config.final_deterministic

        # "mean_minus_std(return_mean)" -- the metric AND the rule that turns
        # len(seed_list) of them into one number. A config property, so the
        # study summary in helper.py and the report here cannot disagree.
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
    def run_split(self, seed, params, trial_number=None, deterministic=None):
        """Train once, at these params, from this seed. Returns its scores.

        A FRESH CONFIG EVERY CALL, never self.config. Two reasons, both of
        which have bitten this file: PPOAgent copies every value it needs out
        of the config in __init__, and train() writes the resolved
        mini_batch_size BACK into it -- so a shared config would carry one
        run's state into the next, and a trial that fell back to 32 would
        silently hold every later trial at 32.

        trial_number redirects the checkpoint into hpo/trial_<n>/. Without it
        -- which is how final() calls this -- it lands directly in hpo/. The
        number cannot go in the FILENAME instead: watch.py and load_model
        rebuild that from the config alone and have no trial number to put in
        it. The SEED is in the filename, which is why one trial directory
        holds one checkpoint per entry in seed_list rather than one file
        overwritten three times.

        deterministic overrides the eval mode for this run: None means "use
        config.eval_deterministic", which is what every trial does. final()
        passes config.final_deterministic instead. Whichever it is gets written
        onto the config BEFORE the agent is built, so the learning curve and
        the closing evaluation are measured the same way -- see below.

        Returns {"seed", "aulc", "success_rate", "return_mean",
        "return_std_episodes", "deterministic"}.
        """
        config = make_config(self.feature_extractor)

        # BEFORE PPOAgent, not after. The agent reads lr and wd into the
        # optimizer and hidden_size / d_model into the encoder it builds, and
        # never looks at the config again -- applied afterwards, the trial
        # would train the defaults and report them as the drawn params.
        config.apply_params(params)

        if trial_number is not None:
            config.dir_pretrained_model = config.build_hpo_trial_dir(trial_number)
        else:
            # final(): the study's OUTPUT, so it belongs beside the search that
            # produced it rather than a level up. That keeps
            # pretrained_model_<ENC>/ to exactly two entries, hpo/ and no_hpo/,
            # which is what says at a glance how a checkpoint was arrived at.
            config.dir_pretrained_model = config.build_hpo_dir()

        # ONE EVAL MODE FOR THE WHOLE RUN, set BEFORE the agent is built so it
        # reaches everything. PPOAgent copies eval_deterministic out of the
        # config in __init__ and train_agent()'s periodic evaluations read it,
        # so writing it here is what makes the LEARNING CURVE and the CLOSING
        # EVALUATION the same kind of measurement.
        #
        # That is the whole point: aulc, success_rate and return_mean are then
        # never measured differently from one another. Previously the curve was
        # sampled while the endpoint was argmax, so switching hpo_objective
        # silently switched the measuring instrument too.
        if deterministic is None:
            deterministic = config.eval_deterministic
        config.eval_deterministic = bool(deterministic)

        agent = PPOAgent(config, seed=seed)
        try:
            history = agent.train_agent(logger=self.logger)

            # no deterministic= argument: it uses config.eval_deterministic,
            # which is the single mode set just above
            final = agent.evaluate()

            # THE LEARNING CURVE, not just its last point. Index 0 is dropped
            # because it is essentially the untrained policy -- the same floor
            # for every trial, so keeping it only adds a constant to every
            # score and compresses the differences the search is looking for.
            curve = [h["eval_success_rate"] for h in history if "eval_success_rate" in h]
            aulc = float(np.mean(curve[1:] or curve))

            return {
                "seed": seed,
                # all three measured under the SAME mode -- see above
                "aulc": aulc,
                "success_rate": float(final["success_rate"]),
                "return_mean": float(final["return_mean"]),
                # the spread over EVAL EPISODES within this one run. Recorded
                # because it describes the run, but deliberately NOT what
                # mean_minus_std subtracts -- see helper.aggregate_scores.
                "return_std_episodes": float(final["return_std"]),
                # so the csv and the json say how these numbers were taken
                "deterministic": bool(deterministic),
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
                scores.append(result[self.hpo_objective])

                # EVERY metric is kept per seed, not only the one being
                # optimized: the csv then answers "what would the other
                # objective have chosen?" without re-running the study, and
                # the per-seed spread is what says whether a trial's score
                # means anything at all.
                for key, value in result.items():
                    if key != "seed":
                        trial.set_user_attr(f"{key}_seed_{seed}", value)

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
                f"per-seed {self.hpo_objective} over {self.seed_list} "
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

        return best

    # ------------------------------------------------------------------
    # the number to report
    # ------------------------------------------------------------------
    def final(self):
        """Retrain at the best params, once per seed, and write the result.

        WHY THIS IS NOT JUST study.best_value. That value is the MAXIMUM of
        n_trials noisy measurements, so it is biased upward by the selection
        itself -- the winner's curse. Some of the winner's margin is real and
        some is that it got the luckiest draw, and the study cannot tell you
        how much of each. Retraining at those params and scoring THAT is the
        clean number, and it is the one that belongs in the writeup.

        Writes final_<ENC>_<env>.json next to the checkpoints, with the
        per-seed values kept alongside the mean: on these tasks the outcome is
        bimodal -- some seeds find the reward and some never do -- so the mean
        of {0.5, 0.5, 0.95} describes no run that happened.

        Returns the summary dict, or None if no trial ever completed.
        """
        try:
            best = self.study.best_trial
        except ValueError:
            self.logger.info("no completed trial -- nothing to retrain")
            return None

        params = best.params
        self.logger.info("")
        self.config.print_separate_lines(self.logger)
        self.logger.info(
            f"FINAL: retraining {self.feature_extractor} at the params of "
            f"trial {best.number} over seeds {self.seed_list}"
        )
        self.logger.info(f"  {params}")

        # trial_number=None, so these land in hpo/ itself rather than in a
        # trial directory -- they are the study's output, not part of the
        # search.
        #
        # deterministic=config.final_deterministic, which is the ONLY thing
        # that distinguishes these runs from a trial's. Default False, so the
        # retrain is measured exactly the way the study was and the two scores
        # are comparable; set it True in config.py for the fully reproducible
        # argmax number instead.
        results = [
            self.run_split(seed, params, deterministic=self.final_deterministic)
            for seed in self.seed_list
        ]

        # EVERY metric summarised, not only the searched one, so the json can
        # be read against a different objective later without retraining. All
        # of them were measured under the same eval mode, so they are directly
        # comparable to each other -- "deterministic" is dropped because it is
        # a flag, not a measurement.
        summary = {}
        for key in sorted(set(results[0]) - {"seed", "deterministic"}):
            values = [r[key] for r in results]
            summary[key] = {
                "mean": float(np.mean(values)),
                # the spread ACROSS SEEDS -- the run-to-run reproducibility.
                # Not to be confused with return_std_episodes, which is the
                # spread across eval episodes inside a single run.
                "std_across_seeds": float(np.std(values)),
                "median": float(np.median(values)),
                "per_seed": values,
            }

        # the study's own score, recomputed here on the RETRAINED runs. This
        # is the headline number: same objective, same aggregation, but
        # measured without the selection that biased best_value upward.
        score = self.config.aggregate_scores([r[self.hpo_objective] for r in results])

        report = {
            "feature_extractor": self.feature_extractor,
            "name_env": self.name_env,
            "objective": self.hpo_objective,
            "aggregation": self.config.hpo_aggregation,
            "hpo_lambda": getattr(self.config, "hpo_lambda", 1.0),
            "score_name": self.score_name,
            # the two numbers to compare. A large gap means the study was
            # mostly fitting noise -- which is a finding about the budget.
            "score_retrained": score,
            "best_value_in_study": best.value,
            "best_trial": best.number,
            "best_params": params,
            "seed_list": self.seed_list,
            "eval_seed": self.config.eval_seed,
            "n_eval_episodes": self.config.n_eval_episodes,
            # how THESE runs were measured, and how the trials were
            "final_deterministic": self.final_deterministic,
            "trials_deterministic": self.config.eval_deterministic,
            "results": results,
            "summary": summary,
        }

        path = os.path.join(self.config.dir_hpo, f"final_{self.config.name_hpo}.json")
        self.config.save_json(path, report)

        mode = "argmax" if self.final_deterministic else "sampled"
        self.logger.info("")
        self.logger.info(
            f"FINAL  {self.feature_extractor} on {self.name_env}   "
            f"(eval: {mode}, {self.config.n_eval_episodes} episodes "
            f"from eval_seed {self.config.eval_seed})"
        )
        self.logger.info(f"{'seed':>8} {'return':>8} {'success':>9} {'aulc':>8}")
        for r in results:
            self.logger.info(
                f"{r['seed']:>8} {r['return_mean']:>8.3f} "
                f"{r['success_rate']:>9.3f} {r['aulc']:>8.3f}"
            )
        for key in ("return_mean", "success_rate", "aulc"):
            s = summary[key]
            self.logger.info(
                f"{key:>14}  mean {s['mean']:.3f}  "
                f"std_across_seeds {s['std_across_seeds']:.3f}  "
                f"median {s['median']:.3f}"
            )
        self.logger.info("")
        self.logger.info(
            f"  SCORE  {self.score_name} = {score:.4f}   "
            f"(the study's own best_value was {best.value:.4f}, "
            f"biased upward by selection)"
        )
        self.logger.info(f"written to {path}")

        return report

