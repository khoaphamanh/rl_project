import random
import os
import numpy as np
import torch

from config.helper import Helper


class Config(Helper):
    """Hyperparameters shared by every encoder. NOT built directly.

    The four architecture configs subclass this and fill in _configure_model()
    with the one encoder they are for:

        config_mlp.py          ConfigMLP          hidden_size, n_layers_mlp
        config_lstm.py         ConfigLSTM         hidden_size
        config_gru.py          ConfigGRU          hidden_size
        config_transformer.py  ConfigTransformer  hidden_size, d_model, n_heads,
                                                  n_layers, d_ff, p_drop, ...

    Everything that stays the SAME whichever encoder runs -- the env, the
    rollout size, the PPO knobs, evaluation -- lives here, so that an ablation
    changes exactly one thing between runs: the encoder. That is the whole
    point of the split; a hyperparameter that belongs here but got copied into
    four files would be four chances for the runs to stop being comparable.

    Build one with make_config("MLP"), never Config() -- see config/__init__.py.
    Config on its own is abstract: _configure_model() raises until a subclass
    overrides it.

    Inheriting Helper means config.build_extractor(), config.zero_hidden() and
    config.is_recurrent read these attributes directly.
    """

    # Set by each subclass in _configure_model(). Named here so is_recurrent
    # and build_model_name have an attribute to read, and so a stray Config()
    # fails in the hook below rather than later on a None.upper().
    #
    # A STRING -- "MLP", "LSTM", "GRU", "TRANSFORMER" -- and NOT the encoder
    # itself. Network.feature_extractor in models/model.py carries the same
    # name for the BUILT MODULE, which is what config.build_extractor()
    # returns:
    #
    #     config.feature_extractor     "GRU"      the choice
    #     network.feature_extractor    GRU(...)   the thing that choice built
    #
    # Both spellings are correct on their own object; only this one has an
    # .upper().
    feature_extractor = None

    def __init__(self):

        # THE SEEDS. A result is this whole list, not any one entry: a single
        # PPO run on a sparse-reward MiniGrid task can land anywhere, and two
        # seeds of the SAME config routinely differ by more than two different
        # encoders do. Anything that reports a number -- main_no_hpo.py, and
        # every HPO trial -- runs all three and reports the spread.
        #
        # Every trial in a study runs this SAME list, deliberately not a seed
        # derived per trial: two trials then differ only in hyperparameters,
        # so the comparison between them is not also a comparison of two
        # different draws of mazes.
        self.seed_list = [0, 26, 98]

        # fix hyperparameters
        self.input_size = 7 * 7 * 20  # 980, NOT 147: flatten_obs one-hots the
        #   observation's three CATEGORY channels (11 objects + 6 colours +
        #   3 door states = 20 numbers per cell). Handing the raw indices to
        #   the encoder instead makes key(5) vs ball(6) a difference of 1.0 in
        #   one of 147 inputs, along the same axis as empty(1) vs wall(2) --
        #   and then no encoder can read the cue, so all three score alike.
        #   See feature_extractor.flatten_obs for the measurements.

        # model hyperparameters shared by every encoder. feature_extractor,
        # hidden_size and the per-architecture knobs (n_layers_mlp, d_model,
        # n_heads, ...) are NOT here -- each subclass sets its own in
        # _configure_model(). hidden_size in particular is per-encoder on
        # purpose: it is the one width every encoder has, so owning it is what
        # lets an LSTM be widened without dragging the MLP along.
        self.lr = 1e-3
        self.wd = 0.0  # Adam's weight_decay. 0.0 = plain Adam, which is what
        #   this was before it became tunable, so nothing changes until a study
        #   moves it. Kept small when tuned: PPO's real regularizer is the clip,
        #   and decay pulls the critic's scale down along with the policy's.

        self.tbptt_length = "max"

        # env
        self.name_env = "MiniGrid-DoorKey-8x8-v0"  # "MiniGrid-DoorKey-5x5-v0"  # "MiniGrid-MemoryS11-v0""MiniGrid-DoorKey-8x8-v0"
        self.force_cue_visible = False  # wrap the env in StartInCueView, which
        #   spawns the agent at (1, height//2) instead of a random x along the
        #   hallway. MiniGrid's own MemoryEnv shows the cue from x = 1 ONLY, so
        #   without this it is unobservable in 7 of 8 episodes -- and an
        #   unobservable cue makes GRU, LSTM and MLP the same agent. This is
        #   what makes the ablation an ablation. See helper.StartInCueView.

        # sampling: W games played in parallel, T steps each per iteration
        self.n_workers = 256  # W

        # T = the env's OWN time limit, 5 * size^2 (245 on S7, 605 on S11),
        # read off the env instead of typed in. NOT a free choice.
        #
        # WHY T = 256 SOLVED S7 AND NOT S11, on identical everything else. The
        # limit is what differs: 245 on S7, 605 on S11. An unsolved episode runs
        # to that limit, so one 8 x 256 = 2048-step rollout contains
        #
        #     S7    2048 / 245  ~  8 complete episodes  -> 8 reward signals
        #     S11   2048 / 605  ~  3 complete episodes  -> 3 reward signals
        #
        # per update. On a task that pays nothing until the right object is
        # touched, three outcomes is not enough to estimate an advantage from,
        # and S7 got nearly three times as many for the same compute. T = 245
        # happened to sit just under 256; that coincidence is the whole reason
        # one size worked and the other did not.
        #
        # Setting T = max_steps makes that structural rather than lucky: the
        # longest episode the env can produce still fits inside one rollout at
        # every size, so each worker contributes at least one complete episode
        # per update and split_pad_mask only ever cuts at a real done -- where
        # truncating BPTT is correct, because the hidden state was zeroed there
        # anyway. The cost is rollout size, which now grows with the maze:
        # 8 * 605 = 4840 steps per iteration on S11 against 1960 on S7.
        self.worker_steps = self.env_max_steps  # T

        # total env steps collected per iteration. NOT called batch_size on
        # purpose: mini_batch_size below is handed to a torch DataLoader, where
        # batch_size counts SEQUENCES, and two sibling attributes counting
        # different things under the same word is a trap. This one counts steps.
        self.n_total_steps = self.n_workers * self.worker_steps  # W * T

        # advantage estimation
        self.gamma = 0.99  # discount: how far ahead a reward still counts
        self.gae_lambda = 0.95  # 0 = TD(0), low variance / high bias
        #                         1 = Monte Carlo, high variance / no bias

        # the update
        self.clip_eps = 0.2  # how far pi_new may drift from pi_old before
        #                      the gradient is cut off. The paper's value.
        self.value_coef = 0.1  # weight of the critic's regression term. Damped,
        #                        or a squared error on returns would dominate
        #                        the shared encoder and the policy learns nothing.
        self.entropy_coef = 0.03  # weight of the exploration bonus. Small on
        #                           purpose: a brake against premature collapse,
        #                           not a goal.
        self.max_grad_norm = 0.5  # rescale the gradient if it is longer than this

        # how hard each rollout is reused
        self.n_epochs = 3  # passes over the SAME batch. The whole reason the
        #                    clip exists: after pass 1 the data is off-policy.
        # DataLoader batch size, counted in SEQUENCES (not timesteps): one
        # sample is one whole padded episode fragment of up to L steps.
        #
        # A LIST OF CANDIDATES, LARGEST FIRST, not a single number. The update
        # runs at the biggest one that fits in memory: helper's
        # run_with_batch_size_fallback tries 128, and on an out-of-memory error
        # frees the allocator and retries at 64, then 32, and so on. That is
        # what stops the experiment definition from depending on which machine
        # it happens to run on -- a laptop resolves to something small, a GPU
        # box to 128, and neither needs this file edited.
        #
        # ON THIS MACHINE (no CUDA) nothing realistically OOMs, so the first
        # candidate is simply used. Note that 128 is not 4: a bigger minibatch
        # means fewer, less noisy optimizer steps per epoch, which is a real
        # change to the training dynamics and not just a memory setting.
        self.mini_batch_size = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
        self.target_kl = None  # e.g. 0.015 to abandon the remaining epochs once
        #                        pi_new has drifted too far. None = off.

        # training length
        self.n_iterations = 500  # sample -> update, this many times
        self.n_iterations_report = 100  # every this many iterations, run a full
        #                                evaluate() and print a line. NOT every
        #                                iteration: 50 unsolved episodes cost up
        #                                to 50 * max_steps env steps, which is
        #                                more than the n_total_steps one
        #                                iteration collects, and the policy
        #                                barely moves in a single update anyway.

        # evaluation: no gradients, no training, COMPLETE episodes only
        self.n_eval_episodes = 50  # played to their own end, never truncated
        self.eval_seed = 10_000  # fixed and far from seed_list, so the same
        #                          50 mazes are replayed at every evaluation.
        #                          That removes the maze draw as a source of
        #                          noise -- but NOT the action sampling, which
        #                          still varies unless deterministic=True.
        self.eval_deterministic = False  # False = sample from pi, the same way
        #                                  training does, so the curve is
        #                                  comparable to train()'s.
        #                                  True = argmax: the cleaner final
        #                                  number, but it deadlocks on a
        #                                  half-trained policy (one action
        #                                  repeated until the time limit).
        #                                  evaluate(deterministic=True)
        #                                  overrides this for one call.
        #
        #   ONE MODE PER RUN, AND IT COVERS EVERY METRIC. Whatever this says is
        #   read by PPOAgent before training starts, so the periodic
        #   evaluations that build the learning curve AND the run's closing
        #   number both use it. return_mean and success_rate are therefore
        #   always measured the same way as each other -- there is no
        #   combination of settings where the curve is sampled and its endpoint
        #   is argmax, which would be a comparison of two different things
        #   dressed up as one metric.
        #
        #   THERE IS NO SEPARATE final_deterministic ANY MORE. It used to pick
        #   the mode for the retrain that hpo_ppo.final() ran; final() no
        #   longer trains anything, and both main_no_hpo.py and final() now
        #   report BOTH modes side by side -- one taken from the run's own
        #   curve, the other measured once at the end. Neither has to be chosen
        #   in advance, and seeing the two together is the point: on this task
        #   they can disagree completely, e.g. a policy that solves 100% of the
        #   eval mazes when sampled and 0% under argmax, which is an argmax
        #   deadlock and a real property of the policy rather than a bug.

        # ----- the search, when there is one -----------------------------
        # Read by agents/hpo_ppo.py. Ignored entirely by main.py, which trains
        # whatever the config already says -- so a project that never runs a
        # study never has to care that these exist.
        self.n_trials = 30  # how many hyperparameter draws the study gets.
        #   Budget is counted in COMPLETE + PRUNED trials, so a trial that
        #   crashes does not spend one and is retried on the next run.
        self.seed_hpo = 42  # seeds the TPE SAMPLER ONLY -- which points in the
        #   search space get proposed, and in what order. It is NOT a training
        #   seed and never reaches the env or the weight init: those come from
        #   seed_list above, which every trial shares. Fixing it is what makes
        #   a resumed or repeated study propose the same sequence of trials.
        self.hpo_direction = "maximize"  # of the score defined by the next two

        # THE SCORE IS ONE STRING WITH THREE FIELDS:
        #
        #     <metric>_<center>_<spread>
        #
        #   metric   ONE NUMBER PER TRAINING RUN (per seed)
        #   center   how the len(seed_list) of those become one number
        #   spread   what is subtracted from that, weighted by hpo_lambda
        #
        # Concretely, with seed_list = [0, 26, 98]:
        #
        #   run seed 0  -> trains -> evaluate() plays n_eval_episodes on the
        #                  eval_seed mazes -> return_mean_0, success_rate_0
        #   run seed 26 -> ...                                  -> ..._26
        #   run seed 98 -> ...                                  -> ..._98
        #   score = center(metric_0, metric_26, metric_98)
        #           - hpo_lambda * spread(metric_0, metric_26, metric_98)
        #
        # So "mean" here IS mean(mean(eval of 0), mean(eval of 26), mean(eval
        # of 98)) -- the mean composes, because a mean of equal-sized means is
        # the mean of the pool. STD DOES NOT COMPOSE THAT WAY, and neither
        # does the median, which is the whole reason the last two fields are
        # spelled out rather than assumed.
        #
        # ONE STRING AND NOT TWO SETTINGS. The metric and the aggregation used
        # to be hpo_objective and hpo_aggregation, two attributes that could be
        # edited independently and printed side by side in a log without
        # anything saying they were halves of one score. config.hpo_metric,
        # .hpo_center, .hpo_spread and .hpo_aggregation are now all derived
        # from this -- read-only properties on Helper, see parse_hpo_objective.
        self.hpo_objective = "return_mean_minus-std"
        #
        # FIELD 1, the metric:
        #   "return"        what the env actually paid, averaged over the eval
        #    (= return_mean) episodes. What "the reward" means here.
        #   "success-rate"  fraction that reached the goal. Cleaner than
        #                   return -- it does not mix in how fast the agent
        #                   walked -- but read off ONE evaluation, whose own
        #                   noise is about +-0.06 at n_eval_episodes = 50.
        #
        # BOTH ARE READ OFF THE RUN'S LAST EVALUATION, which is the policy that
        # was checkpointed -- so the number the study ranks on and the weights
        # on disk are the same thing. There used to be a third, "aulc", the
        # mean of the success_rate curve over the whole run, offered as a way
        # to reward learning FAST rather than merely ending well. It is gone:
        # a mean over the curve does not depend on the ORDER of its points, so
        # a run scoring 1, 1, 0 at its report iterations tied with one scoring
        # 0, 1, 1 -- and only the second one still works at the end. Since the
        # checkpoint kept is the last iteration's, aulc could crown a trial
        # whose saved weights had already collapsed. Speed of learning is
        # visible in the curve plots, which is where it belongs.
        #
        # FIELD 2, the center over seeds:
        #   "mean"          the plain average. Moved a full third by one dead
        #                   seed out of three.
        #   "median"        the TYPICAL seed. Unmoved by that dead seed, which
        #                   is either exactly what you want or exactly what you
        #                   do not -- see below.
        #
        # FIELD 3, the spread over seeds, subtracted:
        #   "minus-std"     penalise the standard deviation across seeds.
        #   "minus-iqr"     penalise the interquartile range instead, the
        #                   robust partner of the median. With three seeds it
        #                   is an interpolation between two of them, so read it
        #                   as "a spread", not as a quantile estimate.
        #   "None"          no penalty: rank on the center alone.
        #
        # WHICH PAIR TO USE IS A REAL DECISION, not a formatting preference:
        #
        #   return_mean_minus-std          "works on EVERY seed" -- one failure
        #                                  is punished twice, once through the
        #                                  mean and once through the std
        #   success-rate_median_minus-iqr  "works on MOST seeds" -- a single
        #                                  dead seed of three moves neither term
        #   success-rate_median_None       "the typical seed solves it", with no
        #                                  reproducibility term at all
        #
        # Examples, all valid: return_mean_minus-std, success-rate_median_None,
        # return_mean_median_None, success-rate_mean_minus-iqr. Dashes and
        # underscores are interchangeable; the trailing two fields may be
        # omitted and default to mean / no penalty. A typo raises rather than
        # falling back -- it would otherwise cost a whole study.

        self.hpo_lambda = 1.0  # THE LAMBDA in  score = center - lambda * spread,
        #   where both terms are taken ACROSS SEEDS. Read by
        #   helper.aggregate_scores, and ignored entirely when the objective
        #   ends in None -- there is no penalty term to weight in that case.
        #
        #       lambda = 0.0   identical to a "_None" objective
        #       lambda = 0.5   spread as a tiebreak
        #       lambda = 1.0   the strict reading of "mean minus std"
        #
        #   READ THIS BEFORE LEAVING IT AT 1.0. With three seeds, a config that
        #   works on exactly ONE of them scores NEGATIVE at lambda = 1, i.e.
        #   BELOW a config that never works at all:
        #
        #       [0, x, 0]  ->  mean x/3,  std x*sqrt(2)/3  ->  -0.138 x
        #       [0, 0, 0]  ->  mean 0,    std 0            ->   0
        #
        #   minus-iqr does not escape this: [0, x, 0] has median 0 and IQR x/2,
        #   so it scores -x/2, which is worse still. Only "_None" avoids it
        #   entirely -- at the cost of not penalising spread at all.
        #
        #   That is measured, not hypothetical -- an early trial here scored
        #   -0.0118 on [0.0, 0.0855, 0.0] while a dead trial scored 0.0. Early
        #   in a study almost everything is near zero, so this can steer the
        #   sampler AWAY from the first configs that show any life at all.
        #
        #   The sign flips at lambda = 1/sqrt(2) ~ 0.707, so any value below
        #   that keeps "one seed worked" ahead of "nothing worked" while still
        #   penalising spread. 0.5 is the safe choice if the study looks like
        #   it is avoiding the promising region; 1.0 is the strict reading of
        #   "mean minus std" and is kept as the default because it is what was
        #   asked for. With a task that gets solved reliably, the two agree.

        # THE STD IS ACROSS SEEDS, NEVER ACROSS EVAL EPISODES, and that is not
        # a detail. The within-run spread of a bimodal return (0 on failure,
        # ~0.9 on success) is pinned at about 0.9*sqrt(p(1-p)) -- a function of
        # the success rate itself, carrying no independent information. Worse,
        # subtracting it is actively perverse: at p = 0 it gives 0 - 0 = 0,
        # while at p = 0.2 it gives 0.18 - 0.36 = -0.18, so a policy that
        # learned something would score BELOW one that learned nothing. The
        # across-seed spread has no such pathology: it is 0 for any config that
        # behaves the same way every time, whatever its score.

        # WHICH KNOBS THE STUDY TURNS. Filled per encoder in _configure_model()
        # -- an MLP has no n_heads to tune -- as a list of dicts passed
        # straight through to trial.suggest_*:
        #
        #     {"type": "float", "name": "lr", "low": 1e-5, "high": 1e-2,
        #      "log": True}
        #
        # "type" is popped, everything else is forwarded verbatim, so any
        # argument optuna's suggest_int / suggest_float / suggest_categorical
        # accepts (step=, log=, choices=) works with no change to the helper.
        # Every "name" must already be an attribute of the config -- apply_params
        # raises otherwise, which turns a typo into an immediate error instead
        # of a silently untuned run.
        #
        # THESE ARE THE KNOBS EVERY ENCODER HAS. Each subclass APPENDS its own
        # architecture knobs in _configure_model() below -- an MLP has no
        # n_heads to tune -- so the shared PPO half is written once here rather
        # than copied into four files that would then drift apart.
        #
        # WHAT IS DELIBERATELY NOT IN HERE. gamma, because on a task that pays
        # once at the end, lowering it changes what "solved" means rather than
        # how well it is solved. n_workers / worker_steps, because they set how
        # much experience a trial gets and a search would simply buy the best
        # score with compute. n_iterations, for the same reason.
        self.search_space = [
            # the one that matters most, and the one worth the log scale: the
            # useful range spans three orders of magnitude and the interesting
            # part is near the bottom of it
            {"type": "float", "name": "lr", "low": 1e-5, "high": 1e-2, "log": True},
            # exploration. Also log: 0.001 vs 0.01 is the interesting question,
            # 0.05 vs 0.06 is not
            {
                "type": "float",
                "name": "entropy_coef",
                "low": 1e-4,
                "high": 1e-1,
                "log": True,
            },
            # how loud the critic is relative to the policy. Too high and the
            # squared error on returns dominates the SHARED encoder
            {
                "type": "float",
                "name": "value_coef",
                "low": 0.05,
                "high": 1.0,
                "log": True,
            },
            # The trust region. Narrow band on purpose -- outside roughly
            # [0.1, 0.3] this stops being PPO.
            #
            # NO log=. Log scale is for ranges spanning ORDERS OF MAGNITUDE,
            # where the interesting question is 0.0001 vs 0.001; over a single
            # factor of three it does nothing but tilt the density slightly.
            # step=0.01 instead: 21 values, which is finer than any training
            # run at this noise level could tell apart anyway, and it makes
            # trials land on numbers that are readable in the csv.
            {
                "type": "float",
                "name": "clip_eps",
                "low": 0.1,
                "high": 0.3,
                "step": 0.01,
            },
            # GAE's bias/variance knob, likewise near its usable range and
            # likewise not a log quantity.
            #
            # BE AWARE THIS GRID IS NOT UNIFORM IN EFFECT. What lambda controls
            # is roughly a horizon of 1/(1 - lambda): 0.90 -> 10 steps,
            # 0.95 -> 20, 0.99 -> 100. So the top of this range is where the
            # behaviour changes fastest, and equal steps in lambda are not
            # equal steps in what it does. Sampling log(1 - lambda) uniformly
            # would fix that, and optuna cannot express it directly -- it would
            # need a derived attribute the way d_ff is derived from d_model.
            # Left as a linear grid because 10 values over this range is
            # already finer than the seeds can resolve.
            {
                "type": "float",
                "name": "gae_lambda",
                "low": 0.9,
                "high": 0.99,
                "step": 0.01,
            },
            # How hard one rollout is reused before it is too off-policy.
            # NO step= and no log=: suggest_int already steps by 1, which is
            # the only sensible granularity for a count of passes.
            {"type": "int", "name": "n_epochs", "low": 1, "high": 8},
            # Adam decay, log over a wide range because the honest answer here
            # is often "none" and 1e-8 is how the search says that
            {"type": "float", "name": "wd", "low": 1e-8, "high": 1e-2, "log": True},
            # Gradient clipping norm. Rescales the gradient if it exceeds this.
            # Log scale because 0.1 vs 1.0 is the interesting question.
            {
                "type": "float",
                "name": "max_grad_norm",
                "low": 0.1,
                "high": 2.0,
                "log": True,
            },
        ]

        # ----- the encoder, filled by the subclass -----------------------
        # ConfigMLP / ConfigLSTM / ConfigGRU / ConfigTransformer override this
        # to set feature_extractor and whatever architecture knobs their encoder
        # needs. Called HERE, near the end, for three reasons that decide the
        # ordering:
        #   - AFTER worker_steps exists, because the transformer's
        #     max_seq_length is derived from it;
        #   - AFTER search_space, so a subclass can append to it;
        #   - BEFORE name_model and the directories below, both of which spell
        #     feature_extractor, which does not exist until this runs.
        self._configure_model()

        # ----- where things are written ----------------------------------
        # ONE DIRECTORY PER ENCODER, agents/pretrained_model_GRU/, because
        # everything below it is named after the encoder and the env and
        # nothing else -- so two encoders sharing a directory is fine, but two
        # RUNS of one encoder is not, and there are three kinds of run:
        #
        #     pretrained_model_GRU/
        #         hpo/                   the search itself
        #             hpo_csv_GRU_<env>.csv    every trial, exported each trial
        #             hpo_db_GRU_<env>.db      the study, so it can resume
        #             hpo_sampler_GRU_<env>.pkl the TPE state, likewise
        #             trial_0/  trial_1/  ...  one checkpoint per seed, per trial
        #             best_trial/              a copy of the winner's, plus
        #                                      final_GRU_<env>.json -- the result
        #         no_hpo/                config_no_hpo.py's hand-picked run
        #
        # Relative, like logs/, so everything lands under the repo root --
        # which means python must be run FROM the repo root.
        self.dir_model = os.path.join(
            "agents", f"pretrained_model_{self.feature_extractor.upper()}"
        )

        # what save_model writes to. The DEFAULT is the encoder's top level;
        # hpo_ppo.py repoints it at hpo/trial_<n>/ for the duration of a trial
        # and ConfigNoHPO repoints it at no_hpo/. It has to be redirectable
        # rather than baked into the filename because watch.py and load_model
        # rebuild the path from the config alone -- they have no trial number.
        self.dir_pretrained_model = self.dir_model
        self.dir_hpo = os.path.join(self.dir_model, "hpo")

        # ppo_GRU_MiniGrid-MemoryS11-v0.pth -- BOTH THE ENCODER AND THE ENV ARE
        # IN THE NAME, because both change what the weights mean. One fixed
        # ppo_feature_extractor.pth would have every run overwrite the last;
        # keying on the encoder alone still lets a MemoryS11 run land on top of
        # a finished DoorKey run of the same encoder, which is what nearly
        # happened here with three checkpoints sitting under three envs.
        #
        # Spelled by helper.build_model_name(), not by an f-string here, so
        # watch.py cannot end up with a different idea of the filename.
        # Relative, like logs/, so everything is written under the repo root --
        # which means python must be run FROM the repo root.
        # name_model and path_model are PROPERTIES on Helper, not attributes
        # set here, so both are re-read every time they are asked for:
        #
        #   name_model  carries the SEED, which set_seed() only fills in later,
        #               inside PPOAgent.__init__
        #   path_model  carries the DIRECTORY, which hpo_ppo.py repoints at
        #               hpo/trial_7/ per trial and ConfigNoHPO at no_hpo/
        #
        # Frozen here instead, every trial and every seed would keep writing to
        # whatever name and directory happened to be set at construction.
        #
        # The directory itself is made by helper.build_model_path(), at save
        # time, not here.

    def _configure_model(self):
        """Set feature_extractor and this encoder's architecture knobs.

        Overridden by ConfigMLP / ConfigLSTM / ConfigGRU / ConfigTransformer.
        Config itself is abstract, so building one directly stops here with a
        clear message instead of failing later inside build_extractor or on a
        None.upper() in build_model_name.
        """
        raise NotImplementedError(
            "Config is abstract -- build a ConfigMLP / ConfigLSTM / ConfigGRU / "
            "ConfigTransformer, or call make_config('MLP'). Each one fills in "
            "feature_extractor and its own encoder hyperparameters in "
            "_configure_model()."
        )

    # ------------------------------------------------------------------
    # the study's name and its files
    #
    # Properties, not attributes, for the same reason build_model_name() is a
    # method: watch.py and hpo_ppo.py can set feature_extractor from the
    # command line, and a name frozen in __init__ would still spell whichever
    # encoder was set at import time.
    # ------------------------------------------------------------------
    @property
    def name_hpo(self):
        """GRU_MiniGrid-DoorKey-8x8-v0 -- the stem every HPO file is built on.

        Both halves are in it for the same reason they are in the checkpoint
        name: a study is over ONE encoder on ONE env, and two of them sharing
        a csv would be two different experiments in one table.
        """
        env = self.name_env.replace("/", "-")
        return f"{self.feature_extractor.upper()}_{env}"

    @property
    def name_study(self):
        """The study_name inside the sqlite file.

        Distinct from the file name because one .db CAN hold several studies;
        keying it on encoder + env means load_if_exists=True resumes the right
        one even if the databases are ever merged.
        """
        return f"ppo_{self.name_hpo}"

    @property
    def path_hpo_csv(self):
        """Every trial as a table, rewritten at the START of each trial.

        Written by csv_study_export. A csv only at the end would be a csv you
        never get when a study is interrupted, which is precisely when you want
        to see how far it got.
        """
        return os.path.join(self.dir_hpo, f"hpo_csv_{self.name_hpo}.csv")

    @property
    def path_hpo_db(self):
        """The study itself, as a SQLAlchemy URL (not a plain path).

        This is what makes a study resumable: optuna keeps every trial, its
        params and its state here, so a crashed run picks up its remaining
        budget instead of starting over.
        """
        path = os.path.join(self.dir_hpo, f"hpo_db_{self.name_hpo}.db")
        return "sqlite:///" + path

    @property
    def path_hpo_sampler(self):
        """The TPE sampler, pickled after every trial.

        SEPARATE FROM THE DATABASE AND NOT OPTIONAL. The db holds what the
        study has TRIED; this holds what the sampler has LEARNED. Resume
        without it and optuna rebuilds a fresh TPESampler(seed=seed_hpo),
        which re-draws its n_startup_trials random probes -- so a study
        resumed at trial 25 spends the next 10 trials exploring at random
        instead of exploiting what the first 25 already showed.
        """
        return os.path.join(self.dir_hpo, f"hpo_sampler_{self.name_hpo}.pkl")

    def set_seed(self, env=None, seed=None):
        """Seed every source of randomness and return the seed that was used.

        seed=None falls back to self.seed_list[0], so set_seed() with no
        argument is always reproducible. Anything that reports a RESULT should
        pass each of seed_list in turn instead of relying on that default --
        one seed is an anecdote on these tasks.

        The CUDA calls are no-ops on a machine without a GPU. They stay in so
        the exact same config also reproduces on a machine that has one.

        env is optional. In Gymnasium the env RNG is seeded through
        reset(seed=...), so this consumes one reset and throws its
        observation away. Call set_seed first, then reset() again without a
        seed for the real first observation -- that keeps the stream going
        from the seeded state instead of restarting it.
        """
        if seed is None:
            seed = self.seed_list[0]
        self.seed = seed

        # affects subprocesses only (the parallel workers), not this process
        os.environ["PYTHONHASHSEED"] = str(seed)

        random.seed(seed)  # python's own random
        np.random.seed(seed)  # numpy, used by gymnasium internally

        torch.manual_seed(seed)  # cpu tensors, weight init, dropout
        torch.cuda.manual_seed(seed)  # one gpu
        torch.cuda.manual_seed_all(seed)  # all gpus

        if torch.cuda.is_available():
            # without these two, cudnn picks algorithms by benchmarking and
            # the same seed can still give different numbers
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if env is not None:
            env.reset(seed=seed)  # seeds the env RNG (layout, cue position)
            env.action_space.seed(seed)  # seeds env.action_space.sample()
            env.observation_space.seed(seed)

        return seed
