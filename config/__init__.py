"""The config package's front door: MODEL_CHOICES lists the encoders the CLIs
accept, and make_config turns one of those names into a built config. The tuned
subclasses live in config_mlp.py / config_gru.py, the hand-picked one in
config_no_hpo.py, and everything they share in config.py / helper.py."""

MODEL_CHOICES = ("MLP", "GRU")


def make_config(name, tbptt_length=None):
    """A CLI model name -> the matching *tuned* Config subclass, built.

    Never returns ConfigNoHPO: the hand-picked config takes its encoder as a
    constructor argument instead (ConfigNoHPO("GRU")).

    Args:
        name (str): encoder name, case-insensitive, one of MODEL_CHOICES.
        tbptt_length (int | None): the GRU's backward reach in steps, which
            also names the directory the run writes to. None = full BPTT. An
            MLP has no recurrence to truncate, so anything but None raises.

    Returns:
        Config: a ConfigMLP or ConfigGRU instance.

    Raises:
        ValueError: on an unknown name, or a length given for the MLP.
    """
    key = name.upper()

    # deferred to avoid a circular import
    from config.config_mlp import ConfigMLP
    from config.config_gru import ConfigGRU

    table = {
        "MLP": ConfigMLP,
        "GRU": ConfigGRU,
    }

    if key not in table:
        raise ValueError(
            f"unknown model {name!r}. choose one of {', '.join(MODEL_CHOICES)}"
        )

    return table[key](tbptt_length=tbptt_length)
