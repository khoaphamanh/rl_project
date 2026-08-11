MODEL_CHOICES = ("MLP", "GRU")


def make_config(name, tbptt_length=None):
    """Name (case-insensitive) -> the matching Config subclass, built.
    tbptt_length is the GRU's backward reach: None (full BPTT) or an int, which
    also decides the directory the study writes to. MLP has no time dependence
    to truncate, so anything but None raises."""
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
