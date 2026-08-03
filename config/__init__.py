MODEL_CHOICES = ("MLP", "LSTM", "GRU", "TRANSFORMER")


def make_config(name):
    """Name (case-insensitive) -> the matching Config subclass, built."""
    key = name.upper()

    # deferred to avoid a circular import

    from config.config_mlp import ConfigMLP
    from config.config_lstm import ConfigLSTM
    from config.config_gru import ConfigGRU
    from config.config_transformer import ConfigTransformer

    table = {
        "MLP": ConfigMLP,
        "LSTM": ConfigLSTM,
        "GRU": ConfigGRU,
        "TRANSFORMER": ConfigTransformer,
    }

    if key not in table:
        raise ValueError(
            f"unknown model {name!r}. choose one of {', '.join(MODEL_CHOICES)}"
        )

    return table[key]()
