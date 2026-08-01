"""The config package.

config.py holds every hyperparameter that is the same across encoders. The
four architecture files subclass it and fill in only what differs:

    config_mlp.py          ConfigMLP          n_layers_mlp
    config_lstm.py         ConfigLSTM         (nn.LSTM needs no extra knobs)
    config_gru.py          ConfigGRU          (nn.GRU needs no extra knobs)
    config_transformer.py  ConfigTransformer  d_model, n_heads, n_layers, ...

make_config("MLP") returns a ConfigMLP(). main.py and watch.py both go through
it, so "python main.py MLP" and "python watch.py MLP" build the same object and
there is one place -- not two -- that maps a name to a class.
"""

# every encoder this project knows how to build, uppercase. argparse in main.py
# and watch.py reads this for its choices, so adding a fifth encoder is a
# one-line change here plus its config file and a build_extractor branch.
MODEL_CHOICES = ("MLP", "LSTM", "GRU", "TRANSFORMER")


def make_config(name):
    """Name -> the matching Config subclass, built.

        config = make_config("MLP")     # -> ConfigMLP()
        config = make_config("gru")     # case-insensitive

    The imports are INSIDE the function on purpose. Each config_*.py does
    `from config.config import Config`, which imports this package first; a
    top-level import of them here would be a cycle. Deferred to call time,
    the package finishes importing before any submodule asks for Config.
    """
    key = name.upper()

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
