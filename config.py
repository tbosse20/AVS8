import encoders, decoders
from types import SimpleNamespace

def dict_to_namespace(d):
    """Recursively convert dictionary into SimpleNamespace."""
    namespace = SimpleNamespace(**d)
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
    return namespace

config = dict_to_namespace({
    "trainer": {
        "learning_rate":    1e-3,
        "epochs":           5,
        "accelerator":      "cpu",
    },
    "model": {
        "encoder":          encoders.Encoder,
        "decoder":          decoders.Decoder,
        "parameters": {
            "input_dim":        1,
            "output_dim":       1,
            "hidden_dim":       256,
            "num_layers":       2,
        },
    },
    "data_loader": {
        "batch_size":   32,
        "data_dir": "./data",
    },
})