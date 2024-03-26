import model.encoders, model.decoders

config = {
    "learning_rate":    1e-3,
    "epochs":           5,
    "accelerator":      "cpu",
    "input_dim":        1,
    "output_dim":       1,
    "hidden_dim":       256,
    "num_layers":       2,
    "encoder":          model.encoders.Encoder(),
    "decoder":          model.decoders.Decoder(),
    "batch_size":       32,
}

information = {
    "data_dir": "./data",
}