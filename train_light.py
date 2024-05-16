import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset, Dataset
import wandb
import encoders, decoders, importlib
from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader
from data_loader.data_loaders import LIBRITTS_Dataset, train_valid_split, load_data, CustomLIBRITTS
from torchaudio.datasets import LIBRITTS
import os
import torch.nn as nn

# wandb.init(
#     project = "AVSP8",
    
#     config = {
#         "learning_rate":    1e-3,
#         "epochs":           5,
#         "accelerator":      "cpu",
#         "encoder":          encoders,
#         "decoder":          decoders.
#     }
# )

def main():
    
    input_dim = 1  # Size of the vocabulary for the source language
    output_dim = 1  # Size of the vocabulary for the target language
    hidden_dim = 256  # Hidden dimension size for the LSTM units
    num_layers = 2  # Number of layers in the LSTM encoder and decoder

    pl.seed_everything(42, workers=True)
    
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")
    
    train_set, test_set = LIBRITTS_Dataset(data_dir, test=True, download=True)
    train_set, valid_set = train_valid_split(train_set)
    train_loader, valid_loader, test_loader = [load_data(x, 1) for x in [train_set, valid_set, test_set]]

    encoder = encoders.Encoder(input_dim, hidden_dim, num_layers)
    decoder = decoders.Decoder(input_dim, hidden_dim, output_dim, num_layers)
    # encoder = getattr(importlib.import_module('encoders'), wandb.config.encoder)()
    # decoder = getattr(importlib.import_module('decoders'), wandb.config.decoder)()
    criterion = nn.MSELoss()

    model = Seq2Seq(encoder, decoder, criterion)
    # model.lr = wandb.config['learning_rate']
    model.lr = 1e-3

    trainer = pl.Trainer(
        # max_epochs=wandb.config.epochs,
        max_epochs=2,
        # accelerator=wandb.config.accelerator,
        accelerator="cpu",
        # deterministic=True,
        log_every_n_steps=100,
        # logger=WandbLogger(log_model="all"),
    )

    trainer.fit(model, train_loader, valid_loader)
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Print the test accuracy
    print(f'Test Loss: {test_results[0]["test_loss"]:.4f}')
    print(f'Test Accuracy: {test_results[0]["test_accuracy"]:.4f}')
    
    # wandb.finish()

if __name__ == '__main__':
    main()