import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import importlib
from Seq2Seq import Seq2Seq
from data_loader.data_loaders import LIBRITTS_Dataset
import os
import torch.nn as nn
from config import config, information

wandb.init(
    project = "AVSP8",
    config = config
)

def main():

    # Set the seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    work_dir = os.getcwd()
    dataset_dir = os.path.join(work_dir, information.data_dir)
    
    ds = LIBRITTS_Dataset(data_dir=dataset_dir, batch_size=config.batch_size) 
    ds.prepare_data()
    ds.setup()
    
    # Get the train, validation and test dataloaders from the dataset
    train_loader, valid_loader, test_loader = ds.train_dataloader(), ds.val_dataloader(), ds.test_dataloader()

    # Get the encoder class from the encoders module
    encoders = importlib.import_module('encoders')
    encoder_class = getattr(encoders, config.encoder)
    encoder = encoder_class(
        config.input_dim, config.hidden_dim, config.num_layers)
    
    # Get the decoder class from the decoders module
    decoders = importlib.import_module('decoders')
    decoder_class = getattr(decoders, config.decoder)
    decoder = decoder_class(
        config.input_dim, config.hidden_dim,
        config.output_dim, config.num_layers)
    
    # Define the loss function
    criterion = nn.MSELoss()

    # Create the model with the en- and decoder
    model = Seq2Seq(encoder, decoder, criterion)
    model.lr = config.learning_rate

    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        deterministic=True,
        logger=WandbLogger(log_model="all"),
    )

    # Fit and test the model
    trainer.fit(model, train_loader, valid_loader)
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Print the test accuracy
    print(f'Test Loss: {test_results[0]["test_loss"]:.4f}')
    print(f'Test Accuracy: {test_results[0]["test_accuracy"]:.4f}')
    
    wandb.finish()

if __name__ == '__main__':
    main()