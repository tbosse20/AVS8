import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import encoders, decoders, importlib
from Seq2Seq import Seq2Seq
from data_loader.data_loaders import LIBRITTS_Dataset
import os
import torch.nn as nn

wandb.init(
    project = "AVSP8",
    
    config = {
        "learning_rate":    1e-3,
        "epochs":           5,
        "accelerator":      "cpu",
        "input_dim":        1,
        "output_dim":       1,
        "hidden_dim":       256,
        "num_layers":       2,
        "encoder":          encoders.Encoder(),
        "decoder":          decoders.Decoder(),
    }
)

def main():

    pl.seed_everything(42, workers=True)
    
    work_dir = os.getcwd()
    dataset_dir = os.path.join(work_dir, "data")
    
    ds = LIBRITTS_Dataset(data_dir=dataset_dir, batch_size=1) 
    ds.prepare_data()
    ds.setup()
    
    train_loader = ds.train_dataloader()
    valid_loader = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    encoder = getattr(
        importlib.import_module('encoders'),
        wandb.config.encoder)(
            wandb.config.input_dim, wandb.config.hidden_dim,
            wandb.config.num_layers)
    decoder = getattr(
        importlib.import_module('decoders'),
        wandb.config.decoder)(
            wandb.config.input_dim, wandb.config.hidden_dim,
            wandb.config.output_dim, wandb.config.num_layers)
    criterion = nn.MSELoss()

    model = Seq2Seq(encoder, decoder, criterion)
    model.lr = wandb.config['learning_rate']

    trainer = pl.Trainer(
        max_epochs=wandb.config.epochs,
        accelerator=wandb.config.accelerator,
        deterministic=True,
        logger=WandbLogger(log_model="all"),
    )

    trainer.fit(model, train_loader, valid_loader)
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Print the test accuracy
    print(f'Test Loss: {test_results[0]["test_loss"]:.4f}')
    print(f'Test Accuracy: {test_results[0]["test_accuracy"]:.4f}')
    
    wandb.finish()

if __name__ == '__main__':
    main()