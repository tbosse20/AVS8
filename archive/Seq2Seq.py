
import pytorch_lightning as pl
import torch
import decoders, encoders
import os
from data_loader.data_loaders import LIBRITTS_Dataset
import torch.nn as nn

class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder, criterion):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

    def forward(self, source, target):
        encoder_output = self.encoder(source)
        output = self.decoder(target, encoder_output)
        return output

    def _loss_acc_step(self, batch, batch_idx):
        source, target = batch
        
        # Compute loss
        output = self(source, target)
        loss = self.criterion(output, target)
        
        # Compute accuracy
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        accuracy = correct / len(target)
        
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._loss_acc_step(batch, batch_idx)
        
        # Log
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._loss_acc_step(batch, batch_idx)
        
        # Log
        self.log('valid_loss', loss)
        self.log('valid_accuracy', accuracy)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, accuracy = self._loss_acc_step(batch, batch_idx)
        
        output = {"test_loss": loss, "test_accuracy": accuracy}
        self.test_step_outputs.append(output)
        
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        
        # Calculate aggregate metrics, log them, or perform any other actions
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_accuracy = torch.tensor([x['test_accuracy'] for x in outputs]).mean()
        
        # Log
        self.log('test_loss', test_loss)
        self.log('test_accuracy', test_accuracy)
        
        # Clear the list for the next epoch
        self.test_step_outputs = []

if __name__ == '__main__':
    
    input_dim = 1  # Size of the vocabulary for the source language
    output_dim = 1  # Size of the vocabulary for the target language
    hidden_dim = 256  # Hidden dimension size for the LSTM units
    num_layers = 2  # Number of layers in the LSTM encoder and decoder

    pl.seed_everything(42, workers=True)
    
    work_dir = os.getcwd()
    dataset_dir = os.path.join(work_dir, "data")
    ds = LIBRITTS_Dataset(data_dir=dataset_dir, batch_size=1) 
    ds.prepare_data()
    ds.setup()
    
    train_loader = ds.train_dataloader()
    valid_loader = ds.val_dataloader()
    test_loader = ds.test_dataloader()
    
    encoder = encoders.Encoder(input_dim, hidden_dim, num_layers)
    decoder = decoders.Decoder(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.MSELoss()
    
    model = Seq2Seq(encoder, decoder, criterion)
    model.lr = 1e-3

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
        log_every_n_steps=100,
    )

    trainer.fit(model, train_loader, valid_loader)
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Print the test accuracy
    print(f'Test Loss: {test_results[0]["test_loss"]:.4f}')
    print(f'Test Accuracy: {test_results[0]["test_accuracy"]:.4f}')