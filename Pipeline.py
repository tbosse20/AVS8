
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class Pipeline(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def _loss_acc_step(self, batch, batch_idx):
        x, y = batch
        
        # Compute loss
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        accuracy = correct / len(y)
        
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

    def __repr__(self):
        return self._get_name()

if __name__ == '__main__':
    Pipeline()