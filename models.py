import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
import torch

class ResNet18(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.resnet = models.resnet18(weights="DEFAULT")
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        self.test_step_outputs = []
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
    
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

    def forward(self, x):
        return self.resnet(x)

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
    
class NewCNN(pl.LightningModule):
    
    def __repr__(self):
        return self._get_name()