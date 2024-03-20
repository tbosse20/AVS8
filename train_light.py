import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
    
pl.seed_everything(42, workers=True)

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform)
mnist_train = torch.utils.data.Subset(mnist_train, range(10))
mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform)
mnist_test = torch.utils.data.Subset(mnist_test, range(10))

class ResNetModule(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.resnet = models.resnet18(weights="DEFAULT")
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        self.validation_step_outputs = []  # Empty list to store outputs

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        accuracy = torch.tensor(correct / len(batch))
        output = {"test_loss": loss, "test_accuracy": accuracy}
        self.validation_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.validation_step_outputs
        
        # Calculate aggregate metrics, log them, or perform any other actions
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('test_loss', test_loss)
        self.log('test_accuracy', test_accuracy)
        
        # Clear the list for the next epoch
        self.validation_step_outputs = []

model = ResNetModule(10) # TODO : UPDATE
train_dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True) # TODO : UPDATE
# val_dataloader = None # TODO : UPDATE
test_dataloader = DataLoader(mnist_test, batch_size=1000, shuffle=False) # TODO : UPDATE

logger = TensorBoardLogger("logs/", name="AVS8")

trainer = pl.Trainer(
    max_epochs=5,
    accelerator="cpu",
    deterministic=True,
    log_every_n_steps=100,
    logger=logger
)

trainer.fit(model, train_dataloader)
test_results = trainer.test(model, dataloaders=test_dataloader, verbose=True)

# Access the test accuracy from the test results
test_accuracy = test_results[0]['test_accuracy']

# Print the test accuracy
print(f'Test Accuracy: {test_accuracy:.4f}')