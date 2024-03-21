import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb
import models, importlib

# Initialize wandb
wandb.init(
    project = "AVSP8",
    
    config = {
        "learning_rate":    1e-3,
        "epochs":           5,
        "accelerator":      "cpu",
        "model":            models.ResNet18()
    }
)

def main():

    pl.seed_everything(42, workers=True)

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    mnist_train = torch.utils.data.Subset(mnist_train, range(10))
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    mnist_test = torch.utils.data.Subset(mnist_test, range(10))

    model = getattr(importlib.import_module('models'), wandb.config.model)()
    model.lr = wandb.config['learning_rate']
    train_dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True) # TODO : UPDATE
    # val_dataloader = None # TODO : UPDATE
    test_dataloader = DataLoader(mnist_test, batch_size=1000, shuffle=False) # TODO : UPDATE

    trainer = pl.Trainer(
        max_epochs=wandb.config['epochs'],
        accelerator=wandb.config['accelerator'],
        # deterministic=True,
        log_every_n_steps=1,
        logger=WandbLogger(log_model="all"),
    )

    trainer.fit(model, train_dataloader)
    test_results = trainer.test(model, dataloaders=test_dataloader, verbose=True)

    # Print the test accuracy
    print(f'Test Loss: {test_results[0]["test_loss"]:.4f}')
    print(f'Test Accuracy: {test_results[0]["test_accuracy"]:.4f}')
    
    wandb.finish()

if __name__ == '__main__':
    main()