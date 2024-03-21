import torch.nn as nn
import torchvision.models as models
from Pipeline import Pipeline

class ResNet18(Pipeline):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.resnet = models.resnet18(weights="DEFAULT")
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        self.test_step_outputs = []
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.resnet(x)
    
class NewCNN(Pipeline):
    pass

if __name__ == '__main__':
    model = ResNet18()