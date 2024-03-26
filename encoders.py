import torch.nn as nn
import torch
import os
from data_loader.data_loaders import LIBRITTS_Dataset

class Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.Linear(x.size(1), self.embedding_dim)(x)
        return x

    def __repr__(self):
        return self._get_name()

if __name__ == '__main__':
    
    work_dir = os.getcwd()
    dataset_dir = os.path.join(work_dir, "data")
    
    ds = LIBRITTS_Dataset(data_dir=dataset_dir, batch_size=1) 
    ds.prepare_data()
    ds.setup()
    audios, labels = next(iter(ds.test_dataloader()))
    print(f'"Audios" shape: {audios.shape}')
    
    encoder = Encoder()
    dummy_input = torch.rand(1, 1, 1600)
    # audios = dummy_input
    embedding = encoder(audios.float())
    print(f'Embedding (i=0) shape: {embedding[0].size()}')