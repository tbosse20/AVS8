import torch.nn as nn
import torch
import os
from data_loader.data_loaders import LIBRITTS_Dataset

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        print(h0.shape)
        print(c0.shape)

        out, _ = self.lstm(x, (h0, c0))
        return out

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
    
    encoder = Encoder(1, 256, 2)
    # dummy_input = torch.randn(3, 10, 1)
    embedding = encoder(audios)
    print(f'Embedding: {embedding}')
    print(f'Embedding shape: {embedding.shape}')