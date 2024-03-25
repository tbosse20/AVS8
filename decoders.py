import torch.nn as nn
import torch
import torch
import os
from data_loader.data_loaders import LIBRITTS_Dataset
import encoders

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_output):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
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
    
    embedding = encoders.Encoder(1, 256, 2)(audios)
    
    decoder = Decoder(1, 256, 1, 2)
    output = decoder(embedding)
    print(output)
    print(output.shape)