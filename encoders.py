import torch.nn as nn
import torch
import os
from data_loader.data_loaders import LIBRITTS_Dataset

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        embedded = embedded.unsqueeze(0)
        output, hidden = self.gru(embedded)
        return output.squeeze(0), hidden.squeeze(0)

    def __repr__(self):
        return self._get_name()

if __name__ == '__main__':
    
    # work_dir = os.getcwd()
    # dataset_dir = os.path.join(work_dir, "data")
    
    # ds = LIBRITTS_Dataset(data_dir=dataset_dir, batch_size=1) 
    # ds.prepare_data()
    # ds.setup()
    # audios, labels = next(iter(ds.test_dataloader()))
    # print(f'"Audios" shape: {audios.shape}')
    
    encoder = Encoder(10000, 10)
    dummy_input = torch.randint(0, 10000, (1, 307920))
    audios = dummy_input
    embedding, _ = encoder(audios.long())
    print(f'Embedding 0: {embedding[0]}')
    print(f'Embedding 0 shape: {embedding[0].size()}')