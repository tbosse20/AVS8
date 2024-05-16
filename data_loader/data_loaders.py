from torchaudio.datasets import LIBRITTS
from torch.utils.data import random_split, DataLoader, Dataset
from torch import Generator
import os
import torch
from torch.nn.utils.rnn import pad_sequence

def LIBRITTS_Dataset(root, train_url="train-clean-100", test=False, test_url="test-clean", download=False):
    '''
    Function that fetches LIBRITTS dataset.
    :root: Root directory of the dataset
    :train_url: Which train set to use
    :test: Fetch test set
    :test_url: Which test set to use
    :download: Download the dataset from web

    Return -> train_set or train_set, test_set
    '''
    train_set = CustomLIBRITTS(root, url=train_url, download=download)
    if not test: return train_set
    test_set = CustomLIBRITTS(root, url=test_url, download=download)
    return train_set, test_set


def train_valid_split(train_set, train_ratio=0.8, random_seed=42):
    seed = Generator().manual_seed(random_seed)
    train_set, valid_set = random_split(train_set, [train_ratio, 1-train_ratio], generator=seed)

    return train_set, valid_set

def load_data(data, batch_size, shuffle=True, num_workers=4):
    return DataLoader(data, batch_size, shuffle=shuffle, num_workers=num_workers)

class CustomLIBRITTS(Dataset):
    def __init__(self, root, url, download=True):
        self.libritts_dataset = LIBRITTS(root=root, url=url, download=download)

    def __len__(self):
        return len(self.libritts_dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id = self.libritts_dataset[idx]
        return waveform, normalized_text

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

if __name__ == '__main__':
    
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")
    
    train_set, test_set = LIBRITTS_Dataset(data_dir, test=True, download=True)
    train_set, valid_set = train_valid_split(train_set)
    train_loader, valid_loader, test_loader = [load_data(x, 2) for x in [train_set, valid_set, test_set]]

    print(len(next(iter(train_loader))))