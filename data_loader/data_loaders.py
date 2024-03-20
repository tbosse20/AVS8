from torchaudio.datasets import LIBRITTS
from torch.utils.data import random_split, DataLoader
from torch import Generator

def LIBRITTS_Dataset(root, train_url ="train-clean-100", test=False, test_url="test-clean", download=False):
    '''
    Function that fetches LIBRITTS dataset.
    :root: Root directory of the dataset
    :train_url: Which train set to use
    :test: Fetch test set
    :test_url: Which test set to use
    :download: Download the dataset from web

    Return -> train_set or train_set, test_set
    '''
    train_set = LIBRITTS(root, url=train_url, download=download)
    if not test:
        return train_set
    test_set = LIBRITTS(root, url=test_url, download=download)
    return train_set, test_set


def train_valid_split(train_set, train_ratio=0.8, random_seed=42):
    
    seed = Generator().manual_seed(random_seed)
    train_set, valid_set = random_split(train_set, [train_ratio, 1-train_ratio], generator=seed)

    return train_set, valid_set


def load_data(data, batch_size, shuffle=True, num_workers=4):
    return DataLoader(data, batch_size, shuffle=shuffle, num_workers=num_workers)
# print(train_loader)


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
