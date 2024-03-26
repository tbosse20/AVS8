from torchaudio.datasets import LIBRITTS
from torch.utils.data import random_split, DataLoader, Dataset
import lightning as L
import os

class LIBRITTS_Dataset(L.LightningDataModule):
    '''
    Dataset class. 
    Handles dataset downloading, preparing, splitting and loading.
    '''
    def __init__(self,
                data_dir: str = "./data",
                train_url: str = "train-clean-100",
                test_url: str = "test-clean",
                train_ratio: float = 0.8,
                batch_size: int = 32,
                num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.train_url = train_url
        self.test_url = test_url
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        CustomLIBRITTS(root=self.data_dir, url=self.train_url, download=True)
        CustomLIBRITTS(root=self.data_dir, url=self.test_url, download=True)

    def setup(self):
        libritts_full = CustomLIBRITTS(root=self.data_dir, url=self.train_url)
        self.libritts_train, self.libritts_val = random_split(libritts_full, [self.train_ratio, 1-self.train_ratio])
        self.libritts_test = CustomLIBRITTS(root=self.data_dir, url=self.test_url)

    def train_dataloader(self):
        return DataLoader(self.libritts_train, self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.libritts_val, self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.libritts_test, self.batch_size, num_workers=self.num_workers)

class CustomLIBRITTS(Dataset):
    def __init__(self, root, url, download=False):
        self.libritts_dataset = LIBRITTS(root=root, url=url, download=download)

    def __len__(self):
        return len(self.libritts_dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id = self.libritts_dataset[idx]
        return waveform, normalized_text

if __name__ == "__main__":
    work_dir = os.getcwd()
    dataset_dir = os.path.join(work_dir, "data")
    # TODO : More than 1 batch doesn't match audio size
    ds = LIBRITTS_Dataset(data_dir=dataset_dir, batch_size=1) 
    ds.prepare_data()
    ds.setup(stage="fit")
    train_data = ds.train_dataloader()
    audios, labels = next(iter(train_data))
    print(f'{audios=}')
    print(f'{labels=}')


# def LIBRITTS_Dataset(root, train_url ="train-clean-100", test=False, test_url="test-clean", download=False):
#     '''
#     Function that fetches LIBRITTS dataset.
#     :root: Root directory of the dataset
#     :train_url: Which train set to use
#     :test: Fetch test set
#     :test_url: Which test set to use
#     :download: Download the dataset from web

#     Return -> train_set or train_set, test_set
#     '''
#     train_set = LIBRITTS(root, url=train_url, download=download)
#     if not test:
#         return train_set
#     test_set = LIBRITTS(root, url=test_url, download=download)
#     return train_set, test_set


# def train_valid_split(train_set, train_ratio=0.8, random_seed=42):
    
#     seed = Generator().manual_seed(random_seed)
#     train_set, valid_set = random_split(train_set, [train_ratio, 1-train_ratio], generator=seed)

#     return train_set, valid_set


# def load_data(data, batch_size, shuffle=True, num_workers=4):
#     return DataLoader(data, batch_size, shuffle=shuffle, num_workers=num_workers)
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