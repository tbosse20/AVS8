import torch
from archive.data_loader import data_loaders as dl
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)





# ds = dl.LIBRITTS_Dataset(batch_size=1, num_workers=1) 
# ds.prepare_data()
# ds.setup()
# train_data, val_data, test_data = ds.train_dataloader(), ds.val_dataloader(), ds.test_dataloader()
# embed_model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder', device="cpu")

# embedding = embed_model(next(iter(train_data))[0])
# print(embedding.shape) # IT WOOOOORKS!!!!!!!




