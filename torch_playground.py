from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
from torchaudio import load
from librosa.core import resample
import torch
import numpy as np
import librosa

# Make this a function.


def spk_embedding(audio, sr:int = 16000, feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv"), 
                  model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")) -> torch.Tensor:
    audio = resample(np.array(audio), orig_sr=sr, target_sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**inputs).embeddings

    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    return embeddings

# Load audio using torchaudio.load() or any other method
# audio, sr = load("/Users/chengyulin/Desktop/CSIE/Aalborg/Project/AVS8/libriTTS/LibriTTS/train-clean-100/1624/142933/1624_142933_000005_000002.wav")
file_name = "/Users/chengyulin/Desktop/CSIE/Aalborg/Project/AVS8/libriTTS/LibriTTS/train-clean-100/1624/142933/1624_142933_000005_000002.wav"
audio, sr = load(file_name)
# audio, _ = librosa.load(file_name)
embeddings = spk_embedding(audio)
print(embeddings.shape)
