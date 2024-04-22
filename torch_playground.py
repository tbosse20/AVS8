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
# file_name = "/Users/chengyulin/Desktop/CSIE/Aalborg/Project/AVS8/libriTTS/LibriTTS/train-clean-100/8238/283452/8238_283452_000006_000000.wav"
# audio, sr = load(file_name)
# # audio, _ = librosa.load(file_name)
# embeddings = spk_embedding(audio)
# print(embeddings.shape)

import torch
import random


def mask_tensor(tensor, num_zeros):
    """
    Randomly masks a 1D PyTorch tensor with zeros for a specified number of zeros.

    Args:
    - tensor: Input 1D PyTorch tensor
    - num_zeros: Number of zeros to mask

    Returns:
    - masked_tensor: Tensor with randomly masked zeros
    """

    # Get the length of the input tensor
    tensor_length = tensor.size(1)

    # Create a copy of the input tensor
    masked_tensor = tensor.clone()

    # Generate random indices to mask
    zero_indices = random.sample(range(tensor_length), num_zeros)

    # Mask the selected indices with zeros
    masked_tensor[:, zero_indices] = 0

    return masked_tensor

# Example usage:
input_tensor = torch.rand(1,10)
print(input_tensor.shape)
print(input_tensor.size(1))
num_zeros_to_mask = 3
masked_result = mask_tensor(input_tensor, num_zeros_to_mask)
print("Input Tensor:", input_tensor)
print("Masked Tensor:", masked_result)

###########################-_____________-------######3
# what_does_troch_Stack_do = []
# for i in range(5):
    # what_does_troch_Stack_do.append(torch.rand(1,10))
# print(torch.stack(what_does_troch_Stack_do).shape)

# normal_list = [1,2,3,4]
# print(normal_list[::-1])

# Assuming tensor is your tensor of shape [4, 1, 512]
tensor = torch.stack([torch.randn(1, 512) for x in range(4)], dim=0)
print("HAHAHA", tensor.shape)


# # Reverse along the first dimension
# reversed_tensor = torch.flip(tensor, [0])

# print("SQUEEZE:", torch.squeeze(tensor, dim=1).shape)
# print(reversed_tensor.shape)  # Output: torch.Size([4, 1, 512])

# import matplotlib.pyplot as plt

# # Calculate cosine similarity between embeddings
cos_sim = torch.nn.CosineSimilarity(dim=2)
cos_sim = cos_sim(tensor, tensor)
print(cos_sim.shape)
weird_cos = (cos_sim -1) / 2
print(f"{torch.sum(weird_cos, dim=0)[0]}")
# Plot the cosine similarity matrix
# plt.imshow(cos_sim, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Cosine Similarity Matrix')
# plt.xlabel('Embedding Index')
# plt.ylabel('Embedding Index')
# plt.show()


def find_duplicates(lst):
    duplicates = []
    seen = set()
    for i, item in enumerate(lst):
        if item in seen:
            duplicates.append(i)
        else:
            seen.add(item)
    return duplicates

# Example usage:
my_list = [1, 2, 3, 4, 2, 5, 3, 6]
duplicate_indices = find_duplicates(my_list)
print(duplicate_indices)