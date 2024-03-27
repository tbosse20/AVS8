# %%
import os
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt

# %%
# Load audio file example and compute mel spectrogram

# Load audio file example
audio_file = '84_121123_000069_000000.wav'

audio, sr = librosa.load(audio_file, sr=22050)  # Adjust sr as needed
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=5)
mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

print(f'{mel_spectrogram.shape=}')
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
plt.colorbar()
plt.savefig('mel_spectrogram.png')

# %%

from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, "runs")

# define model config
config = Tacotron2Config(
    batch_size=4,
    eval_batch_size=4,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    precompute_num_workers=0,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1,
    print_step=1,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    # datasets=[dataset_config],
    use_speaker_embedding=True,
    min_text_len=0,
    max_text_len=500,
    min_audio_len=0,
    max_audio_len=500000,
)

ap = AudioProcessor.init_from_config(config, verbose=False)
tokenizer, config = TTSTokenizer.init_from_config(config)
tacotron2_model = Tacotron2(config, ap, tokenizer, speaker_manager=SpeakerManager())

# Assuming mel_spectrogram is your preprocessed mel spectrogram
reshaped_mel_spectrogram = torch.tensor(mel_spectrogram)

# Reshape mel spectrogram to (512, 512) using interpolation
target_shape = (512, 512)
reshaped_mel_spectrogram = reshaped_mel_spectrogram.unsqueeze(0).permute(0, 2, 1)
reshaped_mel_spectrogram = TF.resize(reshaped_mel_spectrogram, size=target_shape, interpolation=TF.InterpolationMode.BILINEAR)

input_lengths = torch.tensor([reshaped_mel_spectrogram.shape[-1]])
encoder_output_tensor  = tacotron2_model.encoder(reshaped_mel_spectrogram, input_lengths)
# print(encoder_outputs.shape)

# # Extract the embedding
# embedding = encoder_output_tensor["encoder_out"]
embedding = encoder_output_tensor
print(embedding[:, :160, :160].shape)

# Pass the embedding through the decoder
decoder_outputs, _ = tacotron2_model.decoder(
    embedding, memories=encoder_output_tensor[:, :160, :], mask=None)

# Get the predicted mel spectrogram from the decoder outputs
predicted_mel_spectrogram = decoder_outputs["mel_outputs"]

# You can also get the stop token predictions if needed
stop_token_predictions = decoder_outputs["stop_token_predictions"]

print(predicted_mel_spectrogram.shape)

# %%
# Decoder analysis

import torch

# Create dummy input tensors
embedding = torch.randn(1, 1024, 1024)
memories = torch.randn(1, 1024, 160)

# Print shapes of dummy input tensors
print("Shapes of dummy input tensors:")
print("Embedding:", embedding.shape)
print("Memories:", memories.shape)
print()

# Pass the dummy input tensors to the decoder
decoder_outputs = tacotron2_model.decoder(
    embedding,
    memories=memories,
    mask=None,
)

for i, content in enumerate(decoder_outputs):
    print(f'Element {i}')
    example = content.detach().cpu().numpy().flatten()
    print('Content:', example[:5], '...' if len(example) > 1 else '')
    print('Shape:', content.shape)
    print()


# %%
from TTS.tts.layers.tacotron.tacotron2 import Encoder
from torchvision.transforms import functional as TF

encoder = Encoder()

# Assuming mel_spectrogram is your preprocessed mel spectrogram
reshaped_mel_spectrogram = torch.tensor(mel_spectrogram)

# Reshape mel spectrogram to (512, 512) using interpolation
target_shape = (512, 512)
reshaped_mel_spectrogram = reshaped_mel_spectrogram.unsqueeze(0).permute(0, 2, 1)
reshaped_mel_spectrogram = TF.resize(reshaped_mel_spectrogram, size=target_shape, interpolation=TF.InterpolationMode.BILINEAR)

input_lengths = torch.tensor([reshaped_mel_spectrogram.shape[-1]])
encoder.in_out_channels = input_lengths.detach().cpu().numpy()[0]

# Pass mel spectrogram and input lengths through the encoder
embedding = encoder(reshaped_mel_spectrogram, input_lengths)
print(f'{embedding.shape=}')

plt.figure(figsize=(10, 4))
plt.imshow(embedding.permute(1, 2, 0).detach().numpy(), aspect='auto', origin='lower')
plt.colorbar()
plt.savefig('embedding.png')

# %%
from TTS.tts.layers.tacotron.tacotron2 import Decoder

decoder = Decoder()
synth = decoder(embedding, mel_spectrogram.shape[-1])
print(synth.shape)


# %% 


# load audio processor and speaker encoder
ap = AudioProcessor(**config.audio)

manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)

# load a sample audio and compute embedding
waveform = ap.load_wav(sample_wav_path)

mel = ap.melspectrogram(waveform)

d_vector = manager.compute_embeddings(mel.T)

# %%
