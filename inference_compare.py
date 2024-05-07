import os
from dataset import dataset_util
from custom_tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
import wandb
import matplotlib.pyplot as plt
import torchaudio

# Dataset and save path
current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, "runs")
dataset_path = os.path.join(current_path, "libriTTS")

# Dataset names
dataset_subsets = {
    "libri-tts-clean-100": 'train-clean-100',   # 這裡是Ollie做的
    "libri-tts-test-clean": 'test-clean',       # 這裡是Tonko做的
}

# define model config
config = {
    "batch_size": 8,
    "eval_batch_size": 8,
    "num_loader_workers": 0,
    "num_eval_loader_workers": 0,
    "precompute_num_workers": 0,
    "run_eval": True,
    "test_delay_epochs": -1,
    "epochs": 2,
    "lr": 1e-4,
    "print_step": 50,
    "print_eval": True,
    "mixed_precision": False,
    "output_path": output_path,
    "use_speaker_embedding": True,
    "min_text_len": 0,
    "max_text_len": 500,
    "min_audio_len": 5000,
    "max_audio_len": 250000,
    "double_decoder_consistency": True,
    "text_cleaner": "english_cleaners",
    "shuffle": True,
}

# wandb.init(
#     project="AVSP8",                            # Project name
#     entity="qwewef",                            # Entity name
#     config=config,                              # Configuration dictionary
# )

# download the dataset if not downloaded
dataset_configs = dataset_util.download_dataset("LibriTTS", dataset_path, dataset_subsets, formatter="libri_tts")

ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config(config)
# ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config("weights/config_5256.json")
train_samples, eval_samples, test_samples = dataset_util.load_samples(dataset_configs, tacotron2_config)
    
# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples + test_samples, parse_key="speaker_name")

# Load models
tacotron2 = Tacotron2(tacotron2_config, ap, tokenizer, speaker_manager=speaker_manager)

import torch
state_dict_1 = torch.load("tacotron2.pth", map_location=torch.device('cpu'))
state_dict_2 = torch.load("weights/best_model_5256.pth", map_location=torch.device('cpu'))
# del state_dict_2['model']['speaker_embedding.weight']
# Make a copy of the state dictionary
state_dict_copy = state_dict_2.copy()

# # Create a list of keys to delete
# keys_to_delete = []
# for key in state_dict_2['model'].keys():
#     if 'coarse' in key:
#         keys_to_delete.append(key)

# # Delete the specified keys from the state dictionary copy
# for key in keys_to_delete:
#     del state_dict_copy['model'][key]

# state_dict_copy['model']['speaker_embedding.weight'] = state_dict_copy['model']['speaker_embedding.weight'][:247, :]
target = torch.ones(276, 512)
target[:247, :] = state_dict_copy['model']['speaker_embedding.weight']
state_dict_copy['model']['speaker_embedding.weight'] = target
# state_dict_copy['model']['decoder.attention_rnn.weight_ih'] = state_dict_copy['model']['decoder.attention_rnn.weight_ih'][:, :768]
# state_dict_copy['model']['decoder.attention.inputs_layer.linear_layer.weight'] = state_dict_copy['model']['decoder.attention.inputs_layer.linear_layer.weight'][:, :512]
# state_dict_copy['model']['decoder.decoder_rnn.weight_ih'] = state_dict_copy['model']['decoder.decoder_rnn.weight_ih'][:, :1536]
# state_dict_copy['model']['decoder.linear_projection.linear_layer.weight'] = state_dict_copy['model']['decoder.linear_projection.linear_layer.weight'][:, :1536]

# Save the modified state dictionary
torch.save(state_dict_copy, "weights/new_best_model_5256.pth")

# state_dict_2 = state_dict_copy['model']
# # Check if the keys (parameter names) are the same in both state dictionaries
# if state_dict_1.keys() == state_dict_2.keys():
#     # Iterate over the keys (parameter names) and compare the parameters
#     for key in state_dict_1.keys():
#         param1 = state_dict_1[key]
#         param2 = state_dict_2[key]
#         if param1.shape != param2.shape:
#             print(f"'{key}' are not equal.")
#             print(f"Model 1: {param1.shape}")
#             print(f"Model 2: {param2.shape}")
#             print()
# else:
#     print("The models have different architectures or parameters.")
#     print(f'Model 1: {len(list(state_dict_1.keys()))} keys')
#     print(f'Model 2: {len(list(state_dict_2.keys()))} keys')
#     for i in range(len(list(state_dict_2.keys()))):
#         state_dict_1_key = list(state_dict_1.keys())[i]
#         state_dict_2_key = list(state_dict_2.keys())[i]
#         if state_dict_1_key != state_dict_2_key:
#             print(i, state_dict_1_key, '-', state_dict_2_key)
# exit()

# Load checkpoint
# CLAAUDIA: /home/student.aau.dk/lk83xy/avs8/AVS8/runs/run-May-07-2024_12+08AM-3f6f821/
# Marko Local: /home/putak/university/8semester/Project/
# weights_config = Tacotron2Config("/home/student.aau.dk/lk83xy/avs8/AVS8/runs/run-May-07-2024_12+08AM-3f6f821/config.json")
# tacotron2.load_checkpoint(config=weights_config, checkpoint_path="/home/student.aau.dk/lk83xy/avs8/AVS8/runs/run-May-07-2024_12+08AM-3f6f821/best_model_5256.pth")
# weights_config = Tacotron2Config("weights/config_5256.json")
# tacotron2.load_checkpoint(config=tacotron2_config, checkpoint_path="weights/new_best_model_5256.pth", eval=True)
print("LOADED!")

# Load dataloader with test samples
test_dataloader = tacotron2.get_data_loader(
    config=tacotron2_config,
    assets=None,
    is_eval=False,
    samples=test_samples,
    verbose=False,
    num_gpus=0
)
batch = next(iter(test_dataloader))

# Display first sample as text and audio
# print(f'\nraw_text sample:')
# raw_text = batch["raw_text"][0]
# print(f'> {raw_text}')
# waveform = batch["waveform"][0]
# input_file = os.path.join('output', 'input_wav.wav')
# torchaudio.save(input_file, waveform, 22050)

# Format batch and get all values
batch = tacotron2.format_batch(batch)
text_input = batch["text_input"]
text_lengths = batch["text_lengths"]
mel_input = batch["mel_input"]
mel_lengths = batch["mel_lengths"]
speaker_ids = batch["speaker_ids"]
d_vectors = batch["d_vectors"]
spk_emb1 = batch["spk_emb"]
pos_emb = batch["pos_emb"]
aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
inference_outputs = tacotron2.inference(text_input, aux_input, spk_emb1, save_wav=True)
print("INFERENCE DONE")

# Create a figure and axes for subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Plot mel_input
im1 = axes[0].imshow(mel_input[0].numpy().T, aspect='auto', origin='lower')
axes[0].set_title('Input Mel Spectrogram')
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Mel Filter')
plt.colorbar(im1, ax=axes[0])
# Plot model output
im2 = axes[1].imshow(inference_outputs['model_outputs'][0].numpy().T, aspect='auto', origin='lower')
axes[1].set_title('Output Mel Spectrogram')
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Mel Filter')
plt.colorbar(im2, ax=axes[1])
# Add a common title for the whole figure
plt.suptitle('Comparison of Input and Output Mel Spectrograms')
# Save the figure
plt.savefig(os.path.join('output', 'mel_spectrogram_comparison.png'))
