import os
from dataset import dataset_util
# from TTS.tts.models.tacotron2 import Tacotron2
from custom_tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
from trainer import Trainer, TrainerArgs
import wandb
import argparse

# Python cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test_only",      action="store_true",    help="Run test phase only")
args = parser.parse_args()

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
    "batch_size": 16,
    "eval_batch_size": 8,
    "num_loader_workers": 0,
    "num_eval_loader_workers": 0,
    "precompute_num_workers": 0,
    "run_eval": True,
    "test_delay_epochs": -1,
    "epochs": 1,
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
}

# wandb.init(
#     project="AVSP8",                            # Project name
#     entity="qwewef",                            # Entity name
#     config=config,                              # Configuration dictionary
# )

# download the dataset if not downloaded
dataset_configs = dataset_util.download_dataset("LibriTTS", dataset_path, dataset_subsets, formatter="libri_tts")

ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config(config)
train_samples, eval_samples, test_samples = dataset_util.load_samples(dataset_configs, tacotron2_config)
    
# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples + test_samples, parse_key="speaker_name")

# Load models
tacotron2 = Tacotron2(tacotron2_config, ap, tokenizer, speaker_manager=speaker_manager)

# Load checkpoint
# tacotron2.load_checkpoint( TODO: Insert correct weight path )

# trainer = Trainer(
#     config=tacotron2_config,
#     output_path=output_path,
#     model=tacotron2,
#     train_samples=train_samples,
#     eval_samples=eval_samples,
#     test_samples=test_samples,
#     args=TrainerArgs(
#         # skip_train_epoch=True,
#         small_run=4,
#     ),
# )
# trainer.fit()
# exit()

# test_dataloader = trainer.get_test_dataloader(None, test_samples, False)
test_dataloader = tacotron2.get_data_loader(
    config=tacotron2_config,
    assets=None,
    is_eval=False,
    samples=test_samples,
    verbose=False,
    num_gpus=0
)
batch = next(iter(test_dataloader))
batch = tacotron2.format_batch(batch)

# # Run 'trainer_eval_outputs'
# from trainer.generic_utils import KeepAverage
# trainer.keep_avg_eval = KeepAverage()
# trainer_eval_outputs, _ = trainer.eval_step(batch, step=0)
# print('trainer_eval_outputs')
# print(trainer_eval_outputs.keys())
# print()

# # Run 'forward pass'
# text_input = batch["text_input"]
# text_lengths = batch["text_lengths"]
# mel_input = batch["mel_input"]
# mel_lengths = batch["mel_lengths"]
# speaker_ids = batch["speaker_ids"]
# d_vectors = batch["d_vectors"]
# spk_emb1 = batch["spk_emb"]
# pos_emb = batch["pos_emb"]
# aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
# forward_outputs = tacotron2.forward(text_input, text_lengths, mel_input, mel_lengths, aux_input, spk_emb1)
# print('forward_outputs')
# print(forward_outputs.keys())
# print()

# # Run 'tacotron2_eval_outputs'
# tacotron2.eval()
# tacotron2_eval_outputs = tacotron2.eval_step(batch, criterion=tacotron2.get_criterion())
# print('tacotron2_eval_outputs')
# print(tacotron2_eval_outputs)

# Run inference
tacotron2.eval()
import torch
print(batch["text_input"].shape)
tacotron2_inference_outputs = tacotron2.inference(batch["text_input"], batch)
print('tacotron2_inference_outputs')
print(tacotron2_inference_outputs)