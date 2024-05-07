import os
from trainer import Trainer, TrainerArgs
# import numpy as np
from TTS.tts.models.tacotron2 import Tacotron2
from custom_tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
import wandb
# from TTS.api import load_config
# from TTS.tts.utils.synthesis import synthesis
# from TTS.vocoder.models.gan import GAN
# import torchaudio
# import logging
# logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow INFO and WARNING messages
import argparse
import dataset.dataset_util as dataset_util
import gc

# Python cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--notes", type=str,      help="Notes for the run")
parser.add_argument("-dev", action="store_true",    help="Enable development mode")
args = parser.parse_args()

# Vocoder
VOCODER_MODEL = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/model_file.pth"
VOCODER_CONFIG = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/config.json"

# Tacotron2
TACO_MODEL = "./tts_model/tts_models--en--ek1--tacotron2/model_file.pth"
TACO_CONFIG = "./tts_model/tts_models--en--ek1--tacotron2/config.json"

# Dataset and save path
current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, "runs")
dataset_path = os.path.join(current_path, "libriTTS")

# Dataset names
dataset_subsets = {
    "libri-tts-clean-100": 'train-clean-100',   # 這裡是Ollie做的
    "libri-tts-test-clean": 'test-clean',       # 這裡是Tonko做的
}

# download the dataset if not downloaded
dataset_configs = dataset_util.download_dataset("LibriTTS", dataset_path, dataset_subsets, formatter="libri_tts")
gc.collect()

# # define audio config
# # ❗ resample the dataset externally using `TTS/bin/resample.py` and set `resample=False` for faster training
# audio_config = BaseAudioConfig(sample_rate=24000, resample=False, do_trim_silence=False)

# define model config
config = {
    "batch_size": 16,
    "eval_batch_size": 8,
    "num_loader_workers": 4,
    "num_eval_loader_workers": 4,
    "precompute_num_workers": 4,
    "run_eval": True,
    "test_delay_epochs": -1,
    "epochs": 2 if args.dev else 100,
    "lr": 1e-4,
    "print_step": 25,
    "print_eval": True,
    "mixed_precision": False,
    "output_path": output_path,
    "use_speaker_embedding": True,
    "min_text_len": 0,
    "max_text_len": 500,
    "min_audio_len": 10000,
    "max_audio_len": 250000,
    "double_decoder_consistency": True,
    "text_cleaner": "english_cleaners",
    # "infoNCE_alpha": 0.2,
}

wandb.init(
    project="AVSP8",                            # Project name
    entity="qwewef",                            # Entity name
    config=config,                              # Configuration dictionary
    # notes=args.notes if args.notes else "",     # Notes
    tags=[
        "dev" if args.dev else "full"           # Run mode tag
    ]
)

ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config(config)
train_samples, eval_samples, test_samples = dataset_util.load_samples(dataset_configs, tacotron2_config)
    
# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
gc.collect()

# init model
model = Tacotron2(tacotron2_config, ap, tokenizer, speaker_manager=speaker_manager)
# model.load_checkpoint(config=TACO_CONFIG, checkpoint_path=TACO_MODEL)

# # INITIALIZE THE TRAINER
# # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# # distributed training, etc.
trainer = Trainer(
    config=tacotron2_config,
    output_path=output_path,
    args=TrainerArgs(),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    test_samples=test_samples,
)
gc.collect()

# Dev mode: reduce the number of samples
if args.dev:
    trainer.setup_small_run(8)

trainer.fit()
