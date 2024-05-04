import os
from trainer import Trainer, TrainerArgs
# import numpy as np
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from dataset import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from custom_tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
# from TTS.api import load_config
# from TTS.tts.utils.synthesis import synthesis
# from TTS.vocoder.models.gan import GAN
# import torchaudio
import wandb
# import logging
# logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow INFO and WARNING messages
import argparse
from dataset.dataset_util import download_dataset
import gc

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--notes", type=str, help="Notes for the run")
parser.add_argument("-dev", action="store_true", help="Enable development mode")
args = parser.parse_args()

VOCODER_MODEL = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/model_file.pth"
VOCODER_CONFIG = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/config.json"

TACO_MODEL = "./tts_model/tts_models--en--ek1--tacotron2/model_file.pth"
TACO_CONFIG = "./tts_model/tts_models--en--ek1--tacotron2/config.json"

# set experiment paths
current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, "runs")
dataset_path = os.path.join(current_path, "libriTTS")


# download the dataset if not downloaded
if not os.path.exists(dataset_path):
    from TTS.utils.downloaders import download_libri_tts

    download_libri_tts(dataset_path, subset="libri-tts-clean-100") #這裡是Ollie做的

print("downloaded data")
gc.collect()
# define dataset config
dataset_config = BaseDatasetConfig(formatter="libri_tts", meta_file_train="", path=dataset_path)

# define audio config
# ❗ resample the dataset externally using `TTS/bin/resample.py` and set `resample=False` for faster training
audio_config = BaseAudioConfig(sample_rate=24000, resample=False, do_trim_silence=False)

# define model config
# config = load_config(TACO_CONFIG)
config = {
    "batch_size": 16,
    "eval_batch_size": 8,
    "num_loader_workers": 0,
    "num_eval_loader_workers": 0,
    "precompute_num_workers": 0,
    "run_eval": True,
    "test_delay_epochs": -1,
    "epochs": 3 if args.dev else 100,
    "lr": 1e-4,
    "print_step": 1,
    "print_eval": True,
    "mixed_precision": False,
    "output_path": output_path,
    "datasets": [dataset_config],
    "use_speaker_embedding": True,
    "min_text_len": 0,
    "max_text_len": 500,
    "min_audio_len": 0,
    "max_audio_len": 500000,
    "double_decoder_consistency": True,
    "text_cleaner": "english_cleaners",
    # "infoNCE_alpha": 0.2,
}

wandb.init(
    project="AVSP8",                            # Project name
    entity="qwewef",                            # Entity name
    config=config,                              # Configuration dictionary
    notes=args.notes if args.notes else "",     # Notes
    tags=[
        "dev" if args.dev else "full"           # Run mode tag
    ]
)

tacotron2_config = Tacotron2Config(**config)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(tacotron2_config)  #, verbose=False)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(tacotron2_config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=tacotron2_config.eval_split_max_size,
    eval_split_size=tacotron2_config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
gc.collect()
# init model
model = Tacotron2(tacotron2_config, ap, tokenizer, speaker_manager=speaker_manager)
# model.load_checkpoint(config=TACO_CONFIG, checkpoint_path=TACO_MODEL)

# output = synthesis(model=model, text="My name is Jeff.", CONFIG=config, use_cuda=False)
# output = output['outputs']['model_outputs']

# vocoder = GAN(VOCODER_CONFIG)
# vocoder.load_checkpoint(config=VOCODER_CONFIG, checkpoint_path=VOCODER_MODEL, eval=True)

# output = output.permute(0,2,1)
# postnet_outputs = vocoder.inference(output)
# torchaudio.save("output.wav", postnet_outputs[0], 22050)

# voice = model.inference("My name is Jeff")
# quit()
# print("start training  ")
# # INITIALIZE THE TRAINER
# # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# # distributed training, etc.

# train_samples = train_samples[:32]
# eval_samples = eval_samples[:32]

trainer = Trainer(
    config=tacotron2_config,
    output_path=output_path,
    args=TrainerArgs(),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    # train_loader=model.get_data_loader(config=tacotron2_config, assets=None, is_eval=False, samples=train_samples, verbose=False, num_gpus=1),
    # eval_loader=model.get_data_loader(config=tacotron2_config, assets=None, is_eval=True, samples=eval_samples, verbose=False, num_gpus=1),
    test_samples=eval_samples, # TODO: Load and change this to test_samples
)
gc.collect()

# Dev mode: reduce the number of samples
if args.dev:    
    trainer.setup_small_run(8)

trainer.fit()
