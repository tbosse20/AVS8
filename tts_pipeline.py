import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.api import load_config


VOCODER_MODEL = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/model_file.pth"
VOCODER_CONFIG = "./vocoder/vocoder_models--universal--libri-tts--fullband-melgan/config.json"

TACO_MODEL = "./tts_model/tts_models--en--ek1--tacotron2/model_file.pth"
TACO_CONFIG = "./tts_model/tts_models--en--ek1--tacotron2/config.json"

# voc_config_from_path = load_config(VOCODER_CONFIG)


# voc_model = GAN(voc_config_from_path)

# print(voc_model.model_d)
# print(voc_model.model_g)

# set experiment paths
current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, "runs")
dataset_path = os.path.join(current_path, "libriTTS")

# download the dataset if not downloaded
if not os.path.exists(dataset_path):
    from TTS.utils.downloaders import download_libri_tts

    download_libri_tts(dataset_path, subset="libri-tts-clean-100") #這裡是Ollie做的

print("downloaded data")
# define dataset config
dataset_config = BaseDatasetConfig(formatter="libri_tts", meta_file_train="", path=dataset_path)

# define audio config
# ❗ resample the dataset externally using `TTS/bin/resample.py` and set `resample=False` for faster training
audio_config = BaseAudioConfig(sample_rate=24000, resample=False, do_trim_silence=False)

# define model config
config = load_config(TACO_CONFIG)
# config = Tacotron2Config(
#     batch_size=4,
#     eval_batch_size=4,
#     num_loader_workers=0,
#     num_eval_loader_workers=0,
#     precompute_num_workers=0,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=1,
#     print_step=1,
#     print_eval=True,
#     mixed_precision=False,
#     output_path=output_path,
#     datasets=[dataset_config],
#     use_speaker_embedding=True,
#     min_text_len=0,
#     max_text_len=500,
#     min_audio_len=0,
#     max_audio_len=500000,
#     double_decoder_consistency=True,
# )

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers

# init model
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)
model.load_checkpoint(config=TACO_CONFIG, checkpoint_path=TACO_MODEL)

voice = model.inference("My name is Jeff.")
# quit()
# print("start training  ")
# # INITIALIZE THE TRAINER
# # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# # distributed training, etc.
# trainer = Trainer(
#     TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
# )

# # AND... 3,2,1... 🚀
# trainer.fit()