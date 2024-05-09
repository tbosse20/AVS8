import os
from trainer import Trainer, TrainerArgs
# import numpy as np
# from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.configs.tacotron2_config import Tacotron2Config
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
import test_and_inference

# Python cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--notes",    type=str,               help="Notes for the run")
parser.add_argument("--checkpoint_run", type=str,               help="Path to run checkpoint")
parser.add_argument("--dev",            action="store_true",    help="Enable development mode")
parser.add_argument("--base",           action="store_true",    help="Model baseline mode")
parser.add_argument("--unstaffed",      action="store_true",    help="Disable workers")

# Select mode of operation
parser.add_argument("--train",          action="store_true",    help="Train model only")
parser.add_argument("--test",           action="store_true",    help="Run test phase only")
parser.add_argument("--inference",      action="store_true",    help="Run inference on single sample")
args = parser.parse_args()

assert not (args.test and args.inference), "Cannot run test and inference at the same time."

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
    "num_loader_workers": 0 if args.unstaffed else 4,
    "num_eval_loader_workers": 0 if args.unstaffed else 4,
    "precompute_num_workers": 0 if args.unstaffed else 4,
    "run_eval": True,
    "test_delay_epochs": 100,
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
    "infoNCE_alpha": 0.0 if args.base else 0.25,
    "similarity_loss_alpha": 0.0 if args.base else 0.25,
    "shuffle": True,
    "return_wav": True,
}

# Initialize wandb
if args.train or args.test:
    wandb.init(
        project="AVSP8",                                        # Project name
        entity="qwewef",                                        # Entity name
        config=config,                                          # Configuration dictionary
        notes=args.notes if args.notes else "",                 # Notes
        tags=[
            "dev" if args.dev else "product",                   # Run development mode
            "only_test" if args.test else "train_test",    # Phases of the run
            "baseline" if args.base else "new_model",      # Model type
        ]
    )

ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config(config)
train_samples, eval_samples, test_samples = dataset_util.load_samples(dataset_configs, tacotron2_config)
    
# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples + test_samples, parse_key="speaker_name")
gc.collect()

# init model
model = Tacotron2(tacotron2_config, ap, tokenizer, speaker_manager=speaker_manager)

# Load weights
if args.checkpoint_run:
    config = Tacotron2Config()
    config.load_json(os.path.join(args.checkpoint_run, "config.json"))
    model = Tacotron2.init_from_config(config)
    model.load_checkpoint(
        config=config,
        checkpoint_path=os.path.join(args.checkpoint_run, "checkpoint.pth"),
        eval=True,
    )
    print(80*"*" + '\nModel loaded from checkpoint:', args.checkpoint_run + "\n" + 80*"*")
    
# # INITIALIZE THE TRAINER
# # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# # distributed training, etc.
trainer = Trainer(
    config=tacotron2_config,
    output_path=output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    test_samples=test_samples,
    args=TrainerArgs(
        # skip_train_epoch=args.test,    # Skip training phase
        small_run=8 if args.dev else None,  # Reduce number of samples
    ),
)
gc.collect()

# Run the selected phase
if args.train:
    trainer.fit()
elif args.test:
    test_and_inference.test_cos_sim(model, test_samples, tacotron2_config)
elif args.inference:
    test_and_inference.inference(model, test_samples, tacotron2_config)
else:
    print("No phase selected.")