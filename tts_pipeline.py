import os
from custom_trainer import Trainer, TrainerArgs
from custom_tacotron2_config import Tacotron2Config
from custom_tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
import wandb
import argparse
import custom_dataset.dataset_util as dataset_util
import gc
import re
import custom_inference
import analysis.collected_losses.plot_funcs as plot_funcs
import numpy as np
import random

# Python cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--notes",    type=str,               help="Notes for the run")
parser.add_argument("--checkpoint_run", type=str,               help="Path to run checkpoint")
parser.add_argument("--dev",            action="store_true",    help="Enable development mode")
parser.add_argument("--base",           action="store_true",    help="Model baseline mode")
parser.add_argument("--unstaff",        action="store_true",    help="Disable workers")

# Select mode of operation
parser.add_argument("--train",          action="store_true",    help="Train model only")
parser.add_argument("--test",           action="store_true",    help="Run test phase only")
parser.add_argument("--inference",      action="store_true",    help="Run inference on single sample")
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
    "num_loader_workers": 0 if args.unstaff else 4,
    "num_eval_loader_workers": 0 if args.unstaff else 4,
    "precompute_num_workers": 0 if args.unstaff else 4,
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
    "return_wav": False,
    "max_decoder_steps": 1000,
}

# Make notes to add to the wandb config
notes = ""
if args.checkpoint_run:
    notes += f'checkpoint={args.checkpoint_run}'
if args.notes:
    notes += f', {args.notes}'

# Initialize wandb
if args.train or args.test:
    wandb.init(
        project="AVSP8",                                   # Project name
        entity="qwewef",                                   # Entity name
        config=config,                                     # Configuration dictionary
        notes=notes,                                       # Add notes to config
        tags=[
            "dev" if args.dev else "product",              # Run development mode
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


def get_largest(f):
    s = re.findall(r'\d+$', f)
    return (int(s[0]) if s else -1, f)


# Load weights
if args.checkpoint_run:
    pth_list = os.listdir(args.checkpoint_run)
    checkpoint_list = [
        f
        for f in pth_list
        if f.startswith('checkpoint') or f.startswith('best_model')
    ]
    try:
        largest_checkpoint = max(checkpoint_list, key=get_largest)
    except ValueError:
        largest_checkpoint = "best_model.pth"

    # Load model from checkpoint
    tacotron2_config = Tacotron2Config()
    tacotron2_config.load_json(os.path.join(args.checkpoint_run, "config.json"))

    # Update config values as needed
    if args.unstaff:
        tacotron2_config.num_loader_workers = 0
        tacotron2_config.num_eval_loader_workers = 0
        tacotron2_config.precompute_num_workers = 0

    # Update 'speaker.pth' path automatically
    tacotron2_config.speakers_file = os.path.join(args.checkpoint_run, 'speakers.pth')

    # Load model from checkpoint
    model = Tacotron2.init_from_config(tacotron2_config)
    model.load_checkpoint(
        config=tacotron2_config,
        checkpoint_path=os.path.join(args.checkpoint_run, largest_checkpoint),
        eval=True,
    )
    print(f'\n{80*"*"}' + '\nModel loaded from checkpoint:', args.checkpoint_run + "\n" + 80 * "*")


# Run the selected phase
if args.train or args.test:
    # # INITIALIZE THE TRAINER
    # # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
    # # distributed training, etc. continue_path=args.checkpoint_run
    trainer = Trainer(
        config=tacotron2_config,
        output_path=output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        test_samples=test_samples,
        args=TrainerArgs(
            continue_path=args.checkpoint_run,  # Such an elegant way to continue training
            # skip_train_epoch=args.test,       # Skip training phase
            small_run=16 if args.dev else None,  # Reduce number of samples
        ),
    )
    gc.collect()

if args.train:
    trainer.fit()
if args.test:
    collected_losses = trainer.test()
    plot_funcs.save_collected_losses(collected_losses, tacotron2_config.infoNCE_alpha, args.checkpoint_run)
    
if args.inference:
    custom_inference.inference(
        model,
        test_samples,
        tacotron2_config,
        checkpoint_run=args.checkpoint_run,
        idx=35,
    )