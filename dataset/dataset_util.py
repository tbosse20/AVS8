import os
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.shared_configs import BaseDatasetConfig
    
def download_dataset(folder_name, dataset_path, subsets, formatter):

    subfolders = os.listdir(os.path.join(dataset_path, folder_name))
    
    dataset_configs = []
    
    for subset_name, subset_folder in subsets.items():
        if not subset_folder in subfolders:
            from TTS.utils.downloaders import download_libri_tts
            print(f'Downloading: "{subset_name}" -> "{subset_folder}"')
            download_libri_tts(dataset_path, subset=subset_name)
        
        subset_folder_path = os.path.join(dataset_path, folder_name, subset_folder)
        dataset_config = BaseDatasetConfig(formatter=formatter, path=subset_folder_path)
        dataset_configs.append(dataset_config)
        
    print("Datasets downloaded successfully!")
    
    return dataset_configs

def load_tacotron2_config(config):
    tacotron2_config = Tacotron2Config(**config)

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(tacotron2_config)  #, verbose=False)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # If characters are not defined in the config, default characters are passed to the config
    tokenizer, config = TTSTokenizer.init_from_config(tacotron2_config)
    
    return ap, tokenizer, tacotron2_config

def load_samples(dataset_configs, tacotron2_config):
    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_configs[0],
        eval_split=True,
        eval_split_max_size=tacotron2_config.eval_split_max_size,
        eval_split_size=tacotron2_config.eval_split_size,
    )
    test_samples = load_tts_samples(dataset_configs[1], eval_split_max_size=tacotron2_config.eval_split_max_size)
    
    return train_samples, eval_samples, test_samples

if __name__ == '__main__':

    current_path = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.dirname(current_path)
    dataset_path = os.path.join(main_path, "libriTTS")
    
    subsets = {
        "libri-tts-clean-100": 'train-clean-100',   # 這裡是Ollie做的
        "libri-tts-test-clean": 'test-clean',       # 這裡是Tonko做的
    }
    dataset_configs = download_dataset("LibriTTS", dataset_path, subsets, formatter="libri_tts")
    
    config = {
        "batch_size": 16,
        "eval_batch_size": 8,
        "num_loader_workers": 2,
        "num_eval_loader_workers": 2,
        "precompute_num_workers": 2,
        "run_eval": True,
        "test_delay_epochs": -1,
        "lr": 1e-4,
        "print_step": 1,
        "print_eval": True,
        "mixed_precision": False,
        "datasets": [dataset_configs],
        "use_speaker_embedding": True,
        "min_text_len": 0,
        "max_text_len": 500,
        "min_audio_len": 5000,
        "max_audio_len": 500000,
        "double_decoder_consistency": True,
        "text_cleaner": "english_cleaners",
        # "infoNCE_alpha": 0.2,
    }
    
    ap, tokenizer, tacotron2_config = load_tacotron2_config(config)
    train_samples, eval_samples, test_samples = load_samples(dataset_configs, tacotron2_config)