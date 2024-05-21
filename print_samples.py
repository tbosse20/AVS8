
import os
import custom_dataset.dataset_util as dataset_util
    
current_path = os.path.dirname(os.path.abspath(__file__))
main_path = os.path.dirname(current_path)
dataset_path = os.path.join(main_path, "AVS8", "libriTTS")

subsets = {
    "libri-tts-clean-100": 'train-clean-100',   # 這裡是Ollie做的
    "libri-tts-test-clean": 'test-clean',       # 這裡是Tonko做的
}
dataset_configs = dataset_util.download_dataset("LibriTTS", dataset_path, subsets, formatter="libri_tts")

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

ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config(config)
train_samples, eval_samples, test_samples = dataset_util.load_samples(dataset_configs, tacotron2_config)

for i, sample in enumerate(test_samples[:50]):
    text = sample['text'].replace("\n", "")
    print(f'{i}: {text}')
print()

idx = 20
print(f'{idx=}:', test_samples[idx]['text'])