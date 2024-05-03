import os

def download_dataset(dataset_path):
    from TTS.utils.downloaders import download_libri_tts

    subsets = {
        "libri-tts-clean-100": 'train-clean-100',  # 這裡是Ollie做的
        # "libri-tts-test-clean": 'test-clean',  # 這裡是Tonko做的
    }

    subfolders = os.listdir(os.path.join(dataset_path, "LibriTTS"))

    for subset_name, subset_folder in subsets.items():
        if subset_folder in subfolders: continue
        print(f'Downloading: "{subset_name}" -> "{subset_folder}"')
        download_libri_tts(dataset_path, subset=subset_name)

    print("Datasets downloaded successfully!")