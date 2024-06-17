# *Crying in constrastive
Semester project
Computer Engineering - AI, Vision & Sound - 8. semester (CE-AVS8)
Group 841, Aalborg University 2024

## Contributors:
- Marko Putak
- Sameer Aqib Hashmi
- Tonko Bossen

## Installation
1. Download:
- [Google Drive - "tts_model--en--ek1--tacotron2.zip"](https://drive.google.com/file/d/1d4hlUnKMkcJh8SkNOmNaexy0MlEaKblI/view?usp=sharing)
- [Google Drive - "vocoder_models-universal--libri-tts--fullband-melgan.zip"](https://drive.google.com/file/d/1qtb_gN4IGcQWrKpZ4-GGv6xoJsN5FbQP/view?usp=sharing)

2. Put folders in new folders "tts_model" and "vocoder":
```
AVS8
├── tts_model/tts_model--en--ek1--tacotron2
│   │
│   ├── config.json - 
│   └── model_file.pth - 
│
└── vocoder/vocoder_models-universal--libri-tts--fullband-melgan
    ├── config.json - 
    ├── model_file.pth - 
    └── scale_stats.npy -
```


# Usage
- "tts_pipeline.py"

# Load Weights
## Weights
- Baseline (12k steps): [OneDrive - baseline_12k](https://aaudk-my.sharepoint.com/:f:/r/personal/lk83xy_student_aau_dk/Documents/baseline_12k?csf=1&web=1&e=TRvQw3)
- New version (3k steps): [OneDrive - about8ksteps](https://aaudk-my.sharepoint.com/:f:/r/personal/lk83xy_student_aau_dk/Documents/about8ksteps?csf=1&web=1&e=3rLqLW)

- Download folder with weights.
- Run `tts_pipeline.py` with --checkpoint_run "/path/to/folder".

## License