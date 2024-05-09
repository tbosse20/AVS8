# *Crying in constrastive
Semester project
Computer Engineering - AI, Vision & Sound - 8. semester (CE-AVS8)
Group 841, Aalborg University 2024

## Description


## Contributors:
- Marko Putak
- Yu-ling Cheng
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

# Load weights
- Download the specific `run` folder from `runs`
- Rename the `speaker.pth` path in the `config.json` file
- Rename the `checkpoint_XXXX.pth` in the `run` folder to `checkpoint.pth`

Weights example:
https://aaudk-my.sharepoint.com/personal/lk83xy_student_aau_dk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flk83xy_student_aau_dk%2FDocuments%2Fcheckpoint_2418%2Epth&parent=%2Fpersonal%2Flk83xy_student_aau_dk%2FDocuments&ga=1

## License