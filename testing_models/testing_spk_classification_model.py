# "/media/sameerhashmi36/New_drive_hdd/Aalborg_University_Lectures/AVS8/main_project/AVS8/libriTTS/LibriTTS/train-clean-100/60/121082/60_121082_000003_000000.wav"

######### pip install librosa==0.8.1

import soundfile as sf
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pretrained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio file
audio_file_path = "/media/sameerhashmi36/New_drive_hdd/Aalborg_University_Lectures/AVS8/main_project/AVS8/libriTTS/LibriTTS/train-clean-100/60/121082/60_121082_000003_000000.wav"
audio_input, source_sample_rate = sf.read(audio_file_path)
print("audio_input: ", audio_input.shape)
print("source_sample_rate ",source_sample_rate)

# Resample the audio to match the expected sample rate (16000 for the model)
target_sample_rate = 16000
audio_input_resampled = librosa.resample(audio_input, source_sample_rate, target_sample_rate)

# Preprocess the resampled audio
input_values = processor(audio_input_resampled, sampling_rate=target_sample_rate, return_tensors="pt").input_values

# Inference
logits = model(input_values).logits  # Pass input through the model to get logits
predicted_ids = torch.argmax(logits, dim=-1)  # Take argmax to get predicted ids
transcription = processor.decode(predicted_ids[0])  # Decode predicted ids to get transcription

print("Predicted Transcription:", transcription)
