from transformers import Wav2Vec2ForXVector, AutoFeatureExtractor

# Save the Wav2Vec2ForXVector model
model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
model.save_pretrained("./encoderwav2vec2")

# Save the AutoFeatureExtractor associated with the model
feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
feature_extractor.save_pretrained("./featureextractorwav2vec2")