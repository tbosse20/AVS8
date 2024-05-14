def retry_load_models(model_name, cache_dir="./.transformers", attempts: int = 5, retry_delay: int = 3):
    
    for attempt in range(attempts):
        try:
            # Try loading from local cache
            model = Wav2Vec2ForXVector.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                revision="main",
            )   
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                revision="main",
            )
            break
        
        except OSError as e:
            # Download if not found locally
            time.sleep(retry_delay)

    return model, feature_extractor