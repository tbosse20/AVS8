import os
from custom_tacotron2 import Tacotron2
import matplotlib.pyplot as plt
import torchaudio
from datetime import datetime
import json
from random import randint

def inference(tacotron2: Tacotron2, samples: list, config, idx=0, checkpoint_run=None):

    # Set model to evaluation mode
    tacotron2.eval()

    # Load dataloader with test samples
    test_dataloader = tacotron2.get_data_loader(
        config=config,
        assets=None,
        is_eval=True,
        # Only load one sample (idx)
        samples=samples[idx : idx + 1],
        verbose=True,
        num_gpus=0,
    )
    
    # Get batch from dataloader
    batch = next(iter(test_dataloader))

    # Get model type
    model_type = "CLmodel" if config.infoNCE_alpha > 0 else "Baseline"
    # Get raw text and format file name
    raw_text = batch["raw_text"][0]
    # Remove new lines and quotes
    text = raw_text
    for element in ["\n", '"', "'", "?", "!", ".", ",", ":", ";"]:
        text = text.replace(element, "")
    
    # Setup file name
    file_name = model_type
    # Shorten text with dots
    file_name += '_' + text[:10] + '..'
    # Add timestamp to file name
    file_name += "_" + datetime.now().strftime("%m%d_%H%M")
    # Add random numbers to file name
    file_name += "_" + "".join(["{}".format(randint(0, 9)) for num in range(0, 4)])

    # Make folder for speaker output
    folder_path = os.path.join("output", file_name)
    os.makedirs(folder_path, exist_ok=True)

    # Get speaker name and id and add random numbers
    speaker_name = batch["speaker_names"][0]
    speaker_id = batch["speaker_ids"][0]
    speaker = f"{speaker_name}_({speaker_id})"

    # Display first sample as text and audio
    print(f"\nraw_text sample:\n> {raw_text}")

    # Save the waveform input
    waveform = batch["waveform"][0].T
    input_file = os.path.join(folder_path, f"input.wav")
    torchaudio.save(input_file, waveform.cpu(), 22050)
    
    # Save specific config elements in folder
    manual_dict = {
        "infoNCE_alpha": config.infoNCE_alpha,
        "similarity_loss_alpha": config.similarity_loss_alpha,
        "model_type": model_type,
        "raw_text": batch["raw_text"][0],
        "weights": os.path.basename(checkpoint_run) if checkpoint_run else "None",
        "speaker": speaker,
        "test_sample_idx": idx,
    }
    json_file = os.path.join(folder_path, "config.json")
    with open(json_file, "w") as f:
        json.dump(manual_dict, f, indent=4)

    # Format batch and get all values
    batch = tacotron2.format_batch(batch)

    usage = ['inference', 'forward']
    using = usage[0]
    
    # Perform inference on single sample
    if using == 'inference':
        inference_outputs = tacotron2.inference(
            text=batch["text_input"][0].clone().detach().unsqueeze(0),
            spk_emb1=(batch["spk_emb"][0]),
            save_wav=True,
            output_path=os.path.join(folder_path),
        )
    
    if using == 'forward':
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        speaker_ids = batch["speaker_ids"]
        d_vectors = batch["d_vectors"]
        spk_emb1 = batch["spk_emb"]
        aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
        
        inference_outputs = tacotron2.forward(
            text_input, text_lengths, mel_input, mel_lengths, aux_input, spk_emb1
        )
    
        # NEW INFERENCE USING VOCODER#
        waveform = tacotron2.vocoder.inference(inference_outputs['model_outputs'].permute(0, 2, 1))
        os.makedirs(folder_path, exist_ok=True)
        output_file = os.path.join(folder_path, f"output{idx}.wav")
        torchaudio.save(output_file, waveform[0], 22050)

    # Plot mel_input
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    im1 = ax1.imshow(batch["mel_input"][0].numpy().T, aspect="auto", origin="lower")
    ax1.set_title("Ground truth Mel-Spectrogram")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Mel Filter")
    # Save the first subplot individually
    plt.savefig(os.path.join(folder_path, f"ground_truth_mel_spectrogram.png"))
    plt.close(fig1)  # Close the figure to free up resources

    # Plot model output
    for key, value in inference_outputs.items():
        try:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(value[0].clone().detach().numpy().T, origin="lower", aspect="auto")
            ax2.set_title(f"{model_type} {key} shape={value.shape}")
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Mel Filter")
            # Save the second subplot individually
            plt.savefig(os.path.join(folder_path, f"{model_type}_{key}.png"))
            plt.close(fig2)  # Close the figure to free up resources
        except:
            pass

