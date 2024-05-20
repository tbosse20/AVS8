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

    # Get raw text and format file name
    raw_text = batch["raw_text"][0]
    # Remove new lines and quotes
    text = raw_text
    for element in ["\n", '"', "'", "?", "!", ".", ",", ":", ";"]:
        text = raw_text.replace(element, "")
    # Shorten text with dots
    file_name = text[:10] + ".."
    # Add timestamp to file name
    file_name += "_" + datetime.now().strftime("%m%d_%H%M")
    # Add random numbers to file name
    file_name += "_" + "".join(["{}".format(randint(0, 9)) for num in range(0, 4)])

    # Get speaker name and id and add random numbers
    speaker_name = batch["speaker_names"][0]
    speaker_id = batch["speaker_ids"][0]
    speaker = f"{speaker_name}_({speaker_id})"

    # Make folder for speaker output
    folder_path = os.path.join("output", file_name)
    os.makedirs(folder_path, exist_ok=True)

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

    # Perform inference one single sample
    inference_outputs = tacotron2.inference(
        text=batch["text_input"][0].clone().detach().unsqueeze(0),
        aux_input={"speaker_ids": (batch["speaker_ids"][0])},
        spk_emb1=(batch["spk_emb"][0]),
        save_wav=True,
        output_path=os.path.join(folder_path),
    )

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot mel_input
    im1 = axes[0].imshow(batch["mel_input"][0].numpy().T, aspect="auto", origin="lower")
    axes[0].set_title("Ground truth Mel-Spectrogram")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Mel Filter")

    # Plot model output
    im2 = axes[1].imshow(
        inference_outputs["model_outputs"][0].numpy().T, aspect="auto", origin="lower"
    )
    axes[1].set_title("Output Mel-Spectrogram")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Filter")

    # Create a colorbar in a separate axis
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])

    # Use the colorbar from the first image
    fig.colorbar(im1, cax=cax)

    # Add a common title for the whole figure
    plt.suptitle(f"Compare Input and Output Mel-Spectrograms - {file_name}")

    # Save the figure
    plt.savefig(os.path.join(folder_path, f"mel_spect_comp.png"))
