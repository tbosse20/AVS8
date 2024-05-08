import os
from dataset import dataset_util
from custom_tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
import wandb
import matplotlib.pyplot as plt
import torchaudio
import torch
import numpy as np
import plot_funcs
import gc


def test_cos_sim(tacotron2: Tacotron2, samples: list, config):
    """
    Run cosine similarity between two speaker embeddings.

    Args:
        tacotron2 (Tacotron2): Tacotron2 model
        samples (list): List of samples
        config (dict): Configuration dictionary

        Returns:
            None

    """

   # Load dataloader with test samples
    test_dataloader = tacotron2.get_data_loader(
        config=config,
        assets=None,
        is_eval=True,
        samples=samples,
        verbose=True,
        num_gpus=0
    )

    # Make empty list to store cosine similarities
    cos_sims = []

    # Set model to evaluation mode
    tacotron2.eval()

    # Iterate each batch and perform inference
    for batch_num, batch in enumerate(test_dataloader):

        # Format batch and get all values
        batch = tacotron2.format_batch(batch)

        # Get speaker embeddings
        spk_emb1s = batch["spk_emb"]
        spk_emb2s = []

        # Iterate each sample in the batch and perform inference
        for i in range(len(batch["text_input"])):
            inference_outputs = tacotron2.inference(
                text=batch["text_input"][i].clone().detach().unsqueeze(0),
                aux_input={"speaker_ids": (batch["speaker_ids"][i])},
                spk_emb1=(spk_emb1s[i]),
                save_wav=False
            )
            spk_emb2s.append(inference_outputs['spk_emb2'])

        # Stack speaker embeddings and calculate cosine similarity
        stck_spk_emb1s = torch.stack(spk_emb1s)
        stck_spk_emb2s = torch.stack(spk_emb2s).squeeze(dim=2)
        cos_similarity = torch.nn.CosineSimilarity(dim=2)(stck_spk_emb1s, stck_spk_emb2s)
        cos_sims.extend(cos_similarity.tolist())

        print(f"Inference done on batch {batch_num}")
        # print(f"> Cosine Similarity: {cos_similarity}")

        # Clear memory
        gc.collect()

        # Break after n batch
        if batch_num >= 2:
            break

    # Convert list to numpy array
    cos_sims_np = np.array(cos_sims).flatten()
    # Assuming cos_sims_np is your NumPy array
    np.save('output/cos_similarity.npy', cos_sims_np)
    # Call the function to plot the boxplot
    plot_funcs.plot_boxplot(cos_sims_np, 'output/cos_similarity_boxplot.png')


def inference(tacotron2: Tacotron2, samples: list, config, idx=0):

    # Set model to evaluation mode
    tacotron2.eval()

   # Load dataloader with test samples
    test_dataloader = tacotron2.get_data_loader(
        config=config,
        assets=None,
        is_eval=True,
        samples=samples,
        verbose=True,
        num_gpus=0
    )
    batch = next(iter(test_dataloader))

    speaker_name = batch["speaker_names"][idx]
    speaker_id = batch["speaker_ids"][idx]
    speaker = f'{speaker_name} ({speaker_id})'
    
    # Display first sample as text and audio
    print(f'\nraw_text sample:\n> {batch["raw_text"][idx]}')
    # Save txt file
    with open(os.path.join('output', 'raw_text_{speaker}.txt'), 'w') as f:
        f.write(batch["raw_text"][idx])

    # Convert mel_input to waveform and save it
    waveform = batch["waveform"][idx].T
    input_file = os.path.join('output', f'input__{speaker}.wav')
    torchaudio.save(input_file, waveform.cpu(), 22050)

    # Format batch and get all values
    batch = tacotron2.format_batch(batch)

    # Perform inference
    inference_outputs = tacotron2.inference(
        text        = batch["text_input"][idx].clone().detach().unsqueeze(0),
        aux_input   = {"speaker_ids": (batch["speaker_ids"][idx])},
        spk_emb1    = (batch["spk_emb"][idx]),
        save_wav    = True,
        speaker     = speaker,
    )

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot mel_input
    im1 = axes[0].imshow(batch["mel_input"][idx].numpy().T, aspect='auto', origin='lower')
    axes[0].set_title('Ground truth Mel-Spectrogram')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mel Filter')

    # Plot model output
    im2 = axes[1].imshow(inference_outputs['model_outputs'][0].numpy().T, aspect='auto', origin='lower')
    axes[1].set_title('Output Mel-Spectrogram')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mel Filter')

    # Create a colorbar in a separate axis
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])

    # Use the colorbar from the first image
    fig.colorbar(im1, cax=cax)

    # Add a common title for the whole figure
    plt.suptitle('Comparison of Input and Output Mel-Spectrograms ({speaker})')

    # Save the figure
    plt.savefig(os.path.join('output', f'mel_spectrogram_comparison_{speaker}.png'))


if __name__ == "__main__":

    # Dataset and save path
    current_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_path, "runs")
    dataset_path = os.path.join(current_path, "libriTTS")

    # Dataset names
    dataset_subsets = {
        "libri-tts-clean-100": 'train-clean-100',   # 這裡是Ollie做的
        "libri-tts-test-clean": 'test-clean',       # 這裡是Tonko做的
    }

    # define model config
    config = {
        "batch_size": 8,
        "eval_batch_size": 8,
        "num_loader_workers": 0,
        "num_eval_loader_workers": 0,
        "precompute_num_workers": 0,
        "run_eval": True,
        "test_delay_epochs": -1,
        "epochs": 2,
        "lr": 1e-4,
        "print_step": 50,
        "print_eval": True,
        "mixed_precision": False,
        "output_path": output_path,
        "use_speaker_embedding": True,
        "min_text_len": 0,
        "max_text_len": 500,
        "min_audio_len": 5000,
        "max_audio_len": 250000,
        "double_decoder_consistency": True,
        "text_cleaner": "english_cleaners",
        "infoNCE_alpha": 0.25,
        "similarity_loss_alpha": 0.25,
        "shuffle": True,
    }

    # wandb.init(
    #     project="AVSP8",                            # Project name
    #     entity="qwewef",                            # Entity name
    #     config=config,                              # Configuration dictionary
    # )

    # download the dataset if not downloaded
    dataset_configs = dataset_util.download_dataset("LibriTTS", dataset_path, dataset_subsets, formatter="libri_tts")

    ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config(config)
    # ap, tokenizer, tacotron2_config = dataset_util.load_tacotron2_config("weights/config_5256.json")
    train_samples, eval_samples, test_samples = dataset_util.load_samples(dataset_configs, tacotron2_config)

    # init speaker manager for multi-speaker training
    # it maps speaker-id to speaker-name in the model and data-loader
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples + test_samples, parse_key="speaker_name")

    # Load models
    tacotron2 = Tacotron2(tacotron2_config, ap, tokenizer, speaker_manager=speaker_manager)

    # Load checkpoint
    # weights_path = "/home/student.aau.dk/lk83xy/avs8/AVS8/runs/run-May-07-2024_12+08AM-3f6f821/"  # CLAAUDIA
    # weights_path = "/home/putak/university/8semester/Project/"                                    # Marko Local    
    tacotron2.load_checkpoint(
        config=tacotron2_config,
        checkpoint_path="runs/run-May-08-2024_05+08PM-a4d5ca5/best_model_1.pth",
        eval=True
    )

    ## Proper way of loading the model from TTS demo! ##
    # # from TTS.tts.configs.tacotron2_config import Tacotron2Config
    # # from TTS.tts.models.tacotron2 import Tacotron2
    # # config = Tacotron2Config()
    # # config.load_json("/home/putak/university/8semester/Project/AVS8/runs/run-May-08-2024_10+21AM-42ddebb/config.json")
    # # TACOTRON_MODEL = Tacotron2.init_from_config(config)
    # # print("Loading TACOTRON model! ")
    # # TACOTRON_MODEL.load_checkpoint(config, checkpoint_path="/home/putak/university/8semester/Project/AVS8/runs/run-May-08-2024_10+21AM-42ddebb/best_model_1.pth")
    # # if torch.cuda.is_available():
    # #     TACOTRON_MODEL.cuda()

    # # print("Model Loaded!")
    print("LOADED!")

    # Perform test
    test_cos_sim(tacotron2, test_samples, tacotron2_config)

    # Perform inference
    inference(tacotron2, test_samples, tacotron2_config)
