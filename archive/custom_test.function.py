import torch
import numpy as np
import plot_funcs
import gc
from custom_tacotron2 import Tacotron2

def test_cos_sim(tacotron2: Tacotron2, samples: list, config, dev=False):
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
        if batch_num >= 2 and dev:
            break

    # Convert list to numpy array
    cos_sims_np = np.array(cos_sims).flatten()
    # Assuming cos_sims_np is your NumPy array
    np.save('output/cos_similarity.npy', cos_sims_np)
    # Call the function to plot the boxplot
    plot_funcs.plot_boxplot(cos_sims_np, 'output/cos_similarity_boxplot.png')
