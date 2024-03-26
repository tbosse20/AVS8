import torch.nn.functional as F
import torch
import random


def infoNCE_loss(outputs, temperature=1.0):
    bs_size = outputs.size(0) #(bs, 2) -> (1, embedding_aize), (1, speaker_id)
    pos_similarities = []
    neg_similarities = []

    for idx, output in enumerate(outputs):
        samples = output[:idx] + output[idx+1:]
        random.shuffle(samples)
        samples = samples[:bs_size//2]
        pos_similarity = []
        neg_similarity = []
       # Compute cosine similarity between query and samples
        similarities = torch.cosine_similarity(output[:, 0], samples, dim=1)
        
        # Split similarities into positive and negative samples
        pos_similarity = similarities[output[1] == samples[:, 1]]
        neg_similarity = similarities[output[1] != samples[:, 1]]
        
        pos_similarities.append(pos_similarity)
        neg_similarities.append(neg_similarity)

    pos_logits = torch.stack(pos_similarities)
    neg_logits = torch.stack(neg_similarities)        
    logits = torch.cat([pos_logits, neg_logits], dim=1) /temperature

    num_positives = pos_logits.size(1)
    num_negatives = neg_logits.size(1)
    positive_labels = torch.arange(num_positives, device=outputs.device).unsqueeze(0).repeat(outputs.size(0), 1)
    negative_labels = torch.arange(num_positives, num_positives + num_negatives, device=outputs.device).unsqueeze(0).repeat(outputs.size(0), 1)
    labels = torch.cat([positive_labels, negative_labels], dim=1)

    # Compute InfoNCE loss
    loss = F.cross_entropy(logits, labels)

    return loss

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

outputs = torch.randn(10, 2, requires_grad=True)  # Example outputs with shape (batch_size, 2)
loss = infoNCE_loss(outputs, temperature=1.0)
print("InfoNCE Loss:", loss.item())
loss.backward()  