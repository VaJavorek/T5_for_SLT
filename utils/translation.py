import torch
import torch.nn.functional as F

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# Add collate_fn to DataLoader
# Pad and batch target references in a new collate‑fn so labels has shape [B, P, L] (batch, paraphrases, tokens)
def collate_fn(batch):
    """
    Pads and stacks a list of dataset samples.
    Works for any paraphrase mode:
        labels.shape == [P, L]  (P may differ per example)
    Returns:
        sign_inputs       [B, 250, 208]   – unchanged
        attention_mask    [B, 250]
        labels            [B, max_P, max_L]
    """
    # --- static inputs (same as before) ---------------------------
    sign_inputs = torch.stack([
        torch.cat((s["sign_inputs"], torch.zeros(250 - s["sign_inputs"].shape[0], 208)),
                dim=0)
        for s in batch
    ])

    attention_mask = torch.stack([
        torch.cat((s["attention_mask"],
                torch.zeros(250 - s["attention_mask"].shape[0])),
                dim=0) if s["attention_mask"].shape[0] < 250
        else s["attention_mask"]
        for s in batch
    ])

    # --- NEW: 2‑D padding for labels ------------------------------
    max_p = max(s["labels"].shape[0] for s in batch)      # paraphrases
    max_l = max(s["labels"].shape[1] for s in batch)      # tokens

    labels = torch.stack([
    F.pad(s["labels"],                           # [P, L]
            pad=(0, max_l - s["labels"].shape[1],  # pad tokens (right)
                0, max_p - s["labels"].shape[0]), # pad paraphrases (bottom)
            value=0)                               # 0 is `pad_token_id` for T5
        for s in batch
    ]).to(torch.long)                                # [B, max_P, max_L]

    return {"sign_inputs": sign_inputs,
            "attention_mask": attention_mask,
            "labels": labels}