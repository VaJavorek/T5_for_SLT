from transformers import Seq2SeqTrainer
import torch
import torch.nn.functional as F

class MinLossSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Trainer that, for each source sample, computes the loss against every
    paraphrase target [P] and back‑propagates only the minimum.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")          # shape [B, P, L]  (or [B, L])
        # fallback to vanilla path when P == 1
        if labels.ndim == 2:                   # [B, L]
            return super().compute_loss(model,
                                         {**inputs, "labels": labels},
                                         return_outputs)

        B, P, L = labels.size()
        flat_labels = labels.reshape(B * P, L)     # [B*P, L]

        # repeat encoder inputs P times along batch dim
        # enc_inputs = {k: v.repeat_interleave(P, dim=0) for k, v in inputs.items()} # differentiable repeat
        enc_inputs = {k: v.unsqueeze(1).expand(-1, P, *v.shape[1:]).reshape(B*P, *v.shape[1:]) for k, v in inputs.items()} # new version using expand() saves mem

        outputs = model(**enc_inputs, labels=flat_labels, return_dict=True)  # one forward pass to obtain logits
        logits = outputs.logits                     # [B*P, L, V]
        V = logits.size(-1)
        
        token_loss = F.cross_entropy(
            logits.view(-1, V),                     # [B*P*L, V]
            flat_labels.view(-1),                   # [B*P*L]
            ignore_index=-100,                      # HF’s default pad for labels
            reduction="none",
        ).view(B * P, L)                            # [B*P, L]
        
        mask = flat_labels.ne(-100).float()         # 1 where target token is valid
        seq_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B*P]
        
        loss_per_p = seq_loss.view(B, P)            # [B, P]
        min_loss = loss_per_p.min(dim=1).values     # [B]

        loss = min_loss.mean()                      # scalar
        return (loss, outputs) if return_outputs else loss