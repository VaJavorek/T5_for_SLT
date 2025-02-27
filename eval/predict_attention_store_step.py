import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from transformers import T5Tokenizer
from model.configuration_t5 import SignT5Config
from transformers import T5Config
from model.modeling_t5 import T5ModelForSLT
from utils.translation import postprocess_text
import evaluate
import yaml
from dataset.generic_sl_dataset import SignFeatureDataset as DatasetForSLT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned T5 model for SLT with full per-step attention logging using beam search")
    # Configuration
    parser.add_argument("--config_file", type=str, required=True, default='predict_config_attention.yaml')
    # Model and data paths
    parser.add_argument("--model_name", type=str, default=None, help="Model name or folder inside model_dir.")
    parser.add_argument("--output_dir", type=str, default=None)
    # Data processing
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Max number of frames for sign inputs.")
    parser.add_argument("--max_token_length", type=int, default=None, help="Max token length for labels.")
    parser.add_argument("--skip_frames", default=None)
    # Generation parameters
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the directory containing the fine-tuned model and config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference.")
    # Evaluation arguments
    parser.add_argument("--num_beams", type=int, default=None, help="Number of beams for beam search.")
    parser.add_argument("--length_penalty", type=float, default=None, help="Length penalty for generation.")
    parser.add_argument("--early_stopping", type=bool, default=None, help="Use early stopping in generation.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None, help="No repeat ngram size.")
    # Running arguments
    parser.add_argument("--dev", action="store_true", help="Use dev mode.")
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--is_normalized", action="store_true", help="If the data is normalized.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    for param, value in cfg['EvaluationArguments'].items():
        if value == 'none' or value == 'None':
            cfg['EvaluationArguments'][param] = None
    return cfg


def update_config(cfg, args):
    for k, v in vars(args).items():
        if k in cfg['EvaluationArguments'] and v is not None:
            cfg['EvaluationArguments'][k] = v
            if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
                print(f'Config value updated by args - {k}: {v}')
    return cfg


def collate_fn(batch, max_sequence_length, max_token_length, pose_dim):
    # Handle sign_inputs as dict (for multimodal inputs) or tensor
    if isinstance(batch[0]["sign_inputs"], dict):
        sign_inputs_list = []
        for sample in batch:
            if "pose" in sample["sign_inputs"]:
                pose_tensor = sample["sign_inputs"]["pose"]
                if pose_tensor.shape[0] < max_sequence_length:
                    padded_tensor = torch.cat(
                        (pose_tensor, torch.zeros(max_sequence_length - pose_tensor.shape[0], pose_dim)),
                        dim=0
                    )
                else:
                    padded_tensor = pose_tensor[:max_sequence_length]
                sign_inputs_list.append(padded_tensor)
            else:
                sign_inputs_list.append(torch.zeros(max_sequence_length, pose_dim))
        sign_inputs = torch.stack(sign_inputs_list)
    else:
        sign_inputs = torch.stack([
            torch.cat(
                (sample["sign_inputs"], torch.zeros(max_sequence_length - sample["sign_inputs"].shape[0], pose_dim)),
                dim=0
            ) if sample["sign_inputs"].shape[0] < max_sequence_length else sample["sign_inputs"][:max_sequence_length]
            for sample in batch
        ])
    
    return {
        "sign_inputs": sign_inputs,
        "attention_mask": torch.stack([
            torch.cat(
                (sample["attention_mask"], torch.zeros(max_sequence_length - sample["attention_mask"].shape[0])),
                dim=0
            ) if sample["attention_mask"].shape[0] < max_sequence_length else sample["attention_mask"][:max_sequence_length]
            for sample in batch
        ]),
        "labels": torch.stack([
            torch.cat(
                (sample["labels"].squeeze(0), torch.zeros(max_token_length - sample["labels"].shape[0])),
                dim=0
            ) if sample["labels"].shape[0] < max_token_length else sample["labels"][:max_token_length]
            for sample in batch
        ]).squeeze(0).to(torch.long),
    }


def generate_with_beam_attentions(model, batch, tokenizer, max_length):
    """
    Custom beam search that generates tokens step-by-step while capturing, for each beam and each step,
    the decoder self-attentions and cross-attentions (per token, per head).
    Returns:
      - final_sequences: Tensor of shape (batch_size, num_beams, sequence_length)
      - encoder_attentions: encoder attention data (saved once)
      - beam_attentions: nested list [batch][beam][step] = {"decoder_attentions": [...], "cross_attentions": [...]}
      - final_scores: beam scores (as a list)
    """
    device = batch["sign_inputs"].device
    batch_size = batch["sign_inputs"].size(0)
    num_beams = model.config.num_beams

    # Encode inputs once
    encoder_outputs = model.encoder(
         batch["sign_inputs"],
         attention_mask=batch["attention_mask"],
         output_attentions=True,
         return_dict=True,
    )
    # Save encoder attentions (these are common to all beams)
    encoder_attentions = [attn.detach().cpu().tolist() for attn in encoder_outputs.attentions] if hasattr(encoder_outputs, "attentions") else None

    # Expand encoder outputs for beam search (repeat each sample num_beams times)
    def expand(tensor, num_beams):
        shape = tensor.shape
        return tensor.unsqueeze(1).expand(shape[0], num_beams, *shape[1:]).reshape(shape[0] * num_beams, *shape[1:])
    encoder_hidden_states = expand(encoder_outputs.last_hidden_state, num_beams)
    encoder_attention_mask = batch["attention_mask"].unsqueeze(1).expand(batch_size, num_beams, batch["attention_mask"].size(-1)).reshape(batch_size * num_beams, -1)

    # Initialize decoder input with the start token.
    start_token = tokenizer.pad_token_id  # or use model.config.decoder_start_token_id if available
    decoder_input_ids = torch.full((batch_size * num_beams, 1), start_token, device=device, dtype=torch.long)

    # Initialize beam scores: first beam gets 0, others -inf.
    beam_scores = torch.zeros(batch_size, num_beams, device=device)
    beam_scores[:, 1:] = -1e9

    # Data structure for logging attentions:
    # For each sample (batch_size), for each beam (num_beams), maintain a list over generation steps.
    beam_attentions = [[[] for _ in range(num_beams)] for _ in range(batch_size)]

    # Main generation loop.
    for step in range(max_length - 1):
        # Run the decoder for current sequences.
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        # Get logits for the last token.
        last_hidden = decoder_outputs.last_hidden_state  # shape: (batch_size*num_beams, seq_len, hidden_dim)
        logits = model.lm_head(last_hidden[:, -1, :])      # shape: (batch_size*num_beams, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)           # shape: (batch_size*num_beams, vocab_size)

        # Add the previous beam scores.
        curr_scores = log_probs + beam_scores.view(-1, 1)     # shape: (batch_size*num_beams, vocab_size)
        # Reshape scores to (batch_size, num_beams * vocab_size) for selection.
        curr_scores = curr_scores.view(batch_size, num_beams * log_probs.size(-1))
        top_scores, top_indices = torch.topk(curr_scores, num_beams, dim=-1)

        # Decode the flattened indices: beam index and token id.
        next_beam_indices = top_indices // log_probs.size(-1)  # which beam the candidate came from
        next_tokens = top_indices % log_probs.size(-1)         # the token id

        # Prepare new decoder sequences and update beam scores.
        new_decoder_input_ids = []
        new_beam_scores = []
        new_decoder_input_ids_list = decoder_input_ids.view(batch_size, num_beams, -1)
        for i in range(batch_size):
            beams = []
            scores = []
            # For each new beam candidate, update the sequence and log attentions.
            for j in range(num_beams):
                beam_idx = next_beam_indices[i, j].item()
                token = next_tokens[i, j].unsqueeze(0).unsqueeze(0)  # shape: (1, 1)
                prev_seq = new_decoder_input_ids_list[i, beam_idx]     # shape: (seq_len,)
                new_seq = torch.cat([prev_seq, token.squeeze(0)], dim=0)  # shape: (seq_len+1,)
                beams.append(new_seq.unsqueeze(0))
                scores.append(top_scores[i, j].unsqueeze(0))
                # Log attentions for this beam candidate.
                # The decoder outputs provide attentions for all beams in the flattened batch.
                global_beam_idx = i * num_beams + beam_idx
                step_dec_attns = []
                step_cross_attns = []
                if hasattr(decoder_outputs, "decoder_attentions") and decoder_outputs.decoder_attentions is not None:
                    for layer_attn in decoder_outputs.decoder_attentions:
                        # layer_attn: shape (batch_size*num_beams, num_heads, seq_len, seq_len)
                        attn_last = layer_attn[global_beam_idx, :, -1, :].detach().cpu().tolist()
                        step_dec_attns.append(attn_last)
                if hasattr(decoder_outputs, "cross_attentions") and decoder_outputs.cross_attentions is not None:
                    for layer_attn in decoder_outputs.cross_attentions:
                        # layer_attn: shape (batch_size*num_beams, num_heads, seq_len, encoder_seq_len)
                        attn_last = layer_attn[global_beam_idx, :, -1, :].detach().cpu().tolist()
                        step_cross_attns.append(attn_last)
                # Append the logged attentions to the beam slot (use the candidate's beam index).
                beam_attentions[i][beam_idx].append({
                    "decoder_attentions": step_dec_attns,
                    "cross_attentions": step_cross_attns
                })
            new_decoder_input_ids.append(torch.cat(beams, dim=0).unsqueeze(0))  # shape: (1, num_beams, seq_len+1)
            new_beam_scores.append(torch.cat(scores, dim=0).unsqueeze(0))
        # Reshape new decoder inputs back to (batch_size*num_beams, seq_len+1)
        decoder_input_ids = torch.cat(new_decoder_input_ids, dim=0).view(batch_size * num_beams, -1)
        beam_scores = torch.cat(new_beam_scores, dim=0)  # shape: (batch_size, num_beams)

        # Optionally, break if all beams in every sample have generated the eos token.
        eos_token = tokenizer.eos_token_id
        finished = True
        decoder_input_ids_reshaped = decoder_input_ids.view(batch_size, num_beams, -1)
        for seq in decoder_input_ids_reshaped:
            if not torch.all(seq[:, -1] == eos_token):
                finished = False
                break
        if finished:
            break

    final_sequences = decoder_input_ids.view(batch_size, num_beams, -1)
    final_scores = beam_scores.tolist()
    return final_sequences, encoder_attentions, beam_attentions, final_scores


def evaluate_model(model, dataloader, tokenizer, evaluation_config):
    """
    Run inference on the dataset using the custom beam search with attention logging.
    Saves predictions, labels, and attention data.
    """
    def log_message(message, log_file):
        current_time = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{current_time} GMT+1] {message}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    all_predictions = []
    all_labels = []
    all_encoder_attn = []
    all_beam_attn = []  # nested list from generate_with_beam_attentions
    with open("evaluation.log", "a") as log_file:
        log_message("Starting model evaluation", log_file)
        model.eval()
        total_batches = len(dataloader)
        log_message(f"Beginning inference loop with {total_batches} batches", log_file)

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if step % 10 == 0:
                    log_message(f"Processing batch {step}/{total_batches}", log_file)
                batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
                if len(batch['labels'].shape) < 2:
                    batch['labels'] = batch['labels'].unsqueeze(0)
                # Use the custom beam search function.
                final_seqs, enc_attn, beam_attn, beam_scores = generate_with_beam_attentions(
                    model, batch, tokenizer, evaluation_config['max_sequence_length']
                )
                # Save encoder attentions (for this batch) once.
                all_encoder_attn.append(enc_attn)
                all_beam_attn.append(beam_attn)
                # For predictions, choose the best beam (highest score) per sample.
                best_sequences = []
                final_seqs = final_seqs.cpu().numpy()  # shape: (batch_size, num_beams, seq_len)
                for seqs, scores in zip(final_seqs, beam_scores):
                    best_idx = np.argmax(scores)
                    best_sequences.append(seqs[best_idx])
                decoded_preds = tokenizer.batch_decode(best_sequences, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                all_predictions.extend(decoded_preds)
                all_labels.extend([[lbl] for lbl in decoded_labels])
        log_message("Completed model evaluation", log_file)
    # Save attention data
    attn_data = {
        "encoder_attentions": all_encoder_attn,
        "beam_attentions": all_beam_attn,
    }
    attn_output_path = os.path.join(evaluation_config['output_dir'], "attentions.json")
    os.makedirs(evaluation_config['output_dir'], exist_ok=True)
    with open(attn_output_path, "w") as attn_file:
        json.dump(attn_data, attn_file)
    return all_predictions, all_labels


def get_sign_input_dim(config):
    sign_input_dim = 0
    for mod in config['SignDataArguments']['visual_features']:
        if config['SignDataArguments']['visual_features'][mod]['enable_input']:
            sign_input_dim += config['SignModelArguments']['projectors'][mod]['dim']
    return sign_input_dim


def main():
    def log_message(message, log_file):
        current_time = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{current_time} GMT+1] {message}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    with open("evaluation.log", "a") as log_file:
        log_message("Starting script", log_file)
        args = parse_args()
        if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
            log_message('Loading config...', log_file)
        config = load_config(args.config_file)
        config = update_config(config, args)

        log_message("Loading model configuration", log_file)
        evaluation_config = config['EvaluationArguments']
        model_config = config['ModelArguments']
        model_config['sign_input_dim'] = get_sign_input_dim(config)

        t5_config = SignT5Config()
        for param, value in model_config.items():
            if param not in vars(t5_config):
                log_message(f'{param} not in SignT5Config. It may be ignored...', log_file)
            t5_config.__setattr__(param, value)

        log_message("Loading model and tokenizer", log_file)
        model = T5ModelForSLT.from_pretrained(evaluation_config['model_dir'], config=t5_config)
        model.config.output_attentions = True
        for param in model.parameters():
            param.data = param.data.contiguous()
        tokenizer = T5Tokenizer.from_pretrained(model.config.base_model_name, clean_up_tokenization_spaces=True)

        log_message("Preparing dataset", log_file)
        dataset = DatasetForSLT(
            tokenizer=tokenizer,
            sign_data_args=config['SignDataArguments'],
            split=evaluation_config['split'],
            skip_frames=evaluation_config['skip_frames'],
            max_token_length=evaluation_config['max_token_length'],
            max_sequence_length=evaluation_config['max_sequence_length'],
            max_samples=evaluation_config['max_val_samples'],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=evaluation_config['batch_size'],
            collate_fn=lambda batch: collate_fn(
                batch,
                max_sequence_length=evaluation_config['max_sequence_length'],
                max_token_length=evaluation_config['max_token_length'],
                pose_dim=config['SignModelArguments']['projectors']['pose']['dim'],
            ),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        log_message(f"Using device: {device}", log_file)
        if device == "cuda":
            log_message(f"GPU: {torch.cuda.get_device_name(0)}", log_file)
            log_message(f"CUDA available: {torch.cuda.is_available()}", log_file)
            log_message(f"CUDA device count: {torch.cuda.device_count()}", log_file)
        param = next(model.parameters())
        log_message(f"Model parameter device: {param.device}", log_file)
        log_message(f"Model parameter dtype: {param.dtype}", log_file)

        log_message("Running model evaluation", log_file)
        predictions, labels = evaluate_model(model, dataloader, tokenizer, evaluation_config)

        log_message("Post-processing predictions", log_file)
        decoded_preds, decoded_labels = postprocess_text(predictions, [ref[0] for ref in labels])
        if args.verbose:
            for i in range(min(5, len(decoded_preds))):
                log_message(f"Prediction: {decoded_preds[i]}", log_file)
                log_message(f"Reference: {decoded_labels[i]}", log_file)
                log_message("-" * 50, log_file)

        log_message("Computing metrics", log_file)
        sacrebleu = evaluate.load('sacrebleu')
        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {
            "bleu": result["score"],
            'bleu-1': result['precisions'][0],
            'bleu-2': result['precisions'][1],
            'bleu-3': result['precisions'][2],
            'bleu-4': result['precisions'][3],
        }
        result = {k: round(v, 4) for k, v in result.items()}
        if args.verbose:
            for key, value in result.items():
                log_message(f"{key}: {value:.4f}", log_file)

        log_message("Saving predictions", log_file)
        all_predictions = [
            {"prediction": pred, "reference": ref}
            for pred, ref in zip(decoded_preds, decoded_labels)
        ]
        all_predictions = {'metrics': result, 'predictions': all_predictions[:100]}
        os.makedirs(evaluation_config['output_dir'], exist_ok=True)
        prediction_file = os.path.join(evaluation_config['output_dir'], "predictions.json")
        with open(prediction_file, "w") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=4)
        log_message(f"Predictions saved to {prediction_file}", log_file)
        log_message("Script completed", log_file)


if __name__ == "__main__":
    main()