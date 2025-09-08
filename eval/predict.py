import os
import json
import torch
import argparse
import numpy as np
from dotenv import load_dotenv
from transformers import T5Tokenizer
from model.configuration_t5 import SignT5Config
from transformers import T5Config
from model.modeling_t5 import T5ModelForSLT
from utils.translation import postprocess_text
import evaluate
from sacrebleu.metrics import BLEU
import yaml
from dataset.generic_sl_dataset import SignFeatureDataset as DatasetForSLT
from utils.keypoint_dataset import KeypointDatasetJSON
import torch.nn.functional as F

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned T5 model for SLT")

    # Configuration
    parser.add_argument("--config_file", type=str, required=True, default='config.yaml')

    # Model and data paths
    parser.add_argument("--model_name", type=str, default=None, help="Model name or folder inside model_dir.")
    parser.add_argument("--output_dir", type=str, default=None)

    # Data processing
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Max number of frames for sign inputs.")
    parser.add_argument("--max_token_length", type=int, default=None, help="Max token length for labels.")
    parser.add_argument("--skip_frames", default=None)
    parser.add_argument("--paraphrase_mode", type=str, default=None, help='Paraphrase mode: "none" | "random" | "min_loss"')

    # Generation parameters
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the directory containing the fine-tuned model and config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference.")
    # parser.add_argument("--pose_dim", type=int, default=208, help="Dimension of the pose embeddings.")

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
    """
    Load config from a yaml file. 'none' and 'None' values are replaced by None value.
    Args:
        cfg_path: Path to config file
    Returns:
        config (dict): Config
    """
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    for param, value in cfg['EvaluationArguments'].items():
        if value == 'none' or value == 'None':
            cfg['EvaluationArguments'][param] = None

    # Default paraphrase behavior for evaluation: include all references so we can report
    # metrics with and without paraphrases. If missing, fall back to "min_loss".
    if 'paraphrase_mode' not in cfg['EvaluationArguments'] or cfg['EvaluationArguments']['paraphrase_mode'] is None:
        cfg['EvaluationArguments']['paraphrase_mode'] = 'min_loss'

    return cfg


def update_config(cfg, args):
    """
    Update config with args passed. Default None arguments are ignored.
    Args:
        cfg (dict): Config
        args (argparse.Namespace): Argument parsed from the command-line
    Returns:
        cfg (dict): Updated config
    """
    for k, v in vars(args).items():
        if k in cfg['EvaluationArguments'] and v is not None:
            cfg['EvaluationArguments'][k] = v
            if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
                print('Config value updated by args - {}: {}'.format(k, v))
    return cfg

# def collate_fn(batch, max_sequence_length, max_token_length, pose_dim):
#     return {
#         "sign_inputs": torch.stack([
#             torch.cat((sample["sign_inputs"], torch.zeros(max_sequence_length - sample["sign_inputs"].shape[0], pose_dim)), dim=0)
#             for sample in batch
#         ]),
#         "attention_mask": torch.stack([
#             torch.cat((sample["attention_mask"], torch.zeros(max_sequence_length - sample["attention_mask"].shape[0])), dim=0)
#             if sample["attention_mask"].shape[0] < max_sequence_length
#             else sample["attention_mask"]
#             for sample in batch
#         ]),
#         "labels": torch.stack([
#             torch.cat((sample["labels"].squeeze(0), torch.zeros(max_token_length - sample["labels"].shape[0])), dim=0)
#             if sample["labels"].shape[0] < max_token_length
#             else sample["labels"]
#             for sample in batch
#         ]).squeeze(0).to(torch.long),
#     }

def evaluate_model(model, dataloader, tokenizer, evaluation_config):
    model.eval()
    predictions, canonical_refs, all_refs = [], [], []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # Move tensors to device; keep labels as tensor
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
            # Do not pass labels into generate
            gen_batch = {k: v for k, v in batch.items() if k != "labels"}
            outputs = model.generate(
                **gen_batch,
                early_stopping=model.config.early_stopping,
                no_repeat_ngram_size=model.config.no_repeat_ngram_size,
                max_length=evaluation_config['max_sequence_length'],
                num_beams=model.config.num_beams,
                bos_token_id=tokenizer.pad_token_id,
                length_penalty=model.config.length_penalty,
            )
            # Replace invalid tokens with <unk>
            if len(np.where(outputs.cpu().numpy() > len(tokenizer) - 1)[1]) > 0:
                print(f'Replacing <unk> for illegal tokens found on indexes {np.where(outputs.cpu().numpy() > len(tokenizer) - 1)[1]}')
            outputs[outputs > len(tokenizer) - 1] = tokenizer.unk_token_id

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels_tensor = batch["labels"].detach().cpu()

            if labels_tensor.dim() == 2:  # [B, L]
                decoded_labels = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                canonical_refs.extend(decoded_labels)
                all_refs.extend([[ref] for ref in decoded_labels])
            elif labels_tensor.dim() == 3:  # [B, P, L]
                B, P, L = labels_tensor.size()
                flat = labels_tensor.view(B * P, L)
                decoded_flat = tokenizer.batch_decode(flat, skip_special_tokens=True)
                # regroup per sample
                refs_per_sample = [decoded_flat[i * P:(i + 1) * P] for i in range(B)]
                # In dataset, when paraphrase_mode == "min_loss", canonical is last
                canonical = [refs[-1] for refs in refs_per_sample]

                predictions.extend(decoded_preds)
                canonical_refs.extend(canonical)
                all_refs.extend(refs_per_sample)
            else:
                raise ValueError(f"Unexpected labels shape: {labels_tensor.size()}")
    return predictions, canonical_refs, all_refs


def get_sign_input_dim(config):
    sign_input_dim = 0
    for mod in config['SignDataArguments']['visual_features']:
        if config['SignDataArguments']['visual_features'][mod]['enable_input']:
            sign_input_dim += config['SignModelArguments']['projectors'][mod]['dim']
    return sign_input_dim


def main():
    args = parse_args()
    if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
        print('Loading config...')
    config = load_config(args.config_file)
    config = update_config(config, args)

    evaluation_config = config['EvaluationArguments']
    model_config = config['ModelArguments']
    model_config['sign_input_dim'] = get_sign_input_dim(config)

    # Initialize the custom model
    t5_config = SignT5Config()
    for param, value in model_config.items():
        if param not in vars(t5_config):
            print('f{param} not in SignT5Config. It may be ignored...}')
        t5_config.__setattr__(param, value)

    # Load model and tokenizer
    model = T5ModelForSLT.from_pretrained(evaluation_config['model_dir'], config=t5_config)
    for param in model.parameters(): param.data = param.data.contiguous()
    tokenizer = T5Tokenizer.from_pretrained(model.config.base_model_name, clean_up_tokenization_spaces=True)

    # Add collate_fn to DataLoader
    def collate_fn(batch):
        # Add padding to the inputs
        # YT-ASL paper:
            # "inputs" must be 250 frames long
            # "attention_mask" must be 250 frames long
            # "labels" must be 128 tokens long
        max_seq_len = evaluation_config['max_sequence_length']
        # Note: labels may be 2D [L] or 2D per-sample [P, L]; we'll pad both dims dynamically

        # List of enabled modalities (based on config)
        modalities = [
            mod for mod in config['SignDataArguments']['visual_features']
            if config['SignDataArguments']['visual_features'][mod]['enable_input']
        ]
        # Get the dimensionality for enabled modalities
        modality_dim = {
            mod: config['SignModelArguments']['projectors'][mod]['dim']
            for mod in modalities
        }

        # Process each enabled modality
        # Use actual feature width per modality from the first sample; truncate then pad to max_seq_len
        sign_inputs = []
        for mod in modalities:
            feat_dim = batch[0]["sign_inputs"][mod].shape[1]
            stacked = torch.stack([
                torch.cat(
                    (
                        sample["sign_inputs"][mod][:max_seq_len],
                        torch.zeros(
                            max(0, max_seq_len - sample["sign_inputs"][mod].shape[0]),
                            feat_dim,
                            dtype=sample["sign_inputs"][mod].dtype,
                        ),
                    ),
                    dim=0,
                )
                for sample in batch
            ])
            sign_inputs.append(stacked)

        # Process attention mask
        attention_mask = torch.stack([
            torch.cat(
                (
                    sample["attention_mask"][:max_seq_len],
                    torch.zeros(max(0, max_seq_len - sample["attention_mask"].shape[0])),
                ),
                dim=0,
            )
            for sample in batch
        ])

        # Process labels
        # Each sample["labels"] may be [L] or [P, L]. Normalize to [P, L] and pad P and L.
        # Determine max paraphrases (P) and max tokens (L) in this batch
        def get_PL(x):
            if x.dim() == 1:
                return 1, x.size(0)
            return x.size(0), x.size(1)

        max_p = max(get_PL(sample["labels"])[0] for sample in batch)
        max_l = max(get_PL(sample["labels"])[1] for sample in batch)

        padded_labels = []
        for sample in batch:
            lab = sample["labels"]
            if lab.dim() == 1:  # [L] -> [1, L]
                lab = lab.unsqueeze(0)
            lab = F.pad(
                lab,
                pad=(0, max_l - lab.size(1),   # tokens
                     0, max_p - lab.size(0)),  # paraphrases
                value=tokenizer.pad_token_id,
            )
            padded_labels.append(lab)
        labels = torch.stack(padded_labels).to(torch.long)  # [B, max_P, max_L]

        return {
            "sign_inputs": torch.cat(sign_inputs, dim=-1),
            "attention_mask": attention_mask,
            "labels": labels
        }
    # Prepare dataset
    pose_config = config['SignDataArguments']['visual_features']['pose']
    test_raw_pose_data_path = pose_config['normalization']['test_json_dir']

    if os.path.isdir(test_raw_pose_data_path):
        test_pose_dataset = KeypointDatasetJSON(json_folder=test_raw_pose_data_path,
                                           kp_normalization=(
                                               "global-pose_landmarks",
                                               "local-right_hand_landmarks",
                                               "local-left_hand_landmarks",
                                               "local-face_landmarks",),
                                           kp_normalization_method=pose_config['normalization']['normalization_method'],
                                           data_key=pose_config['normalization']['data_key'],
                                           missing_values=pose_config['missing_values'],
                                           augmentation_configs=[],
                                           load_from_raw=evaluation_config['load_from_raw'],
                                           interpolate=pose_config['interpolate'],
                                           )
        print('Train raw pose data path: {}'.format(test_raw_pose_data_path))
    else:
        test_pose_dataset = None
        print('Raw poses not found in {}'.format(test_raw_pose_data_path))

    dataset = DatasetForSLT(tokenizer= tokenizer,
                                sign_data_args=config['SignDataArguments'],
                                split=evaluation_config['split'],
                                skip_frames=evaluation_config['skip_frames'],
                                max_token_length=evaluation_config['max_token_length'],
                                max_sequence_length=evaluation_config['max_sequence_length'],
                                max_samples=evaluation_config['max_val_samples'],
                                pose_dataset=test_pose_dataset,
                                float32=evaluation_config['float32'],
                                decimal_points=evaluation_config['decimal_points'],
                                paraphrase_mode=evaluation_config['paraphrase_mode'],
                                )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=evaluation_config['batch_size'],
        collate_fn=collate_fn,
        # collate_fn=lambda batch: collate_fn(
        #     batch,
        #     max_sequence_length=evaluation_config['max_sequence_length'],
        #     max_token_length=evaluation_config['max_token_length'],
        #     pose_dim=config['SignModelArguments']['projectors']['pose']['dim'],
        # ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Evaluating...")
    predictions, canonical_refs, all_refs = evaluate_model(model, dataloader, tokenizer, evaluation_config)

    # Postprocess predictions and references
    # Strip whitespace
    decoded_preds = [p.strip() for p in predictions]
    decoded_labels_canonical = [r.strip() for r in canonical_refs]
    decoded_labels_all = [[r.strip() for r in refs] for refs in all_refs]

    if args.verbose:
        for i in range(min(5, len(decoded_preds))):
            print("Prediction:", decoded_preds[i])
            print("Reference (canonical):", decoded_labels_canonical[i])
            print("References (all):", decoded_labels_all[i])
            print("-" * 50)

    # Compute metrics
    # Prepare references for corpus BLEU
    # Without paraphrases: single canonical reference corpus [1 x N]
    refs_without = [decoded_labels_canonical]

    # With paraphrases: build reference corpora [R_max x N], padding with canonical when needed
    max_r = max(len(refs) for refs in decoded_labels_all) if len(decoded_labels_all) > 0 else 1
    refs_with = []
    for i in range(max_r):
        col = [refs[i] if i < len(refs) else refs[-1] for refs in decoded_labels_all]
        refs_with.append(col)

    # Compute metrics without paraphrases
    bleu1_wo = BLEU(max_ngram_order=1).corpus_score(decoded_preds, refs_without)
    bleu2_wo = BLEU(max_ngram_order=2).corpus_score(decoded_preds, refs_without)
    bleu3_wo = BLEU(max_ngram_order=3).corpus_score(decoded_preds, refs_without)
    bleu4_wo = BLEU(max_ngram_order=4).corpus_score(decoded_preds, refs_without)

    # Compute metrics with paraphrases
    bleu1_w = BLEU(max_ngram_order=1).corpus_score(decoded_preds, refs_with)
    bleu2_w = BLEU(max_ngram_order=2).corpus_score(decoded_preds, refs_with)
    bleu3_w = BLEU(max_ngram_order=3).corpus_score(decoded_preds, refs_with)
    bleu4_w = BLEU(max_ngram_order=4).corpus_score(decoded_preds, refs_with)

    result = {
        "without_paraphrases": {
            "bleu-1": bleu1_wo.score,
            "bleu-2": bleu2_wo.score,
            "bleu-3": bleu3_wo.score,
            "bleu-4": bleu4_wo.score,
            "bleu-1_precision": bleu4_wo.precisions[0],
            "bleu-2_precision": bleu4_wo.precisions[1],
            "bleu-3_precision": bleu4_wo.precisions[2],
            "bleu-4_precision": bleu4_wo.precisions[3],
        },
        "with_paraphrases": {
            "bleu-1": bleu1_w.score,
            "bleu-2": bleu2_w.score,
            "bleu-3": bleu3_w.score,
            "bleu-4": bleu4_w.score,
            "bleu-1_precision": bleu4_w.precisions[0],
            "bleu-2_precision": bleu4_w.precisions[1],
            "bleu-3_precision": bleu4_w.precisions[2],
            "bleu-4_precision": bleu4_w.precisions[3],
        }
    }

    # Round nested floats to 4 decimals
    for group in ["without_paraphrases", "with_paraphrases"]:
        for k in list(result[group].keys()):
            result[group][k] = round(result[group][k], 4)

    if args.verbose:
        print("BLEU without paraphrases:")
        for key, value in result["without_paraphrases"].items():
            print(f"  {key}: {value:.4f}")
        print("BLEU with paraphrases:")
        for key, value in result["with_paraphrases"].items():
            print(f"  {key}: {value:.4f}")

    # Save predictions
    # Build per-sample best paraphrase info (by sentence BLEU-4)
    bleu_sent = BLEU(max_ngram_order=4)
    per_sample = []
    for pred, canon, refs in zip(decoded_preds, decoded_labels_canonical, decoded_labels_all):
        best_ref = None
        best_score = -1.0
        for r in refs:
            s = bleu_sent.sentence_score(pred, [r]).score
            if s > best_score:
                best_score = s
                best_ref = r
        per_sample.append({
            "prediction": pred,
            "reference_canonical": canon,
            "best_reference": best_ref,
            "best_bleu4": round(best_score, 4),
        })

    all_predictions = {'metrics': result, 'predictions': per_sample[:100]}

    os.makedirs(evaluation_config['output_dir'], exist_ok=True)
    prediction_file = os.path.join(evaluation_config['output_dir'], "predictions.json")
    with open(prediction_file, "w") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {prediction_file}")

if __name__ == "__main__":
    main()
