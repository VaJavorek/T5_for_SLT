import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
import csv
from dataset.generic_sl_dataset import SignFeatureDataset as DatasetForSLT
from utils.keypoint_dataset import KeypointDatasetJSON

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
    predictions, labels = [], []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
            if len(batch['labels'].shape) < 2:
                batch['labels'] = batch['labels'].unsqueeze(0)
            outputs = model.generate(
                **batch,
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
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            predictions.extend(decoded_preds)
            labels.extend([[translation] for translation in decoded_labels])
    return predictions, labels


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
        max_token_len = evaluation_config['max_token_length']

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
        sign_inputs = [
            torch.stack([
                torch.cat((sample["sign_inputs"][mod],
                           torch.zeros(max_seq_len - sample["sign_inputs"][mod].shape[0], modality_dim[mod])), dim=0)
                for sample in batch
            ])
            for mod in modalities
        ]

        # Process attention mask
        attention_mask = torch.stack([
            torch.cat((sample["attention_mask"], torch.zeros(max_seq_len - sample["attention_mask"].shape[0])), dim=0)
            if sample["attention_mask"].shape[0] < max_seq_len else sample["attention_mask"]
            for sample in batch
        ])

        # Process labels
        labels = torch.stack([
            torch.cat((sample["labels"].squeeze(0), torch.zeros(max_token_len - sample["labels"].shape[0])), dim=0)
            if sample["labels"].shape[0] < max_token_len else sample["labels"]
            for sample in batch
        ]).squeeze(0).to(torch.long)

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
    predictions, labels = evaluate_model(model, dataloader, tokenizer, evaluation_config)

    # Postprocess predictions and references
    decoded_preds, decoded_labels = postprocess_text(predictions, [ref[0] for ref in labels])

    if args.verbose:
        for i in range(min(5, len(decoded_preds))):
            print("Prediction:", decoded_preds[i])
            print("Reference:", decoded_labels[i])
            print("-" * 50)

    # Compute metrics
    wer_metric = evaluate.load("wer")
    # Flatten references for WER and handle potential empty references
    flat_labels = [lab[0] if isinstance(lab, (list, tuple)) else lab for lab in decoded_labels]
    try:
        wer_score = wer_metric.compute(predictions=decoded_preds, references=flat_labels)
    except ValueError as e:
        wer_score = None
    print(f"Word Error Rate: {wer_score}")
    
    decoded_labels = [list(x) for x in zip(*decoded_labels)]
    bleu1 = BLEU(max_ngram_order=1).corpus_score(decoded_preds,  decoded_labels)
    bleu2 = BLEU(max_ngram_order=2).corpus_score(decoded_preds,  decoded_labels)
    bleu3 = BLEU(max_ngram_order=3).corpus_score(decoded_preds,  decoded_labels)
    bleu4 = BLEU(max_ngram_order=4).corpus_score(decoded_preds,  decoded_labels)
    result = {
        "bleu-1": bleu1.score,
        "bleu-2": bleu2.score,
        "bleu-3": bleu3.score,
        "bleu-4": bleu4.score,
        "bleu-1_precision": bleu4.precisions[0],
        "bleu-2_precision": bleu4.precisions[1],
        "bleu-3_precision": bleu4.precisions[2],
        "bleu-4_precision": bleu4.precisions[3],
    }
    result["wer"] = wer_score
    result = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in result.items()}

    if args.verbose:
        for key, value in result.items():
            print(f"{key}: {value:.4f}")

    # Save predictions
    all_predictions = [
        {
            "prediction": pred,
            "reference": ref
        }
        for pred, ref in zip(decoded_preds, decoded_labels[0])
    ]
    all_predictions = {'metrics': result, 'predictions': all_predictions[:100]}

    os.makedirs(evaluation_config['output_dir'], exist_ok=True)
    prediction_file = os.path.join(evaluation_config['output_dir'], f"predictions_{os.path.splitext(os.path.basename(config['SignDataArguments']['annotation_path'][evaluation_config['split']]))[0].split('.')[-1]}.json")
    with open(prediction_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {prediction_file}")
    # Save submission CSV file
    csv_file = os.path.join(evaluation_config['output_dir'], f"{os.path.splitext(os.path.basename(config['SignDataArguments']['annotation_path'][evaluation_config['split']]))[0].split('.')[-1]}.csv")
    with open(csv_file, "w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["id", "gloss"])
        ids = [dataset.clip_order_from_int[vid][cid] for vid, cid in dataset.list_data]
        writer.writerows(zip(ids, decoded_preds))
    print(f"Submission CSV saved to {csv_file}")

if __name__ == "__main__":
    main()
