import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import argparse
import numpy as np
from datetime import datetime
from types import SimpleNamespace
from dotenv import load_dotenv
from transformers import T5Tokenizer
from model.configuration_t5 import SignT5Config
from transformers import T5Config
from model.modeling_t5 import T5ModelForSLT
from utils.translation import postprocess_text
import evaluate
import yaml
from torch.utils.data import DataLoader
from dataset.generic_sl_dataset import SignFeatureDataset as DatasetForSLT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# set KMP_DUPLICATE_LIB_OK=TRUE

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned T5 model for SLT with full attention logging")

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

    # Uni-Sign specific optional overrides
    parser.add_argument("--model_type", type=str, default=None, help="t5 or unisign")
    parser.add_argument("--finetune", type=str, default=None, help="Path to Uni-Sign checkpoint (.pth)")
    parser.add_argument("--dataset", type=str, default=None, help="Uni-Sign dataset name, e.g. YTASL")
    parser.add_argument("--task", type=str, default=None, help="Uni-Sign task, e.g. SLT")
    parser.add_argument("--normalization", type=str, default=None, help="Uni-Sign normalization, e.g. signspace")
    parser.add_argument("--n_registers", type=int, default=None, help="Uni-Sign register tokens")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers for Uni-Sign")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length in Uni-Sign dataset loader")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max generated tokens in Uni-Sign generation")
    parser.add_argument("--mt5_path", type=str, default=None, help="Path/name of mT5 backbone for Uni-Sign")
    parser.add_argument("--layout", type=str, default=None, help="Pose layout for Uni-Sign")
    parser.add_argument("--register_position", type=str, default=None, help="Register token placement for Uni-Sign")

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

    def _normalize_none_values(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    _normalize_none_values(value)
                elif value in ("none", "None"):
                    obj[key] = None
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, (dict, list)):
                    _normalize_none_values(value)
                elif value in ("none", "None"):
                    obj[i] = None

    _normalize_none_values(cfg)
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
        if v is None:
            continue
        if k in cfg.get('EvaluationArguments', {}):
            cfg['EvaluationArguments'][k] = v
            if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
                print(f'Config value updated by args - EvaluationArguments.{k}: {v}')
        if k in cfg.get('UniSignArguments', {}):
            cfg['UniSignArguments'][k] = v
            if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
                print(f'Config value updated by args - UniSignArguments.{k}: {v}')
    return cfg


def collate_fn(batch, max_sequence_length, max_token_length, pose_dim):
    # If sign_inputs is a dictionary (e.g. for multimodal inputs), extract the pose component.
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


def evaluate_model(model, dataloader, tokenizer, evaluation_config):
    """
    Run inference and store predictions, labels, and all attention data:
    - encoder_attentions: from the encoder.
    - decoder_attentions: from the decoder (self-attention).
    - cross_attentions: attention over encoder outputs from the decoder.
    """
    def log_message(message, log_file):
        current_time = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{current_time} GMT+1] {message}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    def save_attention_batch(batch_idx, enc_attn, dec_attn, cross_attn, reference_translations, predictions, output_dir):
        """Save attention data for current batch"""
        # Decode tokens for each sequence in the batch, skipping the first token
        decoded_tokens = [
            tokenizer.batch_decode(seq[1:], skip_special_tokens=True)  # Skip first token
            for seq in sequences
        ]
        
        batch_data = {
            "encoder_attentions": enc_attn,
            "decoder_attentions": dec_attn,
            "cross_attentions": cross_attn,
            "reference_translations": reference_translations,
            "predictions": predictions,
            "decoded_tokens": decoded_tokens  # Add decoded tokens to the output
        }
        os.makedirs(os.path.join(output_dir, "attention_batches"), exist_ok=True)
        with open(os.path.join(output_dir, "attention_batches", f"batch_{batch_idx}.json"), "w") as f:
            json.dump(batch_data, f)

    predictions, labels = [], []
    
    with open("evaluation.log", "a") as log_file:
        model_dtype = next(model.parameters()).dtype
        log_message(f"Model dtype: {model_dtype}", log_file)
        log_message("Starting model evaluation", log_file)
        model.eval()
        total_batches = len(dataloader)
        log_message(f"Beginning inference loop with {total_batches} batches", log_file)

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if step % 10 == 0:
                    log_message(f"Processing batch {step}/{total_batches}", log_file)
                
                # Convert batch to model's dtype and move to device
                batch = {k: v.to(model.base_model.device).to(model_dtype) for k, v in batch.items()}
                
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
                    output_attentions=True,
                    return_dict_in_generate=True
                )

                sequences = outputs.sequences

                # Process attention weights for current batch
                enc_attn = []
                dec_attn = []
                cross_attn = []

                if hasattr(outputs, "encoder_attentions"):
                    # Each element: tensor of shape [batch_size, num_heads, seq_len, seq_len]
                    enc_attn = [layer_attn.detach().cpu().tolist() for layer_attn in outputs.encoder_attentions]
                
                if hasattr(outputs, "decoder_attentions"):
                    # Decoder attentions are tuples for each generation step
                    # Each tuple contains tensors for each layer
                    for step_attentions in outputs.decoder_attentions:
                        step_attn = [layer_attn.detach().cpu().tolist() for layer_attn in step_attentions]
                        dec_attn.append(step_attn)
                
                if hasattr(outputs, "cross_attentions"):
                    # Cross attentions are also tuples for each generation step
                    for step_attentions in outputs.cross_attentions:
                        step_attn = [layer_attn.detach().cpu().tolist() for layer_attn in step_attentions]
                        cross_attn.append(step_attn)

                # Process predictions
                if len(np.where(sequences.cpu().numpy() > len(tokenizer) - 1)[1]) > 0:
                    log_message(f'Replacing <unk> for illegal tokens found on indexes {np.where(sequences.cpu().numpy() > len(tokenizer) - 1)[1]}', log_file)
                sequences[sequences > len(tokenizer) - 1] = tokenizer.unk_token_id

                decoded_preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

                # Save attention weights for current batch with reference texts and predictions
                save_attention_batch(step, enc_attn, dec_attn, cross_attn, decoded_labels, decoded_preds, evaluation_config['output_dir'])

                # Clear variables to free memory
                del enc_attn, dec_attn, cross_attn

                predictions.extend(decoded_preds)
                labels.extend([[translation] for translation in decoded_labels])

                # Clear CUDA cache periodically
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                
        log_message("Completed model evaluation", log_file)
    
    return predictions, labels


def get_sign_input_dim(config):
    sign_input_dim = 0
    for mod in config['SignDataArguments']['visual_features']:
        if config['SignDataArguments']['visual_features'][mod]['enable_input']:
            sign_input_dim += config['SignModelArguments']['projectors'][mod]['dim']
    return sign_input_dim


def _move_unisign_batch_to_device(src_input, device):
    moved = {}
    for key, value in src_input.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _build_unisign_args(config, evaluation_config):
    uni_cfg = config.get("UniSignArguments", {})
    return SimpleNamespace(
        # Core Uni-Sign args required by model / dataset
        hidden_dim=uni_cfg.get("hidden_dim", 256),
        max_length=uni_cfg.get("max_length", 256),
        normalization=uni_cfg.get("normalization", "signspace"),
        n_registers=uni_cfg.get("n_registers", 0),
        register_position=uni_cfg.get("register_position", "before_all"),
        no_adaptive_gcn=uni_cfg.get("no_adaptive_gcn", False),
        dataset=uni_cfg.get("dataset", "YTASL"),
        layout=uni_cfg.get("layout", "default"),
        task=uni_cfg.get("task", "SLT"),
        rgb_support=uni_cfg.get("rgb_support", False),
        label_smoothing=uni_cfg.get("label_smoothing", 0.2),
        # Runtime / dataloader args
        batch_size=evaluation_config.get("batch_size", 2),
        num_workers=uni_cfg.get("num_workers", 0),
        pin_mem=uni_cfg.get("pin_mem", False),
        # Model init / checkpoint
        finetune=uni_cfg.get("finetune"),
        mt5_path=uni_cfg.get("mt5_path"),
    )


def evaluate_model_unisign(model, dataloader, tokenizer, evaluation_config, log_message, log_file):
    def save_attention_batch(batch_idx, enc_attn, dec_attn, cross_attn, reference_translations, predictions, sequences, output_dir):
        decoded_tokens = [
            tokenizer.batch_decode(seq[1:], skip_special_tokens=True)
            for seq in sequences
        ]

        batch_data = {
            "encoder_attentions": enc_attn,
            "decoder_attentions": dec_attn,
            "cross_attentions": cross_attn,
            "reference_translations": reference_translations,
            "predictions": predictions,
            "decoded_tokens": decoded_tokens,
        }
        os.makedirs(os.path.join(output_dir, "attention_batches"), exist_ok=True)
        with open(os.path.join(output_dir, "attention_batches", f"batch_{batch_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False)

    predictions, labels = [], []
    model.eval()
    total_batches = len(dataloader)
    log_message(f"Beginning Uni-Sign inference loop with {total_batches} batches", log_file)

    with torch.no_grad():
        for step, (src_input, tgt_input) in enumerate(dataloader):
            if step % 10 == 0:
                log_message(f"Processing Uni-Sign batch {step}/{total_batches}", log_file)

            src_input = _move_unisign_batch_to_device(src_input, model.mt5_model.device)
            pre_compute = model(src_input, tgt_input)

            # Forced attention extraction for compatibility with existing visualizers.
            outputs = model.mt5_model.generate(
                inputs_embeds=pre_compute["inputs_embeds"],
                attention_mask=pre_compute["attention_mask"],
                max_new_tokens=evaluation_config.get("max_new_tokens", 256),
                num_beams=evaluation_config.get("num_beams", 5),
                early_stopping=evaluation_config.get("early_stopping", True),
                no_repeat_ngram_size=evaluation_config.get("no_repeat_ngram_size", 3),
                length_penalty=evaluation_config.get("length_penalty", 0.6),
                output_attentions=True,
                return_dict_in_generate=True,
            )

            sequences = outputs.sequences

            enc_attn = []
            dec_attn = []
            cross_attn = []

            if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
                enc_attn = [layer_attn.detach().cpu().tolist() for layer_attn in outputs.encoder_attentions]

            if hasattr(outputs, "decoder_attentions") and outputs.decoder_attentions is not None:
                for step_attentions in outputs.decoder_attentions:
                    step_attn = [layer_attn.detach().cpu().tolist() for layer_attn in step_attentions]
                    dec_attn.append(step_attn)

            if hasattr(outputs, "cross_attentions") and outputs.cross_attentions is not None:
                for step_attentions in outputs.cross_attentions:
                    step_attn = [layer_attn.detach().cpu().tolist() for layer_attn in step_attentions]
                    cross_attn.append(step_attn)

            decoded_preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            decoded_labels = tgt_input["gt_sentence"]

            save_attention_batch(
                step,
                enc_attn,
                dec_attn,
                cross_attn,
                decoded_labels,
                decoded_preds,
                sequences,
                evaluation_config["output_dir"],
            )

            predictions.extend(decoded_preds)
            labels.extend([[translation] for translation in decoded_labels])

            if step % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return predictions, labels


def run_unisign_pipeline(config, evaluation_config, log_message, log_file):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    unisign_root = os.path.join(repo_root, "_unisign_source")
    if unisign_root not in sys.path:
        sys.path.insert(0, unisign_root)

    import importlib.util

    def _load_module(module_name, file_name):
        module_path = os.path.join(unisign_root, file_name)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    # Ensure Uni-Sign local modules are used even when T5 utils package is already imported.
    _load_module("utils", "utils.py")
    _load_module("normalization", "normalization.py")
    unisign_config = _load_module("config", "config.py")
    uni_cfg = config.get("UniSignArguments", {})

    if uni_cfg.get("mt5_path"):
        unisign_config.mt5_path = uni_cfg["mt5_path"]
    dataset_name = uni_cfg.get("dataset", "YTASL")
    if uni_cfg.get("train_label_path"):
        unisign_config.train_label_paths[dataset_name] = uni_cfg["train_label_path"]
    if uni_cfg.get("dev_label_path"):
        unisign_config.dev_label_paths[dataset_name] = uni_cfg["dev_label_path"]
    if uni_cfg.get("test_label_path"):
        unisign_config.test_label_paths[dataset_name] = uni_cfg["test_label_path"]
    if "rgb_dir" in uni_cfg:
        unisign_config.rgb_dirs[dataset_name] = uni_cfg["rgb_dir"]
    if uni_cfg.get("pose_dir"):
        unisign_config.pose_dirs[dataset_name] = uni_cfg["pose_dir"]

    unisign_models = _load_module("models", "models.py")
    unisign_datasets = _load_module("unisign_datasets_runtime", "datasets.py")

    uni_args = _build_unisign_args(config, evaluation_config)
    if not uni_args.finetune:
        raise ValueError("UniSignArguments.finetune must be set to a valid checkpoint path.")

    if uni_args.task != "SLT":
        raise ValueError("This inference script currently supports Uni-Sign task='SLT' only.")

    split = evaluation_config.get("split", "test")
    if split == "train":
        label_path = unisign_config.train_label_paths[uni_args.dataset]
    elif split == "dev":
        label_path = unisign_config.dev_label_paths[uni_args.dataset]
    else:
        label_path = unisign_config.test_label_paths[uni_args.dataset]

    log_message(f"Loading Uni-Sign dataset from: {label_path}", log_file)
    if uni_args.dataset == "YTASL":
        dataset = unisign_datasets.S2T_Dataset_YTASL(label_path, uni_args, split)
    else:
        dataset = unisign_datasets.S2T_Dataset(label_path, uni_args, split)

    if evaluation_config.get("max_val_samples") is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            list(range(min(len(dataset), int(evaluation_config["max_val_samples"])))),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=evaluation_config.get("batch_size", 2),
        num_workers=uni_args.num_workers,
        pin_memory=uni_args.pin_mem,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"Using device: {device}", log_file)

    model = unisign_models.Uni_Sign(uni_args)
    checkpoint = torch.load(uni_args.finetune, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_result = model.load_state_dict(state_dict, strict=False)
    log_message(
        f"Loaded Uni-Sign checkpoint with missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}",
        log_file,
    )
    model.to(device)

    tokenizer = model.mt5_tokenizer
    predictions, labels = evaluate_model_unisign(model, dataloader, tokenizer, evaluation_config, log_message, log_file)
    decoded_preds, decoded_labels = postprocess_text(predictions, [ref[0] for ref in labels])

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

    all_predictions = [
        {"prediction": pred, "reference": ref}
        for pred, ref in zip(decoded_preds, decoded_labels)
    ]
    all_predictions = {'metrics': result, 'predictions': all_predictions[:100]}
    os.makedirs(evaluation_config['output_dir'], exist_ok=True)
    prediction_file = os.path.join(evaluation_config['output_dir'], "predictions.json")
    with open(prediction_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)
    log_message(f"Uni-Sign predictions saved to {prediction_file}", log_file)


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
        model_type = str(evaluation_config.get("model_type", "t5")).lower()

        if model_type in {"unisign", "uni-sign"}:
            log_message("Running Uni-Sign inference path (T5 path kept intact).", log_file)
            run_unisign_pipeline(config, evaluation_config, log_message, log_file)
            log_message("Script completed", log_file)
            return

        model_config = config['ModelArguments']
        model_config['sign_input_dim'] = get_sign_input_dim(config)

        # Initialize custom T5 configuration.
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
