from transformers import T5Tokenizer
from model.configuration_t5 import SignT5Config
from model.modeling_t5 import T5ModelForSLT
import torch
import os
import argparse
from train.run_finetuning import parse_args, load_config, update_config


if __name__ == "__main__":
    args = parse_args()
    if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
        print('Loading config from {}...'.format(args.config_file))
    config = load_config(args.config_file)
    config = update_config(config, args)

    training_config = config['TrainingArguments']
    model_config = config['ModelArguments']

    # 1. Load your config
    t5_config = SignT5Config()
    for param, value in model_config.items():
        if param not in vars(t5_config):
            print(f'{param} not in SignT5Config. It may be ignored...')
        t5_config.__setattr__(param, value)

    # 2. Load the checkpoint model (weights only)
    model = T5ModelForSLT.from_pretrained(training_config['resume_from_checkpoint'], config=t5_config)

    # 3. Replace the projector layer
    # Assuming the model has an attribute like model.projector = nn.Linear(input_dim, hidden_dim)

    # Example: Check old projector
    print("Old projector:", model.custom_linear)

    # Create a new projector with different input dim (e.g., 172)
    import torch.nn as nn
    new_input_dim = 172
    hidden_dim = model.custom_linear[0].out_features  # same output dim as old projector

    model.custom_linear = nn.Sequential(
        nn.Linear(new_input_dim, hidden_dim, bias=False),
        nn.Dropout(model.config.hidden_dropout_prob),
        nn.GELU(),
    )

    # Initialize weights
    for layer in model.custom_linear:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    print("New projector:", model.custom_linear)

    # 5. Save the updated model
    save_path = "/media/zeleznyt/DATA/repo/T5_for_SLT/results/T5_YTASL_pretrain/mt5-checkpoint-with-new-projector"
    model.save_pretrained(save_path)
    tokenizer = T5Tokenizer.from_pretrained(model.config.base_model_name)
    tokenizer.save_pretrained(save_path)
