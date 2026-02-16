# Unified Attention Visualizer

This document describes how to use `scripts/attention_visualizer_unified.py` and how to read its outputs.

## What It Does

The script loads attention batch JSON files (from `predict_attention_store_batch*`) and generates:

- attention heatmaps for encoder / decoder / cross attentions
- averaged variants across layers and/or heads
- histogram summaries
- a text summary file with GT vs prediction for each batch

It supports processing a single JSON file or a whole directory of JSON files.

## Input Format

Expected JSON keys per batch:

- `encoder_attentions`
- `decoder_attentions`
- `cross_attentions`
- `reference_translations`
- `predictions`
- `decoded_tokens` (used mainly for cross-attention y-axis labels)

## Basic Usage

```bash
python scripts/attention_visualizer_unified.py \
  --input results/Uni-Sign/attention_batches \
  --output results/Uni-Sign/plots \
  --type all \
  --mode all \
  --crop auto
```

## Arguments

- `--input`: path to one JSON file or directory with batch JSONs
- `--output`: output root directory
- `--type`: `encoder | decoder | cross | all`
- `--mode`: comma-separated list:
  - `layer_heads`
  - `avg_layers`
  - `avg_heads_per_layer`
  - `avg_all`
  - `hist`
  - `all`
- `--crop`: `auto | square | fixed | none`
- `--threshold`: threshold used for non-zero region detection
- `--max_files`: limit number of processed files (useful for smoke test)

## Output Structure

Outputs are organized by batch in subfolders:

- `<output>/batch_text_comparison.txt`
- `<output>/batch_0/...png`
- `<output>/batch_1/...png`
- etc.

The text file is generated before plotting and contains, for each batch:

1. batch name
2. `GT: ...`
3. `Pred: ...`

## Output File Types

For batch `batch_X`:

### Encoder

- `batch_X_enc_L{N}.png`  
  Per-layer, all heads (`layer_heads`)
- `batch_X_enc_avg_heads.png`  
  Per-head, averaged across layers (`avg_layers`)
- `batch_X_enc_avg_heads_per_layer.png`  
  Per-layer, averaged across heads (`avg_heads_per_layer`)
- `batch_X_enc_avg_all_layers_heads.png`  
  Single global average across layers and heads (`avg_all`)

### Decoder

- `batch_X_dec_L{N}.png`  
  Per-layer, all heads (`layer_heads`)
- `batch_X_dec_avg_heads.png`  
  Per-head, averaged across layers (`avg_layers`)
- `batch_X_dec_avg_heads_per_layer.png`  
  Per-layer, averaged across heads (`avg_heads_per_layer`)
- `batch_X_dec_avg_all_layers_heads.png`  
  Single global average across layers and heads (`avg_all`)
- `batch_X_dec_hist.png`  
  Histogram summary (`hist`)

### Cross

- `batch_X_cross_avg_layers.png`  
  Per-head, averaged across layers (`avg_layers`)
- `batch_X_cross_avg_heads_per_layer.png`  
  Per-layer, averaged across heads (`avg_heads_per_layer`)
- `batch_X_cross_avg_all_layers_heads.png`  
  Single global average across layers and heads (`avg_all`)
- `batch_X_cross_hist.png`  
  Global histogram summary (`hist`)

Note: cross per-layer-per-head grids are not part of the unified script at the moment.

## Recommended Analysis Flow

If you want fast and stable interpretation:

1. Start with `cross_avg_layers` and `cross_avg_all_layers_heads`
2. Check `cross_hist` for frame concentration
3. Inspect `dec_avg_heads` and `enc_avg_heads`
4. Drill into `*_L{N}.png` only when something looks suspicious

## Useful Command Recipes

### Smoke test (first 2 batches)

```bash
python scripts/attention_visualizer_unified.py \
  --input results/Uni-Sign/attention_batches \
  --output results/Uni-Sign/plots_smoke \
  --type all \
  --mode all \
  --crop auto \
  --max_files 2
```

### Only new global and per-layer-head averages

```bash
python scripts/attention_visualizer_unified.py \
  --input results/Uni-Sign/attention_batches \
  --output results/Uni-Sign/plots_avgs \
  --type all \
  --mode avg_heads_per_layer,avg_all \
  --crop auto
```

### Cross-only focused pass

```bash
python scripts/attention_visualizer_unified.py \
  --input results/Uni-Sign/attention_batches \
  --output results/Uni-Sign/plots_cross \
  --type cross \
  --mode avg_layers,avg_heads_per_layer,avg_all,hist \
  --crop auto
```

## Notes

- If plots look too wide or too empty, tune `--threshold` (e.g. `1e-8` or `1e-10`)
- `--crop square` may improve visual comparability for self-attention blocks
- `--crop none` is useful when you need full context for every matrix
