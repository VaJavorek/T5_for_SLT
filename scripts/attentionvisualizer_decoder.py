import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Change directory if needed (adjust path as required)
os.chdir('/auto/brno2/home/javorek/T5_for_SLT/')

# Folder containing JSON files with attention batches
folder_path = os.path.join('results', 'attention_batches_simple_labeled')

# Create output folders for decoder self-attention plots
decoder_plot_folder = os.path.join(folder_path, "decoder_plots_auto_crop")
os.makedirs(decoder_plot_folder, exist_ok=True)

decoder_avg_plot_folder = os.path.join(folder_path, "decoder_avg_plots_auto_crop")
os.makedirs(decoder_avg_plot_folder, exist_ok=True)

decoder_avg_hist_folder = os.path.join(folder_path, "decoder_avg_histograms")
os.makedirs(decoder_avg_hist_folder, exist_ok=True)

decoder_layer_avg_plot_folder = os.path.join(folder_path, "decoder_layer_avg_plots_auto_crop")
os.makedirs(decoder_layer_avg_plot_folder, exist_ok=True)

decoder_layer_avg_hist_folder = os.path.join(folder_path, "decoder_layer_avg_histograms")
os.makedirs(decoder_layer_avg_hist_folder, exist_ok=True)

# Threshold for cropping (if a row/column sum is below this, treat as zero)
CROP_THRESHOLD = 1e-9

def find_nonzero_bbox_2d(matrix, threshold=CROP_THRESHOLD):
    """
    Find bounding box (min_row, max_row, min_col, max_col) of non-zero values in a 2D matrix.
    Returns None if the matrix is effectively zero.
    """
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    nonempty_rows = np.where(row_sums > threshold)[0]
    nonempty_cols = np.where(col_sums > threshold)[0]
    if len(nonempty_rows) == 0 or len(nonempty_cols) == 0:
        return None
    return (nonempty_rows[0], nonempty_rows[-1], nonempty_cols[0], nonempty_cols[-1])

def crop_2d(matrix, bbox):
    """Crop a 2D numpy array to the bounding box."""
    min_r, max_r, min_c, max_c = bbox
    return matrix[min_r:max_r+1, min_c:max_c+1]

def create_title_with_translation(filename, layer_info, translation, prediction, max_chars=50):
    """Create a multi-line title including file, layer info, reference and prediction."""
    if len(translation) > max_chars:
        translation = translation[:max_chars] + "..."
    if len(prediction) > max_chars:
        prediction = prediction[:max_chars] + "..."
    title = f"File: {filename}\n{layer_info}\nReference: {translation}\nPrediction: {prediction}"
    return title

def pad_row(row, target_length):
    """Pad a 1D list (row) with zeros to match target_length."""
    padded = np.zeros(target_length)
    padded[:len(row)] = row
    return padded

def build_matrix_for_layer_head(dec_attentions, num_steps, layer_index, head_index):
    """
    For a fixed decoder layer and head, extract a 2D matrix (num_steps x max_key_length)
    where each row i comes from dec_attentions[i][layer_index][0][head_index][0].
    Since key lengths may vary over steps, each row is padded with zeros.
    """
    rows = []
    lengths = []
    for step in range(num_steps):
        # Each step: dec_attentions[step] is a list over layers.
        # For batch index 0 and head head_index, get the 1D list (query_len=1 is squeezed).
        row = dec_attentions[step][layer_index][0][head_index][0]
        rows.append(row)
        lengths.append(len(row))
    max_len = max(lengths)
    # Build padded matrix: each row padded to max_len
    matrix = np.stack([pad_row(r, max_len) for r in rows])
    return matrix

def build_aggregated_matrix_across_layers(dec_attentions, num_steps, num_layers, head_index):
    """
    For a fixed head, average the self-attention matrices over layers for each generation step.
    Because key lengths may vary both over steps and (if any) across layers, each step is processed separately.
    Returns a 2D matrix of shape (num_steps, max_key_length_over_steps).
    """
    agg_rows = []
    lengths = []
    for step in range(num_steps):
        layer_rows = []
        layer_lengths = []
        for layer in range(num_layers):
            row = dec_attentions[step][layer][0][head_index][0]
            layer_rows.append(row)
            layer_lengths.append(len(row))
        max_len = max(layer_lengths)
        # Pad each layer's row for this step and average elementwise
        padded = np.stack([pad_row(r, max_len) for r in layer_rows])
        agg_row = np.mean(padded, axis=0)
        agg_rows.append(agg_row)
        lengths.append(max_len)
    overall_max = max(lengths)
    # Pad each aggregated row to overall_max
    agg_matrix = np.stack([pad_row(r, overall_max) for r in agg_rows])
    return agg_matrix

def build_layer_avg_matrix_across_heads(dec_attentions, num_steps, layer_index, num_heads):
    """
    For a fixed layer, average the self-attention matrices over heads for each generation step.
    Returns a 2D matrix of shape (num_steps, max_key_length_over_steps).
    """
    avg_rows = []
    lengths = []
    for step in range(num_steps):
        head_rows = []
        head_lengths = []
        for head in range(num_heads):
            row = dec_attentions[step][layer_index][0][head][0]
            head_rows.append(row)
            head_lengths.append(len(row))
        max_len = max(head_lengths)
        padded = np.stack([pad_row(r, max_len) for r in head_rows])
        avg_row = np.mean(padded, axis=0)
        avg_rows.append(avg_row)
        lengths.append(max_len)
    overall_max = max(lengths)
    avg_matrix = np.stack([pad_row(r, overall_max) for r in avg_rows])
    return avg_matrix

# Process each JSON file in the folder
json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

for filename in json_files:
    json_file_path = os.path.join(folder_path, filename)
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract decoder self-attentions and translations
    dec_attentions = data['decoder_attentions']
    reference_translation = data['reference_translations'][0]
    prediction = data['predictions'][0]
    
    num_steps = len(dec_attentions)
    num_layers = len(dec_attentions[0])
    # Assume batch_size is 1; determine number of heads from first step/layer
    num_heads = len(dec_attentions[0][0][0])
    
    base_filename = os.path.splitext(filename)[0]
    
    # 1) PER-LAYER VISUALIZATION (Auto-Cropped) for each decoder layer
    for layer_index in range(num_layers):
        # Create subplot grid (using 4 columns)
        n_cols = 4
        n_rows = int(np.ceil(num_heads / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        # Flatten axes for easy iteration
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        title = create_title_with_translation(
            filename,
            f"Decoder Self-Attention - Layer {layer_index+1} (Auto-Cropped)",
            reference_translation,
            prediction
        )
        fig.suptitle(title, fontsize=12, wrap=True)
        
        for head in range(num_heads):
            matrix = build_matrix_for_layer_head(dec_attentions, num_steps, layer_index, head)
            bbox = find_nonzero_bbox_2d(matrix, threshold=CROP_THRESHOLD)
            if bbox is None:
                cropped_matrix = np.zeros((1, 1))
            else:
                cropped_matrix = crop_2d(matrix, bbox)
            axes[head].imshow(cropped_matrix, cmap='viridis', aspect='auto')
            axes[head].set_title(f"Head {head+1}", fontsize=10)
            axes[head].axis('off')
        # Remove any unused subplots
        for ax in axes[num_heads:]:
            ax.remove()
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        out_filename = f"{base_filename}_decoder_layer{layer_index+1}_auto_crop.png"
        out_filepath = os.path.join(decoder_plot_folder, out_filename)
        plt.savefig(out_filepath, dpi=150)
        plt.close(fig)
        print(f"Saved per-layer decoder attention plot (auto-cropped): {out_filepath}")
    
    # 2) AGGREGATED/Averaged Visualization (Averaged Across Layers) for each head
    # Create a subplot grid for heads
    n_cols = 4
    n_rows = int(np.ceil(num_heads / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    title = create_title_with_translation(
        filename,
        "Aggregated Decoder Self-Attention (Averaged Across Layers, Auto-Cropped)",
        reference_translation,
        prediction
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    # For each head, build aggregated matrix (average over layers)
    agg_matrices = []
    for head in range(num_heads):
        agg_matrix = build_aggregated_matrix_across_layers(dec_attentions, num_steps, num_layers, head)
        agg_matrices.append(agg_matrix)
        bbox = find_nonzero_bbox_2d(agg_matrix, threshold=CROP_THRESHOLD)
        if bbox is None:
            cropped_matrix = np.zeros((1, 1))
        else:
            cropped_matrix = crop_2d(agg_matrix, bbox)
        axes[head].imshow(cropped_matrix, cmap='viridis', aspect='auto')
        axes[head].set_title(f"Head {head+1}", fontsize=10)
        axes[head].axis('off')
    for ax in axes[num_heads:]:
        ax.remove()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{base_filename}_decoder_avg_heads_auto_crop.png"
    out_filepath = os.path.join(decoder_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved aggregated decoder attention plot (averaged across layers, auto-cropped): {out_filepath}")
    
    # 2.5) HISTOGRAM OF COLUMN SUMS FOR AGGREGATED MATRICES
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    title = create_title_with_translation(
        filename,
        "Attention Distribution (Column Sums of Averaged Decoder Self-Attention)",
        reference_translation,
        prediction
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    for head in range(num_heads):
        matrix = agg_matrices[head]
        # Sum along the rows (over generation steps) to get attention distribution per key token
        col_sums = np.sum(matrix, axis=0)
        non_zero_indices = np.where(col_sums > CROP_THRESHOLD)[0]
        if len(non_zero_indices) > 0:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1] + 1
            cropped_col_sums = col_sums[start_idx:end_idx]
            axes[head].bar(range(len(cropped_col_sums)), cropped_col_sums)
            axes[head].set_xlabel(f"Tokens {start_idx}-{end_idx-1}", fontsize=8)
        else:
            axes[head].bar([0], [0])
            axes[head].set_title(f"Head {head+1} (No significant attention)", fontsize=10)
        axes[head].set_title(f"Head {head+1}", fontsize=10)
        axes[head].set_xticks([])
        if head % n_cols == 0:
            axes[head].set_ylabel("Attention Sum")
    for ax in axes[num_heads:]:
        ax.remove()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{base_filename}_decoder_avg_histograms.png"
    out_filepath = os.path.join(decoder_avg_hist_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved histogram of aggregated decoder attention column sums: {out_filepath}")
    
    # 3) AVERAGE PER LAYER ACROSS HEADS VISUALIZATION
    # Create subplot grid for layers (e.g. 4 columns)
    n_cols_layer = 4
    n_rows_layer = int(np.ceil(num_layers / n_cols_layer))
    fig, axes = plt.subplots(n_rows_layer, n_cols_layer, figsize=(12, 3*n_rows_layer))
    if n_rows_layer * n_cols_layer > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    title = create_title_with_translation(
        filename,
        "Layer-wise Decoder Self-Attention (Averaged Across Heads, Auto-Cropped)",
        reference_translation,
        prediction
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    layer_avg_matrices = []
    for layer in range(num_layers):
        avg_matrix = build_layer_avg_matrix_across_heads(dec_attentions, num_steps, layer, num_heads)
        layer_avg_matrices.append(avg_matrix)
        bbox = find_nonzero_bbox_2d(avg_matrix, threshold=CROP_THRESHOLD)
        if bbox is None:
            cropped_matrix = np.zeros((1, 1))
        else:
            cropped_matrix = crop_2d(avg_matrix, bbox)
        axes[layer].imshow(cropped_matrix, cmap='viridis', aspect='auto')
        axes[layer].set_title(f"Layer {layer+1}", fontsize=10)
        axes[layer].axis('off')
    for ax in axes[num_layers:]:
        ax.remove()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{base_filename}_decoder_avg_layers_auto_crop.png"
    out_filepath = os.path.join(decoder_layer_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved layer-averaged decoder attention plot (auto-cropped): {out_filepath}")
    
    # 3.5) HISTOGRAM OF COLUMN SUMS FOR LAYER-AVERAGED MATRICES
    fig, axes = plt.subplots(n_rows_layer, n_cols_layer, figsize=(12, 3*n_rows_layer))
    if n_rows_layer * n_cols_layer > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    title = create_title_with_translation(
        filename,
        "Layer-wise Attention Distribution (Column Sums of Layer-Averaged Decoder Self-Attention)",
        reference_translation,
        prediction
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    for layer in range(num_layers):
        matrix = layer_avg_matrices[layer]
        col_sums = np.sum(matrix, axis=0)
        non_zero_indices = np.where(col_sums > CROP_THRESHOLD)[0]
        if len(non_zero_indices) > 0:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1] + 1
            cropped_col_sums = col_sums[start_idx:end_idx]
            axes[layer].bar(range(len(cropped_col_sums)), cropped_col_sums)
            axes[layer].set_xlabel(f"Tokens {start_idx}-{end_idx-1}", fontsize=8)
        else:
            axes[layer].bar([0], [0])
            axes[layer].set_title(f"Layer {layer+1} (No significant attention)", fontsize=10)
        axes[layer].set_title(f"Layer {layer+1}", fontsize=10)
        axes[layer].set_xticks([])
        if layer % n_cols_layer == 0:
            axes[layer].set_ylabel("Attention Sum")
    for ax in axes[num_layers:]:
        ax.remove()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{base_filename}_decoder_layer_avg_histograms.png"
    out_filepath = os.path.join(decoder_layer_avg_hist_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved histogram of layer-averaged decoder attention column sums: {out_filepath}")