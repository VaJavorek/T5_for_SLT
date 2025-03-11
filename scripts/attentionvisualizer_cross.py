import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Change directory if needed
os.chdir('/auto/brno2/home/javorek/T5_for_SLT/')

folder_path = os.path.join('results', 'attention_batches_simple')
cross_plot_folder = os.path.join(folder_path, "cross_plots_auto_crop")
os.makedirs(cross_plot_folder, exist_ok=True)

cross_avg_plot_folder = os.path.join(folder_path, "cross_avg_plots_auto_crop")
os.makedirs(cross_avg_plot_folder, exist_ok=True)

# Threshold for deciding if a row/column is non-zero
CROP_THRESHOLD = 1e-9

def find_nonzero_bbox_2d(matrix, threshold=CROP_THRESHOLD):
    """
    Given a 2D NumPy array 'matrix', find the minimal bounding box
    containing all values above 'threshold'.
    
    Returns (min_row, max_row, min_col, max_col), or None if all values
    are effectively zero.
    """
    # Sum rows and columns
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    # Identify non-empty rows and columns
    nonempty_rows = np.where(row_sums > threshold)[0]
    nonempty_cols = np.where(col_sums > threshold)[0]

    if len(nonempty_rows) == 0 or len(nonempty_cols) == 0:
        return None  # Entire matrix is effectively zero

    return (nonempty_rows[0], nonempty_rows[-1],
            nonempty_cols[0], nonempty_cols[-1])

def crop_2d(matrix, bbox):
    """
    Crop a 2D NumPy array 'matrix' to the bounding box 'bbox' = (min_r, max_r, min_c, max_c).
    """
    min_r, max_r, min_c, max_c = bbox
    return matrix[min_r:max_r+1, min_c:max_c+1]

json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

for filename in json_files:
    json_file_path = os.path.join(folder_path, filename)
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # cross_attentions shape: (num_steps, num_layers, batch_size, num_heads, 1, seq_len)
    cross_attentions = data['cross_attentions']
    cross_attentions = np.array(cross_attentions)  # Convert to np array for easier slicing
    
    num_steps, num_layers, batch_size, num_heads, _, seq_len = cross_attentions.shape
    batch_index = 0  # always 0 if batch size is 1

    # 1) PER-LAYER VISUALIZATION (Auto-cropped)
    for layer_index in range(num_layers):
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        axes = axes.flatten()

        fig.suptitle(
            f"File: {filename}\nCross-Attention - Layer {layer_index+1} (Auto-Cropped)",
            fontsize=14
        )

        for head in range(num_heads):
            # cross_matrix has shape (num_steps, seq_len)
            cross_matrix = cross_attentions[:, layer_index, batch_index, head, 0, :]
            
            # Find bounding box of non-zero values
            bbox = find_nonzero_bbox_2d(cross_matrix, threshold=CROP_THRESHOLD)
            if bbox is None:
                # Entire matrix is zero; use a 1x1 zero matrix
                cropped_matrix = np.zeros((1, 1))
            else:
                cropped_matrix = crop_2d(cross_matrix, bbox)

            axes[head].imshow(cropped_matrix, cmap='viridis', aspect='auto')
            axes[head].set_title(f"Head {head+1}", fontsize=10)
            axes[head].axis('off')

        # Remove unused subplots if fewer than 12 heads
        for ax in axes[num_heads:]:
            ax.remove()

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        out_filename = f"{os.path.splitext(filename)[0]}_cross_layer{layer_index+1}_auto_crop.png"
        out_filepath = os.path.join(cross_plot_folder, out_filename)
        plt.savefig(out_filepath, dpi=150)
        plt.close(fig)
        print(f"Saved per-layer cross attention plot (auto-cropped): {out_filepath}")

    # 2) AGGREGATED/AVERAGED VISUALIZATION (Auto-cropped)
    # Average each head's matrix across layers => shape (num_steps, seq_len)
    avg_matrices = []
    for head in range(num_heads):
        # Collect all layers for this head
        matrices = []
        for layer_index in range(num_layers):
            mat = cross_attentions[:, layer_index, batch_index, head, 0, :]
            matrices.append(mat)
        # Average over layers
        avg_matrix = np.mean(matrices, axis=0)  # shape (num_steps, seq_len)
        avg_matrices.append(avg_matrix)

    # Now plot the averaged matrices
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.flatten()
    fig.suptitle(
        f"File: {filename}\nAggregated Cross-Attention (Averaged Across Layers, Auto-Cropped)",
        fontsize=14
    )

    for head in range(num_heads):
        # Auto-crop each head's averaged matrix
        matrix = avg_matrices[head]
        bbox = find_nonzero_bbox_2d(matrix, threshold=CROP_THRESHOLD)
        if bbox is None:
            cropped_matrix = np.zeros((1, 1))
        else:
            cropped_matrix = crop_2d(matrix, bbox)

        axes[head].imshow(cropped_matrix, cmap='viridis', aspect='auto')
        axes[head].set_title(f"Head {head+1}", fontsize=10)
        axes[head].axis('off')

    for ax in axes[num_heads:]:
        ax.remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{os.path.splitext(filename)[0]}_cross_avg_heads_auto_crop.png"
    out_filepath = os.path.join(cross_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved aggregated cross attention plot (auto-cropped): {out_filepath}")