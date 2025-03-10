import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Set working directory if needed
os.chdir('/auto/brno2/home/javorek/T5_for_SLT/')

# Folder containing the JSON files
folder_path = os.path.join('results', 'attention_batches_simple')

# Create output folders for cross-attention plots
cross_plot_folder = os.path.join(folder_path, "cross_plots")
os.makedirs(cross_plot_folder, exist_ok=True)

cross_avg_plot_folder = os.path.join(folder_path, "cross_avg_plots")
os.makedirs(cross_avg_plot_folder, exist_ok=True)

# Get a sorted list of JSON files in the folder
json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

# Process each JSON file
for filename in json_files:
    json_file_path = os.path.join(folder_path, filename)
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Retrieve cross-attentions.
    # Expected shape: (num_steps, num_layers, batch_size, num_heads, 1, seq_len)
    cross_attentions = data['cross_attentions']
    cross_attentions = np.array(cross_attentions)  # for easier slicing
    num_steps, num_layers, batch_size, num_heads, _, seq_len = cross_attentions.shape
    batch_index = 0  # given batch size is 1

    # ================================================
    # 1. Per-Layer Visualization for Cross-Attention
    # ================================================
    # For each decoder layer, plot a 3×4 grid (assuming 12 heads)
    for layer_index in range(num_layers):
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        axes = axes.flatten()

        # Main title: file name and layer information
        fig.suptitle(f"File: {filename}\nCross-Attention - Layer {layer_index+1} (All Steps for Each Head)", fontsize=14)

        for head in range(num_heads):
            # For a given layer and head, gather the matrix across all decoding steps.
            # Each slice is of shape (num_steps, seq_len)
            cross_matrix = cross_attentions[:, layer_index, batch_index, head, 0, :]
            axes[head].imshow(cross_matrix, cmap='viridis', aspect='auto')
            axes[head].set_title(f"Head {head+1}", fontsize=10)
            axes[head].axis('off')

        # Remove any extra axes if there are fewer than 12 heads
        for ax in axes[num_heads:]:
            ax.remove()

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        out_filename = f"{os.path.splitext(filename)[0]}_cross_layer{layer_index+1}.png"
        out_filepath = os.path.join(cross_plot_folder, out_filename)
        plt.savefig(out_filepath, dpi=150)
        plt.close(fig)
        print(f"Saved cross attention plot: {out_filepath}")

    # =====================================================
    # 2. Aggregated/Averaged Cross-Attention Across Layers
    # =====================================================
    # For each head, average the cross-attention matrices over all layers.
    # Each matrix is (num_steps, seq_len).
    avg_matrices = []
    for head in range(num_heads):
        matrices = []
        for layer_index in range(num_layers):
            matrix = cross_attentions[:, layer_index, batch_index, head, 0, :]
            matrices.append(matrix)
        # Average over layers (axis=0) results in shape (num_steps, seq_len)
        avg_matrix = np.mean(matrices, axis=0)
        avg_matrices.append(avg_matrix)
    
    # Plot the averaged matrices in a 3×4 grid
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.flatten()

    fig.suptitle(f"File: {filename}\nAggregated Cross-Attention (Averaged Across Layers), Batch 1", fontsize=14)

    for head in range(num_heads):
        axes[head].imshow(avg_matrices[head], cmap='viridis', aspect='auto')
        axes[head].set_title(f"Head {head+1}", fontsize=10)
        axes[head].axis('off')
    
    for ax in axes[num_heads:]:
        ax.remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{os.path.splitext(filename)[0]}_cross_avg_heads.png"
    out_filepath = os.path.join(cross_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved aggregated cross attention plot: {out_filepath}")