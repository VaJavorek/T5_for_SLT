import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Change directory if needed
os.chdir('/auto/brno2/home/javorek/T5_for_SLT/')

folder_path = os.path.join('results', 'attention_batches_simple_labeled')
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

def create_title_with_translation(filename, layer_info, translation, max_chars=50):
    """
    Creates a title with the translation text, wrapping if too long.
    """
    # Truncate and add ellipsis if translation is too long
    if len(translation) > max_chars:
        translation = translation[:max_chars] + "..."
    
    # Split title into two lines
    title = f"File: {filename}\n{layer_info}\nTranslation: {translation}"
    return title

json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

for filename in json_files:
    json_file_path = os.path.join(folder_path, filename)
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # cross_attentions shape: (num_steps, num_layers, batch_size, num_heads, 1, seq_len)
    cross_attentions = data['cross_attentions']
    cross_attentions = np.array(cross_attentions)  # Convert to np array for easier slicing
    
    # Get the reference translation (assuming one translation per file)
    reference_translation = data['reference_translations'][0]
    
    num_steps, num_layers, batch_size, num_heads, _, seq_len = cross_attentions.shape
    batch_index = 0  # always 0 if batch size is 1

    # 1) PER-LAYER VISUALIZATION (Auto-cropped)
    for layer_index in range(num_layers):
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # Increased height for title
        axes = axes.flatten()

        title = create_title_with_translation(
            filename,
            f"Cross-Attention - Layer {layer_index+1} (Auto-Cropped)",
            reference_translation
        )
        fig.suptitle(title, fontsize=12, wrap=True)

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
        plt.subplots_adjust(top=0.85)  # Increased top margin for title

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
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # Increased height for title
    axes = axes.flatten()
    
    title = create_title_with_translation(
        filename,
        "Aggregated Cross-Attention (Averaged Across Layers, Auto-Cropped)",
        reference_translation
    )
    fig.suptitle(title, fontsize=12, wrap=True)

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
    plt.subplots_adjust(top=0.85)  # Increased top margin for title
    out_filename = f"{os.path.splitext(filename)[0]}_cross_avg_heads_auto_crop.png"
    out_filepath = os.path.join(cross_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved aggregated cross attention plot (auto-cropped): {out_filepath}")
    
    # 2.5) HISTOGRAM OF COLUMN SUMS FOR AVERAGED MATRICES
    # Create a folder for these histogram plots
    cross_avg_hist_folder = os.path.join(folder_path, "cross_avg_histograms")
    os.makedirs(cross_avg_hist_folder, exist_ok=True)
    
    # Create a new figure for the histograms
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    title = create_title_with_translation(
        filename,
        "Attention Distribution (Column Sums of Averaged Cross-Attention)",
        reference_translation
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    for head in range(num_heads):
        # Get the averaged matrix for this head
        matrix = avg_matrices[head]
        
        # Sum along the rows to get column sums (attention distribution across input sequence)
        col_sums = np.sum(matrix, axis=0)
        
        # Find non-zero regions to crop the histogram
        non_zero_indices = np.where(col_sums > CROP_THRESHOLD)[0]
        if len(non_zero_indices) > 0:
            start_idx = max(0, non_zero_indices[0])
            end_idx = min(len(col_sums), non_zero_indices[-1] + 1)
            
            # Crop the column sums
            cropped_col_sums = col_sums[start_idx:end_idx]
            
            # Plot as a bar chart/histogram
            axes[head].bar(range(len(cropped_col_sums)), cropped_col_sums)
            axes[head].set_title(f"Head {head+1}", fontsize=10)
            
            # Add x-axis label showing the range
            axes[head].set_xlabel(f"Tokens {start_idx}-{end_idx-1}", fontsize=8)
        else:
            # If no significant values, plot empty
            axes[head].bar([0], [0])
            axes[head].set_title(f"Head {head+1} (No significant attention)", fontsize=10)
        
        # Remove x-axis ticks to reduce clutter
        axes[head].set_xticks([])
        
        # Add y-axis label only for leftmost plots
        if head % 4 == 0:
            axes[head].set_ylabel("Attention Sum")
    
    # Remove unused subplots
    for ax in axes[num_heads:]:
        ax.remove()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{os.path.splitext(filename)[0]}_cross_avg_histograms.png"
    out_filepath = os.path.join(cross_avg_hist_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved histogram of column sums: {out_filepath}")

    # 3) AVERAGE PER LAYER ACROSS HEADS VISUALIZATION (Auto-cropped)
    # Create a folder for these plots
    cross_layer_avg_plot_folder = os.path.join(folder_path, "cross_layer_avg_plots_auto_crop")
    os.makedirs(cross_layer_avg_plot_folder, exist_ok=True)
    
    # Average across heads for each layer => shape (num_layers, num_steps, seq_len)
    layer_avg_matrices = []
    for layer_index in range(num_layers):
        # Collect all heads for this layer
        matrices = []
        for head in range(num_heads):
            mat = cross_attentions[:, layer_index, batch_index, head, 0, :]
            matrices.append(mat)
        # Average over heads
        layer_avg_matrix = np.mean(matrices, axis=0)  # shape (num_steps, seq_len)
        layer_avg_matrices.append(layer_avg_matrix)
    
    # Calculate how many rows and columns we need for the subplot grid
    n_rows = (num_layers + 3) // 4  # Ceiling division to ensure enough space
    
    # Now plot the layer-averaged matrices
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3*n_rows))  # Adjust height based on number of rows
    axes = axes.flatten()
    
    title = create_title_with_translation(
        filename,
        "Layer-wise Cross-Attention (Averaged Across Heads, Auto-Cropped)",
        reference_translation
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    for layer_index in range(num_layers):
        # Auto-crop each layer's averaged matrix
        matrix = layer_avg_matrices[layer_index]
        bbox = find_nonzero_bbox_2d(matrix, threshold=CROP_THRESHOLD)
        if bbox is None:
            cropped_matrix = np.zeros((1, 1))
        else:
            cropped_matrix = crop_2d(matrix, bbox)
            
        axes[layer_index].imshow(cropped_matrix, cmap='viridis', aspect='auto')
        axes[layer_index].set_title(f"Layer {layer_index+1}", fontsize=10)
        axes[layer_index].axis('off')
    
    # Remove unused subplots
    for ax in axes[num_layers:]:
        ax.remove()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust top margin for title
    out_filename = f"{os.path.splitext(filename)[0]}_cross_avg_layers_auto_crop.png"
    out_filepath = os.path.join(cross_layer_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved layer-averaged cross attention plot (auto-cropped): {out_filepath}")
    
    # 3.5) HISTOGRAM OF COLUMN SUMS FOR LAYER-AVERAGED MATRICES
    # Create a folder for these histogram plots
    cross_layer_avg_hist_folder = os.path.join(folder_path, "cross_layer_avg_histograms")
    os.makedirs(cross_layer_avg_hist_folder, exist_ok=True)
    
    # Create a new figure for the histograms
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3*n_rows))
    axes = axes.flatten()
    
    title = create_title_with_translation(
        filename,
        "Layer-wise Attention Distribution (Column Sums of Layer-Averaged Cross-Attention)",
        reference_translation
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    for layer_index in range(num_layers):
        # Get the layer-averaged matrix
        matrix = layer_avg_matrices[layer_index]
        
        # Sum along the rows to get column sums (attention distribution across input sequence)
        col_sums = np.sum(matrix, axis=0)
        
        # Find non-zero regions to crop the histogram
        non_zero_indices = np.where(col_sums > CROP_THRESHOLD)[0]
        if len(non_zero_indices) > 0:
            start_idx = max(0, non_zero_indices[0])
            end_idx = min(len(col_sums), non_zero_indices[-1] + 1)
            
            # Crop the column sums
            cropped_col_sums = col_sums[start_idx:end_idx]
            
            # Plot as a bar chart/histogram
            axes[layer_index].bar(range(len(cropped_col_sums)), cropped_col_sums)
            axes[layer_index].set_title(f"Layer {layer_index+1}", fontsize=10)
            
            # Add x-axis label showing the range
            axes[layer_index].set_xlabel(f"Tokens {start_idx}-{end_idx-1}", fontsize=8)
        else:
            # If no significant values, plot empty
            axes[layer_index].bar([0], [0])
            axes[layer_index].set_title(f"Layer {layer_index+1} (No significant attention)", fontsize=10)
        
        # Remove x-axis ticks to reduce clutter
        axes[layer_index].set_xticks([])
        
        # Add y-axis label only for leftmost plots
        if layer_index % 4 == 0:
            axes[layer_index].set_ylabel("Attention Sum")
    
    # Remove unused subplots
    for ax in axes[num_layers:]:
        ax.remove()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_filename = f"{os.path.splitext(filename)[0]}_cross_layer_avg_histograms.png"
    out_filepath = os.path.join(cross_layer_avg_hist_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f"Saved histogram of layer-averaged column sums: {out_filepath}")