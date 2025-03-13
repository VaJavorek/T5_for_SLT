import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Change directory if needed
os.chdir('/auto/brno2/home/javorek/T5_for_SLT/')

folder_path = os.path.join('results', 'attention_batches_simple_labeled_tokens')
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

def create_title_with_translation(filename, layer_info, translation, prediction, max_chars=100):
    """
    Creates a title with both the reference translation and prediction text, wrapping if too long.
    """
    # Truncate and add ellipsis if texts are too long
    if len(translation) > max_chars:
        translation = translation[:max_chars] + "..."
    if len(prediction) > max_chars:
        prediction = prediction[:max_chars] + "..."
    
    # Create title with both translation and prediction
    title = f"File: {filename}\n{layer_info}\nReference: {translation}\nPrediction: {prediction}"
    return title

def create_labeled_plot(matrix, ax, frame_indices, tokens, title, show_y_labels=False):
    """
    Create a labeled attention plot with frame numbers on x-axis and tokens on y-axis.
    """
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    
    # Add token labels on y-axis only if requested
    ax.set_yticks(range(len(tokens)))
    if show_y_labels:
        ax.set_yticklabels(tokens, fontsize=8)
    else:
        ax.set_yticklabels([])
    
    # Add frame number labels on x-axis
    step = max(len(frame_indices) // 10, 1)  # Show ~10 labels
    selected_frames = frame_indices[::step]
    selected_positions = range(0, len(frame_indices), step)
    ax.set_xticks(selected_positions)
    ax.set_xticklabels(selected_frames, rotation=45, fontsize=8)
    
    # Set title
    ax.set_title(title, fontsize=10)
    return im

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
    prediction = data['predictions'][0]
    
    # Get the decoded tokens (skip special tokens)
    decoded_tokens = data['decoded_tokens']
    
    # Create frame indices for x-axis
    frame_indices = list(range(cross_attentions.shape[-1]))
    
    num_steps, num_layers, batch_size, num_heads, _, seq_len = cross_attentions.shape
    batch_index = 0  # always 0 if batch size is 1

    # 1) PER-LAYER VISUALIZATION (Auto-cropped)
    for layer_index in range(num_layers):
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))  # Wider figure for labels
        plt.subplots_adjust(left=0.2)  # Add more space on the left for labels
        axes = axes.flatten()

        title = create_title_with_translation(
            filename,
            f"Cross-Attention - Layer {layer_index+1} (Auto-Cropped)",
            reference_translation,
            prediction
        )
        fig.suptitle(title, fontsize=12, wrap=True)

        for head in range(num_heads):
            # cross_matrix has shape (num_steps, seq_len)
            cross_matrix = cross_attentions[:, layer_index, batch_index, head, 0, :]
            
            # Find bounding box of non-zero values
            bbox = find_nonzero_bbox_2d(cross_matrix, threshold=CROP_THRESHOLD)
            if bbox is None:
                cropped_matrix = np.zeros((1, 1))
                cropped_tokens = ['<empty>']
                cropped_frames = [0]
            else:
                cropped_matrix = crop_2d(cross_matrix, bbox)
                # Crop tokens and frames to match the matrix
                cropped_tokens = decoded_tokens[bbox[0]:bbox[1]+1]
                cropped_frames = frame_indices[bbox[2]:bbox[3]+1]

            # Only show y-labels for the leftmost plots in each row
            show_labels = (head % 4 == 0)
            
            create_labeled_plot(
                cropped_matrix, 
                axes[head],
                cropped_frames,
                cropped_tokens,
                f"Head {head+1}",
                show_y_labels=show_labels
            )

        # Remove unused subplots if fewer than 12 heads
        for ax in axes[num_heads:]:
            ax.remove()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Increased top margin for title

        out_filename = f"{os.path.splitext(filename)[0]}_cross_layer{layer_index+1}_auto_crop.png"
        out_filepath = os.path.join(cross_plot_folder, out_filename)
        plt.savefig(out_filepath, dpi=150, bbox_inches='tight')
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
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))  # Increased size for labels
    axes = axes.flatten()
    
    title = create_title_with_translation(
        filename,
        "Aggregated Cross-Attention (Averaged Across Layers, Auto-Cropped)",
        reference_translation,
        prediction
    )
    fig.suptitle(title, fontsize=12, wrap=True)

    for head in range(num_heads):
        # Auto-crop each head's averaged matrix
        matrix = avg_matrices[head]
        bbox = find_nonzero_bbox_2d(matrix, threshold=CROP_THRESHOLD)
        if bbox is None:
            cropped_matrix = np.zeros((1, 1))
            cropped_tokens = ['<empty>']
            cropped_frames = [0]
        else:
            cropped_matrix = crop_2d(matrix, bbox)
            cropped_tokens = decoded_tokens[bbox[0]:bbox[1]+1]
            cropped_frames = frame_indices[bbox[2]:bbox[3]+1]

        create_labeled_plot(
            cropped_matrix,
            axes[head],
            cropped_frames,
            cropped_tokens,
            f"Head {head+1}"
        )

    for ax in axes[num_heads:]:
        ax.remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Increased top margin for title
    out_filename = f"{os.path.splitext(filename)[0]}_cross_avg_heads_auto_crop.png"
    out_filepath = os.path.join(cross_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150, bbox_inches='tight')
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
        reference_translation,
        prediction
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
            cropped_frames = frame_indices[start_idx:end_idx]
            
            # Plot as a bar chart/histogram
            axes[head].bar(range(len(cropped_col_sums)), cropped_col_sums)
            axes[head].set_title(f"Head {head+1}", fontsize=10)
            
            # Add x-axis label showing frame numbers
            # Show every nth frame to avoid overcrowding
            step = max(len(cropped_frames) // 5, 1)  # Show ~5 labels
            selected_frames = cropped_frames[::step]
            selected_positions = range(0, len(cropped_col_sums), step)
            axes[head].set_xticks(selected_positions)
            axes[head].set_xticklabels(selected_frames, rotation=45, fontsize=8)
            
            # Add x-axis label showing the range
            axes[head].set_xlabel(f"Frames {start_idx}-{end_idx-1}", fontsize=8)
        else:
            # If no significant values, plot empty
            axes[head].bar([0], [0])
            axes[head].set_title(f"Head {head+1} (No significant attention)", fontsize=10)
        
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
    plt.savefig(out_filepath, dpi=150, bbox_inches='tight')
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
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3*n_rows))  # Adjust height based on number of rows
    axes = axes.flatten()
    
    title = create_title_with_translation(
        filename,
        "Layer-wise Cross-Attention (Averaged Across Heads, Auto-Cropped)",
        reference_translation,
        prediction
    )
    fig.suptitle(title, fontsize=12, wrap=True)
    
    for layer_index in range(num_layers):
        # Auto-crop each layer's averaged matrix
        matrix = layer_avg_matrices[layer_index]
        bbox = find_nonzero_bbox_2d(matrix, threshold=CROP_THRESHOLD)
        if bbox is None:
            cropped_matrix = np.zeros((1, 1))
            cropped_tokens = ['<empty>']
            cropped_frames = [0]
        else:
            cropped_matrix = crop_2d(matrix, bbox)
            cropped_tokens = decoded_tokens[bbox[0]:bbox[1]+1]
            cropped_frames = frame_indices[bbox[2]:bbox[3]+1]
            
        create_labeled_plot(
            cropped_matrix,
            axes[layer_index],
            cropped_frames,
            cropped_tokens,
            f"Layer {layer_index+1}"
        )
    
    # Remove unused subplots
    for ax in axes[num_layers:]:
        ax.remove()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust top margin for title
    out_filename = f"{os.path.splitext(filename)[0]}_cross_avg_layers_auto_crop.png"
    out_filepath = os.path.join(cross_layer_avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150, bbox_inches='tight')
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
        reference_translation,
        prediction
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
            cropped_frames = frame_indices[start_idx:end_idx]
            
            # Plot as a bar chart/histogram
            axes[layer_index].bar(range(len(cropped_col_sums)), cropped_col_sums)
            axes[layer_index].set_title(f"Layer {layer_index+1}", fontsize=10)
            
            # Add x-axis label showing frame numbers
            # Show every nth frame to avoid overcrowding
            step = max(len(cropped_frames) // 5, 1)  # Show ~5 labels
            selected_frames = cropped_frames[::step]
            selected_positions = range(0, len(cropped_col_sums), step)
            axes[layer_index].set_xticks(selected_positions)
            axes[layer_index].set_xticklabels(selected_frames, rotation=45, fontsize=8)
            
            # Add x-axis label showing the range
            axes[layer_index].set_xlabel(f"Frames {start_idx}-{end_idx-1}", fontsize=8)
        else:
            # If no significant values, plot empty
            axes[layer_index].bar([0], [0])
            axes[layer_index].set_title(f"Layer {layer_index+1} (No significant attention)", fontsize=10)
        
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
    plt.savefig(out_filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram of layer-averaged column sums: {out_filepath}")
    
    # 4) GLOBAL AVERAGE HISTOGRAM (AVERAGED ACROSS ALL HEADS AND LAYERS)
    # Create a folder for these global average histogram plots
    cross_global_avg_hist_folder = os.path.join(folder_path, "cross_global_avg_histograms")
    os.makedirs(cross_global_avg_hist_folder, exist_ok=True)
    
    # Compute the global average directly from the original attention matrices
    # This is mathematically equivalent to both averaging methods (heads-first or layers-first)
    global_avg_matrix = np.mean(cross_attentions, axis=(1, 3))  # Average across layers and heads
    global_avg_matrix = global_avg_matrix.squeeze()  # Remove singleton dimensions
    
    # Sum along the rows to get column sums (attention distribution across input sequence)
    global_col_sums = np.sum(global_avg_matrix, axis=0)
    
    # Find non-zero regions to crop the histogram
    non_zero_indices = np.where(global_col_sums > CROP_THRESHOLD)[0]
    if len(non_zero_indices) > 0:
        start_idx = max(0, non_zero_indices[0])
        end_idx = min(len(global_col_sums), non_zero_indices[-1] + 1)
        
        # Crop the column sums
        cropped_col_sums = global_col_sums[start_idx:end_idx]
        cropped_frames = frame_indices[start_idx:end_idx]
        
        # Dynamically adjust figure width based on number of frames
        # Base width of 12 inches for up to 100 frames, then scale linearly
        num_frames = len(cropped_frames)
        fig_width = max(12, 12 * (num_frames / 100))
        
        # Create a new figure for the global histogram with dynamic width
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        title = create_title_with_translation(
            filename,
            "Global Average Attention Distribution (Across All Heads and Layers)",
            reference_translation,
            prediction
        )
        fig.suptitle(title, fontsize=12, wrap=True)
        
        # Plot as a bar chart/histogram
        ax.bar(range(len(cropped_col_sums)), cropped_col_sums)
        
        # Add x-axis label showing frame numbers
        # For large numbers of frames, use a different approach
        if num_frames > 100:
            # Show every second frame to avoid overcrowding
            step = 2
            selected_frames = cropped_frames[::step]
            selected_positions = range(0, len(cropped_col_sums), step)
            ax.set_xticks(selected_positions)
            ax.set_xticklabels(selected_frames, rotation=90, fontsize=8)
            
            # Add minor ticks for all frames (without labels)
            ax.set_xticks(range(len(cropped_col_sums)), minor=True)
        else:
            # For fewer frames, show all labels
            ax.set_xticks(range(len(cropped_col_sums)))
            ax.set_xticklabels(cropped_frames, rotation=90, fontsize=8)
        
        ax.set_xlabel("Frame Number", fontsize=10)
        ax.set_ylabel("Average Attention", fontsize=10)
        
        # Add grid lines to help track from bars to axis labels
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)  # Adjust margins for labels
        
        out_filename = f"{os.path.splitext(filename)[0]}_cross_global_avg_histogram.png"
        out_filepath = os.path.join(cross_global_avg_hist_folder, out_filename)
        plt.savefig(out_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved global average histogram: {out_filepath}")
    else:
        print(f"No significant attention values found for global average histogram in {filename}")