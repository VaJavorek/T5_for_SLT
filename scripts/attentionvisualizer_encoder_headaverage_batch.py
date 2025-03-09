import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Path to the folder with JSON files
folder_path = '/auto/brno2/home/javorek/T5_for_SLT/results/attention_batches_simple/'

# Create a subfolder for saving average plots
avg_plot_folder = os.path.join(folder_path, "average_plots_auto_crop_square")
os.makedirs(avg_plot_folder, exist_ok=True)

# Threshold for deciding if a row/column is "non-zero" (helps with floating precision)
NONZERO_THRESHOLD = 1e-9

def find_global_bounding_box(matrices, threshold=NONZERO_THRESHOLD):
    """
    Given a list of 2D NumPy arrays (all from the same head across different layers),
    find the minimal bounding box that contains all values above 'threshold'.
    
    Returns (min_row, max_row, min_col, max_col).
    If all values are effectively zero, returns None.
    """
    nonempty_rows_all = []
    nonempty_cols_all = []

    # Collect non-empty row/col indices across all matrices
    for mat in matrices:
        row_sums = np.sum(mat, axis=1)
        col_sums = np.sum(mat, axis=0)

        # Indices where the sum of a row/column is above threshold
        rows = np.where(row_sums > threshold)[0]
        cols = np.where(col_sums > threshold)[0]

        if len(rows) > 0:
            nonempty_rows_all.extend(rows)
        if len(cols) > 0:
            nonempty_cols_all.extend(cols)

    # If we never found any non-zero rows/cols, return None
    if not nonempty_rows_all or not nonempty_cols_all:
        return None

    return (min(nonempty_rows_all), max(nonempty_rows_all),
            min(nonempty_cols_all), max(nonempty_cols_all))

def enforce_square_bbox(bbox):
    """
    Given (min_r, max_r, min_c, max_c), produce a square bounding box
    by truncating the larger dimension (height or width) so the result
    remains anchored at (min_r, min_c) and is square.
    """
    (min_r, max_r, min_c, max_c) = bbox
    height = max_r - min_r + 1
    width = max_c - min_c + 1

    # Determine the side of the square (the smaller dimension)
    side = min(height, width)

    # We keep the top-left corner the same (min_r, min_c),
    # and truncate the larger dimension so the region is square.
    max_r = min_r + side - 1
    max_c = min_c + side - 1

    return (min_r, max_r, min_c, max_c)

def crop_to_bounding_box(matrix, bbox):
    """
    Crop a 2D NumPy array 'matrix' to the bounding box 'bbox' = (min_row, max_row, min_col, max_col).
    """
    min_r, max_r, min_c, max_c = bbox
    return matrix[min_r:max_r+1, min_c:max_c+1]

# Gather JSON files
json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

for filename in json_files:
    json_file_path = os.path.join(folder_path, filename)

    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Retrieve encoder attentions
    encoder_attentions = data['encoder_attentions']
    num_layers = len(encoder_attentions)

    # We know batch size is 1, so batch_index = 0
    batch_index = 0

    # Number of heads (assume consistent across layers)
    num_heads = len(encoder_attentions[0][batch_index])

    # Prepare a list to hold the averaged matrices for each head
    avg_matrices = []

    # -- For each head, find a global bounding box, enforce square, then average --
    for head_index in range(num_heads):
        # Collect this head's matrices across all layers
        layer_mats = []
        for layer_index in range(num_layers):
            mat = np.array(encoder_attentions[layer_index][batch_index][head_index])
            layer_mats.append(mat)

        # Find one bounding box that covers all non-zero values across layers
        bbox = find_global_bounding_box(layer_mats, threshold=NONZERO_THRESHOLD)

        if bbox is None:
            # Everything is zero for this head across layers; create a 1×1 zero matrix
            avg_matrix = np.zeros((1, 1))
        else:
            # Enforce square shape
            bbox = enforce_square_bbox(bbox)

            # Sum up all cropped matrices, then divide by num_layers
            min_r, max_r, min_c, max_c = bbox
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            sum_matrix = np.zeros((height, width))

            for mat in layer_mats:
                cropped = crop_to_bounding_box(mat, bbox)
                sum_matrix += cropped
            avg_matrix = sum_matrix / num_layers

        avg_matrices.append(avg_matrix)

    # -- Plot the averaged heads in a 3×4 grid (square-cropped) --
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    # Main title
    fig.suptitle(
        f"File: {filename}\nAveraged Attention (Auto-Cropped Square), Batch 1",
        fontsize=14
    )

    # Plot each head's averaged matrix
    for head in range(num_heads):
        if head >= rows * cols:
            # If more than 12 heads, stop or adjust the grid
            break
        axes[head].imshow(avg_matrices[head], cmap='viridis')
        axes[head].set_title(f"Head {head+1}", fontsize=10)
        axes[head].axis('off')

    # Remove unused subplots if fewer than rows*cols heads
    for ax in axes[num_heads:]:
        ax.remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)

    # Save the figure
    out_filename = f"{os.path.splitext(filename)[0]}_avg_heads_auto_crop_square.png"
    out_filepath = os.path.join(avg_plot_folder, out_filename)
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)

    print(f"Saved average auto-cropped square plot: {out_filepath}")