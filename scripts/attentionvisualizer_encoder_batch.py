import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the folder with JSON files
folder_path = '/auto/brno2/home/javorek/T5_for_SLT/results/attention_batches_simple/'

# Toggle to show only the top-left 100×100 region of each matrix
crop_attention = True
crop_size = 100

# Create a subfolder for saving plots
plot_folder = os.path.join(folder_path, "plots")
os.makedirs(plot_folder, exist_ok=True)

# Grid dimensions: 3 rows, 4 columns for 12 heads
rows, cols = 3, 4

# Get a sorted list of JSON files and process only the first 10
json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])[:10]

for filename in json_files:
    json_file_path = os.path.join(folder_path, filename)

    # Load the JSON file containing the saved attention data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Retrieve encoder attentions
    encoder_attentions = data['encoder_attentions']
    num_layers = len(encoder_attentions)
    
    # Batch size is 1, so always use batch_index 0
    batch_index = 0

    # Loop through each encoder layer in this file
    for layer_index in range(num_layers):
        # Create a figure with a 3×4 grid of subplots
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        axes = axes.flatten()

        # Main title indicating the file and layer being visualized
        fig.suptitle(
            f"File: {filename}\nEncoder Layer {layer_index+1} - Batch Item {batch_index+1}",
            fontsize=14
        )

        # Number of heads for this layer
        num_heads = len(encoder_attentions[layer_index][batch_index])

        # Plot each head
        for head in range(num_heads):
            if head >= rows * cols:
                # Only plot up to 12 heads if there are more
                break

            # Convert the attention matrix to a NumPy array
            attn_matrix = np.array(encoder_attentions[layer_index][batch_index][head])
            
            # Optionally crop the attention matrix to 100×100
            if crop_attention:
                attn_matrix = attn_matrix[:crop_size, :crop_size]

            # Plot the matrix
            axes[head].imshow(attn_matrix, cmap='viridis')
            axes[head].set_title(f"Head {head+1}", fontsize=10)
            axes[head].axis('off')

        # Remove any unused subplots if there are fewer than rows*cols heads
        for ax in axes[num_heads:]:
            ax.remove()

        # Tight layout to reduce extra spacing
        plt.tight_layout()
        # Adjust top margin to accommodate the main title
        plt.subplots_adjust(top=0.80)

        # Construct output filename and save the figure
        out_filename = f"{os.path.splitext(filename)[0]}_layer{layer_index+1}.png"
        out_filepath = os.path.join(plot_folder, out_filename)
        plt.savefig(out_filepath, dpi=150)
        plt.close(fig)

        print(f"Saved: {out_filepath}")