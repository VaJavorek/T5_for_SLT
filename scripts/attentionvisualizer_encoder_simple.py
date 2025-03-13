import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Change to your working directory if needed
os.chdir('/auto/brno2/home/javorek/T5_for_SLT/')

# Define parameters for selecting the JSON file and indices
batch_file = 'batch_0.json'   # JSON filename in results/attention_batches_simple/
layer_index = 0               # Index of the encoder layer (0-indexed)
batch_index = 0               # Index of the batch item (0-indexed)

# Toggle to show only the top-left 100×100 region of each matrix
crop_attention = True
crop_size = 100

# Construct the full path to the JSON file
json_file_path = os.path.join(
    '/auto/brno2/home/javorek/T5_for_SLT/',
    'results', 
    'attention_batches_simple', 
    batch_file
)

# Load the JSON file containing the saved attention data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Retrieve the encoder attentions from the data
encoder_attentions = data['encoder_attentions']

# Number of heads in this layer/batch (usually 12 for T5-base, can vary for other models)
num_heads = len(encoder_attentions[layer_index][batch_index])

# Create a 3×4 grid for plotting 12 heads
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
axes = axes.flatten()

# Add a main title indicating which layer and batch item we are visualizing
fig.suptitle(f'Encoder Layer {layer_index+1} - Batch Item {batch_index+1}', fontsize=14)

for head in range(num_heads):
    # Convert each head's attention matrix to a NumPy array
    attn_matrix = np.array(encoder_attentions[layer_index][batch_index][head])
    
    # Optionally crop the attention matrix to 100×100
    if crop_attention:
        attn_matrix = attn_matrix[:crop_size, :crop_size]
    
    # Plot the attention matrix
    axes[head].imshow(attn_matrix, cmap='viridis')
    axes[head].set_title(f'Head {head+1}', fontsize=10)
    axes[head].axis('off')

# Remove any unused subplots if num_heads < rows*cols
for ax in axes[num_heads:]:
    ax.remove()

# Tighten layout to reduce empty space
plt.tight_layout()

# Adjust top to accommodate the suptitle
plt.subplots_adjust(top=0.88)

plt.show()