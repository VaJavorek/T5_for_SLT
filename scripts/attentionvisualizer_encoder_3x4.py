import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Functions for bounding box operations
def find_global_bounding_box(matrices, threshold=1e-9):
    rows, cols = [], []
    for mat in matrices:
        row_sums = np.sum(mat, axis=1)
        col_sums = np.sum(mat, axis=0)
        rows.extend(np.where(row_sums > threshold)[0])
        cols.extend(np.where(col_sums > threshold)[0])
    if not rows or not cols:
        return None
    return min(rows), max(rows), min(cols), max(cols)

def enforce_square_bbox(bbox):
    min_r, max_r, min_c, max_c = bbox
    side = min(max_r - min_r + 1, max_c - min_c + 1)
    return min_r, min_r + side - 1, min_c, min_c + side - 1

def crop_to_bounding_box(matrix, bbox):
    min_r, max_r, min_c, max_c = bbox
    return matrix[min_r:max_r+1, min_c:max_c+1]

# Set folder paths
folder_path = '/auto/brno2/home/javorek/T5_for_SLT/results/attention_batches_simple_labeled_tokens/'
output_folder = os.path.join(folder_path, "paper_plots")
os.makedirs(output_folder, exist_ok=True)

# Choose specific file and heads (0 to 11)
json_filename = 'batch_0.json'  # <-- Set your JSON file here
selected_heads = list(range(12))

# Load JSON data
with open(os.path.join(folder_path, json_filename), 'r') as file:
    data = json.load(file)

encoder_attentions = data['encoder_attentions']
reference = data['reference_translations'][0]
prediction = data['predictions'][0]

# Calculate average attentions with cropping
avg_matrices = []
for head_idx in selected_heads:
    matrices = [np.array(layer[0][head_idx]) for layer in encoder_attentions]
    bbox = find_global_bounding_box(matrices)
    if bbox:
        bbox = enforce_square_bbox(bbox)
        cropped_matrices = [crop_to_bounding_box(mat, bbox) for mat in matrices]
        avg_matrix = np.mean(cropped_matrices, axis=0)
    else:
        avg_matrix = np.zeros((1, 1))
    avg_matrices.append(avg_matrix)

# Plotting in a 3x4 grid
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
fig.suptitle(f"\nRef: {reference[:80]}\nPred: {prediction[:80]}", fontsize=12)

for idx, ax in enumerate(axes.flatten()):
    im = ax.imshow(avg_matrices[idx], cmap='viridis', interpolation='none')
    ax.set_title(f"Head {selected_heads[idx] + 1}")
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.93])

# Save figures
output_path_png = os.path.join(output_folder, json_filename.replace('.json', '_avg12heads.png'))
output_path_pdf = os.path.join(output_folder, json_filename.replace('.json', '_avg12heads.pdf'))

plt.savefig(output_path_png, dpi=150)

# Ensure no interpolation for vector image
for ax in axes.flatten():
    for im in ax.get_images():
        im.set_interpolation('none')

plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
plt.close()

print(f"Saved: {output_path_png} and {output_path_pdf}")
