import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# --- Helper Functions ---

def create_title_with_translation(filename, layer_info, translation, prediction, max_chars=100):
    """Create a multi-line title."""
    if len(translation) > max_chars:
        translation = translation[:max_chars] + "..."
    if len(prediction) > max_chars:
        prediction = prediction[:max_chars] + "..."
    return f"File: {filename}\n{layer_info}\nReference: {translation}\nPrediction: {prediction}"

def find_nonzero_bbox(matrix, threshold=1e-9):
    """Find bounding box (min_r, max_r, min_c, max_c) of values > threshold."""
    if matrix is None: return None
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    nonempty_rows = np.where(row_sums > threshold)[0]
    nonempty_cols = np.where(col_sums > threshold)[0]
    if len(nonempty_rows) == 0 or len(nonempty_cols) == 0:
        return None
    return (nonempty_rows[0], nonempty_rows[-1], nonempty_cols[0], nonempty_cols[-1])

def enforce_square_bbox(bbox):
    """Expand/truncate bbox to be square, anchored at top-left."""
    if bbox is None: return None
    min_r, max_r, min_c, max_c = bbox
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    side = min(height, width) # Truncate to smaller dim to ensure data exists, or max to include all? 
    # Original script truncated larger dim to match smaller: side = min(height, width)
    # But usually we want to see the diagonal. Let's follow original logic.
    side = min(height, width) 
    return (min_r, min_r + side - 1, min_c, min_c + side - 1)

def crop_matrix(matrix, bbox):
    """Crop matrix to bbox."""
    if bbox is None: return np.zeros((1,1))
    min_r, max_r, min_c, max_c = bbox
    return matrix[min_r:max_r+1, min_c:max_c+1]

def pad_row(row, target_length):
    """Pad a 1D row with zeros."""
    padded = np.zeros(target_length)
    padded[:len(row)] = row
    return padded

def plot_grid(matrices, titles, main_title, output_path, n_cols=4, show_y_labels=False, x_labels=None, y_labels_list=None):
    """Generic plotting function for a grid of matrices."""
    num_plots = len(matrices)
    n_rows = int(np.ceil(num_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows + 1))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    fig.suptitle(main_title, fontsize=12, wrap=True)

    for i in range(num_plots):
        ax = axes[i]
        mat = matrices[i]
        
        ax.imshow(mat, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_title(titles[i], fontsize=10)
        
        # Axis labels logic
        if x_labels and i < len(x_labels) and x_labels[i]:
            ax.set_xticks(range(len(x_labels[i])))
            ax.set_xticklabels(x_labels[i], rotation=45, fontsize=8)
        else:
            ax.set_xticks([])
            
        if show_y_labels and (i % n_cols == 0) and y_labels_list and i < len(y_labels_list) and y_labels_list[i]:
            ax.set_yticks(range(len(y_labels_list[i])))
            ax.set_yticklabels(y_labels_list[i], fontsize=8)
        else:
            ax.set_yticks([])
            
    # Remove empty subplots
    for i in range(num_plots, len(axes)):
        axes[i].remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85 if len(main_title.split('\n')) > 3 else 0.9)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_histograms(data_list, titles, main_title, output_path, n_cols=4, x_labels_list=None):
    """Generic plotting function for histograms."""
    num_plots = len(data_list)
    n_rows = int(np.ceil(num_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows + 1))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    fig.suptitle(main_title, fontsize=12, wrap=True)

    for i in range(num_plots):
        ax = axes[i]
        data = data_list[i]
        
        ax.bar(range(len(data)), data)
        ax.set_title(titles[i], fontsize=10)
        
        if x_labels_list and i < len(x_labels_list) and x_labels_list[i]:
            # Subsample labels if too many
            labels = x_labels_list[i]
            step = max(1, len(labels) // 10)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_xticklabels(labels[::step], rotation=45, fontsize=8)
        
    for i in range(num_plots, len(axes)):
        axes[i].remove()
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram: {output_path}")

# --- Processing Logic ---

def process_encoder(data, args, output_base, filename_base):
    """
    Encoder Attention: [num_layers, batch_size, num_heads, seq_len, seq_len]
    Batch size assumed 1.
    """
    attentions = data.get('encoder_attentions')
    if not attentions: return

    ref = data['reference_translations'][0]
    pred = data['predictions'][0]
    
    num_layers = len(attentions)
    num_heads = len(attentions[0][0])
    
    # Pre-convert to numpy for easier handling
    # shape: (layers, heads, seq, seq)
    layers_data = []
    for l in range(num_layers):
        head_data = []
        for h in range(num_heads):
            head_data.append(np.array(attentions[l][0][h]))
        layers_data.append(head_data)
    
    # 1. Per-Layer Heads Visualization
    if 'layer_heads' in args.mode or 'all' in args.mode:
        for l in range(num_layers):
            matrices = []
            titles = []
            for h in range(num_heads):
                mat = layers_data[l][h]
                bbox = None
                if args.crop == 'auto':
                    bbox = find_nonzero_bbox(mat, args.threshold)
                elif args.crop == 'square':
                    bbox = enforce_square_bbox(find_nonzero_bbox(mat, args.threshold))
                elif args.crop == 'fixed':
                    bbox = (0, 100, 0, 100)
                
                matrices.append(crop_matrix(mat, bbox) if bbox else (crop_matrix(mat, (0,mat.shape[0]-1,0,mat.shape[1]-1)) if args.crop == 'none' else np.zeros((1,1))))
                titles.append(f"Head {h+1}")
            
            out_path = os.path.join(output_base, f"{filename_base}_enc_L{l+1}.png")
            plot_grid(matrices, titles, 
                      create_title_with_translation(filename_base, f"Encoder Layer {l+1}", ref, pred),
                      out_path)

    # 2. Average Heads (collapsed layers)
    if 'avg_layers' in args.mode or 'all' in args.mode:
        avg_matrices = []
        titles = []
        for h in range(num_heads):
            # stack layers for this head
            head_across_layers = [layers_data[l][h] for l in range(num_layers)]
            
            # Find global bbox if auto/square
            bbox = None
            if args.crop in ['auto', 'square']:
                 bbox = find_nonzero_bbox(np.mean(head_across_layers, axis=0), args.threshold)
                 if args.crop == 'square': bbox = enforce_square_bbox(bbox)
            
            # Crop then average, or average then crop? 
            # Original script: find bbox on individual, then sum cropped. 
            # Simpler: Average full matrices, then crop.
            avg_mat = np.mean(head_across_layers, axis=0)
            
            if args.crop == 'fixed': bbox = (0, 100, 0, 100)
            
            avg_matrices.append(crop_matrix(avg_mat, bbox if bbox else (0,avg_mat.shape[0]-1,0,avg_mat.shape[1]-1)))
            titles.append(f"Head {h+1} (Avg Layers)")

        out_path = os.path.join(output_base, f"{filename_base}_enc_avg_heads.png")
        plot_grid(avg_matrices, titles,
                  create_title_with_translation(filename_base, "Encoder Attention (Avg across Layers)", ref, pred),
                  out_path)


def process_decoder(data, args, output_base, filename_base):
    """
    Decoder Attention: [step][layer][0][head][0] -> list (row)
    Variable length per step.
    """
    attentions = data.get('decoder_attentions') # List of steps
    if not attentions: return
    
    ref = data['reference_translations'][0]
    pred = data['predictions'][0]
    
    num_steps = len(attentions)
    num_layers = len(attentions[0])
    num_heads = len(attentions[0][0][0])
    
    # Helper to build matrix for (layer, head) across steps
    def build_matrix(l, h):
        rows = []
        lengths = []
        for s in range(num_steps):
            row = attentions[s][l][0][h][0]
            rows.append(row)
            lengths.append(len(row))
        max_len = max(lengths) if lengths else 0
        return np.stack([pad_row(r, max_len) for r in rows])

    # 1. Per-Layer
    if 'layer_heads' in args.mode or 'all' in args.mode:
        for l in range(num_layers):
            matrices = []
            titles = []
            for h in range(num_heads):
                mat = build_matrix(l, h)
                bbox = find_nonzero_bbox(mat, args.threshold) if args.crop in ['auto', 'square'] else None
                matrices.append(crop_matrix(mat, bbox) if bbox else mat)
                titles.append(f"Head {h+1}")
            
            out_path = os.path.join(output_base, f"{filename_base}_dec_L{l+1}.png")
            plot_grid(matrices, titles,
                      create_title_with_translation(filename_base, f"Decoder Layer {l+1}", ref, pred),
                      out_path)

    # 2. Avg across layers (for each head)
    if 'avg_layers' in args.mode or 'all' in args.mode:
        matrices = []
        titles = []
        for h in range(num_heads):
            # Average layers for this head
            # Step by step, average the rows from all layers
            step_rows = []
            lengths = []
            for s in range(num_steps):
                layer_rows = [attentions[s][l][0][h][0] for l in range(num_layers)]
                max_l = max(len(r) for r in layer_rows)
                # Pad and average
                padded_layers = np.stack([pad_row(r, max_l) for r in layer_rows])
                avg_row = np.mean(padded_layers, axis=0)
                step_rows.append(avg_row)
                lengths.append(max_l)
            
            overall_max = max(lengths) if lengths else 0
            agg_mat = np.stack([pad_row(r, overall_max) for r in step_rows])
            
            bbox = find_nonzero_bbox(agg_mat, args.threshold) if args.crop in ['auto', 'square'] else None
            matrices.append(crop_matrix(agg_mat, bbox) if bbox else agg_mat)
            titles.append(f"Head {h+1}")

        out_path = os.path.join(output_base, f"{filename_base}_dec_avg_heads.png")
        plot_grid(matrices, titles,
                  create_title_with_translation(filename_base, "Decoder Attention (Avg across Layers)", ref, pred),
                  out_path)
    
    # 3. Histograms (Column Sums of Avg Matrix)
    if 'hist' in args.mode or 'all' in args.mode:
         # Reuse logic from Avg across layers to get matrices
         # ... (Redundant calc, but for clarity kept separate or cached)
         # For brevity, let's assume we want histograms of the Avg-Head matrices calculated above
         # Recalculating for safety/independence
         hists = []
         titles = []
         for h in range(num_heads):
            step_rows = []
            lengths = []
            for s in range(num_steps):
                layer_rows = [attentions[s][l][0][h][0] for l in range(num_layers)]
                max_l = max(len(r) for r in layer_rows)
                padded_layers = np.stack([pad_row(r, max_l) for r in layer_rows])
                step_rows.append(np.mean(padded_layers, axis=0))
                lengths.append(max_l)
            
            overall_max = max(lengths) if lengths else 0
            agg_mat = np.stack([pad_row(r, overall_max) for r in step_rows])
            
            col_sums = np.sum(agg_mat, axis=0)
            
            # Crop histogram
            nz = np.where(col_sums > args.threshold)[0]
            if len(nz) > 0:
                col_sums = col_sums[nz[0]:nz[-1]+1]
            
            hists.append(col_sums)
            titles.append(f"Head {h+1}")

         out_path = os.path.join(output_base, f"{filename_base}_dec_hist.png")
         plot_histograms(hists, titles,
                         create_title_with_translation(filename_base, "Decoder Attention Histogram", ref, pred),
                         out_path)


def process_cross(data, args, output_base, filename_base):
    """
    Cross Attention: [step][layer][0][head][0] -> list of length seq_len (frames)
    Assumed fixed seq_len usually, but format is same as decoder list-of-lists.
    """
    attentions = data.get('cross_attentions')
    if not attentions: return
    
    ref = data['reference_translations'][0]
    pred = data['predictions'][0]
    decoded_tokens = data.get('decoded_tokens', [[]])[0] # Tokens on Y axis
    
    # If using stored batch format, cross_attentions might be a 6D tensor or list of lists
    # The JSON usually has lists.
    # Check if first element is list or what.
    # Based on attentionvisualizer_cross.py: data['cross_attentions'] is [steps, layers, batch, heads, 1, seq_len]
    # But let's check input structure. If it's the output of predict_attention_store_batch.py, it's lists.
    
    # Let's convert to numpy array first for slicing if possible, assuming fixed frame count.
    try:
        arr = np.array(attentions)
        # Shape: (steps, layers, batch=1, heads, 1, seq_len)
        # Squeeze batch and singleton dim
        # New Shape: (steps, layers, heads, seq_len)
        if arr.ndim == 6:
             arr = arr[:, :, 0, :, 0, :]
    except:
        # Irregular shape?
        return 

    num_steps, num_layers, num_heads, seq_len = arr.shape
    
    frame_indices = list(range(seq_len))
    
    # 1. Avg across layers (for each head)
    if 'avg_layers' in args.mode or 'all' in args.mode:
        avg_matrices = []
        titles = []
        y_labels = []
        x_labels = []
        
        for h in range(num_heads):
            # Average over layers
            # arr[:, :, h, :] -> (steps, layers, seq)
            # mean over dim 1 -> (steps, seq)
            avg_mat = np.mean(arr[:, :, h, :], axis=1)
            
            bbox = find_nonzero_bbox(avg_mat, args.threshold) if args.crop in ['auto', 'square'] else None
            
            if bbox:
                mat_crop = crop_matrix(avg_mat, bbox)
                r1, r2, c1, c2 = bbox
                y_lbl = decoded_tokens[r1:r2+1] if r2 < len(decoded_tokens) else []
                x_lbl = [str(i) for i in frame_indices[c1:c2+1]]
            else:
                mat_crop = avg_mat
                y_lbl = decoded_tokens
                x_lbl = [str(i) for i in frame_indices]
                
            avg_matrices.append(mat_crop)
            titles.append(f"Head {h+1}")
            y_labels.append(y_lbl)
            x_labels.append(x_lbl)

        out_path = os.path.join(output_base, f"{filename_base}_cross_avg_layers.png")
        plot_grid(avg_matrices, titles,
                  create_title_with_translation(filename_base, "Cross Attention (Avg Layers)", ref, pred),
                  out_path, show_y_labels=True, x_labels=x_labels, y_labels_list=y_labels)

    # 2. Global Histogram
    if 'hist' in args.mode or 'all' in args.mode:
        # Mean across layers AND heads
        # arr -> (steps, layers, heads, seq)
        # mean(1,2) -> (steps, seq) -> sum(0) -> (seq)
        global_mat = np.mean(arr, axis=(1, 2))
        col_sums = np.sum(global_mat, axis=0)
        
        # Crop
        nz = np.where(col_sums > args.threshold)[0]
        if len(nz) > 0:
            s, e = nz[0], nz[-1]+1
            col_sums = col_sums[s:e]
            labels = [str(i) for i in frame_indices[s:e]]
        else:
            labels = []
            
        out_path = os.path.join(output_base, f"{filename_base}_cross_hist.png")
        plot_histograms([col_sums], ["Global Average"], 
                        create_title_with_translation(filename_base, "Cross Attention Global Histogram", ref, pred),
                        out_path, n_cols=1, x_labels_list=[labels])


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Unified Attention Visualizer")
    parser.add_argument('--input', type=str, required=True, help="Path to JSON file or directory")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    parser.add_argument('--type', type=str, default='all', choices=['encoder', 'decoder', 'cross', 'all'], help="Attention type to visualize")
    parser.add_argument('--mode', type=str, default='all', help="Comma separated modes: layer_heads, avg_layers, hist, all")
    parser.add_argument('--crop', type=str, default='auto', choices=['auto', 'square', 'fixed', 'none'], help="Cropping strategy")
    parser.add_argument('--threshold', type=float, default=1e-9, help="Threshold for zero-value detection")
    parser.add_argument('--max_files', type=int, default=None, help="Max files to process")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    files = []
    if os.path.isdir(args.input):
        files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.json')])
    else:
        files = [args.input]
        
    if args.max_files:
        files = files[:args.max_files]
        
    print(f"Processing {len(files)} files...")
    
    for fpath in tqdm(files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue
            
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        
        if args.type in ['encoder', 'all']:
            process_encoder(data, args, args.output, base_name)
        if args.type in ['decoder', 'all']:
            process_decoder(data, args, args.output, base_name)
        if args.type in ['cross', 'all']:
            process_cross(data, args, args.output, base_name)

if __name__ == "__main__":
    main()
