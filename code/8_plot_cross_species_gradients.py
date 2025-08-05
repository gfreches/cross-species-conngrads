import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Default Configuration ---

# Specify which pairs of gradients (0-indexed) to plot by default
# This can be overridden by a command-line argument.
# e.g., [(0, 1)] means Gradient 1 vs Gradient 2
DEFAULT_GRADIENT_PAIRS_TO_PLOT = [(0, 1)]

# Define colors and markers for each species/hemisphere combination
# Ensure this list matches the order and names in your SPECIES_LIST and HEMISPHERES from the LLE script
species_hem_configs = {
    "human_L": {"color": "blue", "marker": "o", "label": "Human Left"},
    "human_R": {"color": "cornflowerblue", "marker": "o", "label": "Human Right"},
    "chimpanzee_L": {"color": "red", "marker": "s", "label": "Chimpanzee Left"},
    "chimpanzee_R": {"color": "lightcoral", "marker": "s", "label": "Chimpanzee Right"}
}

# Plotting parameters
POINT_SIZE = 20
ALPHA_VALUE = 0.5

# --- Main Script ---

def create_gradient_scatter_plots(npz_file_path, plot_output_dir, gradient_pairs_to_plot):
    """
    Generates and saves scatter plots for specified pairs of cross-species gradients.

    Args:
        npz_file_path (str): Path to the input .npz file with embedding data.
        plot_output_dir (str): Directory to save the output scatter plots.
        gradient_pairs_to_plot (list): List of tuples, where each tuple contains two
                                       0-indexed gradient indices to plot against each other.
    """
    if not os.path.exists(npz_file_path):
        print(f"ERROR: NPZ file not found at {npz_file_path}")
        return

    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"Output directory for plots: {plot_output_dir}")

    # 1. Load the data
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        cross_species_gradients = data['cross_species_gradients']

        # Accommodate different potential keys for segment info from script 6
        if 'segment_info_simple' in data:
            segment_info_raw = data['segment_info_simple']
        elif 'segment_info_detailed_for_remapping' in data:
            segment_info_raw = data['segment_info_detailed_for_remapping']
        elif 'segment_info' in data: # Fallback to original key
            segment_info_raw = data['segment_info']
        else:
            print("ERROR: Could not find segment information in the NPZ file.")
            return

        print(f"Loaded embedding data with shape: {cross_species_gradients.shape}")
        print(f"Loaded segment info for {len(segment_info_raw)} segments.")
    except Exception as e:
        print(f"Error loading data from NPZ file '{npz_file_path}': {e}")
        return

    if cross_species_gradients.ndim != 2 or cross_species_gradients.shape[0] == 0:
        print("ERROR: 'cross_species_gradients' data is not a valid 2D array or is empty.")
        return

    num_available_gradients = cross_species_gradients.shape[1]
    if num_available_gradients == 0:
        print("ERROR: No gradients found in the loaded data.")
        return
    print(f"Number of gradients available in the file: {num_available_gradients}")

    # Convert segment_info back to a list of dicts if it's a structured array
    segment_info = []
    if hasattr(segment_info_raw, 'item') and segment_info_raw.ndim == 0: # Handle 0-d array containing the list
        segment_info = list(segment_info_raw.item())
    elif hasattr(segment_info_raw, 'ndim') and segment_info_raw.ndim == 1:
        segment_info = list(segment_info_raw)
    else:
        print(f"ERROR: segment_info has an unexpected structure (type: {type(segment_info_raw)}).")
        return

    # 2. Generate plots for specified gradient pairs
    for grad_idx_x, grad_idx_y in gradient_pairs_to_plot:
        if grad_idx_x >= num_available_gradients or grad_idx_y >= num_available_gradients:
            print(f"Warning: Gradient pair ({grad_idx_x+1}, {grad_idx_y+1}) exceeds available gradients ({num_available_gradients}). Skipping.")
            continue
        if grad_idx_x == grad_idx_y:
            print(f"Warning: Skipping plot for identical gradients ({grad_idx_x+1} vs {grad_idx_y+1}).")
            continue

        print(f"\nGenerating scatter plot for Gradient {grad_idx_x+1} vs Gradient {grad_idx_y+1}...")

        plt.figure(figsize=(12, 10))
        all_x_values, all_y_values = [], []

        for segment in segment_info:
            species = segment.get('species', 'unknown')
            hem = segment.get('hem', 'U')
            start_row = segment.get('start_row')
            end_row = segment.get('end_row')

            if start_row is None or end_row is None:
                print(f"Warning: Skipping segment due to missing 'start_row' or 'end_row': {segment}")
                continue

            key = f"{species}_{hem}"
            config = species_hem_configs.get(key)
            if not config:
                print(f"Warning: No plot configuration found for '{key}'. Using default.")
                config = {"color": "gray", "marker": ".", "label": key}

            segment_grads = cross_species_gradients[start_row:end_row, :]
            x_values = segment_grads[:, grad_idx_x]
            y_values = segment_grads[:, grad_idx_y]

            all_x_values.extend(x_values)
            all_y_values.extend(y_values)

            plt.scatter(x_values, y_values,
                        color=config["color"], marker=config["marker"],
                        alpha=ALPHA_VALUE, s=POINT_SIZE,
                        label=config["label"], edgecolors='none')

        plt.xlabel(f"Cross-Species Gradient {grad_idx_x+1} Value", fontsize=14)
        plt.ylabel(f"Cross-Species Gradient {grad_idx_y+1} Value", fontsize=14)
        plt.title(f"Cross-Species Temporal Lobe Embedding\n(Gradient {grad_idx_x+1} vs Gradient {grad_idx_y+1})", fontsize=16)

        legend = plt.legend(fontsize=12, markerscale=1.5, loc='best')
        for handle in legend.legend_handles:
            handle.set_alpha(1)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
        plt.axvline(0, color='black', linewidth=0.5, linestyle=':')

        if all_x_values and all_y_values:
            x_min, x_max = np.min(all_x_values), np.max(all_x_values)
            y_min, y_max = np.min(all_y_values), np.max(all_y_values)
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05
            plt.xlim(x_min - x_padding, x_max + x_padding)
            plt.ylim(y_min - y_padding, y_max + y_padding)

        plot_filename = f"cross_species_scatter_G{grad_idx_x+1}_vs_G{grad_idx_y+1}.png"
        plot_save_path = os.path.join(plot_output_dir, plot_filename)
        try:
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved scatter plot to: {plot_save_path}")
        except Exception as e_plot:
            print(f"Error saving scatter plot '{plot_save_path}': {e_plot}")
        plt.close()

    print("\n--- Scatter plot generation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scatter plots from cross-species gradient embedding data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--npz_file', type=str, required=True,
        help="Path to the .npz file containing cross-species gradient data from script 6."
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help="Directory where the output scatter plot images will be saved."
    )
    parser.add_argument(
        '--gradient_pairs', type=str,
        default=','.join([f"{p[0]}_{p[1]}" for p in DEFAULT_GRADIENT_PAIRS_TO_PLOT]),
        help='Comma-separated list of gradient pairs to plot, e.g., "0_1,0_2,1_2" for G1vG2, G1vG3, G2vG3.'
    )

    args = parser.parse_args()

    # --- Parse gradient pairs ---
    parsed_pairs = []
    try:
        pairs_str = args.gradient_pairs.split(',')
        for pair_str in pairs_str:
            indices = [int(i) for i in pair_str.strip().split('_')]
            if len(indices) != 2:
                raise ValueError(f"Each pair must have exactly two indices: '{pair_str}'")
            parsed_pairs.append(tuple(indices))
    except Exception as e:
        print(f"Error: Invalid format for --gradient_pairs. Please use format like '0_1,0_2'. Details: {e}")
        exit(1)

    if not parsed_pairs:
        print("Warning: No valid gradient pairs specified. Nothing to plot.")
        exit(0)

    create_gradient_scatter_plots(
        npz_file_path=args.npz_file,
        plot_output_dir=args.output_dir,
        gradient_pairs_to_plot=parsed_pairs
    )
