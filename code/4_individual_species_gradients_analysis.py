#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D scatter plots

def load_gradient_data_for_hemisphere(
    gradient_input_base_dir,
    mask_input_base_dir,
    species,
    hemisphere_label,
    gradient_filename_pattern,
    mask_filename_pattern,
    num_gradients_to_load
):
    """
    Loads the first N gradients for a given species and hemisphere,
    extracting values only for temporal lobe vertices using the provided mask.

    Args:
        gradient_input_base_dir (str): Base directory for gradient files.
        mask_input_base_dir (str): Base directory for mask files.
        species (str): Name of the species.
        hemisphere_label (str): Hemisphere ('L' or 'R').
        gradient_filename_pattern (str): Filename pattern for gradient files.
        mask_filename_pattern (str): Filename pattern for mask files.
        num_gradients_to_load (int): Number of gradients to load from the file.

    Returns:
        numpy.ndarray or None: Array of shape (n_tl_vertices, num_gradients_to_load)
                                containing gradient values for temporal lobe vertices,
                                or None if loading fails.
    """
    gradient_subdir = os.path.join(gradient_input_base_dir, species)
    gradient_filename = gradient_filename_pattern.format(species_name=species, hemisphere=hemisphere_label)
    gradient_path = os.path.join(gradient_subdir, gradient_filename)

    mask_subdir = os.path.join(mask_input_base_dir, species)
    mask_filename = mask_filename_pattern.format(species_name=species, hemisphere=hemisphere_label)
    mask_path = os.path.join(mask_subdir, mask_filename)

    if not os.path.exists(gradient_path):
        print(f"ERROR: Gradient file not found: {gradient_path}")
        return None
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask file not found: {mask_path}")
        return None

    try:
        gradient_img = nib.load(gradient_path)
        mask_img = nib.load(mask_path)
    except Exception as e:
        print(f"Error loading files for {species} {hemisphere_label}: {e}")
        return None

    if not mask_img.darrays:
        print(f"ERROR: No data arrays in mask file: {mask_path}")
        return None
    mask_data = mask_img.darrays[0].data
    temporal_lobe_indices = np.where(mask_data > 0.5)[0]

    if temporal_lobe_indices.size == 0:
        print(f"ERROR: Mask is empty for {species} {hemisphere_label}.")
        return None
    
    if not gradient_img.darrays or len(gradient_img.darrays) == 0:
        print(f"ERROR: No gradient maps (darrays) in {gradient_path}")
        return None
        
    num_available_gradients = len(gradient_img.darrays)
    if num_available_gradients < num_gradients_to_load:
        print(f"WARNING: Requested {num_gradients_to_load} gradients, but only {num_available_gradients} available in {gradient_path}. Will use {num_available_gradients}.")
        num_gradients_to_load = num_available_gradients # Adjust to what's available
    
    if num_gradients_to_load == 0: # If after adjustment, nothing to load
        print(f"ERROR: No gradients to load for {species} {hemisphere_label} from {gradient_path} after checking availability.")
        return None

    num_total_vertices = gradient_img.darrays[0].data.shape[0]
    if mask_data.shape[0] != num_total_vertices:
        print(f"ERROR: Vertex count mismatch for {species} {hemisphere_label}! Gradient file: {num_total_vertices}, Mask: {mask_data.shape[0]}")
        return None

    gradients_for_tl_vertices = np.zeros((len(temporal_lobe_indices), num_gradients_to_load), dtype=np.float32)

    for i in range(num_gradients_to_load):
        gradient_full_surface = gradient_img.darrays[i].data
        gradients_for_tl_vertices[:, i] = gradient_full_surface[temporal_lobe_indices]
        
    return gradients_for_tl_vertices


def _generate_plots_for_data(plot_data_list, plot_labels_list, plot_colors_list,
                             gradient_indices_to_plot, num_dims_to_plot,
                             title_prefix, filename_infix, species_plot_dir, make_2d_projections):
    """
    Generates 2D or 3D scatter plots based on the provided data.

    Args:
        plot_data_list (list): List of numpy arrays, each (n_verts, n_dims_to_plot).
                               Contains one item for individual plots, two for combined.
        plot_labels_list (list): List of labels for the legend.
        plot_colors_list (list): List of colors for the scatter points.
        gradient_indices_to_plot (list): 0-indexed list of which gradients were selected.
        num_dims_to_plot (int): Number of dimensions to plot (2 or 3).
        title_prefix (str): Prefix for the plot title.
        filename_infix (str): Infix for the output plot filename.
        species_plot_dir (str): Directory to save the plot for the current species.
        make_2d_projections (bool): Whether to generate 2D projections for 3D plots.
    """
    gradient_labels_for_filename = "v".join([f"G{i+1}" for i in gradient_indices_to_plot])
    gradient_labels_for_plot_title = " vs ".join([f"Gradient {i+1}" for i in gradient_indices_to_plot])
    full_plot_title = f"{title_prefix}\n{gradient_labels_for_plot_title}"

    if num_dims_to_plot == 2:
        g_x_idx_val, g_y_idx_val = gradient_indices_to_plot[0] + 1, gradient_indices_to_plot[1] + 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for data, label, color in zip(plot_data_list, plot_labels_list, plot_colors_list):
            ax.scatter(data[:, 0], data[:, 1], color=color, alpha=0.5, label=label, s=15, edgecolors='w', linewidth=0.5)
        
        ax.set_xlabel(f"Gradient {g_x_idx_val} Value", fontsize=12)
        ax.set_ylabel(f"Gradient {g_y_idx_val} Value", fontsize=12)
        ax.set_title(full_plot_title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        
        plot_filename = f"gradient_scatter_2D_{filename_infix}_{gradient_labels_for_filename}.png"
        plt.savefig(os.path.join(species_plot_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved 2D scatter plot: {os.path.join(species_plot_dir, plot_filename)}")

    elif num_dims_to_plot == 3:
        g_x_idx_val, g_y_idx_val, g_z_idx_val = gradient_indices_to_plot[0]+1, gradient_indices_to_plot[1]+1, gradient_indices_to_plot[2]+1

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for data, label, color in zip(plot_data_list, plot_labels_list, plot_colors_list):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                       color=color, alpha=0.3, label=label, s=15, edgecolors='w', linewidth=0.3)

        ax.set_xlabel(f"Gradient {g_x_idx_val}", fontsize=10)
        ax.set_ylabel(f"Gradient {g_y_idx_val}", fontsize=10)
        ax.set_zlabel(f"Gradient {g_z_idx_val}", fontsize=10)
        ax.set_title(full_plot_title, fontsize=14)
        ax.legend(fontsize=10)
        
        plot_filename = f"gradient_scatter_3D_{filename_infix}_{gradient_labels_for_filename}.png"
        plt.savefig(os.path.join(species_plot_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved 3D scatter plot: {os.path.join(species_plot_dir, plot_filename)}")

        if make_2d_projections:
            projection_pairs_indices = [ 
                (gradient_indices_to_plot[0], gradient_indices_to_plot[1]), 
                (gradient_indices_to_plot[0], gradient_indices_to_plot[2]), 
                (gradient_indices_to_plot[1], gradient_indices_to_plot[2])
            ]
            # Column indices in the plot_data_list arrays (which are already selected gradients)
            projection_pairs_columns = [(0,1), (0,2), (1,2)] 

            for (g_p_x_orig_idx, g_p_y_orig_idx), (col_p_x, col_p_y) in zip(projection_pairs_indices, projection_pairs_columns):
                fig2d, ax2d = plt.subplots(figsize=(10, 8))
                for data, label, color in zip(plot_data_list, plot_labels_list, plot_colors_list):
                    ax2d.scatter(data[:, col_p_x], data[:, col_p_y], color=color, alpha=0.5, label=label, s=15, edgecolors='w', linewidth=0.5)
                
                ax2d.set_xlabel(f"Gradient {g_p_x_orig_idx+1} Value", fontsize=12)
                ax2d.set_ylabel(f"Gradient {g_p_y_orig_idx+1} Value", fontsize=12)
                proj_title = f"2D Projection - {title_prefix}\nGradients {g_p_x_orig_idx+1} vs {g_p_y_orig_idx+1}"
                ax2d.set_title(proj_title, fontsize=14)
                ax2d.legend(fontsize=10)
                ax2d.grid(True, linestyle='--', alpha=0.7)
                ax2d.axhline(0, color='black', linewidth=0.8)
                ax2d.axvline(0, color='black', linewidth=0.8)
                
                proj_filename_labels = f"G{g_p_x_orig_idx+1}vG{g_p_y_orig_idx+1}"
                plot_filename_2d_proj = f"gradient_scatter_2D_proj_{filename_infix}_{proj_filename_labels}.png"
                plt.savefig(os.path.join(species_plot_dir, plot_filename_2d_proj), dpi=300, bbox_inches='tight')
                plt.close(fig2d)
                print(f"  Saved 2D projection plot: {os.path.join(species_plot_dir, plot_filename_2d_proj)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2D/3D scatter plots of temporal lobe vertices in gradient space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--gradient_input_dir', type=str, required=True,
                        help="Base directory where Script 3 saved its gradient outputs (containing species subfolders).")
    parser.add_argument('--mask_input_dir', type=str, required=True,
                        help="Base directory for temporal lobe mask files. Assumes species subfolders.")
    parser.add_argument('--plot_output_dir', type=str, required=True,
                        help="Base directory where scatter plots will be saved.")
    parser.add_argument('--species_list', type=str, required=True,
                        help='Comma-separated list of species names to process (e.g., "human,chimpanzee").')
    
    parser.add_argument('--plot_type', type=str, default="combined", choices=["combined", "individual"],
                        help="Type of gradients to plot: 'combined' across hemispheres, or 'individual' per hemisphere.")
    parser.add_argument('--gradient_pattern_combined', type=str,
                        default="all_computed_gradients_{species_name}_COMBINED_{hemisphere}.func.gii",
                        help='Filename pattern for COMBINED gradient files from Script 3. Placeholders: {species_name}, {hemisphere (L/R part)}.')
    parser.add_argument('--gradient_pattern_individual', type=str,
                        default="all_computed_gradients_{species_name}_{hemisphere}_SEPARATE.func.gii",
                        help='Filename pattern for INDIVIDUAL (separate) gradient files from Script 3. Placeholders: {species_name}, {hemisphere}.')
    parser.add_argument('--mask_pattern', type=str, required=True,
                        help='Filename pattern for temporal lobe masks. Placeholders: {species_name}, {hemisphere}.')
    
    parser.add_argument('--gradients_to_plot', type=str, default="1,2,3",
                        help='Comma-separated list of 1-indexed gradient numbers to plot (e.g., "1,2" for G1 vs G2; "1,2,3" for G1 vs G2 vs G3). Max 3.')
    parser.add_argument('--plot_2d_projections', action='store_true',
                        help='If plotting 3 gradients (either type), also generate 2D projection scatter plots.')

    args = parser.parse_args()

    species_to_process = [s.strip() for s in args.species_list.split(',')]
    # Standard hemispheres to iterate over when plotting individual types
    hemispheres_for_individual_plots = ['L', 'R'] 
    
    try:
        # Convert 1-indexed user input to 0-indexed internal list
        gradient_indices_to_plot = sorted(list(set([int(g.strip()) - 1 for g in args.gradients_to_plot.split(',')]))) 
        if not all(idx >= 0 for idx in gradient_indices_to_plot):
            raise ValueError("Gradient indices must be positive integers.")
        if len(gradient_indices_to_plot) > 3:
            print("ERROR: Can plot a maximum of 3 gradients. Please select 2 or 3.")
            exit(1)
        if len(gradient_indices_to_plot) < 2:
            print("ERROR: Need at least 2 gradients for scatter plots.")
            exit(1)
    except ValueError as e:
        print(f"ERROR: Invalid format for --gradients_to_plot. Expected comma-separated positive integers (e.g., '1,2' or '1,2,3'). Details: {e}")
        exit(1)

    num_dims_to_plot = len(gradient_indices_to_plot)
    # Determine the maximum gradient index we need to load from files
    num_gradients_to_load_from_file = max(gradient_indices_to_plot) + 1 if gradient_indices_to_plot else 0


    if not os.path.exists(args.plot_output_dir):
        os.makedirs(args.plot_output_dir, exist_ok=True)

    for species in species_to_process:
        print(f"\n--- Processing scatter plots for {species.capitalize()} (Plot Type: {args.plot_type}) ---")
        
        species_plot_dir = os.path.join(args.plot_output_dir, species)
        if not os.path.exists(species_plot_dir):
            os.makedirs(species_plot_dir, exist_ok=True)

        # These lists will hold the data and metadata for the current plot(s)
        # For 'combined', they'll get two entries (L & R).
        # For 'individual', they'll be repopulated for each hemisphere's separate plot.
        plot_data_list_for_current_figure = [] 
        plot_labels_list_for_current_figure = []
        plot_colors_list_for_current_figure = []
        
        current_title_prefix = ""
        current_filename_infix = ""

        if args.plot_type == "combined":
            current_title_prefix = f"Combined Gradient Space - {species.capitalize()}"
            current_filename_infix = f"COMBINED_{species}"
            active_gradient_pattern = args.gradient_pattern_combined

            # Load L and R parts of the combined gradient
            grad_data_L = load_gradient_data_for_hemisphere(
                args.gradient_input_dir, args.mask_input_dir, species, "L",
                active_gradient_pattern, args.mask_pattern, num_gradients_to_load_from_file
            )
            grad_data_R = load_gradient_data_for_hemisphere(
                args.gradient_input_dir, args.mask_input_dir, species, "R",
                active_gradient_pattern, args.mask_pattern, num_gradients_to_load_from_file
            )

            if grad_data_L is not None and grad_data_R is not None:
                try:
                    plot_data_list_for_current_figure.append(grad_data_L[:, gradient_indices_to_plot])
                    plot_labels_list_for_current_figure.append('Left Hemisphere TL')
                    plot_colors_list_for_current_figure.append('royalblue')
                    plot_data_list_for_current_figure.append(grad_data_R[:, gradient_indices_to_plot])
                    plot_labels_list_for_current_figure.append('Right Hemisphere TL')
                    plot_colors_list_for_current_figure.append('crimson')
                except IndexError:
                    print(f"ERROR: Not enough computed gradients for COMBINED {species} to select for plotting.")
                    plot_data_list_for_current_figure = [] # Clear to prevent plotting
            else:
                print(f"Could not load all required gradient data for COMBINED {species}. Skipping.")
            
            if plot_data_list_for_current_figure: # Proceed to plot if data is ready
                 _generate_plots_for_data(plot_data_list_for_current_figure, 
                                     plot_labels_list_for_current_figure, 
                                     plot_colors_list_for_current_figure,
                                     gradient_indices_to_plot, num_dims_to_plot,
                                     current_title_prefix, current_filename_infix, 
                                     species_plot_dir, args.plot_2d_projections)

        elif args.plot_type == "individual":
            active_gradient_pattern = args.gradient_pattern_individual
            for hem_to_plot in hemispheres_for_individual_plots: 
                print(f"  Processing Individual Plot for Hemisphere: {hem_to_plot}")
                current_title_prefix = f"Individual Gradient Space - {species.capitalize()} - {hem_to_plot} Hemisphere"
                current_filename_infix = f"INDIVIDUAL_{species}_{hem_to_plot}"
                
                # Reset lists for each individual hemisphere plot
                plot_data_list_for_current_figure = [] 
                plot_labels_list_for_current_figure = []
                plot_colors_list_for_current_figure = []

                grad_data_single_hem = load_gradient_data_for_hemisphere(
                    args.gradient_input_dir, args.mask_input_dir, species, hem_to_plot,
                    active_gradient_pattern, args.mask_pattern, num_gradients_to_load_from_file
                )

                if grad_data_single_hem is not None:
                    try:
                        plot_data_list_for_current_figure.append(grad_data_single_hem[:, gradient_indices_to_plot])
                        plot_labels_list_for_current_figure.append(f'{hem_to_plot} Hemisphere TL')
                        plot_colors_list_for_current_figure.append('royalblue' if hem_to_plot == 'L' else 'crimson')
                    except IndexError:
                        print(f"ERROR: Not enough computed gradients for INDIVIDUAL {species} {hem_to_plot} to select for plotting.")
                        plot_data_list_for_current_figure = [] # Clear to prevent plotting
                else:
                    print(f"Could not load gradient data for INDIVIDUAL {species} {hem_to_plot}. Skipping.")
                
                if plot_data_list_for_current_figure: # Proceed to plot if data is ready
                    _generate_plots_for_data(plot_data_list_for_current_figure, 
                                             plot_labels_list_for_current_figure, 
                                             plot_colors_list_for_current_figure,
                                             gradient_indices_to_plot, num_dims_to_plot,
                                             current_title_prefix, current_filename_infix, 
                                             species_plot_dir, args.plot_2d_projections)
            # End of loop for individual hemispheres for the current species

    print("\n--- Scatter plot generation complete ---")