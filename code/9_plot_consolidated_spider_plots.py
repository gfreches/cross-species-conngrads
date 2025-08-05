import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# --- Default Configuration ---
DEFAULT_SPECIES_FOR_PLOTS = ["human", "chimpanzee"]
DEFAULT_HEMISPHERES = ['L', 'R']
DEFAULT_GRADIENTS_TO_ANALYZE = [0, 1] # e.g., Cross-species Gradient 1 and Gradient 2

DEFAULT_N_TRACTS = 20
DEFAULT_TRACT_NAMES = [
    "AC", "AF", "AR","CBD","CBP", "CBT", "CST", "FA",
    "FMI", "FMA", "FX", "IFOF", "ILF", "MDLF",
    "OR", "SLF I", "SLF II", "SLF III", "UF", "VOF"
]

# Plotting configurations for the 4 lines on each plot
PLOT_CONFIGS = {
    "human_L": {"color": "blue", "linestyle": "-", "label": "Human Left"},
    "human_R": {"color": "cornflowerblue", "linestyle": "--", "label": "Human Right"},
    "chimpanzee_L": {"color": "red", "linestyle": "-", "label": "Chimp Left"},
    "chimpanzee_R": {"color": "lightcoral", "linestyle": "--", "label": "Chimp Right"}
}


# --- Helper Functions ---
def get_vertex_blueprint_profile(species_name, hemisphere_label, vertex_id, masked_blueprint_dir, n_tracts):
    """Loads the original average blueprint and returns the profile for a specific vertex."""
    blueprint_filename = f"average_{species_name}_blueprint.{hemisphere_label}_temporal_lobe_masked.func.gii"
    blueprint_path = os.path.join(masked_blueprint_dir, species_name, blueprint_filename)

    if not os.path.exists(blueprint_path):
        print(f"ERROR: Blueprint file for spider plot not found: {blueprint_path}"); return None
    try:
        blueprint_img = nib.load(blueprint_path)
        if not blueprint_img.darrays or len(blueprint_img.darrays) != n_tracts:
            print(f"ERROR: Blueprint {blueprint_path} num darrays ({len(blueprint_img.darrays)}) != {n_tracts}."); return None

        blueprint_all_vertices_data = np.array([d.data for d in blueprint_img.darrays]).T
        num_total_surface_vertices = blueprint_all_vertices_data.shape[0]

        if vertex_id < 0 or vertex_id >= num_total_surface_vertices:
            print(f"ERROR: Vertex ID {vertex_id} out of bounds for {species_name} {hemisphere_label} ({num_total_surface_vertices} verts)."); return None

        return blueprint_all_vertices_data[vertex_id, :]
    except Exception as e:
        print(f"Error loading/processing blueprint {blueprint_path} for vertex {vertex_id}: {e}"); return None

def get_k_from_mask(species, hemisphere, mask_base_dir):
    """Gets the number of vertices from a species' mask to use as k."""
    mask_filename = f"{species}_{hemisphere}.func.gii"
    mask_path = os.path.join(mask_base_dir, species, mask_filename)
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask for {species} {hemisphere} not found at {mask_path}"); return None
    try:
        mask_img = nib.load(mask_path)
        k_val = int(np.sum(mask_img.darrays[0].data > 0.5))
        if k_val == 0:
            print(f"ERROR: Mask {mask_path} is empty."); return None
        return k_val
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}"); return None

# --- Main Plotting Script ---
def create_consolidated_spider_plots(
    npz_file_path,
    masked_blueprint_dir,
    downsampled_data_dir,
    mask_base_dir,
    output_dir,
    species_list,
    hemispheres,
    gradients_to_analyze,
    n_tracts,
    tract_names,
    target_k_species='chimpanzee' # Species that defines k for downsampling
):
    if not os.path.exists(npz_file_path):
        print(f"ERROR: NPZ file not found: {npz_file_path}"); return
    os.makedirs(output_dir, exist_ok=True)

    try:
        npz_data = np.load(npz_file_path, allow_pickle=True)
        cross_species_gradients = npz_data['cross_species_gradients']
        segment_info_list = list(npz_data['segment_info_simple'])
        print(f"Loaded cross-species gradients with shape: {cross_species_gradients.shape}")
    except Exception as e:
        print(f"Error loading NPZ file '{npz_file_path}': {e}"); return

    num_available_gradients = cross_species_gradients.shape[1]

    # Determine k from the target_k_species masks, needed for constructing human filenames
    k_vals = {}
    for hem in hemispheres:
        k = get_k_from_mask(target_k_species, hem, mask_base_dir)
        if k is None:
            print(f"FATAL: Could not determine k from {target_k_species} {hem} mask. Cannot proceed."); return
        k_vals[hem] = k

    for grad_idx in gradients_to_analyze:
        if grad_idx >= num_available_gradients:
            print(f"Warning: Gradient index {grad_idx+1} is out of bounds. Skipping."); continue

        print(f"\nGenerating consolidated spider plots for Cross-Species Gradient {grad_idx+1}...")
        profiles_max, profiles_min = {}, {}
        vertex_ids_max, vertex_ids_min = {}, {}

        for segment in segment_info_list:
            species = segment['species']
            hem = segment['hem']

            if species not in species_list: continue

            start_row, end_row = segment['start_row'], segment['end_row']
            gradient_values = cross_species_gradients[start_row:end_row, grad_idx]

            if gradient_values.size == 0:
                print(f"Warning: No profiles in segment for {species} {hem}. Skipping."); continue

            idx_max = np.argmax(gradient_values)
            idx_min = np.argmin(gradient_values)

            vtx_id_max, vtx_id_min = -1, -1

            if species != target_k_species: # Assumes this is the downsampled species (e.g., human)
                k_current_hem = k_vals[hem]
                labels_path = os.path.join(downsampled_data_dir, species, f"{species}_{hem}_k{k_current_hem}_labels.npy")
                indices_path = os.path.join(downsampled_data_dir, species, f"{species}_{hem}_k{k_current_hem}_original_indices.npy")

                if not (os.path.exists(labels_path) and os.path.exists(indices_path)):
                    print(f"ERROR: Missing labels or indices for {species} {hem} (k={k_current_hem}). Skipping."); continue

                cluster_labels = np.load(labels_path)
                original_tl_indices = np.load(indices_path)

                indices_for_max_centroid = np.where(cluster_labels == idx_max)[0]
                if indices_for_max_centroid.size > 0:
                    vtx_id_max = original_tl_indices[indices_for_max_centroid[0]]

                indices_for_min_centroid = np.where(cluster_labels == idx_min)[0]
                if indices_for_min_centroid.size > 0:
                    vtx_id_min = original_tl_indices[indices_for_min_centroid[0]]

            else: # This is the target_k_species (e.g., chimp), using original vertices
                mask_path = os.path.join(mask_base_dir, species, f"{species}_{hem}.func.gii")
                if not os.path.exists(mask_path): print(f"ERROR: Mask for {species} {hem} not found."); continue
                try:
                    original_tl_indices = np.where(nib.load(mask_path).darrays[0].data > 0.5)[0]
                    if original_tl_indices.size != gradient_values.shape[0]:
                         print(f"ERROR: {species} TL index count mismatch for {hem}. Skipping."); continue
                    vtx_id_max = original_tl_indices[idx_max]
                    vtx_id_min = original_tl_indices[idx_min]
                except Exception as e: print(f"Error getting {species} TL indices: {e}"); continue

            plot_key = f"{species}_{hem}"
            if vtx_id_max != -1:
                profile = get_vertex_blueprint_profile(species, hem, vtx_id_max, masked_blueprint_dir, n_tracts)
                if profile is not None: profiles_max[plot_key] = profile; vertex_ids_max[plot_key] = vtx_id_max
            if vtx_id_min != -1:
                profile = get_vertex_blueprint_profile(species, hem, vtx_id_min, masked_blueprint_dir, n_tracts)
                if profile is not None: profiles_min[plot_key] = profile; vertex_ids_min[plot_key] = vtx_id_min

        # --- Plotting Logic ---
        plot_spider(profiles_max, vertex_ids_max, grad_idx, "Max", output_dir, tract_names)
        plot_spider(profiles_min, vertex_ids_min, grad_idx, "Min", output_dir, tract_names)

    print("\n--- Consolidated spider plot generation complete ---")

def plot_spider(profiles_data, vertex_ids, grad_idx, extreme_type, output_dir, tract_names):
    """Helper function to generate a single spider plot."""
    if not profiles_data: return

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    title = f"{extreme_type}-Expressing Vertices for Cross-Species Gradient {grad_idx+1}"
    plot_filename = f"consolidated_spider_CS_G{grad_idx+1}_{extreme_type.upper()}_extremes.png"

    all_values = []
    for key, profile in profiles_data.items():
        all_values.extend(profile)
        config = PLOT_CONFIGS[key]
        angles = np.linspace(0, 2 * np.pi, len(profile), endpoint=False).tolist()
        values_plot = np.concatenate((profile, [profile[0]]))
        angles_plot = angles + angles[:1]
        ax.plot(angles_plot, values_plot, linewidth=1.5, linestyle=config["linestyle"], color=config["color"], label=f"{config['label']} (Vtx: {vertex_ids.get(key, 'N/A')})")
        ax.fill(angles_plot, values_plot, config["color"], alpha=0.2)

    angles_for_ticks = np.linspace(0, 2 * np.pi, len(tract_names), endpoint=False).tolist()
    ax.set_xticks(angles_for_ticks)
    ax.set_xticklabels(tract_names, fontsize=8)
    ax.set_ylim(0, max(0.05, np.max(all_values) * 1.1) if all_values else 0.05)
    ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.75), fontsize=9)
    plt.title(title, size=16, y=1.1)
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {extreme_type}-extremes spider plot for Gradient {grad_idx+1}: {plot_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate consolidated spider plots for min/max expressing vertices of cross-species gradients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input Paths
    parser.add_argument('--npz_file', type=str, required=True, help="Path to the cross-species .npz file from script 6.")
    parser.add_argument('--masked_blueprint_dir', type=str, required=True, help="Base directory for masked average blueprints (script 2 output).")
    parser.add_argument('--downsampled_data_dir', type=str, required=True, help="Base directory for downsampled data (script 5 output).")
    parser.add_argument('--mask_base_dir', type=str, required=True, help="Base directory for all species' temporal lobe masks.")

    # Output Path
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output spider plot images.")

    # Configuration
    parser.add_argument('--species', type=str, default=",".join(DEFAULT_SPECIES_FOR_PLOTS), help="Comma-separated list of species to include.")
    parser.add_argument('--hemispheres', type=str, default=",".join(DEFAULT_HEMISPHERES), help="Comma-separated list of hemispheres (L,R).")
    parser.add_argument('--gradients', type=str, default=",".join(map(str, DEFAULT_GRADIENTS_TO_ANALYZE)), help="Comma-separated list of 0-indexed gradients to analyze.")
    parser.add_argument('--target_k_species', type=str, default='chimpanzee', help="The species that defines 'k' for downsampling.")
    parser.add_argument('--n_tracts', type=int, default=DEFAULT_N_TRACTS, help="Expected number of tracts in the blueprints.")
    parser.add_argument('--tract_names', type=str, default=",".join(DEFAULT_TRACT_NAMES), help="Comma-separated list of tract names for plot labels.")

    args = parser.parse_args()

    # Process comma-separated string arguments into lists
    species_to_process = [s.strip().lower() for s in args.species.split(',')]
    hemispheres_to_process = [h.strip().upper() for h in args.hemispheres.split(',')]
    gradients_to_process = [int(g.strip()) for g in args.gradients.split(',')]
    tract_names_list = [name.strip() for name in args.tract_names.split(',')]

    if len(tract_names_list) != args.n_tracts:
        print(f"Error: The number of tract names provided ({len(tract_names_list)}) does not match --n_tracts ({args.n_tracts}).")
        exit(1)

    create_consolidated_spider_plots(
        npz_file_path=args.npz_file,
        masked_blueprint_dir=args.masked_blueprint_dir,
        downsampled_data_dir=args.downsampled_data_dir,
        mask_base_dir=args.mask_base_dir,
        output_dir=args.output_dir,
        species_list=species_to_process,
        hemispheres=hemispheres_to_process,
        gradients_to_analyze=gradients_to_process,
        n_tracts=args.n_tracts,
        tract_names=tract_names_list,
        target_k_species=args.target_k_species.lower()
    )
