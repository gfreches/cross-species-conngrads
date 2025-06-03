#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from numba import njit, prange

# --- Default Configuration Constants ---
# These can be overridden by command-line arguments if necessary.
DEFAULT_N_TRACTS_EXPECTED = 20

# --- Numba JITted Helper Function (eta2 for similarity) ---
@njit(parallel=True, fastmath=True)
def eta2(X):
    """
    Calculates the similarity matrix S using the eta2 formula.
    Each row of X is a profile. S_ij is similarity between profile i and profile j.
    """
    n_profiles = X.shape[0]
    if n_profiles == 0:
        return np.empty((0, 0), dtype=np.float64)
    n_features = X.shape[1]
    S = np.zeros((n_profiles, n_profiles), dtype=np.float64)

    for i in prange(n_profiles):
        for j in range(i, n_profiles):
            mi = (X[i, :] + X[j, :]) / 2.0
            mm = 0.0
            if n_features > 0:
                for k_idx in range(n_features):
                    mm += mi[k_idx]
                mm /= n_features
            else:
                mm = 0.0
            
            ssw_term1 = np.square(X[i, :] - mi)
            ssw_term2 = np.square(X[j, :] - mi)
            ssw = np.sum(ssw_term1 + ssw_term2)

            sst_term1 = np.square(X[i, :] - mm)
            sst_term2 = np.square(X[j, :] - mm)
            sst = np.sum(sst_term1 + sst_term2)

            if sst < 1e-9:
                S[i, j] = 1.0 if ssw < 1e-9 else 0.0
            else:
                S[i, j] = 1.0 - (ssw / sst)
    
    # Symmetrize and set diagonal to 1
    for i in range(n_profiles):
        for j in range(i + 1, n_profiles):
            S[j, i] = S[i, j]
    for i in range(n_profiles):
        S[i, i] = 1.0
    return S

# --- Python Helper Functions ---

def get_temporal_lobe_vertex_count(species, hemisphere, mask_base_dir, mask_pattern):
    """
    Counts the number of vertices in a species' temporal lobe mask.

    Args:
        species (str): Name of the species.
        hemisphere (str): Hemisphere label ('L' or 'R').
        mask_base_dir (str): Base directory for mask files (containing species subfolders).
        mask_pattern (str): Filename pattern for mask files.

    Returns:
        int or None: Number of temporal lobe vertices, or None if an error occurs.
    """
    mask_subdir = os.path.join(mask_base_dir, species)
    mask_filename = mask_pattern.format(species_name=species, hemisphere=hemisphere)
    mask_path = os.path.join(mask_subdir, mask_filename)

    if not os.path.exists(mask_path):
        print(f"ERROR: Target mask file for k determination not found: {mask_path}")
        return None
    try:
        mask_img = nib.load(mask_path)
        if not mask_img.darrays:
            print(f"ERROR: No data arrays in target mask file: {mask_path}")
            return None
        mask_data = mask_img.darrays[0].data
        num_tl_vertices = np.sum(mask_data > 0.5) # Binarize mask
        if num_tl_vertices == 0:
            print(f"ERROR: Target mask {mask_path} is empty.")
            return None
        print(f"  Target species {species} hemisphere {hemisphere} has {num_tl_vertices} temporal lobe vertices (this will be k).")
        return int(num_tl_vertices)
    except Exception as e:
        print(f"Error loading or processing target mask {mask_path}: {e}")
        return None

def downsample_blueprint_with_kmeans(
    source_species_name,
    hemisphere_label,
    source_masked_blueprint_path,
    source_temporal_mask_path, # Mask for the source species/hemisphere
    num_clusters_k,
    n_tracts_expected,
    output_dir,
    output_prefix # Prefix for output files, e.g., "human_L_k1234"
):
    """
    Downsamples a source blueprint to k centroids using k-means clustering.
    Saves centroid profiles, cluster labels, original TL indices, a validation plot,
    and a visual downsampled blueprint .func.gii.

    Args:
        source_species_name (str): Name of the source species.
        hemisphere_label (str): Hemisphere ('L' or 'R').
        source_masked_blueprint_path (str): Path to the masked blueprint (output of Script 2).
        source_temporal_mask_path (str): Path to the mask for the source blueprint.
        num_clusters_k (int): Number of clusters for k-means.
        n_tracts_expected (int): Expected number of features/tracts.
        output_dir (str): Directory to save all outputs for this downsampling run.
        output_prefix (str): Prefix for generating output filenames.
    """
    print(f"\n--- Downsampling {source_species_name} Hemisphere {hemisphere_label} to {num_clusters_k} centroids ---")

    # 1. Load source blueprint and its mask
    if not os.path.exists(source_masked_blueprint_path):
        print(f"ERROR: Source blueprint file not found: {source_masked_blueprint_path}"); return
    if not os.path.exists(source_temporal_mask_path):
        print(f"ERROR: Source mask file not found: {source_temporal_mask_path}"); return
    
    try:
        blueprint_img = nib.load(source_masked_blueprint_path)
        mask_img = nib.load(source_temporal_mask_path)
    except Exception as e:
        print(f"Error loading files for {source_species_name} {hemisphere_label}: {e}"); return

    if not mask_img.darrays:
        print(f"ERROR: No darrays in source mask: {source_temporal_mask_path}"); return
    source_mask_data = mask_img.darrays[0].data
    source_tl_indices = np.where(source_mask_data > 0.5)[0]

    if source_tl_indices.size == 0:
        print(f"ERROR: Source mask {source_temporal_mask_path} is empty for {source_species_name} {hemisphere_label}."); return
    if source_tl_indices.size < num_clusters_k:
        print(f"ERROR: Number of vertices in source mask ({source_tl_indices.size}) "
              f"is less than k ({num_clusters_k}). Cannot run k-means. Skipping.")
        return

    if not blueprint_img.darrays or len(blueprint_img.darrays) != n_tracts_expected:
        print(f"ERROR: Blueprint {source_masked_blueprint_path} has {len(blueprint_img.darrays)} darrays, "
              f"expected {n_tracts_expected}."); return
    
    num_total_source_vertices = blueprint_img.darrays[0].data.shape[0]
    if source_mask_data.shape[0] != num_total_source_vertices:
        print(f"ERROR: Vertex count mismatch! Blueprint: {num_total_source_vertices}, Source Mask: {source_mask_data.shape[0]}"); return
        
    # Stack darrays: result is (num_vertices, num_tracts/features)
    source_bp_all_vertices_stacked = np.array([d.data for d in blueprint_img.darrays]).T 
    source_tl_blueprint_data = source_bp_all_vertices_stacked[source_tl_indices, :]
    print(f"  Extracted {source_tl_blueprint_data.shape[0]} source TL vertices with {source_tl_blueprint_data.shape[1]} features for k-means.")

    # 2. Perform k-means clustering
    print(f"  Performing k-means clustering with k={num_clusters_k}...")
    try:
        kmeans = KMeans(n_clusters=num_clusters_k, init='k-means++', random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(source_tl_blueprint_data) 
        centroid_profiles = kmeans.cluster_centers_ 
        print(f"  K-means complete. Centroid profiles shape: {centroid_profiles.shape}")
        sum_of_centroid_features = np.sum(centroid_profiles, axis=1)
        print(f"  Sum of features for each centroid profile (min/mean/max): "
              f"{np.min(sum_of_centroid_features):.4f} / {np.mean(sum_of_centroid_features):.4f} / {np.max(sum_of_centroid_features):.4f}")
    except Exception as e:
        print(f"  Error during k-means clustering: {e}"); return

    # --- Define output filenames ---
    os.makedirs(output_dir, exist_ok=True)
    centroids_path = os.path.join(output_dir, f"{output_prefix}_centroids.npy")
    labels_path = os.path.join(output_dir, f"{output_prefix}_labels.npy")
    tl_indices_path = os.path.join(output_dir, f"{output_prefix}_tl_indices.npy")
    plot_save_path = os.path.join(output_dir, f"{output_prefix}_similarity_validation.png")
    output_viz_funcgii_path = os.path.join(output_dir, f"{output_prefix}_ds_visualization.func.gii") # Defined here
    
    try:
        np.save(centroids_path, centroid_profiles)
        print(f"  Saved centroid profiles to: {centroids_path}")
        np.save(labels_path, cluster_labels)
        print(f"  Saved cluster labels for TL vertices to: {labels_path}")
        np.save(tl_indices_path, source_tl_indices)
        print(f"  Saved original TL vertex indices to: {tl_indices_path}")
    except Exception as e_save_npy:
        print(f"  Error saving k-means .npy outputs: {e_save_npy}")

    # --- 3. Create downsampled representation for similarity comparison plot ---
    print(f"  Creating downsampled data representation for similarity comparison plot...")
    downsampled_tl_blueprint_data_for_plot = centroid_profiles[cluster_labels, :]

    # --- 4. Compare similarity structure (Original vs. Downsampled for plot) ---
    print(f"  Calculating similarity matrix for original TL data (for plot)...")
    S_original_processed = eta2(source_tl_blueprint_data) 

    print(f"  Calculating similarity matrix for plot's downsampled TL data...")
    S_downsampled_processed_for_plot = eta2(downsampled_tl_blueprint_data_for_plot)

    if S_original_processed.shape[0] > 1: 
        vec_S_original = S_original_processed[np.triu_indices_from(S_original_processed, k=1)]
        vec_S_downsampled_for_plot = S_downsampled_processed_for_plot[np.triu_indices_from(S_downsampled_processed_for_plot, k=1)]

        correlation = np.nan
        if len(vec_S_original) > 1 and len(vec_S_downsampled_for_plot) == len(vec_S_original) and \
           np.std(vec_S_original) > 1e-9 and np.std(vec_S_downsampled_for_plot) > 1e-9:
            correlation, _ = pearsonr(vec_S_original, vec_S_downsampled_for_plot)
            print(f"  Correlation (for plot) between original and centroid-mapped similarity structures: R = {correlation:.4f}")
        elif len(vec_S_original) == len(vec_S_downsampled_for_plot) and len(vec_S_original) > 0:
             correlation = 1.0 if np.allclose(vec_S_original, vec_S_downsampled_for_plot) else 0.0
             print(f"  Similarity structures are identical or too uniform for Pearson R; using allclose: {correlation:.4f}")
        else:
            print("  Could not compute similarity correlation for validation (vector length mismatch or zero variance).")

        plt.figure(figsize=(8, 8))
        num_points_to_plot = min(10000, len(vec_S_original)) 
        sample_indices = np.random.choice(len(vec_S_original), num_points_to_plot, replace=False) if len(vec_S_original) > num_points_to_plot else np.arange(len(vec_S_original))
        
        plt.scatter(vec_S_original[sample_indices], vec_S_downsampled_for_plot[sample_indices], 
                    alpha=0.3, s=10, edgecolor='none')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=0.8, label='Identity line') 
        plt.xlabel("Original Pairwise Similarities (eta2)")
        plt.ylabel("Downsampled Pairwise Similarities (eta2 on Centroid-mapped Data)")
        title_corr = f"\nCorrelation: {correlation:.4f}" if not np.isnan(correlation) else ""
        plt.title(f"Similarity Preservation by K-means Downsampling\n{source_species_name} - {hemisphere_label} (k={num_clusters_k}){title_corr}")
        plt.grid(True, linestyle='--', alpha=0.7); plt.axis('square'); plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
        
        try:
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved similarity preservation scatter plot to: {plot_save_path}")
        except Exception as e_plot:
            print(f"  Error saving scatter plot: {e_plot}")
        plt.close()
    else:
        print("  Not enough vertices for pairwise similarity validation plot.")

    # --- 5. Create the visual downsampled blueprint .func.gii ---
    output_data_gii_for_vis = np.zeros((n_tracts_expected, num_total_source_vertices), dtype=np.float32)
    # Assign centroid profiles to the locations of the original temporal lobe vertices
    for i, original_vertex_idx in enumerate(source_tl_indices):
        assigned_cluster_id = cluster_labels[i]
        assigned_centroid_profile = centroid_profiles[assigned_cluster_id, :] 
        output_data_gii_for_vis[:, original_vertex_idx] = assigned_centroid_profile
    # ***** CORRECTED PRINT STATEMENT BELOW *****
    print(f"  Generated data for visual downsampled blueprint .func.gii (to be saved as {os.path.basename(output_viz_funcgii_path)}).")

    output_gifti_darrays = []
    structure_name = 'CortexLeft' if hemisphere_label.upper() == 'L' else 'CortexRight'
    for i in range(n_tracts_expected): 
        darray_meta = nib.gifti.GiftiMetaData()
        darray_meta['Name'] = f"Tract_{i+1}_ds_k{num_clusters_k}"
        darray_meta['AnatomicalStructurePrimary'] = structure_name
        
        darray = nib.gifti.GiftiDataArray(
            data=output_data_gii_for_vis[i, :].astype(np.float32), 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes.code['NIFTI_TYPE_FLOAT32'],
            meta=darray_meta)
        output_gifti_darrays.append(darray)
    
    output_gifti_image_meta = nib.gifti.GiftiMetaData()
    desc_text = f'Downsampled blueprint (k={num_clusters_k}) for {source_species_name} {hemisphere_label}'
    output_gifti_image_meta['Description'] = desc_text
    output_gifti_image_meta['AnatomicalStructurePrimary'] = structure_name
    
    output_gii_img = nib.gifti.GiftiImage(darrays=output_gifti_darrays, meta=output_gifti_image_meta)
    
    try:
        nib.save(output_gii_img, output_viz_funcgii_path)
        print(f"  Successfully saved visual downsampled blueprint to: {output_viz_funcgii_path}")
    except Exception as e:
        print(f"  Error saving visual downsampled GIFTI image: {e}")

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample average connectivity blueprints using k-means clustering. "
                    "The number of clusters 'k' is determined by the temporal lobe "
                    "vertex count of a specified target species.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_masked_blueprint_dir', type=str, required=True,
                        help="Base directory of masked average blueprints (output of Script 2), "
                             "containing species subfolders.")
    parser.add_argument('--input_mask_dir', type=str, required=True,
                        help="Base directory for temporal lobe mask .func.gii files, "
                             "containing species subfolders.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Base directory where all outputs of this script will be saved.")
    parser.add_argument('--source_species_list', type=str, required=True,
                        help='Comma-separated list of source species names to downsample.')
    parser.add_argument('--target_species_for_k', type=str, required=True,
                        help='Name of the target species whose temporal lobe vertex count will define k.')
    parser.add_argument('--hemispheres', type=str, default="L,R",
                        help='Comma-separated list of hemisphere labels to process (e.g., "L,R").')
    
    parser.add_argument('--masked_blueprint_pattern', type=str,
                        default="average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii",
                        help='Filename pattern for masked blueprints from Script 2. Placeholders: {species_name}, {hemisphere}.')
    parser.add_argument('--mask_pattern', type=str, required=True,
                        help='Filename pattern for temporal lobe masks. Placeholders: {species_name}, {hemisphere}.')
    parser.add_argument('--n_tracts_expected', type=int, default=DEFAULT_N_TRACTS_EXPECTED,
                        help="Expected number of tracts/features in the blueprint data.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created main output directory: {args.output_dir}")

    k_values_per_hem = {}
    print(f"\n--- Determining k from {args.target_species_for_k} masks ---")
    hemisphere_list_for_k = [h.strip() for h in args.hemispheres.split(',')]
    for hem in hemisphere_list_for_k:
        k = get_temporal_lobe_vertex_count(args.target_species_for_k, hem, args.input_mask_dir, args.mask_pattern)
        if k is None:
            print(f"FATAL: Could not determine k for hemisphere {hem} from {args.target_species_for_k} mask. Exiting.")
            exit(1)
        k_values_per_hem[hem] = k

    source_species_to_process = [s.strip() for s in args.source_species_list.split(',')]
    hemispheres_to_process = [h.strip() for h in args.hemispheres.split(',')]

    for species in source_species_to_process:
        species_masked_bp_input_base = os.path.join(args.input_masked_blueprint_dir, species)
        species_mask_input_base = os.path.join(args.input_mask_dir, species) # Assuming masks also in species subfolders
        species_downsampled_output_base = os.path.join(args.output_dir, species)
        
        if not os.path.exists(species_downsampled_output_base):
            os.makedirs(species_downsampled_output_base, exist_ok=True)
            print(f"Created species output directory: {species_downsampled_output_base}")

        for hem in hemispheres_to_process:
            num_clusters_for_hem = k_values_per_hem[hem]
            
            source_blueprint_filename = args.masked_blueprint_pattern.format(species_name=species, hemisphere=hem)
            source_blueprint_path = os.path.join(species_masked_bp_input_base, source_blueprint_filename)
            
            source_mask_filename = args.mask_pattern.format(species_name=species, hemisphere=hem)
            source_mask_path = os.path.join(species_mask_input_base, source_mask_filename)
            
            output_file_prefix = f"{species}_{hem}_k{num_clusters_for_hem}"

            downsample_blueprint_with_kmeans(
                source_species_name=species,
                hemisphere_label=hem,
                source_masked_blueprint_path=source_blueprint_path,
                source_temporal_mask_path=source_mask_path,
                num_clusters_k=num_clusters_for_hem,
                n_tracts_expected=args.n_tracts_expected,
                output_dir=species_downsampled_output_base,
                output_prefix=output_file_prefix
            )
                    
    print("\n--- All downsampling processing complete ---")