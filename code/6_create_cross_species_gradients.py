#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
import numpy as np
from sklearn.manifold import SpectralEmbedding
from scipy.sparse.csgraph import connected_components, laplacian as csgraph_laplacian
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from numba import njit, prange, objmode
import scipy.sparse

# --- Default Configuration Constants ---
DEFAULT_MAX_GRADIENTS_TO_TEST = 10
DEFAULT_MAX_K_SEARCH_FOR_KNN = 200
DEFAULT_N_NEIGHBORS_FALLBACK = 30
DEFAULT_MIN_GAIN_FOR_DIM_SELECTION = 0.1
DEFAULT_NUM_GRADIENTS_TO_SAVE = 10

# --- Core Numba-JITted Helper Functions ---
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
            ssw = np.sum(np.square(X[i, :] - mi) + np.square(X[j, :] - mi))
            sst = np.sum(np.square(X[i, :] - mm) + np.square(X[j, :] - mm))
            S[i, j] = 1.0 - (ssw / sst) if sst >= 1e-9 else (1.0 if ssw < 1e-9 else 0.0)
    # Symmetrize and set diagonal to 1
    for i in range(n_profiles):
        for j in range(i + 1, n_profiles):
            S[j, i] = S[i, j]
    for i in range(n_profiles):
        S[i, i] = 1.0
    return S

@njit(fastmath=True)
def build_knn_graph_for_k(k_to_test, similarity_matrix_input):
    """Builds a k-NN graph for a specific k."""
    n_v = similarity_matrix_input.shape[0]
    _current_w = np.zeros_like(similarity_matrix_input)
    if n_v == 0:
        return _current_w
    for i in range(n_v):
        _sim_i = similarity_matrix_input[i, :].copy()
        _sim_i[i] = -np.inf # Exclude self
        _actual_k = min(k_to_test, n_v - 1) # k cannot be more than n_v-1
        if _actual_k <= 0:
            continue
        _neighbor_indices = np.argsort(_sim_i)[-_actual_k:]
        for _neighbor_idx in _neighbor_indices:
            _current_w[i, _neighbor_idx] = similarity_matrix_input[i, _neighbor_idx]
    return np.maximum(_current_w, _current_w.T) # Symmetrize

@njit(fastmath=True)
def calculate_eigenvalues_for_gradients_numba(embedding_matrix, laplacian_dense):
    """Calculates eigenvalues (Rayleigh quotients) for given eigenvectors and Laplacian."""
    n_gradients = embedding_matrix.shape[1]
    eigenvalues = np.zeros(n_gradients, dtype=laplacian_dense.dtype)
    for i in range(n_gradients):
        v_i = np.ascontiguousarray(embedding_matrix[:, i]) # Ensure contiguity
        L_v_i_temp = np.dot(laplacian_dense, v_i)
        eigenvalues[i] = np.dot(v_i, L_v_i_temp)
    return eigenvalues

# --- Python Helper Functions ---

def find_adaptive_knn_graph_binary_search(similarity_matrix, max_k, default_k):
    """Orchestrates binary search for adaptive k to ensure a connected graph."""
    n_vertices = similarity_matrix.shape[0]
    if n_vertices <= 1: # Cannot form a graph
        return np.zeros_like(similarity_matrix)
        
    print(f"    Binary searching for adaptive k (up to {max_k}) for {n_vertices} vertices...")
    low = 1
    high = min(max_k, n_vertices - 1) # k cannot be more than n-1
    best_k_graph = None
    best_k_val = -1
    
    while low <= high:
        mid = max(1, (low + high) // 2) # Ensure mid is at least 1
        current_w_for_mid = build_knn_graph_for_k(mid, similarity_matrix)
        with objmode(n_c='intp'): # Temporarily switch to object mode for SciPy function
            n_c = connected_components(current_w_for_mid, directed=False, connection='weak')[0]
        if n_c == 1: # Found a k that results in a connected graph
            best_k_graph = current_w_for_mid
            best_k_val = mid
            high = mid - 1 # Try smaller k
        else:
            low = mid + 1 # Need larger k
            
    if best_k_graph is not None:
        print(f"    Found connected graph with adaptive k = {best_k_val}")
        return best_k_graph
    
    # Fallback if binary search fails
    print(f"    Warning: Binary search did not find connected graph. Using fallback k={default_k}.")
    actual_default_k = max(1, min(default_k, n_vertices - 1 if n_vertices > 1 else 1))
    if n_vertices > 1 and actual_default_k > 0:
        fallback_w = build_knn_graph_for_k(actual_default_k, similarity_matrix)
        with objmode(n_c_fallback='intp'):
            n_c_fallback = connected_components(fallback_w, directed=False, connection='weak')[0]
        if n_c_fallback > 1:
            print(f"    Warning: Fallback k={actual_default_k} also not connected ({n_c_fallback} components).")
        return fallback_w
    return np.zeros_like(similarity_matrix) # Return empty graph if still not possible

def suggest_dimensionality_with_penalty(dimensions, scores, min_absolute_improvement):
    """Suggests optimal dimensionality based on diminishing returns or score decrease."""
    if not dimensions or not scores or len(dimensions) != len(scores):
        print("Warning (Penalty Suggestion): Invalid input for dimensionality suggestion."); return None
    
    valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
    if not valid_indices:
        print("Warning (Penalty Suggestion): No valid (non-NaN) scores provided."); return None
    
    dims_filtered = [dimensions[i] for i in valid_indices]
    scores_filtered = [scores[i] for i in valid_indices]
    
    if not scores_filtered: # Should be caught by previous check
        print("Warning (Penalty Suggestion): No valid scores after NaN filter."); return None
    if len(dims_filtered) == 1:
        print(f"Suggested (Penalty): {dims_filtered[0]} (Score: {scores_filtered[0]:.4f})")
        return dims_filtered[0]
        
    suggested_dim = dims_filtered[0] # Start with the first valid dimension
    for i in range(1, len(scores_filtered)):
        gain = scores_filtered[i] - scores_filtered[i-1]
        if scores_filtered[i] < (scores_filtered[i-1] - 1e-5): # Score decreased
            suggested_dim = dims_filtered[i-1]
            print(f"    (Penalty Suggestion) Score decreased from {scores_filtered[i-1]:.4f} to {scores_filtered[i]:.4f}. Suggesting: {suggested_dim}")
            break
        if gain < min_absolute_improvement: # Gain too small
            suggested_dim = dims_filtered[i-1]
            print(f"    (Penalty Suggestion) Gain {gain:.4f} from dim {dims_filtered[i-1]} to {dims_filtered[i]} < threshold {min_absolute_improvement}. Suggesting: {suggested_dim}")
            break
        suggested_dim = dims_filtered[i] # Otherwise, current dimension is better or equally good
    else: # Loop completed without break (all gains met threshold or only increases)
        print(f"    (Penalty Suggestion) All gains >= threshold ({min_absolute_improvement}) or no score decrease up to dim {dims_filtered[-1]}. Suggesting last tested: {suggested_dim}")

    # Report the score for the suggested dimension from the original list
    if suggested_dim in dimensions:
        final_score_idx = dimensions.index(suggested_dim)
        # Ensure index is valid for scores list which might contain NaNs originally
        final_score = scores[final_score_idx] if final_score_idx < len(scores) and not np.isnan(scores[final_score_idx]) else np.nan
        print(f"Suggested optimal dimensionality (penalty {min_absolute_improvement}): {suggested_dim} (Score: {final_score:.4f})")
    else: # This case should ideally not be reached if logic is correct
        print(f"Error: Suggested dimension {suggested_dim} (from filtered list) not in original dimensions. Defaulting to first valid dimension.")
        return dims_filtered[0] if dims_filtered else None
    return suggested_dim

def get_k_from_target_species_mask(species_name, hemisphere, mask_base_dir, mask_pattern):
    """Gets the number of TL vertices from a target species' mask to use as k."""
    mask_subdir = os.path.join(mask_base_dir, species_name)
    mask_filename = mask_pattern.format(species_name=species_name, hemisphere=hemisphere)
    mask_path = os.path.join(mask_subdir, mask_filename)
    
    if not os.path.exists(mask_path):
        print(f"ERROR: Target species mask for k determination not found: {mask_path}")
        return None
    try:
        mask_img = nib.load(mask_path)
        if not mask_img.darrays:
            print(f"ERROR: No darrays in target species mask: {mask_path}")
            return None
        k_val = int(np.sum(mask_img.darrays[0].data > 0.5)) # Binarize mask
        if k_val == 0:
            print(f"ERROR: Target species mask {mask_path} is empty.")
            return None
        print(f"    Determined k from {species_name} {hemisphere} = {k_val}")
        return k_val
    except Exception as e:
        print(f"Error loading target species mask {mask_path}: {e}")
        return None

def load_species_hem_data_for_cross_species_lle(
    species_to_load, 
    hem, 
    k_for_downsampling, 
    target_k_species_name, 
    target_species_bp_dir, 
    target_species_bp_pattern, 
    other_species_downsampled_dir, 
    downsampled_centroid_pattern, 
    mask_dir, 
    mask_pattern
):
    """
    Loads data for a given species and hemisphere for cross-species LLE.
    - If species_to_load is the target_k_species_name: loads original masked TL blueprint data.
    - For other species: loads downsampled centroid profiles and their original TL labels.
    Returns profiles, cluster_labels_for_remapping, original_tl_indices, num_total_surface_vertices.
    """
    print(f"Loading data for cross-species LLE: {species_to_load} {hem}...")
    
    species_mask_subdir = os.path.join(mask_dir, species_to_load)
    mask_filename = mask_pattern.format(species_name=species_to_load, hemisphere=hem)
    mask_path = os.path.join(species_mask_subdir, mask_filename)

    if not os.path.exists(mask_path):
        print(f"ERROR: Mask file not found for {species_to_load} {hem}: {mask_path}"); return None, None, None, None
    try:
        mask_img = nib.load(mask_path)
        if not mask_img.darrays:
            print(f"ERROR: No darrays in mask {mask_path}"); return None, None, None, None
        mask_data = mask_img.darrays[0].data
        num_total_surface_vertices = mask_data.shape[0]
        original_tl_indices = np.where(mask_data > 0.5)[0]
        if original_tl_indices.size == 0:
            print(f"ERROR: Original TL mask for {species_to_load} {hem} is empty."); return None, None, None, None
    except Exception as e:
        print(f"Error loading mask for {species_to_load} {hem}: {e}"); return None, None, None, None

    profiles_for_lle = None
    cluster_labels_for_remapping = None 

    if species_to_load == target_k_species_name:
        target_bp_subdir = os.path.join(target_species_bp_dir, species_to_load)
        blueprint_filename = target_species_bp_pattern.format(species_name=species_to_load, hemisphere=hem)
        blueprint_path = os.path.join(target_bp_subdir, blueprint_filename)
        
        if not os.path.exists(blueprint_path):
            print(f"ERROR: Target species ({species_to_load}) blueprint not found: {blueprint_path}"); return None, None, None, None
        try:
            blueprint_img = nib.load(blueprint_path)
            if not blueprint_img.darrays:
                print(f"ERROR: No darrays in target species ({species_to_load}) blueprint {blueprint_path}"); return None, None, None, None
            bp_data_all_vertices = np.array([d.data for d in blueprint_img.darrays]).T # (Vertices, Tracts)
            profiles_for_lle = bp_data_all_vertices[original_tl_indices, :]
            cluster_labels_for_remapping = np.arange(profiles_for_lle.shape[0]) # Each original vertex is its own "cluster"
            print(f"    Loaded {species_to_load} {hem}: {profiles_for_lle.shape[0]} original TL vertex profiles.")
        except Exception as e:
            print(f"Error loading target species ({species_to_load}) blueprint {blueprint_path}: {e}"); return None, None, None, None
    else: # For other (downsampled) species
        if k_for_downsampling is None:
            print(f"ERROR: k_for_downsampling (derived from {target_k_species_name}) not provided for {species_to_load} {hem}."); return None, None, None, None
        
        species_downsampled_subdir = os.path.join(other_species_downsampled_dir, species_to_load)
        centroids_filename = downsampled_centroid_pattern.format(species_name=species_to_load, hemisphere=hem, k_val=k_for_downsampling)
        centroids_path = os.path.join(species_downsampled_subdir, centroids_filename)
        
        labels_pattern = downsampled_centroid_pattern.replace("_centroids.npy", "_labels.npy")
        labels_filename = labels_pattern.format(species_name=species_to_load, hemisphere=hem, k_val=k_for_downsampling)
        labels_path = os.path.join(species_downsampled_subdir, labels_filename)

        if not os.path.exists(centroids_path):
            print(f"ERROR: Centroid file not found for {species_to_load} {hem}: {centroids_path}"); return None, None, None, None
        if not os.path.exists(labels_path):
            print(f"ERROR: Labels file not found for {species_to_load} {hem}: {labels_path}"); return None, None, None, None
        
        try:
            profiles_for_lle = np.load(centroids_path)
            cluster_labels_for_remapping = np.load(labels_path)
            print(f"    Loaded {species_to_load} {hem}: {profiles_for_lle.shape[0]} centroid profiles (k={k_for_downsampling}).")
            if profiles_for_lle.shape[0] != k_for_downsampling:
                 print(f"    WARNING: Loaded centroids count ({profiles_for_lle.shape[0]}) "
                       f"does not match expected k ({k_for_downsampling}) for {species_to_load} {hem}.")
            if len(cluster_labels_for_remapping) != original_tl_indices.size:
                print(f"    ERROR: Mismatch between number of cluster labels ({len(cluster_labels_for_remapping)}) "
                      f"and number of original TL indices ({original_tl_indices.size}) for {species_to_load} {hem}.")
                return None, None, None, None
        except Exception as e:
            print(f"Error loading downsampled data for {species_to_load} {hem}: {e}"); return None, None, None, None
            
    if profiles_for_lle is None: # Should be caught by earlier specific errors
        print(f"ERROR: Failed to load profiles for {species_to_load} {hem}"); return None, None, None, None
        
    return profiles_for_lle, cluster_labels_for_remapping, original_tl_indices, num_total_surface_vertices

def _compute_affinity_and_knn_graph(tl_blueprint_data, id_prefix_for_files, output_dir_for_intermediates, max_k_knn, default_k_knn):
    """Computes similarity matrix and k-NN graph, with checkpointing."""
    s_full_path = os.path.join(output_dir_for_intermediates, f"S_full_processed_{id_prefix_for_files}.npy")
    knn_w_path = os.path.join(output_dir_for_intermediates, f"knn_graph_W_{id_prefix_for_files}.npy")

    # Checkpointing logic
    if os.path.exists(s_full_path) and os.path.exists(knn_w_path):
        try:
            S_full_processed = np.load(s_full_path)
            knn_graph_W = np.load(knn_w_path)
            if S_full_processed.shape[0] == knn_graph_W.shape[0] and \
               S_full_processed.shape[0] == tl_blueprint_data.shape[0]:
                print(f"    Loaded S_full & kNN graph for {id_prefix_for_files} from files.")
                return S_full_processed, knn_graph_W
            else:
                print(f"    Warning: Shape mismatch in loaded intermediates for {id_prefix_for_files}. Recomputing.")
        except Exception as e:
            print(f"    Warning: Error loading intermediates for {id_prefix_for_files}, recomputing. Error: {e}")

    print(f"    Calculating full similarity (S_full) for {id_prefix_for_files}...")
    S_full_raw = eta2(tl_blueprint_data)
    S_full_processed = np.clip(S_full_raw, 0, None) # Ensure non-negative
    if S_full_processed.shape[0] > 0:
        np.fill_diagonal(S_full_processed, 1.0)
    
    try:
        os.makedirs(output_dir_for_intermediates, exist_ok=True)
        np.save(s_full_path, S_full_processed)
        print(f"    Saved S_full to: {s_full_path}")
    except Exception as e:
        print(f"    Error saving S_full for {id_prefix_for_files}: {e}")

    if np.any(np.isnan(S_full_processed)) or np.any(np.isinf(S_full_processed)):
        print(f"ERROR: S_full contains NaNs/Infs for {id_prefix_for_files}."); return None, None
    if S_full_processed.shape[0] <= 1:
        print(f"ERROR: Not enough vertices in S_full for {id_prefix_for_files} to build graph."); return None, None
        
    print(f"    Constructing k-NN graph for {id_prefix_for_files}...")
    knn_graph_W = find_adaptive_knn_graph_binary_search(S_full_processed, max_k_knn, default_k_knn)
    if knn_graph_W is None or knn_graph_W.shape[0] == 0 or np.all(knn_graph_W == 0): # Check if graph is empty
        print(f"ERROR: k-NN graph construction failed or resulted in empty graph for {id_prefix_for_files}."); return None, None
        
    try:
        np.save(knn_w_path, knn_graph_W)
        print(f"    Saved kNN graph to: {knn_w_path}")
    except Exception as e:
        print(f"    Error saving kNN graph for {id_prefix_for_files}: {e}")
    return S_full_processed, knn_graph_W

def _perform_embedding_and_eigval_calc(knn_graph_W, max_dims_to_compute, id_prefix_for_files, output_dir_for_intermediates):
    """Performs spectral embedding and calculates eigenvalues, with checkpointing."""
    embedding_path = os.path.join(output_dir_for_intermediates, f"embedding_{id_prefix_for_files}.npy")
    eigenvalues_path = os.path.join(output_dir_for_intermediates, f"eigenvalues_{id_prefix_for_files}.npy")
    
    # n_components must be < n_samples for spectral embedding
    actual_max_dims = min(max_dims_to_compute, knn_graph_W.shape[0] - 1 if knn_graph_W.shape[0] > 1 else 1)
    if actual_max_dims <= 0:
        print(f"ERROR: Not enough samples ({knn_graph_W.shape[0]}) for gradient computation for {id_prefix_for_files}. Need at least 2 samples."); return None, None, 0

    if os.path.exists(embedding_path) and os.path.exists(eigenvalues_path):
        try:
            all_grads = np.load(embedding_path)
            eigvals = np.load(eigenvalues_path)
            # Check if loaded data has at least the number of dimensions we need now
            if all_grads.ndim == 2 and all_grads.shape[1] >= actual_max_dims and \
               eigvals.ndim == 1 and len(eigvals) >= actual_max_dims:
                print(f"    Loaded embedding & eigenvalues for {id_prefix_for_files} from files. Using first {actual_max_dims} dimensions.")
                return all_grads[:, :actual_max_dims], eigvals[:actual_max_dims], actual_max_dims
            else:
                print(f"    Warning: Loaded embedding/eigenvalue shape mismatch or insufficient dimensions for {id_prefix_for_files}. Required {actual_max_dims} dims. Recomputing.")
        except Exception as e:
            print(f"    Warning: Error loading embedding/eigenvalues for {id_prefix_for_files}, recomputing. Error: {e}")
            
    print(f"    Performing Spectral Embedding for {actual_max_dims} gradients for {id_prefix_for_files}...")
    try:
        L_sparse = csgraph_laplacian(knn_graph_W, normed=True)
        L_dense = L_sparse.toarray() if scipy.sparse.issparse(L_sparse) else L_sparse # For eigenvalue calculation
        
        spec_embed = SpectralEmbedding(
            n_components=actual_max_dims, 
            affinity='precomputed', # knn_graph_W is the affinity matrix
            random_state=42, 
            n_jobs=-1, 
            eigen_solver='lobpcg'
        )
        all_grads = spec_embed.fit_transform(knn_graph_W)
        eigvals = calculate_eigenvalues_for_gradients_numba(all_grads, L_dense) # Use L_dense here
        print(f"    Computed Eigenvalues for {id_prefix_for_files}: {eigvals}")
        
        try:
            os.makedirs(output_dir_for_intermediates, exist_ok=True)
            np.save(embedding_path, all_grads)
            np.save(eigenvalues_path, eigvals)
            print("    Saved embedding & eigenvalues.")
        except Exception as e:
            print(f"    Error saving embedding/eigenvalues for {id_prefix_for_files}: {e}")
        return all_grads, eigvals, actual_max_dims
    except Exception as e:
        print(f"    Error in Spectral Embedding or Eigenvalue calculation for {id_prefix_for_files}: {e}")
        return None, None, 0

def _evaluate_reconstruction_and_plot(all_gradients_embedding, S_full_processed, actual_max_dims, output_plot_path, plot_title_suffix):
    """Evaluates reconstruction quality and saves a plot."""
    original_S_vector = S_full_processed[np.triu_indices_from(S_full_processed, k=1)]
    recon_corrs = []
    dimensions_tested = list(range(1, actual_max_dims + 1))

    for d_val in dimensions_tested:
        embed_Y_d = all_gradients_embedding[:, :d_val]
        if embed_Y_d.shape[1] == 0: # Should not happen if actual_max_dims >=1
            recon_corrs.append(np.nan); continue
        
        S_embed_raw = eta2(embed_Y_d)
        S_embed_proc = np.clip(S_embed_raw, 0, None)
        if S_embed_proc.shape[0] > 0: np.fill_diagonal(S_embed_proc, 1.0)
        S_embed_vec = S_embed_proc[np.triu_indices_from(S_embed_proc, k=1)]
        
        correlation = np.nan
        if len(original_S_vector) > 1 and len(S_embed_vec) == len(original_S_vector) and \
           np.std(original_S_vector) > 1e-9 and np.std(S_embed_vec) > 1e-9:
            correlation, _ = pearsonr(original_S_vector, S_embed_vec)
        elif len(original_S_vector) == len(S_embed_vec) and len(original_S_vector) > 0:
            correlation = 1.0 if np.allclose(original_S_vector, S_embed_vec) else 0.0
        recon_corrs.append(correlation)
            
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_tested, recon_corrs, marker='o', linestyle='-')
    plt.xlabel("Number of Dimensions (Gradients)")
    plt.ylabel("Pearson Corr (Original Similarity vs. Embedded Similarity)")
    plt.title(f"Dimensionality Evaluation - {plot_title_suffix}")
    plt.xticks(dimensions_tested)
    plt.grid(True, linestyle='--', alpha=0.7)
    valid_corrs = [c for c in recon_corrs if not np.isnan(c)]
    min_y_lim = min(0, np.min(valid_corrs) if valid_corrs else 0) - 0.05
    plt.ylim(min_y_lim, 1.05)
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved dimensionality evaluation plot to: {output_plot_path}")
    return dimensions_tested, recon_corrs

def _save_cross_species_gradients(
    all_gradients_embedding_cross_species,
    segment_info_list, 
    computed_eigenvalues_cross_species,
    num_gradients_saved, 
    output_parent_dir,
    target_k_species_name 
):
    """Splits cross-species gradients and remaps them to each original species' surface."""
    print("\nSplitting and remapping cross-species gradients to original surfaces...")
    for seg_info in segment_info_list:
        species = seg_info['species']
        hem = seg_info['hem']
        start_row, end_row = seg_info['start_row_in_concat'], seg_info['end_row_in_concat']
        original_tl_indices = seg_info['original_tl_indices']
        num_total_surface_verts = seg_info['num_total_surface_verts']
        cluster_labels_for_remapping = seg_info['cluster_labels_for_remapping']

        # Gradients corresponding to this segment's input profiles
        gradient_values_for_segment_profiles = all_gradients_embedding_cross_species[start_row:end_row, :num_gradients_saved]

        output_gifti_darrays = []
        structure_name = 'CortexLeft' if hem.upper() == 'L' else 'CortexRight'

        for i in range(num_gradients_saved):
            gradient_map_full = np.zeros(num_total_surface_verts, dtype=np.float32)
            
            if i < gradient_values_for_segment_profiles.shape[1]:
                current_grad_column = gradient_values_for_segment_profiles[:, i]
                
                if cluster_labels_for_remapping is not None and \
                   len(cluster_labels_for_remapping) == len(original_tl_indices):
                    # Max label value should be less than the number of profiles for this segment
                    max_label = np.max(cluster_labels_for_remapping) if cluster_labels_for_remapping.size > 0 else -1
                    if current_grad_column.shape[0] > max_label :
                         mapped_values_to_tl_verts = current_grad_column[cluster_labels_for_remapping]
                         gradient_map_full[original_tl_indices] = mapped_values_to_tl_verts
                    else:
                        print(f"    WARNING: Gradient value count ({current_grad_column.shape[0]}) "
                              f"insufficient for max cluster label ({max_label}) "
                              f"for {species} {hem}, grad {i+1}. Map may be incorrect/empty.")
                else:
                     print(f"    WARNING: Missing or mismatched cluster_labels_for_remapping for {species} {hem}, grad {i+1}. Map will be empty.")
            else:
                print(f"    WARNING: Gradient index {i+1} out of bounds for available gradients for {species} {hem}. Map will be empty.")
            
            darray_meta = nib.gifti.GiftiMetaData()
            map_name = f'CrossSpecies_Grad_{i+1}'
            if computed_eigenvalues_cross_species is not None and i < len(computed_eigenvalues_cross_species):
                map_name = f'CrossSpecies_Grad_{i+1}_eig_{computed_eigenvalues_cross_species[i]:.4e}'
            darray_meta['Name'] = map_name
            darray_meta['AnatomicalStructurePrimary'] = structure_name
            
            darray = nib.gifti.GiftiDataArray(
                data=gradient_map_full.astype(np.float32),
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes.code['NIFTI_TYPE_FLOAT32'],
                meta=darray_meta
            )
            output_gifti_darrays.append(darray)

        output_gifti_image_meta = nib.gifti.GiftiMetaData()
        desc_text = (f"{species.capitalize()} {hem} from Cross-Species (k based on {target_k_species_name}) - "
                     f"{num_gradients_saved} Gradients")
        output_gifti_image_meta['Description'] = desc_text
        output_gifti_image_meta['AnatomicalStructurePrimary'] = structure_name
        
        output_gii_img = nib.gifti.GiftiImage(darrays=output_gifti_darrays, meta=output_gifti_image_meta)
        
        # Save remapped gradients into species-specific subdirectories
        species_output_dir_for_grads = os.path.join(output_parent_dir, species, "cross_species_gradients_remapped")
        os.makedirs(species_output_dir_for_grads, exist_ok=True)
        # Filename includes target_k_species for clarity on how k was derived
        output_gii_path = os.path.join(species_output_dir_for_grads, 
                                       f"{species}_{hem}_from_cs_gradients_k_{target_k_species_name}.func.gii")
        
        try:
            nib.save(output_gii_img, output_gii_path)
            print(f"    Successfully saved cross-species gradients for {species} {hem} to: {output_gii_path}")
        except Exception as e:
            print(f"    Error saving remapped GIFTI for {species} {hem}: {e}")

def run_cross_species_lle(args):
    """Main function to orchestrate the cross-species LLE pipeline."""
    print(f"\n=== Running Cross-Species Combined LLE for {args.species_list_for_lle} using k from {args.target_k_species} ===")
    os.makedirs(args.output_dir, exist_ok=True)

    all_profiles_for_lle_concat = []
    segment_info_list = [] 
    current_row_offset = 0

    k_vals_from_target = {}
    for hem in args.hemispheres_to_process:
        k = get_k_from_target_species_mask(args.target_k_species, hem, args.mask_dir, args.mask_pattern)
        if k is None:
            print(f"FATAL: Could not get k from {args.target_k_species} {hem} mask. Exiting."); return
        k_vals_from_target[hem] = k

    for species_to_load_current in args.species_for_lle:
        for hem in args.hemispheres_to_process: 
            k_for_current_hem = k_vals_from_target[hem] # This k is used for loading downsampled data
            
            profiles, cluster_labels, original_tl_indices, num_total_verts = \
                load_species_hem_data_for_cross_species_lle(
                    species_to_load_current, hem, k_for_current_hem,
                    args.target_k_species, 
                    args.target_species_bp_dir, 
                    args.target_species_bp_pattern, 
                    args.other_species_downsampled_dir,
                    args.downsampled_centroid_pattern,
                    args.mask_dir,
                    args.mask_pattern
                )
            if profiles is None:
                print(f"FATAL: Load failed for {species_to_load_current} {hem}. Cannot proceed."); return
            
            all_profiles_for_lle_concat.append(profiles)
            segment_info_list.append({
                'species': species_to_load_current, 'hem': hem, 
                'profiles_for_lle_shape': profiles.shape, 
                'cluster_labels_for_remapping': cluster_labels,
                'original_tl_indices': original_tl_indices, 
                'num_total_surface_verts': num_total_verts, 
                'k_val_this_hem': k_for_current_hem, # k that was used for this segment's downsampling (if applicable)
                'start_row_in_concat': current_row_offset, 
                'end_row_in_concat': current_row_offset + profiles.shape[0]
            })
            current_row_offset += profiles.shape[0]

    if not all_profiles_for_lle_concat:
        print("ERROR: No data loaded for combined LLE. Exiting."); return
        
    concatenated_blueprint_profiles = np.vstack(all_profiles_for_lle_concat)
    print(f"\nTotal concatenated profiles matrix shape for LLE: {concatenated_blueprint_profiles.shape}")

    id_prefix_for_cs_run = f"{'_'.join(args.species_for_lle)}_CrossSpecies_kRef_{args.target_k_species}"
    intermediates_dir = os.path.join(args.output_dir, "intermediates", id_prefix_for_cs_run)
    os.makedirs(intermediates_dir, exist_ok=True)

    S_full_cs, knn_graph_W_cs = _compute_affinity_and_knn_graph(
        concatenated_blueprint_profiles, 
        id_prefix_for_cs_run,
        intermediates_dir,
        args.max_k_knn, args.default_k_knn
    )
    if knn_graph_W_cs is None:
        print("Cross-species k-NN graph construction failed."); return

    all_grads_cs, eigvals_cs, actual_max_dims_cs = _perform_embedding_and_eigval_calc(
        knn_graph_W_cs, 
        args.max_gradients, 
        id_prefix_for_cs_run,
        intermediates_dir
    )
    if all_grads_cs is None:
        print("Cross-species spectral embedding failed."); return

    plot_path_cs = os.path.join(intermediates_dir, f"dim_eval_plot_{id_prefix_for_cs_run}.png")
    dims_tested_cs, recon_scores_cs = _evaluate_reconstruction_and_plot(
        all_grads_cs, S_full_cs, actual_max_dims_cs, plot_path_cs, id_prefix_for_cs_run
    )
    
    suggested_d_cs = actual_max_dims_cs 
    if dims_tested_cs and recon_scores_cs:
        sugg_d = suggest_dimensionality_with_penalty(dims_tested_cs, recon_scores_cs, args.min_gain_dim_select)
        if sugg_d is not None:
            suggested_d_cs = sugg_d
        print(f"Suggested optimal dimensions for {id_prefix_for_cs_run}: {suggested_d_cs}")
    
    num_to_save = min(args.num_gradients_to_save, actual_max_dims_cs)
    # Ensure num_to_save is positive and sensible
    if num_to_save <= 0 : 
        num_to_save = suggested_d_cs if suggested_d_cs is not None and suggested_d_cs > 0 else actual_max_dims_cs
    if num_to_save <=0 : num_to_save = 1 # Fallback to save at least one if possible
    num_to_save = min(num_to_save, actual_max_dims_cs) # Cannot save more than computed

    print(f"Will save top {num_to_save} cross-species gradients.")
    
    raw_embedding_output_path = os.path.join(intermediates_dir, f"cross_species_embedding_data_{id_prefix_for_cs_run}.npz")
    # Prepare simplified segment info for quick inspection if needed
    simple_segment_info_for_npz = [{'species': s['species'], 'hem': s['hem'], 
                                    'num_profiles': s['profiles_for_lle_shape'][0], 
                                    'start_row': s['start_row_in_concat'], 'end_row': s['end_row_in_concat']} 
                                   for s in segment_info_list]
    np.savez_compressed(raw_embedding_output_path, 
                        cross_species_gradients=all_grads_cs[:, :num_to_save], 
                        segment_info_simple=np.array(simple_segment_info_for_npz, dtype=object),
                        segment_info_detailed_for_remapping=np.array(segment_info_list, dtype=object), 
                        eigenvalues=eigvals_cs[:num_to_save] if eigvals_cs is not None else None)
    print(f"Saved raw cross-species embedding data and segment info to: {raw_embedding_output_path}")

    _save_cross_species_gradients(
        all_grads_cs, segment_info_list, eigvals_cs, num_to_save, 
        args.output_dir, args.target_k_species
    )
            
    print("\n--- All Cross-Species LLE Processing Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform cross-species gradient analysis using Spectral Embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments that define the core logic
    parser.add_argument('--species_list_for_lle', type=str, required=True,
                        help='Comma-separated list of ALL species to include in the joint LLE (e.g., "human,chimpanzee").')
    parser.add_argument('--target_k_species', type=str, required=True,
                        help='The species from --species_list_for_lle that provides its original (non-downsampled) blueprint.')

    # Optional arguments with sensible defaults
    parser.add_argument('--project_root', type=str, default='.',
                        help='Path to the project root directory containing data/ and results/.')
    parser.add_argument('--hemispheres_to_process', type=str, default="L,R",
                        help='Comma-separated list of hemisphere labels to process.')
    
    # Optional technical/algorithm parameters
    parser.add_argument('--max_gradients', type=int, default=DEFAULT_MAX_GRADIENTS_TO_TEST,
                        help="Maximum number of gradients to compute and test.")
    parser.add_argument('--max_k_knn', type=int, default=DEFAULT_MAX_K_SEARCH_FOR_KNN,
                        help="Maximum k for k-NN graph adaptive search.")
    parser.add_argument('--default_k_knn', type=int, default=DEFAULT_N_NEIGHBORS_FALLBACK,
                        help="Default k for k-NN if adaptive search fails.")
    parser.add_argument('--min_gain_dim_select', type=float, default=DEFAULT_MIN_GAIN_FOR_DIM_SELECTION,
                        help="Minimum gain in reconstruction score to suggest an additional dimension.")
    parser.add_argument('--num_gradients_to_save', type=int, default=DEFAULT_NUM_GRADIENTS_TO_SAVE,
                        help="Number of top gradients to save in the final output files.")
    
    parsed_args = parser.parse_args()
    
    # Create a namespace/object to hold all arguments for the main function ---
    # This keeps the main `run_cross_species_lle` function unchanged.
    class RunArgs:
        pass
    args_for_run = RunArgs()

    # --- NEW: Populate arguments by automatically determining paths and patterns ---
    # Copy over essential and technical args
    args_for_run.species_list_for_lle = parsed_args.species_list_for_lle
    args_for_run.target_k_species = parsed_args.target_k_species
    args_for_run.hemispheres_to_process = parsed_args.hemispheres_to_process
    args_for_run.max_gradients = parsed_args.max_gradients
    args_for_run.max_k_knn = parsed_args.max_k_knn
    args_for_run.default_k_knn = parsed_args.default_k_knn
    args_for_run.min_gain_dim_select = parsed_args.min_gain_dim_select
    args_for_run.num_gradients_to_save = parsed_args.num_gradients_to_save

    # Construct paths
    args_for_run.target_species_bp_dir = os.path.join(parsed_args.project_root, 'results', '2_masked_average_blueprints')
    args_for_run.other_species_downsampled_dir = os.path.join(parsed_args.project_root, 'results', '5_downsampled_blueprints')
    args_for_run.mask_dir = os.path.join(parsed_args.project_root, 'data', 'masks')
    args_for_run.output_dir = os.path.join(parsed_args.project_root, 'results', '6_cross_species_gradients')

    # Define fixed filename patterns
    args_for_run.target_species_bp_pattern = "average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii"
    args_for_run.downsampled_centroid_pattern = "{species_name}_{hemisphere}_k{k_val}_centroids.npy"
    args_for_run.mask_pattern = "{species_name}_{hemisphere}.func.gii"

    # --- Validation logic (same as before, but uses the new args object) ---
    args_for_run.species_for_lle = [s.strip().lower() for s in args_for_run.species_list_for_lle.split(',')]
    args_for_run.hemispheres_to_process = [h.strip().upper() for h in args_for_run.hemispheres_to_process.split(',')]
    args_for_run.target_k_species = args_for_run.target_k_species.strip().lower()

    if args_for_run.target_k_species not in args_for_run.species_for_lle:
        print(f"ERROR: target_k_species '{args_for_run.target_k_species}' must be included in --species_list_for_lle.")
        exit(1)
    if not all(h in ['L', 'R'] for h in args_for_run.hemispheres_to_process):
        print(f"ERROR: --hemispheres_to_process must be a comma-separated list of 'L' and/or 'R'. Got: {args_for_run.hemispheres_to_process}")
        exit(1)
        
    # --- Call the main processing function with the fully populated arguments ---
    run_cross_species_lle(args_for_run)