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

# --- Script Constants ---
MAX_GRADIENTS_TO_TEST_DEFAULT = 10
MAX_K_SEARCH_FOR_KNN_DEFAULT = 150
DEFAULT_N_NEIGHBORS_FALLBACK_DEFAULT = 20
MIN_GAIN_FOR_DIM_SELECTION_DEFAULT = 0.1

# --- Core Numba-JITted Helper Functions ---

@njit(parallel=True, fastmath=True)
def eta2(X):
    """
    Calculates the similarity matrix S using the provided eta2 formula.
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
        _sim_i[i] = -np.inf  # Exclude self from neighbors
        _actual_k = min(k_to_test, n_v - 1) # k cannot be more than n_v-1 other vertices
        if _actual_k <= 0:
            continue

        _sorted_indices = np.argsort(_sim_i)
        _neighbor_indices = _sorted_indices[-_actual_k:]

        for _neighbor_idx in _neighbor_indices:
            _current_w[i, _neighbor_idx] = similarity_matrix_input[i, _neighbor_idx]
    _current_w = np.maximum(_current_w, _current_w.T)  # Symmetrize
    return _current_w

@njit(fastmath=True)
def calculate_eigenvalues_for_gradients_numba(embedding_matrix, laplacian_dense):
    """Calculates eigenvalues for given eigenvectors and Laplacian."""
    n_gradients = embedding_matrix.shape[1]
    eigenvalues = np.zeros(n_gradients, dtype=laplacian_dense.dtype)
    for i in range(n_gradients):
        v_i = embedding_matrix[:, i]
        L_v_i = np.dot(laplacian_dense, v_i)
        lambda_i = np.dot(v_i, L_v_i)
        eigenvalues[i] = lambda_i
    return eigenvalues

# --- Python Helper Functions ---

def find_adaptive_knn_graph_binary_search(similarity_matrix, max_k_search, default_k_fallback):
    """Orchestrates binary search for adaptive k, calling JITted graph builder."""
    n_vertices = similarity_matrix.shape[0]
    if n_vertices <= 1: # Cannot form a graph with 0 or 1 vertex
        return np.zeros_like(similarity_matrix)

    print(f"    Binary searching for adaptive k (up to {max_k_search}) for {n_vertices} vertices...")
    low = 1
    high = min(max_k_search, n_vertices - 1) # k cannot be more than n-1
    best_k_graph = None
    best_k_val = -1

    while low <= high:
        mid = (low + high) // 2
        if mid == 0: # Should not happen if low starts at 1
            mid = 1
        
        current_w_for_mid = build_knn_graph_for_k(mid, similarity_matrix)
        
        # Use objmode to call scipy.sparse.csgraph.connected_components
        with objmode(n_c='intp'):
            res_tuple = connected_components(current_w_for_mid, directed=False, connection='weak')
            n_c = res_tuple[0]

        if n_c == 1: # Found a k that results in a connected graph
            best_k_graph = current_w_for_mid
            best_k_val = mid
            high = mid - 1 # Try smaller k
        else:
            low = mid + 1 # Need larger k

    if best_k_graph is not None:
        print(f"    Found connected graph with adaptive k = {best_k_val}")
        return best_k_graph
    else:
        print(f"    Warning: Binary search did not find connected graph. Using fallback k={default_k_fallback}.")
        actual_default_k = min(default_k_fallback, n_vertices - 1 if n_vertices > 1 else 1)
        if actual_default_k <= 0 and n_vertices > 1: # Ensure k is at least 1 if possible
            actual_default_k = 1
        
        if n_vertices > 1 and actual_default_k > 0:
            fallback_w = build_knn_graph_for_k(actual_default_k, similarity_matrix)
            with objmode(n_c_fallback='intp'):
                res_tuple_fb = connected_components(fallback_w, directed=False, connection='weak')
                n_c_fallback = res_tuple_fb[0]
            if n_c_fallback > 1:
                print(f"    Warning: Graph with fallback k={actual_default_k} is also not connected ({n_c_fallback} components).")
            return fallback_w
        else: # Not enough vertices to form a graph
            return np.zeros_like(similarity_matrix)


def suggest_dimensionality_with_penalty(dimensions, scores, min_absolute_improvement):
    """Suggests optimal dimensionality based on diminishing returns or score decrease."""
    if not dimensions or not scores or len(dimensions) != len(scores):
        print("Warning (Penalty Suggestion): Invalid input."); return None
    
    valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
    if not valid_indices:
        print("Warning (Penalty Suggestion): No valid scores."); return None
    
    dims_filtered = [dimensions[i] for i in valid_indices]
    scores_filtered = [scores[i] for i in valid_indices]

    if not scores_filtered:
        print("Warning (Penalty Suggestion): No valid scores after NaN filter."); return None
    if len(dims_filtered) == 1:
        print(f"Suggested (Penalty): {dims_filtered[0]} (Score: {scores_filtered[0]:.4f})")
        return dims_filtered[0]
    
    suggested_dim = dims_filtered[0] # Start with the first valid dimension
    for i in range(1, len(scores_filtered)):
        gain = scores_filtered[i] - scores_filtered[i-1]
        # If score decreases significantly, previous dimension was better
        if scores_filtered[i] < (scores_filtered[i-1] - 1e-5): 
            suggested_dim = dims_filtered[i-1]
            print(f"    (Penalty Suggestion) Score decreased from {scores_filtered[i-1]:.4f} to {scores_filtered[i]:.4f}. Suggesting: {suggested_dim}")
            break
        # If gain is less than threshold, previous dimension is sufficient
        if gain < min_absolute_improvement:
            suggested_dim = dims_filtered[i-1] 
            print(f"    (Penalty Suggestion) Gain {gain:.4f} from dim {dims_filtered[i-1]} to {dims_filtered[i]} < threshold {min_absolute_improvement}. Suggesting: {suggested_dim}")
            break
        suggested_dim = dims_filtered[i] # Otherwise, current dimension is better or equally good
    else: # Loop completed without break
        print(f"    (Penalty Suggestion) All gains >= threshold ({min_absolute_improvement}) or no score decrease up to dim {dims_filtered[-1]}. Suggesting last tested: {suggested_dim}")
    
    # Ensure the suggested_dim is valid and report its score from the original (unfiltered) scores list
    if suggested_dim in dimensions:
        final_score_idx = dimensions.index(suggested_dim)
        final_score = scores[final_score_idx] if final_score_idx < len(scores) and not np.isnan(scores[final_score_idx]) else np.nan
        print(f"Suggested optimal dimensionality (with penalty {min_absolute_improvement}): {suggested_dim} (Score: {final_score:.4f})")
    else: # Should ideally not happen with this logic
        print(f"Error: Suggested dimension {suggested_dim} from filtered list not in original dimensions. Defaulting to first valid dimension.")
        return dims_filtered[0] if dims_filtered else None
    return suggested_dim


def _load_and_prepare_data(masked_avg_blueprint_path, mask_path, species_name, hemisphere_token):
    """Loads masked blueprint and mask, returns TL indices, TL blueprint data, and total surface vertices."""
    print(f"Loading data for {species_name} {hemisphere_token}...")
    if not os.path.exists(masked_avg_blueprint_path):
        print(f"ERROR: Blueprint file not found: {masked_avg_blueprint_path}"); return None, None, None
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask file not found: {mask_path}"); return None, None, None
    try:
        blueprint_img = nib.load(masked_avg_blueprint_path)
        mask_img = nib.load(mask_path)
    except Exception as e:
        print(f"Error loading GIFTI for {species_name} {hemisphere_token}: {e}"); return None, None, None

    if not mask_img.darrays:
        print(f"ERROR: No darrays in mask {mask_path}"); return None, None, None
    mask_data = mask_img.darrays[0].data # Assume mask is in the first darray
    tl_indices = np.where(mask_data > 0.5)[0] # Binarize mask
    if tl_indices.size == 0:
        print(f"ERROR: Mask empty for {species_name} {hemisphere_token}."); return None, None, None
    
    if not blueprint_img.darrays:
        print(f"ERROR: No darrays in blueprint {masked_avg_blueprint_path}"); return None, None, None
    
    num_total_surface_vertices = blueprint_img.darrays[0].data.shape[0]
    if mask_data.shape[0] != num_total_surface_vertices:
        print(f"ERROR: Vertex count mismatch! Blueprint: {num_total_surface_vertices}, Mask: {mask_data.shape[0]}"); return None, None, None
        
    num_tracts = len(blueprint_img.darrays)
    # Blueprint data: vertices as rows, tracts/features as columns
    bp_data_all_vertices = np.zeros((num_total_surface_vertices, num_tracts), dtype=np.float32)
    for i, darray in enumerate(blueprint_img.darrays):
        if darray.data.shape[0] != num_total_surface_vertices:
            print(f"ERROR: Darray {i} in blueprint has inconsistent number of vertices."); return None, None, None
        bp_data_all_vertices[:, i] = darray.data
    
    tl_blueprint_data = bp_data_all_vertices[tl_indices, :]
    print(f"    Extracted {tl_blueprint_data.shape[0]} temporal lobe vertices with {tl_blueprint_data.shape[1]} features.")
    return tl_indices, tl_blueprint_data, num_total_surface_vertices


def _compute_affinity_and_knn_graph(tl_blueprint_data, species_name, id_token, output_dir_for_intermediates,
                                    max_k_search, default_k_fallback):
    """Computes S_full_processed and knn_graph_W, with checkpointing."""
    s_full_path = os.path.join(output_dir_for_intermediates, f"S_full_processed_{species_name}_{id_token}.npy")
    knn_w_path = os.path.join(output_dir_for_intermediates, f"knn_graph_W_{species_name}_{id_token}.npy")

    if os.path.exists(knn_w_path) and os.path.exists(s_full_path):
        print(f"    Loading existing k-NN graph from: {knn_w_path}")
        print(f"    Loading existing S_full_processed from: {s_full_path}")
        knn_graph_W = np.load(knn_w_path)
        S_full_processed = np.load(s_full_path)
        return S_full_processed, knn_graph_W
    
    print(f"    Calculating full similarity matrix (S_full) for {species_name} {id_token}...")
    S_full_raw = eta2(tl_blueprint_data)
    # Ensure similarity is non-negative; eta2 can sometimes yield small negatives due to precision
    S_full_processed = np.clip(S_full_raw, 0, None)
    if S_full_processed.shape[0] > 0: # Ensure diagonal is 1
        np.fill_diagonal(S_full_processed, 1.0) 
    
    np.save(s_full_path, S_full_processed)
    print(f"    Saved S_full_processed to: {s_full_path}")

    if np.any(np.isnan(S_full_processed)) or np.any(np.isinf(S_full_processed)):
        print(f"    ERROR: S_full_processed contains NaNs/Infs for {species_name} {id_token}."); return None, None
    if S_full_processed.shape[0] <= 1: # Need at least 2 vertices to build a graph
        print(f"    ERROR: Not enough vertices in S_full_processed for {species_name} {id_token}."); return None, None
    
    print(f"    Constructing adaptive k-NN graph for {species_name} {id_token}...")
    knn_graph_W = find_adaptive_knn_graph_binary_search(S_full_processed, max_k_search, default_k_fallback)
    if knn_graph_W is None or knn_graph_W.shape[0] == 0: # find_adaptive_knn returns 0-shape array on failure for n_vertices<=1
        print(f"    ERROR: k-NN graph construction failed for {species_name} {id_token}."); return None, None
    
    np.save(knn_w_path, knn_graph_W)
    print(f"    Saved knn_graph_W to: {knn_w_path}")
    return S_full_processed, knn_graph_W


def _perform_embedding_and_eigval_calc(knn_graph_W, max_dims_to_compute, species_name, id_token, output_dir_for_intermediates):
    """Performs spectral embedding and calculates eigenvalues, with checkpointing."""
    embedding_path = os.path.join(output_dir_for_intermediates, f"all_gradients_embedding_{species_name}_{id_token}.npy")
    eigenvalues_path = os.path.join(output_dir_for_intermediates, f"computed_eigenvalues_{species_name}_{id_token}.npy")

    # Number of components must be < n_samples
    actual_max_dims = min(max_dims_to_compute, knn_graph_W.shape[0] - 1 if knn_graph_W.shape[0] > 1 else 1)
    if actual_max_dims <= 0:
        print(f"    ERROR: Not enough samples/rank to compute gradients for {species_name} {id_token} ({knn_graph_W.shape[0]} vertices)."); return None, None, 0

    if os.path.exists(embedding_path) and os.path.exists(eigenvalues_path):
        print(f"    Loading existing embedding from: {embedding_path}")
        print(f"    Loading existing eigenvalues from: {eigenvalues_path}")
        all_gradients_embedding = np.load(embedding_path)
        computed_eigenvalues = np.load(eigenvalues_path)
        # Validate loaded data shape
        if all_gradients_embedding.shape[1] >= actual_max_dims and \
           len(computed_eigenvalues) >= actual_max_dims:
            # Trim to current actual_max_dims if loaded data is larger (e.g. from previous run with higher max_dims)
            return all_gradients_embedding[:, :actual_max_dims], computed_eigenvalues[:actual_max_dims], actual_max_dims
        else:
             print(f"    Warning: Loaded embedding/eigenvalues shape mismatch or insufficient. Expected {actual_max_dims} dims. Recomputing.")


    print(f"    Performing Spectral Embedding for up to {actual_max_dims} gradients for {species_name} {id_token}...")
    try:
        # Normalized graph Laplacian, L = D^-0.5 * (D-W) * D^-0.5
        L_sparse = csgraph_laplacian(knn_graph_W, normed=True) 
        L_dense = L_sparse.toarray() if scipy.sparse.issparse(L_sparse) else L_sparse
        
        spectral_embedder = SpectralEmbedding(
            n_components=actual_max_dims,
            affinity='precomputed', # Input is already a graph/affinity matrix
            random_state=42,
            n_jobs=-1,
            eigen_solver='lobpcg' # Good for large sparse matrices
        )
        all_gradients_embedding = spectral_embedder.fit_transform(knn_graph_W) # knn_graph_W is used as affinity
        computed_eigenvalues = calculate_eigenvalues_for_gradients_numba(all_gradients_embedding, L_dense)
        print(f"    Computed Eigenvalues for {species_name} {id_token}: {computed_eigenvalues}")

        np.save(embedding_path, all_gradients_embedding)
        print(f"    Saved all_gradients_embedding to: {embedding_path}")
        np.save(eigenvalues_path, computed_eigenvalues)
        print(f"    Saved computed_eigenvalues to: {eigenvalues_path}")

        return all_gradients_embedding, computed_eigenvalues, actual_max_dims
    except Exception as e:
        print(f"    Error in Spectral Embedding/Eigenvalue calculation for {species_name} {id_token}: {e}")
        return None, None, 0


def _evaluate_reconstruction_and_plot(all_gradients_embedding, S_full_processed, actual_max_dims, output_plot_path, plot_title_suffix):
    """Evaluates reconstruction quality by comparing similarity from original vs. embedded data."""
    # Use only upper triangle (excluding diagonal) for correlation to avoid redundancy and self-similarity bias
    original_S_full_vector = S_full_processed[np.triu_indices_from(S_full_processed, k=1)]
    reconstruction_correlations = []
    dimensions_tested = list(range(1, actual_max_dims + 1))

    for d_val in dimensions_tested:
        embedding_Y_d = all_gradients_embedding[:, :d_val]
        if embedding_Y_d.shape[1] == 0:
            reconstruction_correlations.append(np.nan)
            continue
        
        S_embed_raw = eta2(embedding_Y_d) # Calculate similarity from embedded data
        S_embed_processed = np.clip(S_embed_raw, 0, None)
        if S_embed_processed.shape[0] > 0:
            np.fill_diagonal(S_embed_processed, 1.0)
        S_embed_vector = S_embed_processed[np.triu_indices_from(S_embed_processed, k=1)]

        if len(original_S_full_vector) > 1 and \
           len(S_embed_vector) == len(original_S_full_vector) and \
           np.std(original_S_full_vector) > 1e-9 and np.std(S_embed_vector) > 1e-9:
            corr, _ = pearsonr(original_S_full_vector, S_embed_vector)
            reconstruction_correlations.append(corr)
        elif len(original_S_full_vector) == len(S_embed_vector) and len(original_S_full_vector) > 0 : 
            # Handle cases with no variance (e.g. if all similarities are identical)
            reconstruction_correlations.append(1.0 if np.allclose(original_S_full_vector, S_embed_vector) else 0.0)
        else: # Mismatch in length or not enough data points
            reconstruction_correlations.append(np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_tested, reconstruction_correlations, marker='o', linestyle='-')
    plt.xlabel("Number of Dimensions (Gradients)")
    plt.ylabel("Pearson Corr (Original Similarity vs. Embedded Similarity)")
    plt.title(f"LLE Dimensionality Evaluation - {plot_title_suffix}")
    plt.xticks(dimensions_tested)
    plt.grid(True)
    # Adjust y-limits for better visualization
    valid_corrs = [c for c in reconstruction_correlations if not np.isnan(c)]
    min_y_lim = min(0, np.min(valid_corrs) if valid_corrs else 0) - 0.05
    plt.ylim(min_y_lim, 1.05)
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path)
    plt.close()
    print(f"    Saved dimensionality evaluation plot to: {output_plot_path}")
    return dimensions_tested, reconstruction_correlations


def _save_gradients_to_func_gii(all_gradients_embedding, tl_indices, num_total_surface_vertices,
                                computed_eigenvalues, actual_max_dims_to_save, hemisphere_label,
                                output_gii_path, description_prefix):
    """Remaps and saves gradients to a func.gii file with appropriate metadata."""
    print(f"    Saving {actual_max_dims_to_save} gradients for {description_prefix} to {output_gii_path}...")
    output_gifti_darrays = []
    
    structure_name = 'CortexLeft' if hemisphere_label.upper() == 'L' else 'CortexRight'

    for i in range(actual_max_dims_to_save):
        gradient_map_full = np.zeros(num_total_surface_vertices, dtype=np.float32)
        gradient_map_full[tl_indices] = all_gradients_embedding[:, i]
        
        darray_meta = nib.gifti.GiftiMetaData()
        map_name = f'Gradient_{i+1}'
        if computed_eigenvalues is not None and i < len(computed_eigenvalues):
            map_name = f'Gradient_{i+1}_eig_{computed_eigenvalues[i]:.4e}'
        darray_meta['Name'] = map_name
        darray_meta['AnatomicalStructurePrimary'] = structure_name # Set for each darray
        
        darray = nib.gifti.GiftiDataArray(
            data=gradient_map_full,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes.code['NIFTI_TYPE_FLOAT32'],
            meta=darray_meta
        )
        output_gifti_darrays.append(darray)

    output_gifti_image_meta = nib.gifti.GiftiMetaData()
    desc_text = f'{description_prefix} - {actual_max_dims_to_save} Gradients'
    output_gifti_image_meta['Description'] = desc_text
    output_gifti_image_meta['AnatomicalStructurePrimary'] = structure_name # Set for the image
    
    output_gii_img = nib.gifti.GiftiImage(darrays=output_gifti_darrays, meta=output_gifti_image_meta)
    os.makedirs(os.path.dirname(output_gii_path), exist_ok=True)
    try:
        nib.save(output_gii_img, output_gii_path)
        print(f"    Successfully saved gradients: {output_gii_path}")
    except Exception as e:
        print(f"    Error saving gradient GIFTI image: {e}")

# --- Main Evaluation Functions ---

def run_single_hemisphere_lle_pipeline( 
    species_name, hemisphere_label, masked_avg_blueprint_gii_path, 
    temporal_mask_gii_path, species_output_dir,
    max_dims_to_test, min_gain_threshold, max_k_search, default_k_fallback
):
    """Orchestrates the LLE pipeline for a single hemisphere."""
    id_token = f"{hemisphere_label}_Separate" # For intermediate file naming
    plot_title_suffix = f"{species_name.capitalize()} Hemisphere {hemisphere_label} (Separate)"
    output_plot_path = os.path.join(species_output_dir, f"dim_eval_plot_{species_name}_{hemisphere_label}_SEPARATE.png")
    output_gradient_gii_path = os.path.join(species_output_dir, f"all_computed_gradients_{species_name}_{hemisphere_label}_SEPARATE.func.gii")

    print(f"\n--- Running Single Hemisphere LLE Pipeline for {species_name} {hemisphere_label} ---")
    
    tl_indices, tl_blueprint_data, num_total_verts = _load_and_prepare_data(
        masked_avg_blueprint_gii_path, temporal_mask_gii_path, species_name, hemisphere_label)
    if tl_indices is None: return None, None, None, None

    S_full_processed, knn_graph_W = _compute_affinity_and_knn_graph(
        tl_blueprint_data, species_name, id_token, species_output_dir, max_k_search, default_k_fallback)
    if knn_graph_W is None: return None, None, None, None
    
    all_grads, eigvals, actual_max_dims = _perform_embedding_and_eigval_calc(
        knn_graph_W, max_dims_to_test, species_name, id_token, species_output_dir)
    if all_grads is None: return None, None, None, None

    dims_tested, recon_scores = _evaluate_reconstruction_and_plot(
        all_grads, S_full_processed, actual_max_dims, 
        output_plot_path, plot_title_suffix)
    if dims_tested is None: return None, None, None, None
        
    suggested_d = suggest_dimensionality_with_penalty(dims_tested, recon_scores, min_gain_threshold)
    
    num_gradients_to_save = actual_max_dims # Save all computed gradients up to max_dims_to_test
    # Optionally, one could choose to save only up to suggested_d by changing num_gradients_to_save
    # For example:
    # if suggested_d is not None and suggested_d > 0:
    #    num_gradients_to_save = suggested_d
        

    _save_gradients_to_func_gii(
        all_grads, tl_indices, num_total_verts, eigvals, num_gradients_to_save,
        hemisphere_label, output_gradient_gii_path, 
        f"{species_name.capitalize()} TL {hemisphere_label} Separate")
        
    return dims_tested, recon_scores, suggested_d, eigvals


def run_combined_hemisphere_lle_pipeline(
    species_name, masked_avg_blueprint_gii_path_L, masked_avg_blueprint_gii_path_R,
    temporal_mask_gii_path_L, temporal_mask_gii_path_R, 
    species_output_dir, 
    max_dims_to_test, min_gain_threshold, max_k_search, default_k_fallback
):
    """Orchestrates the LLE pipeline for combined hemispheres."""
    id_token = "Combined" # For intermediate file naming
    plot_title_suffix = f"{species_name.capitalize()} Combined Hemispheres"
    output_plot_path = os.path.join(species_output_dir, f"dim_eval_plot_{species_name}_COMBINED.png")
    output_gradient_gii_path_L = os.path.join(species_output_dir, f"all_computed_gradients_{species_name}_COMBINED_L.func.gii")
    output_gradient_gii_path_R = os.path.join(species_output_dir, f"all_computed_gradients_{species_name}_COMBINED_R.func.gii")

    print(f"\n--- Running Combined Hemisphere LLE Pipeline for {species_name} ---")

    tl_indices_L, tl_data_L, num_verts_L = _load_and_prepare_data(
        masked_avg_blueprint_gii_path_L, temporal_mask_gii_path_L, species_name, "L_for_Combined")
    tl_indices_R, tl_data_R, num_verts_R = _load_and_prepare_data(
        masked_avg_blueprint_gii_path_R, temporal_mask_gii_path_R, species_name, "R_for_Combined")

    if tl_data_L is None or tl_data_R is None:
        print(f"ERROR: Failed to load data for one or both hemispheres for combined analysis of {species_name}.")
        return None, None, None, None
    
    N_L = tl_data_L.shape[0] # Number of temporal lobe vertices in Left hemisphere
    combined_tl_blueprint_data = np.vstack((tl_data_L, tl_data_R))

    S_full_combined, knn_graph_W_combined = _compute_affinity_and_knn_graph(
        combined_tl_blueprint_data, species_name, id_token, species_output_dir, max_k_search, default_k_fallback)
    if knn_graph_W_combined is None: return None, None, None, None

    all_grads_comb, eigvals_comb, actual_max_dims_comb = _perform_embedding_and_eigval_calc(
        knn_graph_W_combined, max_dims_to_test, species_name, id_token, species_output_dir)
    if all_grads_comb is None: return None, None, None, None

    dims_tested_c, recon_scores_c = _evaluate_reconstruction_and_plot(
        all_grads_comb, S_full_combined, actual_max_dims_comb,
        output_plot_path, plot_title_suffix)
    if dims_tested_c is None: return None, None, None, None
        
    suggested_d_c = suggest_dimensionality_with_penalty(dims_tested_c, recon_scores_c, min_gain_threshold)

    num_gradients_to_save_c = actual_max_dims_comb
    # Save Left hemisphere part of combined gradients
    _save_gradients_to_func_gii(
        all_grads_comb[:N_L, :], tl_indices_L, num_verts_L, eigvals_comb, num_gradients_to_save_c,
        'L', output_gradient_gii_path_L, f"{species_name.capitalize()} TL Combined (Left Part)")
    # Save Right hemisphere part of combined gradients
    _save_gradients_to_func_gii(
        all_grads_comb[N_L:, :], tl_indices_R, num_verts_R, eigvals_comb, num_gradients_to_save_c,
        'R', output_gradient_gii_path_R, f"{species_name.capitalize()} TL Combined (Right Part)")
        
    return dims_tested_c, recon_scores_c, suggested_d_c, eigvals_comb

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute connectivity gradients from masked blueprints using Spectral Embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument('--species_list', type=str, required=True,
                        help='Comma-separated list of species names to process (e.g., "human,chimpanzee").')
    
    # Optional arguments with sensible defaults
    parser.add_argument('--project_root', type=str, default='.',
                        help='Path to the project root directory containing data/ and results/.')
    parser.add_argument('--hemispheres', type=str, default="L,R",
                        help='Comma-separated list of hemisphere labels to process.')
    
    # Technical parameters for the LLE/Embedding process (for advanced users)
    parser.add_argument('--max_gradients', type=int, default=MAX_GRADIENTS_TO_TEST_DEFAULT,
                        help="Maximum number of gradients to compute and test.")
    parser.add_argument('--max_k_knn', type=int, default=MAX_K_SEARCH_FOR_KNN_DEFAULT,
                        help="Maximum k for k-NN graph adaptive search.")
    parser.add_argument('--default_k_knn', type=int, default=DEFAULT_N_NEIGHBORS_FALLBACK_DEFAULT,
                        help="Default k for k-NN if adaptive search fails.")
    parser.add_argument('--min_gain_dim_select', type=float, default=MIN_GAIN_FOR_DIM_SELECTION_DEFAULT,
                        help="Minimum gain in reconstruction score to select an additional dimension.")

    args = parser.parse_args()

    species_to_process = [s.strip() for s in args.species_list.split(',')]
    hemispheres_to_process = [h.strip() for h in args.hemispheres.split(',')]
    
    # --- NEW: Define fixed paths and patterns inside the script ---
    input_masked_bp_base_dir = os.path.join(args.project_root, 'results', '2_masked_average_blueprints')
    input_mask_base_dir = os.path.join(args.project_root, 'data', 'masks')
    output_base_dir = os.path.join(args.project_root, 'results', '3_individual_species_gradients')
    
    os.makedirs(output_base_dir, exist_ok=True)

    all_results_summary = {}

    for species in species_to_process:
        # Define species-specific directories
        species_output_dir = os.path.join(output_base_dir, species)
        os.makedirs(species_output_dir, exist_ok=True)
            
        species_masked_bp_input_dir = os.path.join(input_masked_bp_base_dir, species)
        species_mask_files_dir = os.path.join(input_mask_base_dir, species)
        
        # Define filename patterns for this species
        masked_blueprint_pattern = f"average_{species}_blueprint.{{hemisphere}}_temporal_lobe_masked.func.gii"
        mask_pattern = f"{species}_{{hemisphere}}.func.gii"

        all_results_summary[species] = {}
        
        # --- Separate Hemisphere Analysis ---
        print(f"\n\n=== Running SEPARATE Hemisphere Analysis for {species.capitalize()} ===")
        for hem in hemispheres_to_process:
            masked_avg_bp_path = os.path.join(species_masked_bp_input_dir, masked_blueprint_pattern.format(hemisphere=hem))
            mask_path = os.path.join(species_mask_files_dir, mask_pattern.format(hemisphere=hem))
            
            if not os.path.exists(masked_avg_bp_path) or not os.path.exists(mask_path):
                print(f"WARNING: Input file not found, skipping {species} {hem} separate.")
                continue

            dims, scores, suggested_dim, eigenvalues = run_single_hemisphere_lle_pipeline( 
                species_name=species, hemisphere_label=hem,
                masked_avg_blueprint_gii_path=masked_avg_bp_path, 
                temporal_mask_gii_path=mask_path,
                species_output_dir=species_output_dir,
                max_dims_to_test=args.max_gradients,
                min_gain_threshold=args.min_gain_dim_select,
                max_k_search=args.max_k_knn,
                default_k_fallback=args.default_k_knn
            )
            # (Reporting logic remains the same)
            all_results_summary[species][hem] = {
                'dimensions': dims, 'scores': scores, 
                'suggested_dim_penalized': suggested_dim, 'eigenvalues': eigenvalues 
            }

        # --- Combined Hemisphere Analysis ---
        print(f"\n\n=== Running COMBINED Hemisphere Analysis for {species.capitalize()} ===")
        bp_L_path = os.path.join(species_masked_bp_input_dir, masked_blueprint_pattern.format(hemisphere='L'))
        bp_R_path = os.path.join(species_masked_bp_input_dir, masked_blueprint_pattern.format(hemisphere='R'))
        mask_L_path = os.path.join(species_mask_files_dir, mask_pattern.format(hemisphere='L'))
        mask_R_path = os.path.join(species_mask_files_dir, mask_pattern.format(hemisphere='R'))

        if not (os.path.exists(bp_L_path) and os.path.exists(bp_R_path) and \
                os.path.exists(mask_L_path) and os.path.exists(mask_R_path)):
            print(f"Skipping combined analysis for {species} due to missing input files.")
        else:
            dims_c, scores_c, suggested_dim_c, eigvals_c = run_combined_hemisphere_lle_pipeline(
                species_name=species,
                masked_avg_blueprint_gii_path_L=bp_L_path, masked_avg_blueprint_gii_path_R=bp_R_path,
                temporal_mask_gii_path_L=mask_L_path, temporal_mask_gii_path_R=mask_R_path,
                species_output_dir=species_output_dir,
                max_dims_to_test=args.max_gradients, min_gain_threshold=args.min_gain_dim_select,
                max_k_search=args.max_k_knn, default_k_fallback=args.default_k_knn
            )
            # (Reporting logic remains the same)
            all_results_summary[species]['COMBINED'] = {
                'dimensions': dims_c, 'scores': scores_c, 
                'suggested_dim_penalized': suggested_dim_c, 'eigenvalues': eigvals_c 
            }
                    
    print("\n--- All LLE Dimensionality Evaluation and Gradient Saving Complete ---")
