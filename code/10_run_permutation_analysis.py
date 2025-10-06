import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# --- Default Configuration ---
DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_ALPHA_LEVEL = 0.01
DEFAULT_NUM_GRADIENTS = 3

# --- Helper Functions ---

def permutation_test_mean_difference(sample1, sample2, n_permutations, alternative='two-sided'):
    if sample1 is None or sample2 is None or sample1.size < 2 or sample2.size < 2:
        return np.nan, np.full(n_permutations, np.nan), np.nan
    observed_diff = np.mean(sample1) - np.mean(sample2)
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(combined)
        perm_sample1 = combined[:n1]
        perm_sample2 = combined[n1:]
        perm_diffs[i] = np.mean(perm_sample1) - np.mean(perm_sample2)
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return observed_diff, perm_diffs, p_value

def get_cross_species_gradient_values(cs_grads, seg_info, species, hem, idx):
    for s in seg_info:
        if s['species'] == species and s['hem'] == hem:
            if idx < cs_grads.shape[1]:
                return cs_grads[s['start_row']:s['end_row'], idx]
    return None

def get_individual_gradient_values(grad_path, mask_path, idx):
    if not os.path.exists(grad_path):
        print(f"ERROR: Gradient file not found: {grad_path}")
        return None
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask file not found: {mask_path}")
        return None
    try:
        grad_img = nib.load(grad_path)
        if idx >= len(grad_img.darrays):
            print(f"ERROR: Gradient index {idx+1} is out of bounds for {grad_path}")
            return None
        mask_data = nib.load(mask_path).darrays[0].data
        return grad_img.darrays[idx].data[np.where(mask_data > 0.5)[0]]
    except Exception as e:
        print(f"Error loading from {grad_path}: {e}")
        return None

def perform_permutation_comparison(label1, values1, label2, values2, gradient_name, output_dir, n_permutations, alpha_level, plot_hist=False, hist_color='gray'):
    print(f"\n--- Comparing {label1} vs {label2} for {gradient_name} ---")
    if values1 is None or values2 is None or values1.size < 2 or values2.size < 2:
        print("  Could not perform comparison due to insufficient data.")
        return
    obs_diff, perm_diffs, p_val = permutation_test_mean_difference(values1, values2, n_permutations)
    p_str = f"p = {p_val:.4f}"
    if p_val < 0.0001: p_str = "p < 0.0001"
    print(f"  Observed Mean Difference ({label1} - {label2}): {obs_diff:.4f}")
    print(f"  Empirical p-value: {p_str}")
    if p_val < alpha_level:
        print(f"  Result: Statistically significant difference (p < {alpha_level})")
    else:
        print(f"  Result: No statistically significant difference (p >= {alpha_level})")
    if plot_hist:
        plt.figure(figsize=(8, 6))
        plt.hist(perm_diffs, bins=50, color=hist_color, edgecolor='k', alpha=0.7, label='Permutation Null Distribution')
        plt.axvline(obs_diff, color='b', linestyle='--', linewidth=2, label=f'Observed Diff: {obs_diff:.4f}')
        plt.title(f"Permutation Test: {label1} vs {label2} ({gradient_name})")
        plt.text(0.97, 0.97, p_str, fontsize=12, color='black', ha='right', va='top', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.4))
        plt.legend(loc='upper left')
        hist_filename = f"permtest_hist_{gradient_name}_{label1.replace(' ','')}_vs_{label2.replace(' ','')}.png"
        hist_path = os.path.join(output_dir, hist_filename)
        plt.savefig(hist_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved permutation histogram to: {hist_path}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    for grad_idx in range(args.num_gradients):
        gradient_name = f"G{grad_idx+1}"
        print(f"\n{'='*15} Analyzing {gradient_name} {'='*15}")
        
        common_args = { "gradient_name": f"{gradient_name}_{args.analysis_type}", "output_dir": args.output_dir, "n_permutations": args.n_permutations, "alpha_level": args.alpha, "plot_hist": not args.no_histograms }

        if args.analysis_type == 'cross_species':
            with np.load(args.npz_file, allow_pickle=True) as data:
                cs_grads, seg_info = data['cross_species_gradients'], list(data['segment_info_simple'])
                human_L = get_cross_species_gradient_values(cs_grads, seg_info, "human", "L", grad_idx)
                human_R = get_cross_species_gradient_values(cs_grads, seg_info, "human", "R", grad_idx)
                chimp_L = get_cross_species_gradient_values(cs_grads, seg_info, "chimpanzee", "L", grad_idx)
                chimp_R = get_cross_species_gradient_values(cs_grads, seg_info, "chimpanzee", "R", grad_idx)
            
            perform_permutation_comparison("Human L", human_L, "Human R", human_R, **common_args, hist_color='skyblue')
            perform_permutation_comparison("Chimp L", chimp_L, "Chimp R", chimp_R, **common_args, hist_color='salmon')
            perform_permutation_comparison("Human L", human_L, "Chimp L", chimp_L, **common_args, hist_color='mediumpurple')
            perform_permutation_comparison("Human R", human_R, "Chimp R", chimp_R, **common_args, hist_color='gold')

        elif args.analysis_type == 'individual':
            base_grad_dir = os.path.join(args.project_root, 'results', '3_individual_species_gradients', args.species)
            base_mask_dir = os.path.join(args.project_root, 'data', 'masks', args.species)
            
            L_grad_path = os.path.join(base_grad_dir, f'all_computed_gradients_{args.species}_COMBINED_L.func.gii')
            R_grad_path = os.path.join(base_grad_dir, f'all_computed_gradients_{args.species}_COMBINED_R.func.gii')
            L_mask_path = os.path.join(base_mask_dir, f'{args.species}_L.func.gii')
            R_mask_path = os.path.join(base_mask_dir, f'{args.species}_R.func.gii')

            left_hem_vals = get_individual_gradient_values(L_grad_path, L_mask_path, grad_idx)
            right_hem_vals = get_individual_gradient_values(R_grad_path, R_mask_path, grad_idx)
            
            if left_hem_vals is None or right_hem_vals is None:
                print(f"Stopping analysis for {args.species} due to missing data for {gradient_name}.")
                break
            
            perform_permutation_comparison(f"{args.species.capitalize()} L", left_hem_vals, f"{args.species.capitalize()} R", right_hem_vals, **common_args, hist_color='lightgreen')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run permutation tests on gradient values.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--analysis_type', type=str, required=True, choices=['cross_species', 'individual'])
    parser.add_argument('--num_gradients', type=int, default=DEFAULT_NUM_GRADIENTS, help="Number of top gradients to analyze.")
    
    # Args for cross_species
    parser.add_argument('--species_list_for_run', type=str, help='(For cross_species) e.g., "human,chimpanzee"')
    parser.add_argument('--target_k_species_for_run', type=str, help='(For cross_species) The reference species.')
    
    # Args for individual
    parser.add_argument('--species', type=str, help='(For individual) Species to analyze (e.g., "human").')

    # Common optional args
    parser.add_argument('--project_root', type=str, default='.')
    parser.add_argument('--n_permutations', type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA_LEVEL)
    parser.add_argument('--no_histograms', action='store_true')
    
    args = parser.parse_args()

    # --- Setup paths and validate ---
    if args.analysis_type == 'cross_species':
        if not (args.species_list_for_run and args.target_k_species_for_run):
            parser.error("--species_list_for_run and --target_k_species_for_run are required.")
        run_id = f"{'_'.join(args.species_list_for_run.split(','))}_CrossSpecies_kRef_{args.target_k_species_for_run}"
        interm_dir = os.path.join(args.project_root, 'results', '6_cross_species_gradients', 'intermediates', run_id)
        args.npz_file = os.path.join(interm_dir, f'cross_species_embedding_data_{run_id}.npz')
        args.output_dir = os.path.join(args.project_root, 'results', '10_permutation_analysis', run_id)
    
    elif args.analysis_type == 'individual':
        if not args.species:
            parser.error("--species is required for --analysis_type individual.")
        args.output_dir = os.path.join(args.project_root, 'results', '10_permutation_analysis', f'individual_{args.species}')

    main(args)
    print("\n--- Permutation testing script finished ---")
