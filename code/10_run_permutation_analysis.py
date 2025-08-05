import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- Default Configuration ---
DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_ALPHA_LEVEL = 0.01
DEFAULT_GRADIENT_INDEX_TO_ANALYZE = 0

# --- Helper Functions ---

def permutation_test_mean_difference(sample1, sample2, n_permutations, alternative='two-sided'):
    """
    Permutation test for difference in means.
    Returns the observed difference, the null distribution, and empirical p-value.
    """
    if sample1 is None or sample2 is None or sample1.size < 2 or sample2.size < 2:
        print("Warning: Not enough data in one or both samples.")
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
    elif alternative == 'greater':
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == 'less':
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return observed_diff, perm_diffs, p_value

def get_gradient_values_for_segment(cross_species_gradients, segment_info_list, target_species, target_hem, gradient_idx):
    for segment in segment_info_list:
        if segment['species'] == target_species and segment['hem'] == target_hem:
            start_row, end_row = segment['start_row'], segment['end_row']
            if gradient_idx < cross_species_gradients.shape[1]:
                return cross_species_gradients[start_row:end_row, gradient_idx]
            else:
                print(f"ERROR: Gradient index {gradient_idx+1} is out of bounds.")
                return None
    print(f"ERROR: Segment not found for {target_species} {target_hem}.")
    return None

def perform_permutation_comparison(label1, values1, label2, values2, gradient_name, output_dir, n_permutations, alpha_level, plot_hist=False, hist_color='gray'):
    print(f"\n--- Comparing {label1} vs {label2} for {gradient_name} (Permutation Test) ---")
    if values1 is not None and values2 is not None and values1.size > 1 and values2.size > 1:
        mean1, mean2 = np.mean(values1), np.mean(values2)
        print(f"  Mean {gradient_name} {label1}: {mean1:.4f} (N={len(values1)})")
        print(f"  Mean {gradient_name} {label2}: {mean2:.4f} (N={len(values2)})")

        obs_diff, perm_diffs, p_val = permutation_test_mean_difference(values1, values2, n_permutations)
        print(f"  Observed Mean Difference ({label1} - {label2}): {obs_diff:.4f}")

        p_str = f"{p_val:.2e}" if p_val < 1e-5 else f"{p_val:.5f}"
        print(f"  Empirical p-value: {p_str}")

        if p_val < alpha_level:
            print(f"  Result: Statistically significant difference (p < {alpha_level})")
        else:
            print(f"  Result: No statistically significant difference (p >= {alpha_level})")

        if plot_hist and perm_diffs.size > 1 and not np.all(np.isnan(perm_diffs)):
            plt.figure(figsize=(8, 6))
            plt.hist(perm_diffs, bins=50, color=hist_color, edgecolor='k', alpha=0.7, label='Permutation Null Distribution')
            plt.axvline(0, color='k', linestyle=':')
            plt.axvline(obs_diff, color='b', linestyle='--', linewidth=2, label=f'Observed Diff: {obs_diff:.4f}')
            plt.title(f"Permutation Test: {label1} vs {label2} ({gradient_name})")
            plt.xlabel("Difference in Mean Gradient Value")
            plt.ylabel("Frequency")

            text_str = f"p = {p_str}"
            plt.text(0.97, 0.97, text_str, fontsize=12, color='black', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            plt.legend()

            hist_filename = f"permtest_hist_{gradient_name}_{label1.replace(' ','')}_vs_{label2.replace(' ','')}.png"
            hist_path = os.path.join(output_dir, hist_filename)
            try:
                plt.savefig(hist_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"  Saved permutation histogram to: {hist_path}")
            except Exception as e:
                print(f"  Could not save permutation histogram: {e}")
    else:
        print(f"  Could not perform comparison for {label1} vs {label2} due to insufficient data.")

def main(args):
    if not os.path.exists(args.npz_file):
        print(f"ERROR: NPZ file not found: {args.npz_file}"); return
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        npz_data = np.load(args.npz_file, allow_pickle=True)
        cross_species_gradients = npz_data['cross_species_gradients']
        segment_info_list = list(npz_data['segment_info_simple'])
        print(f"Loaded cross-species gradients with shape: {cross_species_gradients.shape}")

        if args.gradient_index >= cross_species_gradients.shape[1]:
            print(f"ERROR: --gradient_index ({args.gradient_index}) is out of bounds for available gradients ({cross_species_gradients.shape[1]}).")
            return
    except Exception as e:
        print(f"Error loading NPZ file '{args.npz_file}': {e}"); return

    gradient_name = f"G{args.gradient_index+1}"
    print(f"\n--- Permutation Testing for {gradient_name} ---")

    human_L = get_gradient_values_for_segment(cross_species_gradients, segment_info_list, "human", "L", args.gradient_index)
    human_R = get_gradient_values_for_segment(cross_species_gradients, segment_info_list, "human", "R", args.gradient_index)
    chimp_L = get_gradient_values_for_segment(cross_species_gradients, segment_info_list, "chimpanzee", "L", args.gradient_index)
    chimp_R = get_gradient_values_for_segment(cross_species_gradients, segment_info_list, "chimpanzee", "R", args.gradient_index)

    common_args = {
        "gradient_name": gradient_name,
        "output_dir": args.output_dir,
        "n_permutations": args.n_permutations,
        "alpha_level": args.alpha,
        "plot_hist": not args.no_histograms
    }

    # 1. Intra-Species Hemispheric Comparisons
    perform_permutation_comparison("Human L", human_L, "Human R", human_R, **common_args, hist_color='skyblue')
    perform_permutation_comparison("Chimp L", chimp_L, "Chimp R", chimp_R, **common_args, hist_color='salmon')

    # 2. Cross-Species Comparisons
    perform_permutation_comparison("Human L", human_L, "Chimp L", chimp_L, **common_args, hist_color='mediumpurple')
    perform_permutation_comparison("Human R", human_R, "Chimp R", chimp_R, **common_args, hist_color='gold')
    perform_permutation_comparison("Human L", human_L, "Chimp R", chimp_R, **common_args, hist_color='lightgreen')
    perform_permutation_comparison("Human R", human_R, "Chimp L", chimp_R, **common_args, hist_color='lightcoral')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run permutation tests for mean differences in cross-species gradient values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--npz_file', type=str, required=True, help="Path to the cross-species .npz file from script 6.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save analysis results and plots.")
    parser.add_argument('--gradient_index', type=int, default=DEFAULT_GRADIENT_INDEX_TO_ANALYZE, help="0-indexed gradient to analyze.")
    parser.add_argument('--n_permutations', type=int, default=DEFAULT_N_PERMUTATIONS, help="Number of permutations to run.")
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA_LEVEL, help="Alpha level for significance testing.")
    parser.add_argument('--no_histograms', action='store_true', help="If set, do not generate and save histogram plots.")

    parsed_args = parser.parse_args()

    main(parsed_args)
    print("\n--- Permutation testing script finished ---")
