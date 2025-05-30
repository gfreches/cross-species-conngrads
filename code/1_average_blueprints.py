import os
import nibabel as nib
import numpy as np
import argparse 

def create_average_funcgii(list_of_subject_blueprint_files, hemisphere_label, output_dir, species_name):
    """
    Calculates and saves the average connectivity blueprint for a given species and hemisphere
    as a .func.gii file.

    Args:
        list_of_subject_blueprint_files (list): List of full file paths to the .dscalar.nii blueprint files.
        hemisphere_label (str): Hemisphere identifier ('L' or 'R').
        output_dir (str): Directory to save the averaged blueprint.
        species_name (str): Name of the species (e.g., "human", "chimpanzee") for naming outputs.
    """
    print(f"Processing {species_name}, hemisphere: {hemisphere_label}")

    all_subject_data_list = []
    first_subject_img = None

    # --- 1. Load data for all subjects ---
    for file_path in list_of_subject_blueprint_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        try:
            img = nib.load(file_path)
            data = img.get_fdata()

            if first_subject_img is None:
                first_subject_img = img
                expected_shape = data.shape

            if data.shape != expected_shape:
                print(f"Error: Data shape mismatch for file {file_path}.")
                print(f"Expected shape: {expected_shape}, Got shape: {data.shape}")
                continue

            all_subject_data_list.append(data)

        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}")
            continue

    if not all_subject_data_list:
        print(f"No data loaded for {species_name}, hemisphere {hemisphere_label}. Skipping averaging.")
        return

    if first_subject_img is None: # Should be caught by the previous check, but good for robustness
        print(f"Could not load any template image for {species_name}, hemisphere {hemisphere_label}. Skipping.")
        return

    print(f"Successfully loaded data for {len(all_subject_data_list)} subjects for {species_name} hemisphere {hemisphere_label}.")

    # --- 2. Stack and average the data ---
    if not all(arr.shape == all_subject_data_list[0].shape for arr in all_subject_data_list):
        print(f"Error: Not all subject data arrays have the same shape for {species_name}, hemisphere {hemisphere_label}.")
        return

    stacked_data = np.stack(all_subject_data_list, axis=0)
    average_data = np.mean(stacked_data, axis=0)
    print(f"Averaged data shape for {species_name}, hemisphere {hemisphere_label}: {average_data.shape}")

    # Original sum_check based on the initial average
    original_sum_check = np.sum(average_data, axis=0)
    print(f"Min sum of tracts per vertex BEFORE re-normalization ({species_name}, {hemisphere_label}): {np.min(original_sum_check):.6f}")
    print(f"Max sum of tracts per vertex BEFORE re-normalization ({species_name}, {hemisphere_label}): {np.max(original_sum_check):.6f}")
    print(f"Mean sum of tracts per vertex BEFORE re-normalization ({species_name}, {hemisphere_label}): {np.mean(original_sum_check):.6f}")

    if not np.allclose(original_sum_check, 1.0, atol=1e-4):
        print(f"Warning: Sum of tracts BEFORE re-normalization for {species_name}, hemisphere {hemisphere_label} is not consistently 1. Re-normalizing...")

    # --- Re-normalize average_data so each vertex sums to 1 ---
    denominator = original_sum_check.copy()

    # Prevent division by zero for vertices that had all zeros in the input
    zero_sum_mask = np.isclose(denominator, 0.0, atol=1e-9)
    denominator[zero_sum_mask] = 1.0  # Avoid division by zero, result will be 0 if numerator was 0

    renormalized_average_data = average_data / denominator[np.newaxis, :]
    # Ensure that where original sum was zero, the output is also explicitly zero
    # This handles cases where average_data might have had non-zero values due to float precision
    # even if the original sum was zero (though less likely with a direct sum)
    if average_data.ndim == renormalized_average_data.ndim and average_data.shape[0] == renormalized_average_data.shape[0]:
         for i in range(average_data.shape[0]):
            renormalized_average_data[i, zero_sum_mask] = 0.0
    else: # Fallback if shapes are unexpected, though ideally they match
        renormalized_average_data[:, zero_sum_mask] = 0.0


    average_data = renormalized_average_data # Update average_data for saving

    sum_check = np.sum(average_data, axis=0)
    print(f"Min sum of tracts per vertex AFTER re-normalization ({species_name}, {hemisphere_label}): {np.min(sum_check):.6f}")
    print(f"Max sum of tracts per vertex AFTER re-normalization ({species_name}, {hemisphere_label}): {np.max(sum_check):.6f}")
    print(f"Mean sum of tracts per vertex AFTER re-normalization ({species_name}, {hemisphere_label}): {np.mean(sum_check):.6f}")

    non_zero_original_sums_mask = ~zero_sum_mask
    if np.any(non_zero_original_sums_mask):
        if not np.allclose(sum_check[non_zero_original_sums_mask], 1.0, atol=1e-6):
            print(f"Warning: Sum of tracts AFTER re-normalization for {species_name}, hemisphere {hemisphere_label} is still not consistently 1 for some originally non-zero-sum vertices.")
    # For vertices that were originally all zero, their sum should remain zero.
    if np.any(zero_sum_mask):
        if not np.all(np.isclose(sum_check[zero_sum_mask], 0.0, atol=1e-9)):
            print(f"Warning: Some vertices with original zero sums do not have a zero sum after re-normalization for {species_name}, hemisphere {hemisphere_label}.")


    # --- 3. Create and save the average .func.gii file ---
    try:
        num_maps = average_data.shape[0]

        gifti_data_arrays = []
        for i in range(num_maps):
            map_name = f"Tract_{i+1}"
            meta = nib.gifti.GiftiMetaData()
            try: # Handle different nibabel versions for GiftiMetaData.add_meta if necessary
                meta.add_meta('Name', map_name)
            except AttributeError: # Older nibabel versions might need this
                meta.data.append(nib.gifti.GiftiNVPairs('Name', map_name))

            darray = nib.gifti.GiftiDataArray(
                data=average_data[i, :].astype(np.float32),
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes.code['NIFTI_TYPE_FLOAT32'],
                meta=meta
            )
            gifti_data_arrays.append(darray)

        gifti_image_meta = nib.gifti.GiftiMetaData()
        description = f'Average {species_name} blueprint, Hemisphere {hemisphere_label}'
        try: # Handle different nibabel versions for GiftiMetaData.add_meta if necessary
            gifti_image_meta.add_meta('Description', description)
        except AttributeError: # Older nibabel versions might need this
            gifti_image_meta.data.append(nib.gifti.GiftiNVPairs('Description', description))

        average_gii_img = nib.gifti.GiftiImage(
            darrays=gifti_data_arrays,
            meta=gifti_image_meta
        )

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True) # exist_ok=True is safer
                print(f"Created output directory: {output_dir}")
            except Exception as e_makedir:
                print(f"Could not create output directory {output_dir}: {e_makedir}. Files may not be saved.")
                return

        output_filename = f"average_{species_name}_blueprint.{hemisphere_label}.func.gii"
        output_file_path = os.path.join(output_dir, output_filename)

        nib.save(average_gii_img, output_file_path)
        print(f"Successfully saved average {species_name} blueprint to: {output_file_path}")

    except Exception as e:
        print(f"Error creating or saving GiftiImage for {species_name}, hemisphere {hemisphere_label}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an average functional connectivity blueprint for a given species.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--species_name', type=str, required=True,
                        help='Name of the species (e.g., "human", "macaque"). Used for output naming and messages.')
    parser.add_argument('--subject_list_file', type=str, required=True,
                        help='Path to a text file containing subject identifiers, one per line.')
    parser.add_argument('--data_base_dir', type=str, required=True,
                        help='Base directory for this species\' raw input data.')
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help='Base directory where average blueprints will be saved. A subdirectory for species_name will be created here.')
    parser.add_argument('--hemispheres', type=str, default="L,R",
                        help='Comma-separated list of hemisphere labels to process (e.g., "L,R" or "L").')
    parser.add_argument('--subject_dir_pattern', type=str, required=True,
                        help='Pattern for subject-specific data directory relative to data_base_dir. '
                             'Use {subject_id} and {hemisphere} as placeholders. '
                             'Example: "{subject_id}_bp_midthickness_inv_{hemisphere}"')
    parser.add_argument('--blueprint_filename', type=str, required=True,
                        help='Filename of the blueprint file within each subject\'s directory. '
                             'Can use {hemisphere} as a placeholder. Example: "BP.{hemisphere}.dscalar.nii"')

    args = parser.parse_args()

    # --- Prepare species-specific output directory ---
    species_output_dir = os.path.join(args.output_base_dir, args.species_name)
    if not os.path.exists(species_output_dir):
        os.makedirs(species_output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {species_output_dir}")

    # --- Load subject list ---
    try:
        with open(args.subject_list_file, 'r') as f:
            subject_ids = [line.strip() for line in f if line.strip()]
        if not subject_ids:
            print(f"Error: Subject list file '{args.subject_list_file}' is empty or contains only whitespace.")
            exit(1)
        print(f"Loaded {len(subject_ids)} subject IDs from '{args.subject_list_file}'.")
    except FileNotFoundError:
        print(f"Error: Subject list file not found: '{args.subject_list_file}'")
        exit(1)
    except Exception as e:
        print(f"Error reading subject list file '{args.subject_list_file}': {e}")
        exit(1)

    # --- Process each hemisphere ---
    hemisphere_list = [h.strip() for h in args.hemispheres.split(',')]
    print(f"\n--- Starting {args.species_name} Blueprint Processing for hemispheres: {', '.join(hemisphere_list)} ---")

    for hem in hemisphere_list:
        print(f"\nProcessing Hemisphere: {hem}")
        blueprint_files_for_hemisphere = []
        for subject_id in subject_ids:
            try:
                # Construct the path to the subject's specific data directory
                subject_specific_dir_fragment = args.subject_dir_pattern.format(subject_id=subject_id, hemisphere=hem)
                # Construct the blueprint filename, potentially with hemisphere
                current_blueprint_filename = args.blueprint_filename.format(hemisphere=hem)

                file_path = os.path.join(args.data_base_dir, subject_specific_dir_fragment, current_blueprint_filename)
                blueprint_files_for_hemisphere.append(file_path)
            except KeyError as e:
                print(f"Error: Placeholder {e} in --subject_dir_pattern or --blueprint_filename is not being filled correctly. "
                      f"Ensure your patterns match available placeholders ({{subject_id}}, {{hemisphere}}).")
                print(f"Skipping subject {subject_id} for hemisphere {hem}.")
                continue # Skip this subject for this hemisphere if path construction fails

        existing_blueprint_files = [fp for fp in blueprint_files_for_hemisphere if os.path.exists(fp)]

        # Print which files were expected vs. found for clarity, especially if some are missing
        if len(existing_blueprint_files) < len(blueprint_files_for_hemisphere):
            print(f"Expected {len(blueprint_files_for_hemisphere)} files for hemisphere {hem}, found {len(existing_blueprint_files)}.")
            missing_files = set(blueprint_files_for_hemisphere) - set(existing_blueprint_files)
            if missing_files:
                print("Missing files:")
                for mf in sorted(list(missing_files))[:5]: # Print first 5 missing
                     print(f"  {mf}")
                if len(missing_files) > 5:
                    print(f"  ...and {len(missing_files) - 5} more.")


        if existing_blueprint_files:
            print(f"Found {len(existing_blueprint_files)} existing blueprint files for {args.species_name}, hemisphere {hem}.")
            create_average_funcgii(
                list_of_subject_blueprint_files=existing_blueprint_files,
                hemisphere_label=hem,
                output_dir=species_output_dir, # Use the species-specific output directory
                species_name=args.species_name
            )
        else:
            print(f"No existing {args.species_name} blueprint files found for hemisphere {hem} using the provided patterns and subject list under '{args.data_base_dir}'.")

    print(f"\n--- {args.species_name} Blueprint Processing Complete ---")
    print(f"Output for {args.species_name} saved in: {species_output_dir}")
    print("\nOverall processing complete for this species run.")
