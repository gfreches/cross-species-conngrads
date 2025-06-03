import os
import nibabel as nib
import numpy as np
import argparse

def mask_funcgii_with_temporal_lobe(blueprint_path, mask_path, output_path, hemisphere_label):
    """
    Masks a .func.gii blueprint file with a given .func.gii mask file.

    Args:
        blueprint_path (str): Path to the input .func.gii blueprint file.
        mask_path (str): Path to the .func.gii mask file (expects a single darray).
        output_path (str): Path to save the masked .func.gii blueprint file.
        hemisphere_label (str): Hemisphere identifier ('L' or 'R').
    """
    try:
        # --- Load blueprint ---
        blueprint_img = nib.load(blueprint_path)
        blueprint_data = np.stack([darray.data for darray in blueprint_img.darrays])
        # This carries over darray-specific metadata from the input blueprint
        darrays_meta = [darray.meta for darray in blueprint_img.darrays]


        if not blueprint_data.size:
            print(f"Error: No data arrays found in blueprint: {blueprint_path}")
            return

        # --- Load mask ---
        mask_img = nib.load(mask_path)
        if not mask_img.darrays:
            print(f"Error: No data arrays found in mask: {mask_path}")
            return
        mask = mask_img.darrays[0].data.astype(bool)

        print(f"Loaded blueprint: {os.path.basename(blueprint_path)} (shape {blueprint_data.shape}), "
              f"mask: {os.path.basename(mask_path)} (shape {mask.shape}), "
              f"vertices in mask: {np.sum(mask)}.")

        if blueprint_data.shape[1] != mask.shape[0]:
            print(f"Error: Vertex count mismatch between blueprint ({blueprint_data.shape[1]}) "
                  f"and mask ({mask.shape[0]}) for {os.path.basename(blueprint_path)}.")
            return

        # --- Apply mask: outside mask set to zero ---
        masked_data = np.where(mask, blueprint_data, 0)

        # --- Determine anatomical structure and prepare image-level metadata ---
        structure_name = 'CortexLeft' if hemisphere_label.upper() == 'L' else 'CortexRight'
        
        output_img_meta = nib.gifti.GiftiMetaData()
        # Try to copy existing metadata from the input blueprint's image level
        if blueprint_img.meta:
            for k, v in blueprint_img.meta.items():
                output_img_meta[k] = v
        
        # Set/Overwrite AnatomicalStructurePrimary and update Description
        output_img_meta['AnatomicalStructurePrimary'] = structure_name
        original_description = output_img_meta.get('Description', f'Original: {os.path.basename(blueprint_path)}')
        output_img_meta['Description'] = f"{original_description} - Masked with {os.path.basename(mask_path)}"


        # --- Write masked .func.gii ---
        masked_darrays = []
        for i in range(masked_data.shape[0]):
            # Use the darray_meta copied from the input blueprint.
            # Script 1 should have set 'AnatomicalStructurePrimary' correctly here.
            darray = nib.gifti.GiftiDataArray(
                data=masked_data[i].astype(np.float32),
                intent=blueprint_img.darrays[i].intent if i < len(blueprint_img.darrays) else nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=blueprint_img.darrays[i].datatype if i < len(blueprint_img.darrays) else nib.nifti1.data_type_codes.code['NIFTI_TYPE_FLOAT32'],
                meta=darrays_meta[i] if i < len(darrays_meta) else None
            )
            masked_darrays.append(darray)

        gifti_img_out = nib.gifti.GiftiImage(
            header=blueprint_img.header,
            file_map=blueprint_img.file_map,
            meta=output_img_meta, # Set the prepared image-level metadata
            darrays=masked_darrays
        )

        nib.save(gifti_img_out, output_path)
        print(f"Saved masked blueprint to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred while processing {blueprint_path} and {mask_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask averaged .func.gii blueprints with a given mask for a species.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--species_name', type=str, required=True,
                        help='Species name (e.g., "human", "chimpanzee").')
    parser.add_argument('--input_avg_blueprint_basedir', type=str, required=True,
                        help='Directory with averaged blueprints from script 1 (should contain a subfolder named after species).')
    parser.add_argument('--mask_files_basedir', type=str, required=True,
                        help='Directory with mask .func.gii files.')
    parser.add_argument('--output_masked_blueprint_basedir', type=str, required=True,
                        help='Directory where masked blueprints will be saved (species subdir will be created).')
    parser.add_argument('--hemispheres', type=str, default="L,R",
                        help='Comma-separated hemisphere labels (e.g., "L,R" or "L").')
    parser.add_argument('--avg_blueprint_name_pattern', type=str,
                        default="average_{species_name}_blueprint.{hemisphere}.func.gii",
                        help='Filename pattern for averaged blueprints (placeholders: {species_name}, {hemisphere}).')
    parser.add_argument('--mask_name_pattern', type=str, required=True,
                        help='Filename pattern for masks (placeholders: {species_name}, {hemisphere}).')
    parser.add_argument('--output_masked_name_pattern', type=str,
                        default="average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii",
                        help='Filename pattern for output masked blueprints (placeholders: {species_name}, {hemisphere}).')

    args = parser.parse_args()

    species_avg_dir = os.path.join(args.input_avg_blueprint_basedir, args.species_name)
    species_output_dir = os.path.join(args.output_masked_blueprint_basedir, args.species_name)

    if not os.path.exists(species_avg_dir):
        print(f"Error: Input average blueprint directory for {args.species_name} not found: {species_avg_dir}")
        print("Ensure script 1 output is in the expected location.")
        exit(1)
    if not os.path.exists(args.mask_files_basedir):
        print(f"Error: Mask files directory not found: {args.mask_files_basedir}")
        exit(1)
    if not os.path.exists(species_output_dir):
        os.makedirs(species_output_dir, exist_ok=True)
        print(f"Created output directory for masked blueprints: {species_output_dir}")

    hemisphere_list = [h.strip() for h in args.hemispheres.split(',')]
    print(f"\n--- Starting masking for {args.species_name} hemispheres: {', '.join(hemisphere_list)} ---")

    for hem in hemisphere_list:
        try:
            blueprint_filename = args.avg_blueprint_name_pattern.format(species_name=args.species_name, hemisphere=hem)
            blueprint_path = os.path.join(species_avg_dir, blueprint_filename)

            mask_filename = args.mask_name_pattern.format(species_name=args.species_name, hemisphere=hem)
            mask_path = os.path.join(args.mask_files_basedir, mask_filename)

            output_filename = args.output_masked_name_pattern.format(species_name=args.species_name, hemisphere=hem)
            output_path = os.path.join(species_output_dir, output_filename)

            if not os.path.exists(blueprint_path):
                print(f"Blueprint not found, skipping: {blueprint_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"Mask not found, skipping: {mask_path}")
                continue

            print(f"  Input Blueprint: {blueprint_path}")
            print(f"  Mask:           {mask_path}")
            print(f"  Output Masked:  {output_path}")
            mask_funcgii_with_temporal_lobe(blueprint_path, mask_path, output_path, hem)
        except KeyError as e:
            print(f"Error: Placeholder {e} in filename pattern is missing. Skipping {hem}.")
            continue

    print(f"\n--- {args.species_name} masking complete ---")
    print(f"Masked output saved in: {species_output_dir}")
