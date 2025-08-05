import os
import nibabel as nib
import numpy as np
import argparse

def mask_funcgii_with_temporal_lobe(blueprint_path, mask_path, output_path, hemisphere_label):
    """
    Masks a .func.gii blueprint file with a given .func.gii mask file.
    (This function remains unchanged)
    """
    try:
        # --- Load blueprint ---
        blueprint_img = nib.load(blueprint_path)
        blueprint_data = np.stack([darray.data for darray in blueprint_img.darrays])
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
        if blueprint_img.meta:
            for k, v in blueprint_img.meta.items():
                output_img_meta[k] = v
        
        output_img_meta['AnatomicalStructurePrimary'] = structure_name
        original_description = output_img_meta.get('Description', f'Original: {os.path.basename(blueprint_path)}')
        output_img_meta['Description'] = f"{original_description} - Masked with {os.path.basename(mask_path)}"


        # --- Write masked .func.gii ---
        masked_darrays = []
        for i in range(masked_data.shape[0]):
            darray = nib.gifti.GiftiDataArray(
                data=masked_data[i].astype(np.float32),
                intent=blueprint_img.darrays[i].intent,
                datatype=blueprint_img.darrays[i].datatype,
                meta=darrays_meta[i]
            )
            masked_darrays.append(darray)

        gifti_img_out = nib.gifti.GiftiImage(
            header=blueprint_img.header,
            file_map=blueprint_img.file_map,
            meta=output_img_meta,
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
    # --- The species name is the only required argument ---
    parser.add_argument('--species_name', type=str, required=True,
                        help='Species name (e.g., "human", "chimpanzee").')
    
    # --- All other arguments are flexible and have defaults based on the README ---
    parser.add_argument('--input_avg_blueprint_basedir', type=str, default='results/1_average_blueprints',
                        help='Base directory with averaged blueprints (script will look for a species subfolder here).')
    parser.add_argument('--mask_files_basedir', type=str, default='data/masks',
                        help='Base directory with mask files (script will look for a species subfolder here).')
    parser.add_argument('--output_masked_blueprint_basedir', type=str, default='results/2_masked_average_blueprints',
                        help='Base directory where masked blueprints will be saved (species subdir will be created).')
    
    parser.add_argument('--hemispheres', type=str, default="L,R",
                        help='Comma-separated hemisphere labels (e.g., "L,R" or "L").')

    parser.add_argument('--avg_blueprint_name_pattern', type=str,
                        default="average_{species_name}_blueprint.{hemisphere}.func.gii",
                        help='Filename pattern for averaged blueprints.')
    parser.add_argument('--mask_name_pattern', type=str,
                        default="{species_name}_{hemisphere}.func.gii",
                        help='Filename pattern for masks.')
    parser.add_argument('--output_masked_name_pattern', type=str,
                        default="average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii",
                        help='Filename pattern for output masked blueprints.')

    args = parser.parse_args()

    # --- Construct full paths by combining the base directories with the species name ---
    species_avg_dir = os.path.join(args.input_avg_blueprint_basedir, args.species_name)
    species_mask_dir = os.path.join(args.mask_files_basedir, args.species_name)
    species_output_dir = os.path.join(args.output_masked_blueprint_basedir, args.species_name)

    # --- Directory validation and creation ---
    if not os.path.exists(species_avg_dir):
        print(f"Error: Input average blueprint directory not found: {species_avg_dir}")
        exit(1)
    if not os.path.exists(species_mask_dir):
        print(f"Error: Mask files directory not found: {species_mask_dir}")
        exit(1)
    if not os.path.exists(species_output_dir):
        os.makedirs(species_output_dir, exist_ok=True)
        print(f"Created output directory: {species_output_dir}")

    hemisphere_list = [h.strip() for h in args.hemispheres.split(',')]
    print(f"\n--- Starting masking for {args.species_name} hemispheres: {', '.join(hemisphere_list)} ---")

    for hem in hemisphere_list:
        # --- Format the filename patterns with the provided species name ---
        blueprint_filename = args.avg_blueprint_name_pattern.format(species_name=args.species_name, hemisphere=hem)
        blueprint_path = os.path.join(species_avg_dir, blueprint_filename)

        mask_filename = args.mask_name_pattern.format(species_name=args.species_name, hemisphere=hem)
        mask_path = os.path.join(species_mask_dir, mask_filename)

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

    print(f"\n--- {args.species_name} masking complete ---")
    print(f"Masked output saved in: {species_output_dir}")