# Cross-Species Connectivity Gradient Analysis Pipeline

This repository contains a suite of Python scripts designed to process, analyze, and visualize brain connectivity blueprints, with a focus on computing and comparing connectivity gradients across different primate species (e.g., humans and chimpanzees). The pipeline includes steps for averaging blueprints, masking, individual species gradient computation, data downsampling via k-means, cross-species gradient computation, and interactive visualization of the results.
You can also find an online version of the 2-D interactive plot of this work in https://gfreches.pythonanywhere.com/

## Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Setup](#setup)
3.  [Directory Structure](#directory-structure)
4.  [Pipeline Overview & Script Usage](#pipeline-overview--script-usage)
    * [Script 1: Average Blueprints](#script-1-average-blueprints)
    * [Script 2: Mask Blueprints](#script-2-mask-blueprints)
    * [Script 3: Compute Individual Species Gradients](#script-3-compute-individual-species-gradients)
    * [Script 4: Visualize Individual/Combined Gradients (Static Plots)](#script-4-visualize-individualcombined-gradients-static-plots)
    * [Script 5: Downsample Blueprints via K-Means](#script-5-downsample-blueprints-via-k-means)
    * [Script 6: Compute Cross-Species Gradients](#script-6-compute-cross-species-gradients)
    * [Script 7: Interactive Gradient Visualization (Dash App)](#script-7-interactive-gradient-visualization-dash-app)
    * [Script 8: Plot Cross-Species Gradients (Static Scatter Plots)](#script-8-plot-cross-species-gradients-static-scatter-plots)
    * [Script 9: Plot Consolidated Spider Plots](#script-9-plot-consolidated-spider-plots)
5.  [Outputs](#outputs)

## Prerequisites

* Python 3.7+
* Required Python packages (see `requirements.txt`)

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

A recommended directory structure for your data and results:
```
your_project_root/
├── data/
│   ├── subject_lists/              # Text files listing subject IDs per species
│   │   ├── human_subjects.txt
│   │   └── chimpanzee_subjects.txt
│   ├── raw_individual_blueprints/  # Original .dscalar.nii or .func.gii blueprints
│   │   └── <species_name>/         # e.g., human, chimpanzee
│   │       └── <subject_id_specific_directory_pattern>/ # e.g., subject1_bp_midthickness_inv_L
│   │           └── <blueprint_filename>                 # e.g., BP.L.dscalar.nii
│   └── masks/                        # Temporal lobe (or other ROI) masks
│       └── <species_name>/
│           └── <species_name>_.func.gii # e.g., human_L.func.gii
├── results/                        # Main output directory for the pipeline
│   ├── 1_average_blueprints/
│   │   └── <species_name>/         # Output of Script 1
│   ├── 2_masked_average_blueprints/
│   │   └── <species_name>/         # Output of Script 2
│   ├── 3_individual_species_gradients/
│   │   └── <species_name>/         # Output of Script 3
│   ├── 4_static_gradient_plots/
│   │   └── <species_name>/         # Output of Script 4
│   ├── 5_downsampled_blueprints/
│   │   └── <species_name>/         # Output of Script 5 (centroids, labels, etc.)
│   └── 6_cross_species_gradients/
│       ├── intermediates/          # .npy, .npz, plots from cross-species run
│       │   └── <run_identifier>/
│       └── <species_name>/         # Remapped cross-species gradients per species
│           └── cross_species_gradients_remapped/
│   ├── 8_static_cross_species_plots/ # Output of Script 8
│   └── 9_consolidated_spider_plots/  # Output of Script 9
└── code/                        # Where your Python scripts (1-9) reside
├── 1_average_blueprints.py
├── 2_mask_blueprints.py
├── 3_individual_species_gradients.py
├── 4_visualize_gradients.py
├── 5_downsample_blueprints.py
├── 6_cross_species_gradients.py
├── 7_interactive_analyse_cross_species.py
├── 8_plot_cross_species_gradients.py
└── 9_plot_consolidated_spider_plots.py
```

*Adjust paths in the script commands according to your actual structure.*

## Pipeline Overview & Script Usage

This pipeline processes connectivity blueprints through several stages:

### Script 1: Average Blueprints
* **Name**: `1_average_blueprints.py`
* **Function**: Calculates and saves the average connectivity blueprint for a given species and hemisphere from individual subject blueprint files (e.g., `.dscalar.nii`). Output is a `.func.gii` file.
* **Example Command**:
    ```bash
    python scripts/1_average_blueprints.py \
        --species_name "human" \
        --subject_list_file "data/subject_lists/human_subjects.txt" \
        --data_base_dir "data/raw_individual_blueprints/" \
        --output_base_dir "results/1_average_blueprints/" \
        --hemispheres "L,R" \
        --subject_dir_pattern "{subject_id}_bp_midthickness_inv_{hemisphere}" \
        --blueprint_filename "BP.{hemisphere}.dscalar.nii"
    ```
* **Key Arguments**:
    * `--species_name`: Name of the species.
    * `--subject_list_file`: Path to file listing subject IDs.
    * `--data_base_dir`: Base directory for raw input data.
    * `--output_base_dir`: Base directory where averaged blueprints will be saved (a subdirectory for `species_name` will be created).
    * `--subject_dir_pattern`: Pattern for subject-specific data directories.
    * `--blueprint_filename`: Filename of blueprint files within subject directories.

### Script 2: Mask Blueprints
* **Name**: `2_mask_blueprints.py`
* **Function**: Masks the averaged `.func.gii` blueprints (from Script 1) with a given region of interest (ROI) mask (e.g., temporal lobe).
* **Example Command**:
    ```bash
    python code/2_mask_blueprints.py \
        --species_name "human" \
        --input_avg_blueprint_basedir "results/1_average_blueprints/" \
        --mask_files_basedir "data/masks/" \
        --output_masked_blueprint_basedir "results/2_masked_average_blueprints/" \
        --hemispheres "L,R" \
        --mask_name_pattern "{species_name}_{hemisphere}.func.gii" \
        --avg_blueprint_name_pattern "average_{species_name}_blueprint.{hemisphere}.func.gii" \
        --output_masked_name_pattern "average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii"
    ```
* **Key Arguments**:
    * `--input_avg_blueprint_basedir`: Directory with averaged blueprints from Script 1 (expects species subfolders).
    * `--mask_files_basedir`: Directory with mask `.func.gii` files (expects species subfolders).
    * `--output_masked_blueprint_basedir`: Directory where masked blueprints will be saved (species subfolder created).
    * `--mask_name_pattern`: Filename pattern for mask files.

### Script 3: Compute Individual Species Gradients
* **Name**: `3_individual_species_gradients.py` (or `3_compute_gradients.py`)
* **Function**: Computes connectivity gradients within the masked region for each species/hemisphere separately using spectral embedding.
* **Example Command**:
    ```bash
    python code/3_individual_species_gradients.py \
        --input_masked_blueprint_dir "results/2_masked_average_blueprints/" \
        --input_mask_dir "data/masks/" \
        --output_dir "results/3_individual_species_gradients/" \
        --species_list "human,chimpanzee" \
        --hemispheres "L,R" \
        --masked_blueprint_pattern "average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii" \
        --mask_pattern "{species_name}_{hemisphere}.func.gii" \
        --max_gradients 10 
    ```
* **Key Arguments**:
    * `--input_masked_blueprint_dir`: Output directory from Script 2.
    * `--input_mask_dir`: Directory with mask files (used to define TL vertices).
    * `--output_dir`: Base directory for saving gradients, plots, and intermediates.
    * `--species_list`: Comma-separated species to process.
    * `--masked_blueprint_pattern`: Filename pattern for inputs from Script 2.
    * `--mask_pattern`: Filename pattern for mask files.

### Script 4: Visualize Individual/Combined Gradients (Static Plots)
* **Name**: `4_visualize_gradients.py` (or `4_individual_species_gradients_analysis.py`)
* **Function**: Generates static 2D or 3D scatter plots of temporal lobe vertices in gradient space, using outputs from Script 3. Can plot "individual" hemisphere gradients or "combined" (if Script 3 produces such outputs, or adapt to use outputs of Script 6 for combined visualization).
* **Example (Plotting Individual Gradients from Script 3):**
    ```bash
    python code/4_visualize_gradients.py \
        --gradient_input_dir "results/3_individual_species_gradients/" \
        --mask_input_dir "data/masks/" \
        --plot_output_dir "results/4_static_gradient_plots/" \
        --species_list "human,chimpanzee" \
        --mask_pattern "{species_name}_{hemisphere}.func.gii" \
        --plot_type "individual" \
        --gradient_pattern_individual "all_computed_gradients_{species_name}_{hemisphere}_SEPARATE.func.gii" \
        --gradients_to_plot "1,2" 
    ```
* **Key Arguments**:
    * `--gradient_input_dir`: Output directory from Script 3 (or Script 6 if plotting combined cross-species).
    * `--plot_type`: "individual" or "combined".
    * `--gradient_pattern_individual` / `--gradient_pattern_combined`: Patterns for gradient files.
    * `--gradients_to_plot`: Comma-separated 1-indexed gradient numbers (e.g., "1,2").

### Script 5: Downsample Blueprints via K-Means
* **Name**: `5_downsample_blueprints.py`
* **Function**: Downsamples masked average blueprints (from Script 2) for specified source species using k-means clustering. The number of clusters (`k`) is determined by the temporal lobe vertex count of a specified `target_k_species`. Outputs include centroid profiles (`.npy`), vertex labels (`.npy`), and a visual downsampled blueprint (`.func.gii`).
* **Example Command**:
    ```bash
    python code/5_downsample_blueprints.py \
        --input_masked_blueprint_dir "results/2_masked_average_blueprints/" \
        --input_mask_dir "data/masks/" \
        --output_dir "results/5_downsampled_blueprints/" \
        --source_species_list "human" \
        --target_species_for_k "chimpanzee" \
        --hemispheres "L,R" \
        --masked_blueprint_pattern "average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii" \
        --mask_pattern "{species_name}_{hemisphere}.func.gii" \
        --n_tracts_expected 20
    ```
* **Key Arguments**:
    * `--input_masked_blueprint_dir`: Output from Script 2.
    * `--input_mask_dir`: Directory with masks (for source species and for target_k_species).
    * `--output_dir`: Base directory for downsampled outputs.
    * `--source_species_list`: Species to downsample.
    * `--target_species_for_k`: Species whose TL vertex count defines `k`.

### Script 6: Compute Cross-Species Gradients
* **Name**: `6_cross_species_gradients.py`
* **Function**: Performs a joint spectral embedding using a combination of data: original masked blueprints for the `target_k_species` (e.g., chimpanzee, from Script 2) and downsampled centroid profiles for other species (e.g., human, from Script 5). Outputs remapped cross-species gradients as `.func.gii` for each species and an `.npz` archive with detailed embedding information.
* **Example Command**:
    ```bash
    python code/6_cross_species_gradients.py \
        --target_species_bp_dir "results/2_masked_average_blueprints/" \
        --other_species_downsampled_dir "results/5_downsampled_blueprints/" \
        --mask_dir "data/masks/" \
        --output_dir "results/6_cross_species_gradients/" \
        --species_list_for_lle "human,chimpanzee" \
        --target_k_species "chimpanzee" \
        --hemispheres_to_process "L,R" \
        --target_species_bp_pattern "average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii" \
        --downsampled_centroid_pattern "{species_name}_{hemisphere}_k{k_val}_centroids.npy" \
        --mask_pattern "{species_name}_{hemisphere}.func.gii" \
        --num_gradients_to_save 10
    ```
* **Key Arguments**:
    * `--target_species_bp_dir`: Path to masked blueprints of the species defining `k` (Script 2 output).
    * `--other_species_downsampled_dir`: Path to downsampled data for other species (Script 5 output).
    * `--mask_dir`: Path to masks for all species (used for remapping).
    * `--output_dir`: Base directory for cross-species outputs.
    * `--species_list_for_lle`: All species to include in the joint embedding.
    * `--target_k_species`: The reference species.

### Script 7: Interactive Gradient Visualization (Dash App)
* **Name**: `7_interactive_analyse_cross_species.py`
* **Function**: Launches an interactive Dash web application to visualize the cross-species gradients (G1 vs G2) from the `.npz` file generated by Script 6. Allows clicking on points to see their original connectivity profiles (spider plots) and find closest neighbors.
* **Example Command**:
    ```bash
    python code/7_interactive_analyse_cross_species.py \
        --npz_file "results/6_cross_species_gradients/intermediates/human_chimpanzee_CrossSpecies_kRef_chimpanzee/cross_species_embedding_data_human_chimpanzee_CrossSpecies_kRef_chimpanzee.npz" \
        --average_bp_dir "results/2_masked_average_blueprints/" \
        --n_tracts 20 \
        --tract_names "AC,AF,AR,CBD,CBP,CBT,CST,FA,FMI,FMA,FX,IFOF,ILF,MDLF,OR,SLF I,SLF II,SLF III,UF,VOF" \
        --port 8051 
    ```
* **Key Arguments**:
    * `--npz_file`: Path to the `.npz` output file from Script 6.
    * `--average_bp_dir`: Path to masked average blueprints (Script 2 output), needed for spider plots.
    * `--n_tracts`, `--tract_names`: Configuration for spider plots.
    * `--host`, `--port`, `--debug`: For running the Dash application.
* **Accessing the App**: After running, open your web browser and go to `http://<host>:<port>/` (e.g., `http://127.0.0.1:8051/`).

### Script 8: Plot Cross-Species Gradients (Static Scatter Plots)
* **Name**: `8_plot_cross_species_gradients.py`
* **Function**: Generates static 2D scatter plots from the cross-species gradient data (`.npz` file) created by Script 6. This is useful for creating publication-quality figures of specific gradient comparisons (e.g., G1 vs G2, G1 vs G3).
* **Example Command**:
    ```bash
    python code/8_plot_cross_species_gradients.py \
        --npz_file "results/6_cross_species_gradients/intermediates/human_chimpanzee_CrossSpecies_kRef_chimpanzee/cross_species_embedding_data_human_chimpanzee_CrossSpecies_kRef_chimpanzee.npz" \
        --output_dir "results/8_static_cross_species_plots/" \
        --gradient_pairs "0_1,0_2"
    ```
* **Key Arguments**:
    * `--npz_file`: Path to the `.npz` output file from Script 6 containing the cross-species embedding data.
    * `--output_dir`: Directory where the output `.png` scatter plots will be saved.
    * `--gradient_pairs`: Comma-separated list of gradient pairs to plot. The indices are 0-based. For example, `"0_1"` plots Gradient 1 vs Gradient 2. `"0_1,0_2,1_2"` would create three separate plots.

### Script 9: Plot Consolidated Spider Plots
* **Name**: `9_plot_consolidated_spider_plots.py`
* **Function**: For each specified cross-species gradient, this script identifies the vertices (or centroids) that show the minimum and maximum expression of that gradient for each species/hemisphere. It then generates two consolidated spider plots per gradient: one for the max-expressing vertices and one for the min-expressing vertices, overlaying the blueprint profiles from all four species/hemisphere combinations.
* **Example Command**:
    ```bash
    python code/9_plot_consolidated_spider_plots.py \
        --npz_file "results/6_cross_species_gradients/intermediates/human_chimpanzee_CrossSpecies_kRef_chimpanzee/cross_species_embedding_data_human_chimpanzee_CrossSpecies_kRef_chimpanzee.npz" \
        --masked_blueprint_dir "results/2_masked_average_blueprints/" \
        --downsampled_data_dir "results/5_downsampled_blueprints/" \
        --mask_base_dir "data/masks/" \
        --output_dir "results/9_consolidated_spider_plots/" \
        --gradients "0,1" \
        --target_k_species "chimpanzee"
    ```
* **Key Arguments**:
    * `--npz_file`: Path to the cross-species `.npz` file from script 6.
    * `--masked_blueprint_dir`: Directory for masked average blueprints (script 2 output), used to get the blueprint profiles for plotting.
    * `--downsampled_data_dir`: Directory for downsampled data (script 5 output), needed to map centroids back to original vertices for non-target species.
    * `--mask_base_dir`: Directory for all species' masks, used to identify original temporal lobe vertices.
    * `--output_dir`: Directory to save the output spider plot images.
    * `--gradients`: Comma-separated list of 0-indexed gradients to analyze.
    * `--target_k_species`: The species that was used as the reference for k-means downsampling (e.g., 'chimpanzee'). This is crucial for correctly locating data for other species.

## Outputs

The pipeline generates several types of outputs in the specified `results` subdirectories:

* **Averaged Blueprints**: `.func.gii` files (Script 1).
* **Masked Blueprints**: `.func.gii` files, focused on the ROI (Script 2).
* **Individual Species Gradients**: `.func.gii` gradient maps, `.npy` intermediate files, and dimensionality evaluation plots (Script 3).
* **Static Gradient Scatter Plots**: `.png` files (Script 4).
* **Downsampled Blueprint Data**: `.npy` files for centroids and labels, and a visual `.func.gii` (Script 5).
* **Cross-Species Gradients**:
    * Remapped `.func.gii` gradient files for each species.
    * An `.npz` archive containing the raw joint embedding, segment information, and eigenvalues.
    * Intermediate `.npy` files and dimensionality evaluation plots (Script 6).
* **Interactive Visualization**: A web application (Script 7).
* **Cross-Species Scatter Plots**: Static `.png` files showing relationships between different cross-species gradients (Script 8).
* **Consolidated Spider Plots**: `.png` files showing the blueprint profiles of the vertices that express the minimum and maximum values for each cross-species gradient (Script 9).
