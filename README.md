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
    * [Script 10: Run Permutation Analysis](#script-10-run-permutation-analysis)

5.  [Outputs](#outputs)

## Prerequisites

* Python 3.7+
* Required Python packages (see `requirements.txt`)

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/gfreches/cross-species-conngrads.git
    cd cross-species-conngrads
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
│   ├── 9_consolidated_spider_plots/  # Output of Script 9
│   └── 10_permutation_analysis/      # Output of Script 10
└── code/                        # Where your Python scripts (1-10) reside
├── 1_average_blueprints.py
├── 2_mask_blueprints.py
├── 3_individual_species_gradients.py
├── 4_visualize_gradients.py
├── 5_downsample_blueprints.py
├── 6_cross_species_gradients.py
├── 7_interactive_analyse_cross_species.py
├── 8_plot_cross_species_gradients.py
├── 9_plot_consolidated_spider_plots.py
└── 10_run_permutation_analysis.py

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
* **Function**: Masks the averaged `.func.gii` blueprints (from Script 1) with a given region of interest (ROI) mask (e.g., temporal lobe). The script intelligently uses the provided species name to find the correct files and directories, but allows for flexibility by overriding the default paths and patterns.
* **Example Usage**:

    **Basic command (using all default paths):**
    ```bash
    python code/2_mask_blueprints.py --species_name "human"
    ```

    **Advanced command (overriding the output directory):**
    ```bash
    python code/2_mask_blueprints.py \
        --species_name "human" \
        --output_masked_blueprint_basedir "results_custom/masked_data"
    ```
    *(This will use default locations for inputs but save the output to `results_custom/masked_data/human/`)*

* **Key Arguments**:
    * `--species_name`: **(Required)** The name of the species (e.g., "human").
    * `--input_avg_blueprint_basedir`: **(Optional)** Base directory for averaged blueprints. (Default: `results/1_average_blueprints`)
    * `--mask_files_basedir`: **(Optional)** Base directory for mask files. (Default: `data/masks`)
    * `--output_masked_blueprint_basedir`: **(Optional)** Base directory where masked blueprints will be saved. (Default: `results/2_masked_average_blueprints`)
    * `--hemispheres`: **(Optional)** Comma-separated hemispheres to process. (Default: `"L,R"`)
    * `--avg_blueprint_name_pattern`, `--mask_name_pattern`, `--output_masked_name_pattern`: **(Optional)** Arguments to specify custom filename patterns.
  
### Script 3: Compute Individual Species Gradients
* **Name**: `3_individual_species_gradients.py`
* **Function**: Computes connectivity gradients within the masked region for each species/hemisphere separately using spectral embedding. It automatically locates the necessary inputs from Script 2 and saves outputs to the correct directory based on the project's standard structure.
* **Example Usage**:

    **Basic command (processing multiple species):**
    ```bash
    python code/3_individual_species_gradients.py --species_list "human,chimpanzee"
    ```

    **Advanced command (overriding a technical parameter):**
    ```bash
    python code/3_individual_species_gradients.py --species_list "human" --max_gradients 15
    ```

* **Key Arguments**:
    * `--species_list`: **(Required)** A comma-separated list of the species to process.
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: `.`)
    * `--hemispheres`: **(Optional)** Comma-separated list of hemispheres to process. (Default: `"L,R"`)
    * `--max_gradients`: **(Optional)** Maximum number of gradients to compute. (Default: `10`)
    * `--max_k_knn`: **(Optional)** Maximum value of *k* for the k-NN graph search. (Default: `150`)
    * `--default_k_knn`: **(Optional)** Fallback *k* value. (Default: `20`)
    * `--min_gain_dim_select`: **(Optional)** Minimum gain in score to select an additional gradient. (Default: `0.1`)
 
### Script 4: Visualize Individual/Combined Gradients (Static Plots)
* **Name**: `4_individual_species_gradients_analysis.py`
* **Function**: Generates static 2D or 3D scatter plots of vertices in gradient space, using outputs from Script 3. It automatically finds the required files based on the standard project structure.
* **Example Usage**:

    **Plot combined gradients (G1 vs G2) for multiple species:**
    ```bash
    python code/4_individual_species_gradients_analysis.py --species_list "human,chimpanzee" --gradients_to_plot "1,2"
    ```

    **Plot individual hemisphere gradients for one species:**
    ```bash
    python code/4_individual_species_gradients_analysis.py --species_list "human" --plot_type "individual" --gradients_to_plot "1,2"
    ```

    **Generate a 3D plot (G1 vs G2 vs G3) and all its 2D projections:**
    ```bash
    python code/4_individual_species_gradients_analysis.py --species_list "human" --gradients_to_plot "1,2,3" --plot_2d_projections
    ```

* **Key Arguments**:
    * `--species_list`: **(Required)** A comma-separated list of species to process.
    * `--gradients_to_plot`: **(Optional)** Comma-separated 1-indexed gradient numbers to plot (e.g., `"1,2"` or `"1,2,3"`). Defaults to `"1,2,3"`.
    * `--plot_type`: **(Optional)** Type of plot to generate. Choices: `combined`, `individual`. Defaults to `combined`.
    * `--plot_2d_projections`: **(Optional)** When plotting 3 gradients, add this flag to also generate all three corresponding 2D scatter plots.
    * `--project_root`: **(Optional)** Path to the project's root directory. Defaults to the current directory (`.`).p

### Script 5: Downsample Blueprints via K-Means
* **Name**: `5_downsample_blueprints.py`
* **Function**: Downsamples masked average blueprints (from Script 2) for specified source species using k-means clustering. The number of clusters (`k`) is determined by the temporal lobe vertex count of a specified `target_k_species`. Outputs include centroid profiles (`.npy`), vertex labels (`.npy`), and a visual downsampled blueprint (`.func.gii`).
* **Example Command**:
    ```bash
    python code/5_downsample_blueprints.py \
        --source_species_list "human" \
        --target_species_for_k "chimpanzee"
    ```
* **Key Arguments**:
    * `--source_species_list`: **(Required)** Comma-separated list of source species to downsample.
    * `--target_species_for_k`: **(Required)** The species whose temporal lobe vertex count will be used to define `k`.
    * `--project_root`: **(Optional)** Path to the project's root directory. Defaults to the current directory (`.`).
    * `--hemispheres`: **(Optional)** Comma-separated list of hemispheres to process. Defaults to `"L,R"`.
    * `--n_tracts_expected`: **(Optional)** Expected number of features/tracts in the blueprint data. Defaults to `20`.

### Script 6: Compute Cross-Species Gradients
* **Name**: `6_cross_species_gradients.py`
* **Function**: Performs a joint spectral embedding using a combination of data: original masked blueprints for the `target_k_species` (e.g., chimpanzee, from Script 2) and downsampled centroid profiles for other species (e.g., human, from Script 5). Outputs remapped cross-species gradients as `.func.gii` for each species and an `.npz` archive with detailed embedding information.
* **Example Command**:
    ```bash
    python code/6_cross_species_gradients.py \
        --species_list_for_lle "human,chimpanzee" \
        --target_k_species "chimpanzee"
    ```
* **Key Arguments**:
    * `--species_list_for_lle`: **(Required)** Comma-separated list of all species to include in the joint analysis.
    * `--target_k_species`: **(Required)** The species from the list that will provide its original, non-downsampled blueprint as the reference.
    * `--project_root`: **(Optional)** Path to the project's root directory. Defaults to the current directory (`.`).
    * `--hemispheres_to_process`: **(Optional)** Comma-separated list of hemispheres to process. Defaults to `"L,R"`.
    * `--num_gradients_to_save`: **(Optional)** Number of top gradients to save in the final output files. Defaults to `10`.

### Script 7: Interactive Gradient Visualization (Dash App)
* **Name**: `7_interactive_analyse_cross_species.py`
* **Function**: Launches an interactive Dash web application to visualize the cross-species gradients from a specified Script 6 run. It allows clicking on points to see their original connectivity profiles (spider plots) and find closest neighbors. The script automatically finds the required input files.
* **Example Command**:
    ```bash
    python code/7_interactive_analyse_cross_species.py \
        --species_list_for_run "human,chimpanzee" \
        --target_k_species_for_run "chimpanzee" \
        --port 8051
    ```
* **Key Arguments**:
    * `--species_list_for_run`: **(Required)** Comma-separated list of species included in the Script 6 run. **Must be in the same order as the original run.**
    * `--target_k_species_for_run`: **(Required)** The reference species (`target_k_species`) used in the Script 6 run.
    * `--project_root`: **(Optional)** Path to the project's root directory. Defaults to the current directory (`.`).
    * `--tract_names`, `--n_tracts`: **(Optional)** Configuration for the spider plots.
    * `--host`, `--port`, `--debug`: **(Optional)** Arguments to configure the Dash web server.
* **Accessing the App**: After running, open your web browser and go to `http://<host>:<port>/` (e.g., `http://127.0.0.1:8051/`).

### Script 8: Plot Cross-Species Gradients (Static Scatter Plots)
* **Name**: `8_plot_cross_species_gradients.py`
* **Function**: Generates static 2D scatter plots from the cross-species gradient data (`.npz` file) created by Script 6. It automatically finds the correct `.npz` file based on the species used in the Script 6 run. This is useful for creating publication-quality figures of specific gradient comparisons.
* **Example Command**:
    ```bash
    python code/8_plot_cross_species_gradients.py \
        --species_list_for_run "human,chimpanzee" \
        --target_k_species_for_run "chimpanzee" \
        --gradient_pairs "0_1,0_2"
    ```
* **Key Arguments**:
    * `--species_list_for_run`: **(Required)** Comma-separated list of species included in the Script 6 run. **Must be in the same order as the original run.**
    * `--target_k_species_for_run`: **(Required)** The reference species (`target_k_species`) used in the Script 6 run.
    * `--gradient_pairs`: **(Optional)** Comma-separated list of 0-indexed gradient pairs to plot (e.g., `"0_1,0_2"`). Defaults to `"0_1"`.
    * `--project_root`: **(Optional)** Path to the project's root directory. Defaults to the current directory (`.`).

### Script 9: Plot Consolidated Spider Plots
* **Name**: `9_plot_consolidated_spider_plots.py`
* **Function**: For each specified cross-species gradient, this script identifies the vertices (or centroids) that show the minimum and maximum expression of that gradient for each species/hemisphere. It then generates two consolidated spider plots per gradient: one for the max-expressing vertices and one for the min-expressing vertices. It automatically locates the required data from Scripts 2, 5, and 6.
* **Example Command**:
    ```bash
    python code/9_plot_consolidated_spider_plots.py \
        --species_list_for_run "human,chimpanzee" \
        --target_k_species_for_run "chimpanzee" \
        --gradients "0,1"
    ```
* **Key Arguments**:
    * `--species_list_for_run`: **(Required)** Comma-separated list of species included in the Script 6 run. **Must be in the same order as the original run.**
    * `--target_k_species_for_run`: **(Required)** The reference species (`target_k_species`) used in the Script 6 run.
    * `--gradients`: **(Optional)** Comma-separated list of 0-indexed gradients to analyze. Defaults to `"0,1"`.
    * `--project_root`: **(Optional)** Path to the project's root directory. Defaults to the current directory (`.`).
    * `--tract_names`: **(Optional)** Comma-separated list of tract names for the spider plot labels.

### Script 10: Run Permutation Analysis
* **Name**: `10_run_permutation_analysis.py`
* **Function**: Performs permutation testing to compare the mean gradient values between different groups (e.g., Human L vs. Human R, Human L vs. Chimp L). It automatically finds the correct input data from the specified Script 6 run, prints the results to the console, and can optionally save histograms of the null distributions.
* **Example Command**:
    ```bash
    python code/10_run_permutation_analysis.py \
        --species_list_for_run "human,chimpanzee" \
        --target_k_species_for_run "chimpanzee"
    ```
* **Key Arguments**:
    * `--species_list_for_run`: **(Required)** Comma-separated list of species included in the Script 6 run. **Must be in the same order as the original run.**
    * `--target_k_species_for_run`: **(Required)** The reference species (`target_k_species`) used in the Script 6 run.
    * `--gradient_index`: **(Optional)** The 0-indexed gradient to analyze. (Default: `0`)
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: `.`)
    * `--n_permutations`: **(Optional)** The number of permutations to run for the test. (Default: `10000`)
    * `--alpha`: **(Optional)** The significance level for the test. (Default: `0.01`)
    * `--no_histograms`: **(Optional)** A flag to disable saving histogram plots. (Default: Histograms are generated)

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
* **Permutation Analysis**: Console output with statistical results and optional `.png` histograms of null distributions (Script 10).
