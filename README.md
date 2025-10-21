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
    * [Script 4: Visualize Individual/Combined Gradients (Static Plots)](#script-4-visualize-individual-combined-gradients-static-plots)
    * [Script 5: Downsample Blueprints via K-Means](#script-5-downsample-blueprints-via-k-means)
    * [Script 6: Compute Cross-Species Gradients](#script-6-compute-cross-species-gradients)
    * [Script 7: Interactive Gradient Visualization (Dash App)](#script-7-interactive-plot-cross-species)
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
│           └── <species_name>_{hemisphere}.func.gii # e.g., human_L.func.gii (mask per hemisphere)
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
├── 4_individual_species_gradients_analysis.py
├── 5_downsample_blueprints_knn.py
├── 6_cross_species_gradients.py
├── 7_interactive_plot_cross_species.py
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
* **Note**: The results of this script for the example species (human and chimpanzee) are already available in the `results/1_average_blueprints` directory. You only need to run this script if you are processing your own data.
* **Example Command**:
    ```bash
    python code/1_average_blueprints.py \
        --species_name "human" \
        --subject_list_file "data/subject_lists/human_subjects.txt" \
        --data_base_dir "data/raw_individual_blueprints/" \
        --output_base_dir "results/1_average_blueprints/" \
        --hemispheres "L,R" \
        --subject_dir_pattern "{subject_id}_bp_midthickness_inv_{hemisphere}" \
        --blueprint_filename "BP.{hemisphere}.dscalar.nii"
    ```
* **Key Arguments**:
    * `--species_name`: **(Required)** Name of the species.
    * `--subject_list_file`: **(Required)** Path to file listing subject IDs.
    * `--data_base_dir`: **(Required)** Base directory for raw input data.
    * `--output_base_dir`: **(Required)** Base directory where averaged blueprints will be saved (a subdirectory for `species_name` will be created).
    * `--hemispheres`: **(Optional)** Comma-separated list of hemisphere labels to process (e.g., "L,R" or "L"). (Default: "L,R").
    * `--subject_dir_pattern`: **(Required)** Pattern for subject-specific data directories.
    * `--blueprint_filename`: **(Required)** Filename of blueprint files within subject directories.

### Script 2: Mask Blueprints
* **Name**: `2_mask_blueprints.py`
* **Function**: Masks the averaged `.func.gii` blueprints (from Script 1) with a given region of interest (ROI) mask (e.g., temporal lobe). The script intelligently uses the provided species name to find the correct files and directories, but allows for flexibility by overriding the default paths and patterns.
* **Note**: The results of this script for the example species are also pre-calculated and can be found in the `results/2_masked_average_blueprints` directory.
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
    * `--hemispheres`: **(Optional)** Comma-separated hemispheres to process. (Default: "L,R")
    * `--avg_blueprint_name_pattern`: **(Optional)** Filename pattern for averaged blueprints. (Default: "average_{species_name}_blueprint.{hemisphere}.func.gii")
    * `--mask_name_pattern`: **(Optional)** Filename pattern for masks. (Default: "{species_name}_{hemisphere}.func.gii")
    * `--output_masked_name_pattern`: **(Optional)** Filename pattern for output masked blueprints. (Default: "average_{species_name}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii")
  
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
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--hemispheres`: **(Optional)** Comma-separated list of hemispheres to process. (Default: "L,R")
    * `--max_gradients`: **(Optional)** Maximum number of gradients to compute. (Default: 10)
    * `--max_k_knn`: **(Optional)** Maximum value of *k* for the k-NN graph search. (Default: 150)
    * `--default_k_knn`: **(Optional)** Fallback *k* value. (Default: 20)
    * `--min_gain_dim_select`: **(Optional)** Minimum gain in score to select an additional gradient. (Default: 0.1)
 
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
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--plot_type`: **(Optional)** Type of plot to generate. Choices: `combined`, `individual`. (Default: "combined")
    * `--gradients_to_plot`: **(Optional)** Comma-separated 1-indexed gradient numbers to plot (e.g., "1,2" or "1,2,3"). (Default: "1,2,3")
    * `--plot_2d_projections`: **(Optional)** If plotting 3 gradients, also generate 2D projection scatter plots. (Default: False)

### Script 5: Downsample Blueprints via K-Means
* **Name**: `5_downsample_blueprints_knn.py`
* **Function**: Downsamples masked average blueprints (from Script 2) for specified source species using k-means clustering. The number of clusters (`k`) is determined by the temporal lobe vertex count of a specified `target_k_species`. Outputs include centroid profiles (`.npy`), vertex labels (`.npy`), and a visual downsampled blueprint (`.func.gii`).
* **Example Command**:
    ```bash
    python code/5_downsample_blueprints_knn.py \
        --source_species_list "human" \
        --target_species_for_k "chimpanzee"
    ```
* **Key Arguments**:
    * `--source_species_list`: **(Required)** Comma-separated list of source species to downsample.
    * `--target_species_for_k`: **(Required)** The species whose temporal lobe vertex count will be used to define `k`.
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--hemispheres`: **(Optional)** Comma-separated list of hemispheres to process. (Default: "L,R")
    * `--n_tracts_expected`: **(Optional)** Expected number of features/tracts in the blueprint data. (Default: 20)

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
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--hemispheres_to_process`: **(Optional)** Comma-separated list of hemispheres to process. (Default: "L,R")
    * `--num_gradients_to_save`: **(Optional)** Number of top gradients to save in the final output files. (Default: 10)
    * `--max_gradients`: **(Optional)** Maximum number of gradients to compute. (Default: 10)
    * `--max_k_knn`: **(Optional)** Maximum value of *k* for the k-NN graph search. (Default: 200)
    * `--default_k_knn`: **(Optional)** Fallback *k* value. (Default: 30)
    * `--min_gain_dim_select`: **(Optional)** Minimum gain in score to select an additional gradient. (Default: 0.1)

### Script 7: Interactive Gradient Visualization (Dash App)
* **Name**: `7_interactive_plot_cross_species.py`
* **Function**: Launches an interactive Dash web application to visualize the cross-species gradients from a specified Script 6 run. The dashboard allows for in-depth exploration of the gradient space by allowing users to click on any data point (vertex) to instantly visualize its detailed connectivity profile on a spider plot and its anatomical location on a 3D brain surface rendering. It also includes a tool to find the closest neighbor for any selected point, either within the same species or across to the other species. The script automatically finds the required input files for the gradient data.
* **Example Command**:
    ```bash
    python code/7_interactive_plot_cross_species.py \
        --species_list_for_run "human,chimpanzee" \
        --target_k_species_for_run "chimpanzee" \
        --port 8051
    ```
* **Key Arguments**:
    * `--species_list_for_run`: **(Required)** Comma-separated list of species included in the Script 6 run. **Must be in the same order as the original run.**
    * `--target_k_species_for_run`: **(Required)** The reference species (`target_k_species`) used in the Script 6 run.
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--surface_dir`: **(Optional)** Directory with species subfolders containing `.surf.gii` files. (Default: `<project_root>/data/surfaces`)
    * `--n_tracts`: **(Optional)** Expected number of tracts/features. (Default: 20)
    * `--tract_names`: **(Optional)** Comma-separated list of tract names for spider plots. (Default: "AC,AF,AR,CBD,CBP,CBT,CST,FA,FMI,FMA,FX,IFOF,ILF,MDLF,OR,SLF I,SLF II,SLF III,UF,VOF")
    * `--host`: **(Optional)** Host address for the Dash app. (Default: "127.0.0.1")
    * `--port`: **(Optional)** Port for the Dash app. (Default: 8050)
    * `--debug`: **(Optional)** Enable Dash debug mode. (Default: False)
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
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--gradient_pairs`: **(Optional)** Comma-separated list of 0-indexed gradient pairs to plot (e.g., "0_1,0_2"). (Default: "0_1")

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
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--gradients`: **(Optional)** Comma-separated list of 0-indexed gradients to analyze. (Default: "0,1")
    * `--n_tracts`: **(Optional)** Expected number of tracts in the blueprints. (Default: 20)
    * `--tract_names`: **(Optional)** Comma-separated list of tract names for the spider plot labels. (Default: "AC,AF,AR,CBD,CBP,CBT,CST,FA,FMI,FMA,FX,IFOF,ILF,MDLF,OR,SLF I,SLF II,SLF III,UF,VOF")

### Script 10: Run Permutation Analysis
* **Name**: `10_run_permutation_analysis.py`
* **Function**: Performs permutation testing to compare mean gradient values between groups. It supports two primary modes:
    1.  **`cross_species`**: Compares gradients between hemispheres (e.g., Human L vs. R) and across species (e.g., Human L vs. Chimp L) using the output from a **Script 6** run.
    2.  **`individual`**: Compares gradients between the left and right hemispheres for a single species, using the output from a **Script 3** run.
    The script prints statistical results to the console and can save histograms of the null distributions. It automatically finds the required input data based on the specified analysis type and parameters.
* **Example Commands**:

    **1. Cross-Species Analysis (comparing Human vs. Chimpanzee from a Script 6 run for the first 3 gradients):**
    ```bash
    python code/10_run_permutation_analysis.py \
        --analysis_type "cross_species" \
        --species_list_for_run "human,chimpanzee" \
        --target_k_species_for_run "chimpanzee" \
        --num_gradients 3
    ```

    **2. Individual Species Analysis (comparing L vs. R hemisphere for the Human species from a Script 3 run):**
    ```bash
    python code/10_run_permutation_analysis.py \
        --analysis_type "individual" \
        --species "human" \
        --num_gradients 3
    ```

* **Key Arguments**:
    * `--analysis_type`: **(Required)** The type of analysis to run. Choices: `cross_species`, `individual`.
    * `--num_gradients`: **(Optional)** The number of top gradients to analyze (e.g., 3 means G1, G2, G3). (Default: 3)
    * `--species_list_for_run`: **(Required for `cross_species`)** Comma-separated list of species from the Script 6 run (e.g., "human,chimpanzee").
    * `--target_k_species_for_run`: **(Required for `cross_species`)** The reference species used in the Script 6 run.
    * `--species`: **(Required for `individual`)** The species to analyze from the Script 3 run (e.g., "human").
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--n_permutations`: **(Optional)** The number of permutations to run for the test. (Default: 10000)
    * `--alpha`: **(Optional)** The significance level for the test. (Default: 0.01)
    * `--no_histograms`: **(Optional)** A flag to disable saving histogram plots of the null distributions. (Default: False, meaning histograms are generated)

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

## Disclaimer
LLMs such as ChatGPT o3/4o and Gemini 2.5 were used to generate/correct the code in this repository while the authors provided the actual tasks
