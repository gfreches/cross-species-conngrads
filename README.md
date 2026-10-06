# Cross-Species Connectivity Gradient Analysis Pipeline

This repository contains a suite of Python scripts designed to process, analyze, and visualize brain connectivity blueprints, with a focus on computing and comparing connectivity gradients across different primate species (e.g., humans and chimpanzees). The pipeline includes steps for averaging blueprints, masking, individual species gradient computation, data downsampling via k-means, cross-species gradient computation, and interactive visualization of the results.
You can also find an online version of the 2-D interactive plot of this work in https://cross-species-gradients.duckdns.org/

## Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Setup](#setup)
3.  [Directory Structure](#directory-structure)
4.  [Pipeline Overview & Script Usage](#pipeline-overview--script-usage)
    * [Script 1: Average Blueprints](#script-1-average-blueprints)
    * [Script 2: Mask Blueprints](#script-2-mask-blueprints)
    * [Script 3: Compute Individual Species Gradients](#script-3-compute-individual-species-gradients)
    * [Script 4: Downsample Blueprints via K-Means](#script-4-downsample-blueprints-via-k-means)
    * [Script 5: Compute Cross-Species Gradients](#script-5-compute-cross-species-gradients)
    * [Script 6: Supplementary Statistics](#script-6-supplementary-statistics)
    * [Script 7: Figures](#script-7-figures)
5.  [Reproducing the Paper](#reproducing-the-paper)
6.  [Interactive App and Self-Hosted Deployment](#interactive-app-and-self-hosted-deployment)
7.  [Outputs](#outputs)

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
│   ├── masks/                        # Temporal lobe (or other ROI) masks
│   │   └── <species_name>/
│   │       └── <species_name>_{hemisphere}.func.gii # e.g., human_L.func.gii (mask per hemisphere)
│   └── surfaces/                      # Inflated brain surface meshes for visualization
│       └── <species_name>/
│           └── *.surf.gii             # e.g., Human32k.L.inflated.surf.gii
├── results/                        # Main output directory for the pipeline
│   ├── 1_average_blueprints/
│   │   └── <species_name>/         # Output of Script 1
│   ├── 2_masked_average_blueprints/
│   │   └── <species_name>/         # Output of Script 2
│   ├── 3_individual_species_gradients/
│   │   └── <species_name>/         # Output of Script 3
│   ├── 4_downsampled_blueprints/
│   │   └── <species_name>/         # Output of Script 4 (centroids, labels, etc.)
│   ├── 5_cross_species_gradients/
│   │   ├── intermediates/          # .npy, .npz, plots from cross-species run
│   │   │   └── <run_identifier>/
│   │   └── <species_name>/         # Remapped cross-species gradients per species
│   │       └── cross_species_gradients_remapped/
│   ├── 6_supplementary_statistics/  # Output of Script 6 (CSV, NPZ and JSON tables)
│   └── 7_figures/                   # Output of Script 7 (main/ and supplementary/, PNG and PDF)
└── code/                        # Where your Python scripts reside
    ├── 1_average_blueprints.py
    ├── 2_mask_blueprints.py
    ├── 3_individual_species_gradients.py
    ├── 4_downsample_blueprints_knn.py
    ├── 5_create_cross_species_gradients.py
    ├── 6_supplementary_statistics.py
    ├── 7_figures.py
    ├── common.py                   # Loaders shared by scripts 6 and 7
    └── self_hosted_app.py          # Interactive Dash app (local or self-hosted)

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
 
### Script 4: Downsample Blueprints via K-Means
* **Name**: `4_downsample_blueprints_knn.py`
* **Function**: Downsamples masked average blueprints (from Script 2) for specified source species using k-means clustering. The number of clusters (`k`) is determined by the temporal lobe vertex count of a specified `target_k_species`. Outputs include centroid profiles (`.npy`), vertex labels (`.npy`), and a visual downsampled blueprint (`.func.gii`).
* **Example Command**:
    ```bash
    python code/4_downsample_blueprints_knn.py \
        --source_species_list "human" \
        --target_species_for_k "chimpanzee"
    ```
* **Key Arguments**:
    * `--source_species_list`: **(Required)** Comma-separated list of source species to downsample.
    * `--target_species_for_k`: **(Required)** The species whose temporal lobe vertex count will be used to define `k`.
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--hemispheres`: **(Optional)** Comma-separated list of hemispheres to process. (Default: "L,R")
    * `--n_tracts_expected`: **(Optional)** Expected number of features/tracts in the blueprint data. (Default: 20)

### Script 5: Compute Cross-Species Gradients
* **Name**: `5_create_cross_species_gradients.py`
* **Function**: Performs a joint spectral embedding using a combination of data: original masked blueprints for the `target_k_species` (e.g., chimpanzee, from Script 2) and downsampled centroid profiles for other species (e.g., human, from Script 4). Outputs remapped cross-species gradients as `.func.gii` for each species and an `.npz` archive with detailed embedding information.
* **Example Command**:
    ```bash
    python code/5_create_cross_species_gradients.py \
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

### Script 6: Supplementary Statistics
* **Name**: `6_supplementary_statistics.py`
* **Function**: Adds the statistics reported in the paper on top of the outputs of scripts 2, 3 and 5. Nothing is recomputed, so the numbers match the gradients already on disk. Permutations use a fixed random seed, so every p-value is reproducible. It writes to `results/6_supplementary_statistics/`:
    * `reconstruction_scores.csv`: reconstruction score for 1 to 10 gradients (single-species and cross-species), the values behind the dimensionality plots
    * `tests.csv`: the 13 permutation tests (hemispheres within each species, species within each hemisphere) with Cohen's d, repeated on the means of 50 and 100 spatially contiguous parcels per hemisphere (k-means on the inflated surface coordinates)
    * `permutation_nulls.npz`: the null distributions of those 13 tests, used for Figures 4, 7, 11 and 12
    * `g1_spread.csv`: SD of cross-species G1 for human centroids and chimpanzee vertices, with a permutation test
    * `correspondence.csv`: correlations between single-species and cross-species gradients (G1 to G4)
    * `best_match.csv`: median eta2 of each profile's best match in each species and hemisphere (rows are the source group)
    * `summary.json`: share of edges in the cross-species kNN graph that link the two species, and the two hemispheres within each species, share of profile variance kept by the k-means centroids, regression of cross-species G2 on human G1 to G3, quadratic fit of cross-species G2 on G1, and a spatial null (BrainSMASH, geodesic distances on the inflated surface) for the correlation between human G3 and cross-species G2
* **Example Usage**:
    ```bash
    python code/6_supplementary_statistics.py                    # about 45 minutes, mostly the spatial null
    python code/6_supplementary_statistics.py --n_surrogates 0   # skip the spatial null
    ```

### Script 7: Figures
* **Name**: `7_figures.py`
* **Function**: Draws every figure of the paper, in one style, from the outputs of scripts 2, 3, 5 and 6 into `results/7_figures/main` and `results/7_figures/supplementary` (PNG at 300 dpi and PDF):
    * Figure 1: reconstruction score against the number of gradients
    * Figures 2 and 5: chimpanzee and human gradients on the surfaces, and the profiles in gradient space with the locations of Figures 3 and 6
    * Figures 3 and 6: connectivity profiles at the chimpanzee and human locations
    * Figures 4, 7, 11 and 12: permutation tests (null distribution and observed difference) from script 6
    * Figure 8: eta2 between human vertices before and after the k-means step of script 4
    * Figure 9: cross-species G1 and G2 on the surfaces, and all profiles in the G1-G2 space
    * Figure 10: profiles at the ends of cross-species G2 (A: human end, B: the chimpanzee vertex closest to it, C: chimpanzee end)
    * Figures S1-S3: human, chimpanzee and cross-species gradients G1 to G10 on the surfaces
    * Figure S4: cross-species G1 against G2 to G10
    * Figure S5: spread of the human and chimpanzee values along cross-species G1
* **Example Usage**:
    ```bash
    python code/7_figures.py   # run script 6 first
    ```

## Reproducing the Paper

The repository contains the temporal lobe masks, the surfaces and the group-average blueprints (outputs of scripts 1 and 2). It also contains the outputs of scripts 3 and 5 that the paper uses: `results/3_individual_species_gradients/<species>/all_gradients_embedding_<species>_Combined.npy` and the cross-species `.npz` in `results/5_cross_species_gradients/intermediates/human_chimpanzee_CrossSpecies_kRef_chimpanzee/`. With these, scripts 6 and 7 reproduce every number, table and figure of the paper exactly:

```bash
python code/6_supplementary_statistics.py
python code/7_figures.py
```

To rerun the whole pipeline from the blueprints:

```bash
python code/3_individual_species_gradients.py --species_list "human,chimpanzee"
python code/4_downsample_blueprints_knn.py --source_species_list "human" --target_species_for_k "chimpanzee"
python code/5_create_cross_species_gradients.py --species_list_for_lle "human,chimpanzee" --target_k_species "chimpanzee"
python code/6_supplementary_statistics.py
python code/7_figures.py
```

Script 3 reproduces the stored gradients exactly. The k-means step of script 4 can split the right human hemisphere slightly differently with other scikit-learn versions, which shifts the cross-species results a little, and script 5 needs more than 6 GB of memory.

| In the paper | Produced by |
| --- | --- |
| Figures 1 to 12, Supplementary Figures S1 to S5 | Script 7 (`results/7_figures/`) |
| Supplementary Table S1 | Script 6 (`tests.csv`) |
| Reconstruction scores and number of gradients | Script 6 (`reconstruction_scores.csv`) |
| Permutation tests, Cohen's d, parcel-level tests | Script 6 (`tests.csv`, `permutation_nulls.npz`) |
| Spread of cross-species G1 | Script 6 (`g1_spread.csv`) |
| Correlations between single-species and cross-species gradients | Script 6 (`correspondence.csv`) |
| k-means variance kept, regression and quadratic fits, kNN edge shares, spatial null | Script 6 (`summary.json`) |
| Best-match similarity between species and hemispheres | Script 6 (`best_match.csv`) |
| k-means validation (r = 0.9999) | Script 4, and Figure 8 from script 7 |

## Interactive App and Self-Hosted Deployment

`code/self_hosted_app.py` is a Dash app for exploring the gradients. It runs locally and is also deployed at:

> **https://cross-species-gradients.duckdns.org/**

* **Function**: Launches a tabbed interactive Dash web application for visualizing brain connectivity gradients across species. The app combines 3D brain-surface rendering with scatter-plot exploration in three tabs:

    1.  **Individual Gradients (Tab 1)**: View per-species combined-hemisphere gradients painted on 3D brain surfaces (left and right hemispheres side-by-side). Select species, gradient number, and colorscale. Uses output from **Script 3**.
    2.  **Cross-Species Gradients (Tab 2)**: View joint cross-species gradients rendered simultaneously on every species/hemisphere surface. A shared color range is applied across all surfaces for direct comparison. Select gradient number and colorscale. Uses output from **Script 5**.
    3.  **Interactive Explorer (Tab 3)**: Scatter-plot exploration of gradient space with marginal histograms. A data-source selector switches between chimpanzee-only, human-only, and cross-species data. Select which gradients map to the X and Y axes. Click on any data point (vertex) to see its connectivity profile on a spider plot, the selected gradient rendered on its brain surface with the vertex highlighted, and the closest neighbour in another species (or same species). Choose which gradient to display on the surfaces and the colorscale. Supports Euclidean, X-axis only, and Y-axis only distance modes. Uses output from **Scripts 2, 3 and 5**.

    Surface rendering in Tabs 1 and 2 paints gradient values only on temporal-lobe (TL) vertices; the rest of the brain is shown in grey. The TL and non-TL regions are rendered as independent meshes with GPU-accelerated colorscale interpolation.

* **Required Data**:
    * Surface meshes from `data/surfaces/<species>/` (`.surf.gii`)
    * Temporal-lobe masks from `data/masks/<species>/` (`.func.gii`)
    * Individual gradients from `results/3_individual_species_gradients/` (Script 3, for Tab 1)
    * Cross-species `.npz` from `results/5_cross_species_gradients/` (Script 5, for Tabs 2 & 3)
    * Masked average blueprints from `results/2_masked_average_blueprints/` (Script 2, for Tab 3 spider plots)
* **Example Command**:
    ```bash
    python code/self_hosted_app.py \
        --species_list_for_run "human,chimpanzee" \
        --target_k_species_for_run "chimpanzee" \
        --port 8051
    ```
* **Key Arguments**:
    * `--species_list_for_run`: **(Required)** Comma-separated list of species included in the Script 5 run. **Must be in the same order as the original run.**
    * `--target_k_species_for_run`: **(Required)** The reference species (`target_k_species`) used in the Script 5 run.
    * `--project_root`: **(Optional)** Path to the project's root directory. (Default: ".")
    * `--surface_dir`: **(Optional)** Directory with species subfolders containing `.surf.gii` files. (Default: `<project_root>/data/surfaces`)
    * `--n_tracts`: **(Optional)** Expected number of tracts/features. (Default: 20)
    * `--tract_names`: **(Optional)** Comma-separated list of tract names for spider plots. (Default: "AC,AF,AR,CBD,CBP,CBT,CST,FA,FMA,FMI,FX,IFOF,ILF,MDLF,OR,SLF I,SLF II,SLF III,UF,VOF")
    * `--host`: **(Optional)** Host address for the Dash app. (Default: "127.0.0.1")
    * `--port`: **(Optional)** Port for the Dash app. (Default: 8050)
    * `--debug`: **(Optional)** Enable Dash debug mode. (Default: False)
* **Accessing the App**: After running, open your web browser and go to `http://<host>:<port>/` (e.g., `http://127.0.0.1:8051/`).

For WSGI hosting (e.g. Gunicorn + Nginx), edit the `DEPLOY_*` settings at the top of the script; the server uses the flatter layout below.

### Self-hosted directory structure

On the server, data files are organized under the configured `DEPLOY_PROJECT_ROOT/data/` in a flatter layout than the local `results/`-based structure:

```
data/
├── surfaces/                              # Brain surface meshes (same as local)
│   ├── human/
│   └── chimpanzee/
├── gradient_outputs/                      # All gradient files in one flat directory
│   ├── all_computed_gradients_*_.func.gii # Individual species gradients (Script 3)
│   ├── *_from_cs_gradients_k_*.func.gii  # Cross-species gradient maps (Script 5)
│   └── cross_species_embedding_data_*.npz # Cross-species embedding (Script 5)
├── temporal_lobe_average_blueprints/      # Average blueprints for spider plots
│   ├── human/
│   │   ├── average_human_blueprint_L_temporal_lobe.func.gii
│   │   └── average_human_blueprint_R_temporal_lobe.func.gii
│   └── chimpanzee/
│       ├── average_chimpanzee_blueprint_L_temporal_lobe.func.gii
│       └── average_chimpanzee_blueprint_R_temporal_lobe.func.gii
└── downsampled_temporal_lobe_blueprints/
```

## Outputs

The pipeline generates several types of outputs in the specified `results` subdirectories:

* **Averaged Blueprints**: `.func.gii` files (Script 1).
* **Masked Blueprints**: `.func.gii` files, focused on the ROI (Script 2).
* **Individual Species Gradients**: `.func.gii` gradient maps, `.npy` intermediate files, and dimensionality evaluation plots (Script 3).
* **Downsampled Blueprint Data**: `.npy` files for centroids and labels, and a visual `.func.gii` (Script 4).
* **Cross-Species Gradients**:
    * Remapped `.func.gii` gradient files for each species.
    * An `.npz` archive containing the raw joint embedding, segment information, and eigenvalues.
    * Intermediate `.npy` files and dimensionality evaluation plots (Script 5).
* **Statistics**: CSV, NPZ and JSON tables with every statistic reported in the paper (Script 6).
* **Figures**: every figure of the paper as PNG and PDF (Script 7).

## Disclaimer
LLMs such as ChatGPT o3/4o and Gemini 2.5 were used to generate/correct the code in this repository while the authors provided the actual tasks
