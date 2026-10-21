# Cross-species connectivity gradients of the human and chimpanzee temporal lobe

Code and data for *Comparing principles of temporal lobe organization across humans and chimpanzees using connectivity
gradients* (Blazquez Freches, Bryant, Haak, Beckmann and Mars). The pipeline builds connectivity gradients from
group-average connectivity blueprints with Laplacian eigenmaps, within each species and in a joint human-chimpanzee
embedding, and computes every statistic and figure of the paper.

An online version of the interactive app is at https://cross-species-gradients.duckdns.org/

## Setup

Python 3.12 or newer.

```bash
git clone https://github.com/gfreches/cross-species-conngrads.git
cd cross-species-conngrads
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Interactive app

```bash
python code/app.py                # then open http://127.0.0.1:8050
```

The app reads the gradients stored in the repository, so it works straight after cloning. It has three tabs:
single-species gradients on the surfaces of either species, cross-species gradients on the surfaces of both species,
and an explorer that shows all profiles in gradient space. Clicking a point in the explorer shows its connectivity
profile, its location on the surface and its closest neighbour in the other (or the same) species. Use
`--host 0.0.0.0 --port 8080` to serve it on a network; for a production server, point a WSGI server at `app:server`
(e.g. `gunicorn --chdir code app:server`).

## Reproducing the paper

The repository contains the inputs (temporal lobe masks, surfaces and group-average blueprints) and the outputs of
scripts 2-4 that the paper uses. From these, scripts 5 and 6 reproduce every number, table and figure:

```bash
python code/5_supplementary_statistics.py                    # about 45 minutes, mostly the spatial null
python code/5_supplementary_statistics.py --n_surrogates 0   # a few minutes, without the spatial null
python code/6_figures.py
```

| In the paper | Produced by |
| --- | --- |
| Figures 1-4, Supplementary Figures S1-S10 | script 6 (`results/6_figures/`) |
| Number of gradients and reconstruction scores | scripts 2 and 4 (stored in the gradient files), Figure 1 |
| Supplementary Table S1, permutation tests, Cohen's d, parcel-level tests | script 5 (`tests.csv`, `permutation_nulls.npz`) |
| Spread of cross-species G1 | script 5 (`g1_spread.csv`) |
| Profile entropy, similarity of the G1 ends, AF share along G1 | script 5 (`g1_composition.csv`, `g1_tract_share.csv`) |
| Correlations between single-species and cross-species gradients | script 5 (`correspondence.csv`) |
| k-means variance kept, regression and quadratic fits, kNN edge shares, spatial null | script 5 (`summary.json`) |
| Best-match similarity between species and hemispheres | script 5 (`best_match.csv`) |
| k-means check (r = 0.9999) | script 3, and Supplementary Figure S5 |

## Rerunning the pipeline

```bash
python code/2_individual_species_gradients.py   # about 2 minutes
python code/3_kmeans_downsampling.py            # optional, see below
python code/4_cross_species_gradients.py        # about 2 minutes, under 2 GB of memory
python code/5_supplementary_statistics.py
python code/6_figures.py
```

Scripts 2 and 4 reproduce the stored gradients to within 1e-9. The k-means clustering of script 3 is identical for the
left hemisphere, but the right hemisphere comes out slightly differently on different computers (rounding in
float32 k-means depends on the processor), which shifts the cross-species results a little. Skip script 3 to keep
the clustering used in the paper, which is stored in `results/3_kmeans_downsampling/`. Rerunning a script overwrites
its stored output, so `git checkout results` restores the paper's versions.

Script 1 averages the subject blueprints, which are not part of the repository (see Data):

```bash
python code/1_average_blueprints.py --species human --subjects human_subjects.txt \
    --pattern "/data/human/{subject}_bp_midthickness_inv_{hemisphere}/BP.{hemisphere}.dscalar.nii"
```

## Scripts

| Script | What it does | Output |
| --- | --- | --- |
| `1_average_blueprints.py` | averages the subject blueprints and renormalises each vertex to sum to one | `results/1_average_blueprints/` |
| `2_individual_species_gradients.py` | single-species gradients, both hemispheres embedded together | `results/2_individual_species_gradients/` |
| `3_kmeans_downsampling.py` | clusters the human vertices of each hemisphere into as many clusters as the chimpanzee hemisphere has vertices | `results/3_kmeans_downsampling/` |
| `4_cross_species_gradients.py` | joint embedding of the human centroids and the chimpanzee vertices | `results/4_cross_species_gradients/` |
| `5_supplementary_statistics.py` | the statistics reported in the paper | `results/5_supplementary_statistics/` |
| `6_figures.py` | the figures of the paper (PNG and PDF) | `results/6_figures/` |
| `common.py` | file layout, loaders and the gradient method shared by all scripts and the app | |
| `app.py` | the interactive app | |

The gradient method (`common.gradients`) computes the eta2 similarity between all pairs of connectivity profiles,
keeps each profile's k most similar profiles (the smallest k that gives a single connected graph) and takes 10
Laplacian eigenmaps of that graph. The number of gradients kept is the first number after which one more gradient
raises the reconstruction score by less than 0.1.

The gradient files are plain NumPy archives (`np.load(path)`):

* `results/2_individual_species_gradients/gradients_<species>.npz`: `gradients` (vertices of both hemispheres x 10),
  `hemisphere` of each row, `k`, `reconstruction_r` (1-10 gradients)
* `results/3_kmeans_downsampling/human_kmeans_labels.npz`: `L` and `R`, the cluster of each human temporal lobe vertex
* `results/4_cross_species_gradients/cross_species_gradients.npz`: `gradients` (human L, human R, chimpanzee L,
  chimpanzee R x 10), `species` and `hemisphere` of each row, `k`, `reconstruction_r`

Scripts 2 and 4 also write `.func.gii` maps of the gradients for viewing in Connectome Workbench.

## Repository layout

```
data/
    masks/<species>/<species>_<H>.func.gii         temporal lobe masks
    surfaces/<species>/*.inflated*.surf.gii        inflated surfaces (32k_fs_LR human, 20k_fs_LR chimpanzee)
results/
    1_average_blueprints/                          group-average blueprints (20 tracts)
    2_individual_species_gradients/                single-species gradients
    3_kmeans_downsampling/                         human k-means clusters
    4_cross_species_gradients/                     cross-species gradients
    5_supplementary_statistics/                    written by script 5
    6_figures/                                     written by script 6
code/
```

## Data

Human data: 30 subjects of the Human Connectome Project (www.humanconnectome.org). Chimpanzee data: 25 subjects from
the National Chimpanzee Brain Resource (www.chimpanzeebrain.org). The subject blueprints were built with FSL's XTRACT
(see the Methods of the paper) and are not included; the repository starts from their group averages.

## Use of AI tools

Gemini 2.5 (Google), ChatGPT 4.5/5 (OpenAI) and Claude 4/5 (Anthropic) were used to write, document and check the
code. The authors reviewed the code and take responsibility for it.
