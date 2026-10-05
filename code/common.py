"""Shared loaders for scripts 10 and 11.

Everything is read from the standard project layout produced by scripts 1-6:

    data/masks/<species>/<species>_<H>.func.gii
    data/surfaces/<species>/<inflated surface>
    results/2_masked_average_blueprints/<species>/average_<species>_blueprint.<H>_temporal_lobe_masked.func.gii
    results/3_individual_species_gradients/<species>/all_gradients_embedding_<species>_Combined.npy
    results/6_cross_species_gradients/intermediates/<run>/cross_species_embedding_data_<run>.npz
"""
import nibabel as nib
import numpy as np

SPECIES = ['human', 'chimpanzee']
HEMIS = ['L', 'R']
CS_RUN = 'human_chimpanzee_CrossSpecies_kRef_chimpanzee'

# Row order of the blueprints (the order the tracts were passed to create_blueprint.py).
TRACTS = ['AC', 'AF', 'AR', 'CBD', 'CBP', 'CBT', 'CST', 'FA', 'FMA', 'FMI', 'FX', 'IFOF', 'ILF',
          'MDLF', 'OR', 'SLF I', 'SLF II', 'SLF III', 'UF', 'VOF']

SURFACES = {'human': 'Human32k.{h}.inflated.surf.gii',
            'chimpanzee': 'ChimpYerkes29.{h}.inflated.20k_fs_LR.surf.gii'}


def mask_indices(root, species, h):
    """Surface indices of the temporal lobe vertices (ascending, as used by scripts 3-6)."""
    m = nib.load(f'{root}/data/masks/{species}/{species}_{h}.func.gii').darrays[0].data
    return np.where(m > 0)[0]


def profiles(root, species, h):
    """Temporal lobe connectivity profiles, vertices x 20 tracts."""
    f = (f'{root}/results/2_masked_average_blueprints/{species}/'
         f'average_{species}_blueprint.{h}_temporal_lobe_masked.func.gii')
    bp = np.stack([d.data for d in nib.load(f).darrays]).astype(float)
    return bp[:, mask_indices(root, species, h)].T


def surface(root, species, h):
    """Inflated surface (coordinates, faces) used for plotting, parcels and geodesic distances."""
    g = nib.load(f'{root}/data/surfaces/{species}/' + SURFACES[species].format(h=h))
    return g.darrays[0].data.astype(float), g.darrays[1].data.astype(int)


def single_species(root):
    """{(species, H): vertices x 10} single-species gradients (combined-hemisphere run of script 3)."""
    out = {}
    for sp in SPECIES:
        E = np.load(f'{root}/results/3_individual_species_gradients/{sp}/all_gradients_embedding_{sp}_Combined.npy')
        n_left = len(mask_indices(root, sp, 'L'))
        out[(sp, 'L')], out[(sp, 'R')] = E[:n_left], E[n_left:]
    return out


def cross_species(root, run=CS_RUN):
    """Cross-species gradients from script 6.

    Returns
      seg:    {(species, H): rows x 10} as embedded (human rows = k-means centroids, chimpanzee rows = vertices)
      vertex: {(species, H): vertices x 10} on the surface (each human vertex takes its centroid's value)
      labels: {('human', H): centroid label of each human temporal lobe vertex}
    """
    z = np.load(f'{root}/results/6_cross_species_gradients/intermediates/{run}/cross_species_embedding_data_{run}.npz',
                allow_pickle=True)
    G = z['cross_species_gradients']
    seg, vertex, labels = {}, {}, {}
    for s in z['segment_info_detailed_for_remapping']:
        key = (s['species'], s['hem'])
        rows = G[s['start_row_in_concat']:s['end_row_in_concat']]
        lab = np.asarray(s['cluster_labels_for_remapping'])
        seg[key], vertex[key] = rows, rows[lab]
        if key[0] == 'human':
            labels[key] = lab
    return seg, vertex, labels


def centroids(root, labels, h):
    """Human k-means centroid profiles, rebuilt from the vertex profiles and the saved labels."""
    P, lab = profiles(root, 'human', h), labels[('human', h)]
    C = np.zeros((lab.max() + 1, P.shape[1]))
    np.add.at(C, lab, P)
    return C / np.bincount(lab)[:, None]


def vertex_row(root, species, h, vertex):
    """Row of a surface vertex within that hemisphere's temporal lobe arrays."""
    idx = mask_indices(root, species, h)
    return int(np.flatnonzero(idx == vertex)[0])
