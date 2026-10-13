"""Shared code for the scripts and the app: file layout, loaders and the gradient method.

Paths are relative to the project root (the folder that contains code/):

    data/masks/<species>/<species>_<H>.func.gii                              temporal lobe masks
    data/surfaces/<species>/<inflated surface>                               surfaces for plotting
    results/1_average_blueprints/average_<species>_blueprint.<H>.func.gii    script 1
    results/2_individual_species_gradients/gradients_<species>.npz           script 2
    results/3_kmeans_downsampling/human_kmeans_labels.npz                    script 3
    results/4_cross_species_gradients/cross_species_gradients.npz            script 4
"""
import os

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import SpectralEmbedding

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIES = ['human', 'chimpanzee']
HEMIS = ['L', 'R']
# Row order of the blueprints (the order the tracts were passed to create_blueprint.py).
TRACTS = ['AC', 'AF', 'AR', 'CBD', 'CBP', 'CBT', 'CST', 'FA', 'FMA', 'FMI', 'FX', 'IFOF', 'ILF',
          'MDLF', 'OR', 'SLF I', 'SLF II', 'SLF III', 'UF', 'VOF']
SURFACES = {'human': 'Human32k.{h}.inflated.surf.gii',
            'chimpanzee': 'ChimpYerkes29.{h}.inflated.20k_fs_LR.surf.gii'}
N_GRADIENTS = 10  # gradients computed and stored
MIN_GAIN = 0.1    # dimensionality rule: stop when one more gradient raises the reconstruction score by less

BLUEPRINT = 'results/1_average_blueprints/average_{species}_blueprint.{h}.func.gii'
SINGLE = 'results/2_individual_species_gradients/gradients_{species}.npz'
KMEANS = 'results/3_kmeans_downsampling/human_kmeans_labels.npz'
CROSS = 'results/4_cross_species_gradients/cross_species_gradients.npz'


# ---------- loaders ----------

def mask_indices(root, species, h):
    """Surface indices of the temporal lobe vertices (ascending)."""
    m = nib.load(f'{root}/data/masks/{species}/{species}_{h}.func.gii').darrays[0].data
    return np.where(m > 0)[0]


def profiles(root, species, h):
    """Temporal lobe connectivity profiles, vertices x 20 tracts."""
    bp = nib.load(f'{root}/' + BLUEPRINT.format(species=species, h=h))
    return np.stack([d.data for d in bp.darrays], 1).astype(float)[mask_indices(root, species, h)]


def surface(root, species, h):
    """Inflated surface (coordinates, faces) used for plotting, parcels and geodesic distances."""
    g = nib.load(f'{root}/data/surfaces/{species}/' + SURFACES[species].format(h=h))
    return g.darrays[0].data.astype(float), g.darrays[1].data.astype(int)


def single_species(root):
    """{(species, H): vertices x 10} single-species gradients (both hemispheres embedded together)."""
    out = {}
    for sp in SPECIES:
        z = np.load(f'{root}/' + SINGLE.format(species=sp))
        for h in HEMIS:
            out[(sp, h)] = z['gradients'][z['hemisphere'] == h]
    return out


def cross_species(root):
    """Cross-species gradients.

    Returns
      seg:    {(species, H): rows x 10} as embedded (human rows = k-means centroids, chimpanzee rows = vertices)
      vertex: {(species, H): vertices x 10} on the surface (each human vertex takes its centroid's value)
      labels: {('human', H): centroid of each human temporal lobe vertex}
    """
    z, k = np.load(f'{root}/{CROSS}'), np.load(f'{root}/{KMEANS}')
    seg, vertex, labels = {}, {}, {('human', h): k[h] for h in HEMIS}
    for sp in SPECIES:
        for h in HEMIS:
            seg[(sp, h)] = z['gradients'][(z['species'] == sp) & (z['hemisphere'] == h)]
            vertex[(sp, h)] = seg[(sp, h)][labels[(sp, h)]] if sp == 'human' else seg[(sp, h)]
    return seg, vertex, labels


def reconstruction(root):
    """Reconstruction score for 1-10 gradients of each embedding, as stored by scripts 2 and 4."""
    out = {sp: np.load(f'{root}/' + SINGLE.format(species=sp))['reconstruction_r'] for sp in SPECIES}
    out['cross-species'] = np.load(f'{root}/{CROSS}')['reconstruction_r']
    return out


def centroids(root, labels, h):
    """Human k-means centroids: the mean profile of the vertices in each cluster."""
    P, lab = profiles(root, 'human', h), labels[('human', h)]
    C = np.zeros((lab.max() + 1, P.shape[1]))
    np.add.at(C, lab, P)
    return C / np.bincount(lab)[:, None]


def vertex_row(root, species, h, vertex):
    """Row of a surface vertex within that hemisphere's temporal lobe arrays."""
    return int(np.flatnonzero(mask_indices(root, species, h) == vertex)[0])


def save_maps(path, root, species, h, values, names):
    """Write temporal lobe values (vertices x maps) as a whole-surface .func.gii, zero outside the mask."""
    idx = mask_indices(root, species, h)
    n = nib.load(f'{root}/data/masks/{species}/{species}_{h}.func.gii').darrays[0].data.size
    full = np.zeros((values.shape[1], n), np.float32)
    full[:, idx] = values.T
    write_gifti(path, full, h, names)


def write_gifti(path, maps, h, names):
    """Write maps (maps x surface vertices) to a .func.gii, one named data array per map."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    structure = {'AnatomicalStructurePrimary': 'CortexLeft' if h == 'L' else 'CortexRight'}
    arrays = [nib.gifti.GiftiDataArray(np.asarray(m, np.float32), meta=nib.gifti.GiftiMetaData({'Name': name, **structure}))
              for m, name in zip(maps, names)]
    nib.save(nib.gifti.GiftiImage(darrays=arrays, meta=nib.gifti.GiftiMetaData(structure)), path)


# ---------- gradient method ----------

def eta2(A, B):
    """eta2 similarity between the rows of A and the rows of B (Cohen et al., 2008)."""
    p = A.shape[1]
    sa, sb = A.sum(1)[:, None], B.sum(1)[None]
    qa, qb = (A ** 2).sum(1)[:, None], (B ** 2).sum(1)[None]
    ssw = np.maximum((qa + qb - 2 * A @ B.T) / 2, 0)
    sst = ssw if p == 1 else qa + qb - (sa + sb) ** 2 / (2 * p)  # with one feature SST equals SSW
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(sst >= 1e-9, 1 - ssw / sst, (ssw < 1e-9).astype(float))
    return np.clip(S, 0, None)


def similarity(X, block=1000):
    """eta2 between all pairs of profiles, with ones on the diagonal."""
    S = np.empty((len(X), len(X)))
    for a in range(0, len(X), block):
        S[a:a + block] = eta2(X[a:a + block], X)
    np.fill_diagonal(S, 1.0)
    return S


def nearest_neighbours(S, k_max=50, block=1000):
    """Indices of each profile's k_max most similar other profiles (most similar last)."""
    nn = np.empty((len(S), k_max), int)
    for a in range(0, len(S), block):
        rows = S[a:a + block].copy()
        rows[np.arange(len(rows)), np.arange(a, a + len(rows))] = -np.inf
        nn[a:a + block] = np.argsort(rows, axis=1)[:, -k_max:]
    return nn


def knn_graph(S, nn, k):
    """Symmetric k-nearest-neighbour graph weighted by similarity (sparse)."""
    rows, cols = np.repeat(np.arange(len(S)), k), nn[:, -k:].ravel()
    W = sparse.csr_matrix((S[rows, cols], (rows, cols)), shape=S.shape)
    return W.maximum(W.T)


def gradients(X, n=N_GRADIENTS):
    """Laplacian eigenmaps of the profiles in X (rows).

    The kNN graph uses the smallest k that gives a single connected component. Returns (gradients, k).
    """
    S = similarity(X)
    nn = nearest_neighbours(S)
    for k in range(1, nn.shape[1] + 1):
        W = knn_graph(S, nn, k)
        if connected_components(W, directed=False)[0] == 1:
            break
    else:
        raise RuntimeError(f'the kNN graph is not connected with k = {nn.shape[1]}')
    E = SpectralEmbedding(n_components=n, affinity='precomputed', random_state=42, eigen_solver='lobpcg').fit_transform(W)
    return E, k


def reconstruction_scores(X, E, block=500):
    """Pearson r between the upper triangles of eta2(X) and eta2(E[:, :d]) for d = 1 .. number of gradients."""
    n, dims = len(X), E.shape[1]
    sums = np.zeros((dims, 6))
    for a in range(0, n, block):
        b = min(n, a + block)
        upper = np.arange(n)[None] > np.arange(a, b)[:, None]
        x = eta2(X[a:b], X)[upper]
        for d in range(dims):
            y = eta2(E[a:b, :d + 1], E[:, :d + 1])[upper]
            sums[d] += [x.size, x.sum(), y.sum(), x @ x, y @ y, x @ y]
    N, sx, sy, sxx, syy, sxy = sums.T
    return (sxy - sx * sy / N) / np.sqrt((sxx - sx ** 2 / N) * (syy - sy ** 2 / N))


def n_gradients(scores, min_gain=MIN_GAIN):
    """Number of gradients kept: stop at the first d after which the score rises by less than min_gain (or falls)."""
    for d in range(1, len(scores)):
        if scores[d] - scores[d - 1] < min_gain:
            return d
    return len(scores)
