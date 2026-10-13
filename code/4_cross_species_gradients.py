"""Script 4: cross-species connectivity gradients.

The human k-means centroids (script 3) and the chimpanzee vertex profiles of both hemispheres are embedded together
with Laplacian eigenmaps (common.gradients), in the row order human L, human R, chimpanzee L, chimpanzee R. Writes
results/4_cross_species_gradients/cross_species_gradients.npz with the gradients, the species and hemisphere of each
row, the k of the kNN graph and the reconstruction score for 1-10 gradients, plus one .func.gii per species and
hemisphere for viewing the maps (each human vertex takes the values of its centroid).

Usage (from the project root):
    python code/4_cross_species_gradients.py
"""
import os

import numpy as np

import common as c


def main():
    z = np.load(f'{c.ROOT}/{c.KMEANS}')
    labels = {('human', h): z[h] for h in c.HEMIS}
    blocks = {**{('human', h): c.centroids(c.ROOT, labels, h) for h in c.HEMIS},
              **{('chimpanzee', h): c.profiles(c.ROOT, 'chimpanzee', h) for h in c.HEMIS}}
    sizes = [len(b) for b in blocks.values()]
    species = np.repeat([sp for sp, _ in blocks], sizes)
    hemisphere = np.repeat([h for _, h in blocks], sizes)
    profiles = np.vstack(list(blocks.values()))
    E, k = c.gradients(profiles)
    r = c.reconstruction_scores(profiles, E)

    path = f'{c.ROOT}/{c.CROSS}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, gradients=E, species=species, hemisphere=hemisphere, k=k, reconstruction_r=r)
    for sp, h in blocks:
        rows = E[(species == sp) & (hemisphere == h)]
        c.save_maps(path.replace('.npz', f'_{sp}.{h}.func.gii'), c.ROOT, sp, h,
                    rows[labels[(sp, h)]] if sp == 'human' else rows, [f'G{i + 1}' for i in range(E.shape[1])])
    print(f'cross-species: {len(profiles)} profiles, k = {k}, {c.n_gradients(r)} gradients kept, '
          f'reconstruction score {np.round(r, 2)}')


if __name__ == '__main__':
    main()
