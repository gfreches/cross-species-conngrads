"""Script 2: single-species connectivity gradients.

For each species, the temporal lobe profiles of both hemispheres are embedded together with Laplacian eigenmaps
(common.gradients). Writes results/2_individual_species_gradients/gradients_<species>.npz with the gradients, the
hemisphere of each row, the k of the kNN graph and the reconstruction score for 1-10 gradients, plus one .func.gii
per hemisphere for viewing the maps.

Usage (from the project root):
    python code/2_individual_species_gradients.py
"""
import os

import numpy as np

import common as c


def main():
    for sp in c.SPECIES:
        X = {h: c.profiles(c.ROOT, sp, h) for h in c.HEMIS}
        hemisphere = np.repeat(c.HEMIS, [len(X[h]) for h in c.HEMIS])
        profiles = np.vstack([X[h] for h in c.HEMIS])
        E, k = c.gradients(profiles)
        r = c.reconstruction_scores(profiles, E)

        path = f'{c.ROOT}/' + c.SINGLE.format(species=sp)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, gradients=E, hemisphere=hemisphere, k=k, reconstruction_r=r)
        for h in c.HEMIS:
            c.save_maps(path.replace('.npz', f'.{h}.func.gii'), c.ROOT, sp, h, E[hemisphere == h],
                        [f'G{i + 1}' for i in range(E.shape[1])])
        print(f'{sp}: k = {k}, {c.n_gradients(r)} gradients kept, reconstruction score {np.round(r, 2)}', flush=True)


if __name__ == '__main__':
    main()
