"""Script 3: k-means downsampling of the human temporal lobe.

The human profiles of each hemisphere are clustered into as many clusters as the chimpanzee hemisphere has
temporal lobe vertices, so that both species enter the cross-species embedding with the same number of profiles.
The centroid of a cluster is the mean profile of its vertices (common.centroids). Writes the cluster of every human
temporal lobe vertex to results/3_kmeans_downsampling/human_kmeans_labels.npz and prints how well the centroids
keep the similarity between vertices (Figure 8).

The right-hemisphere clustering comes out slightly differently on different computers (rounding in the float32
k-means depends on the processor), so rerunning this script can replace the stored labels, which are the ones used
in the paper, and shift the cross-species results a little.

Usage (from the project root):
    python code/3_kmeans_downsampling.py
"""
import os

import numpy as np
from sklearn.cluster import KMeans

import common as c


def main():
    labels = {}
    for h in c.HEMIS:
        X = c.profiles(c.ROOT, 'human', h)
        k = len(c.mask_indices(c.ROOT, 'chimpanzee', h))
        labels[h] = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42).fit_predict(X.astype(np.float32))
        C = c.centroids(c.ROOT, {('human', h): labels[h]}, h)[labels[h]]
        upper = np.triu_indices(len(X), 1)
        r = np.corrcoef(c.eta2(X, X)[upper], c.eta2(C, C)[upper])[0, 1]
        print(f'human {h}: {len(X)} vertices in {k} clusters, eta2 between vertices before vs after r = {r:.4f}', flush=True)
    path = f'{c.ROOT}/{c.KMEANS}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **labels)


if __name__ == '__main__':
    main()
