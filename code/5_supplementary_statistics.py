"""Script 5: the statistics reported in the paper.

Reads the stored outputs of scripts 1-4 (no gradients are recomputed) and writes, to
results/5_supplementary_statistics/:

  tests.csv                  the 13 permutation tests (hemispheres within species, species within hemispheres) with Cohen's d, plus the same tests
                             on 50 and 100 spatially contiguous parcels per hemisphere
  permutation_nulls.npz      the vertex-level null distributions of those 13 tests (for Figures 4, 7, 11, 12)
  g1_spread.csv              SD of cross-species G1, human centroids vs chimpanzee vertices (permutation test)
  g1_composition.csv         what lies behind the narrower human range on cross-species G1: interquartile range ratio,
                             SD ratio without mask-boundary vertices, profile entropy, and eta2 between the mean profiles
                             of the two ends of G1 (with all 20 tracts and without the AF)
  g1_tract_share.csv         mean share of each tract in each fifth of cross-species G1, per species and hemisphere
  correspondence.csv         Pearson r between single-species and cross-species gradients (G1-G4)
  best_match.csv             median eta2 of each profile's best match in each species and hemisphere (rows = source)
  summary.json               share of kNN graph edges between species and between hemispheres, k-means variance kept, regression of cross-species G2 on human G1-G3,
                             quadratic fit of G2 on G1 (the arch), and the spatial null for human G3 vs
                             cross-species G2 (BrainSMASH, geodesic distances on the surfaces in data/surfaces)

Usage (from the project root):
    python code/5_supplementary_statistics.py
    python code/5_supplementary_statistics.py --n_surrogates 0   # skip the slow spatial null
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans

import common as c

np.row_stack = np.vstack  # brainsmash still calls np.row_stack, which numpy 2 removed


# ---------- similarity between groups ----------

def best_match(X, groups, block=500):
    """Median eta2 of each row's best match within every group (self excluded), as a groups x groups table."""
    names = list(dict.fromkeys(groups))
    best = np.zeros((len(X), len(names)))
    for a in range(0, len(X), block):
        S = c.eta2(X[a:a + block], X)
        S[np.arange(len(S)), np.arange(a, a + len(S))] = -1
        for j, g in enumerate(names):
            best[a:a + len(S), j] = S[:, groups == g].max(1)
    return pd.DataFrame([np.median(best[groups == g], 0) for g in names], index=names, columns=names)


# ---------- tests ----------

def permutation_test(a, b, stat=np.mean, n_perm=10000, seed=0):
    """Two-sided label-shuffling test of stat(a) - stat(b); p = share of |null| >= |observed|."""
    rng = np.random.default_rng(seed)
    pooled, n = np.concatenate([a, b]), len(a)
    observed = stat(a) - stat(b)
    null = np.empty(n_perm)
    for i in range(n_perm):
        x = rng.permutation(pooled)
        null[i] = stat(x[:n]) - stat(x[n:])
    return observed, np.mean(np.abs(null) >= np.abs(observed)), null


def cohen_d(a, b):
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def the_13_tests():
    """(name, kind, (species, H) of group 1, (species, H) of group 2, gradient index) for the 13 tests reported in the paper."""
    T = [(f'chimpanzee L vs R, single-species G{g + 1}', 'single', ('chimpanzee', 'L'), ('chimpanzee', 'R'), g) for g in (0, 1)]
    T += [(f'human L vs R, single-species G{g + 1}', 'single', ('human', 'L'), ('human', 'R'), g) for g in (0, 1, 2)]
    for g in (0, 1):
        T += [(f'{sp} L vs R, cross-species G{g + 1}', 'cross', (sp, 'L'), (sp, 'R'), g) for sp in c.SPECIES]
    for g in (0, 1):
        T += [(f'human {h} vs chimpanzee {h}, cross-species G{g + 1}', 'cross', ('human', h), ('chimpanzee', h), g) for h in c.HEMIS]
    return T


def parcel_means(values, parcels):
    return np.bincount(parcels, weights=values) / np.bincount(parcels)


# ---------- spatial null ----------

def geodesic_distances(root, species, h):
    """Geodesic distance between temporal lobe vertices along the mesh edges (inflated surface)."""
    xyz, faces = c.surface(root, species, h)
    idx = c.mask_indices(root, species, h)
    pos = np.full(len(xyz), -1)
    pos[idx] = np.arange(len(idx))
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = e[(pos[e[:, 0]] >= 0) & (pos[e[:, 1]] >= 0)]
    w = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1)
    A = sparse.coo_matrix((w, (pos[e[:, 0]], pos[e[:, 1]])), shape=(len(idx),) * 2).tocsr()
    return dijkstra(A.maximum(A.T), directed=False)


def spatial_null_correlation(x, y, D, n_surrogates, seed=1):
    """Pearson r(x, y) against variogram-matched surrogates of x (BrainSMASH Sampled)."""
    from brainsmash.mapgen.sampled import Sampled
    order = np.argsort(D, axis=1)
    gen = Sampled(x=x, D=np.take_along_axis(D, order, axis=1), index=order, ns=500, knn=1500, resample=True, seed=seed)
    r = stats.pearsonr(x, y)[0]
    null = np.array([stats.pearsonr(s, y)[0] for s in gen(n=n_surrogates)])
    return dict(r=r, p=(1 + np.sum(np.abs(null) >= abs(r))) / (n_surrogates + 1), n_surrogates=n_surrogates)


# ---------- composition of the profiles along cross-species G1 ----------

def entropy(P):
    """Shannon entropy of each profile across the tracts, divided by its maximum (log of the number of tracts)."""
    logs = np.log(P, where=P > 0, out=np.zeros_like(P))
    return -(P * logs).sum(1) / np.log(P.shape[1])


def interior(root, species, h):
    """True for temporal lobe vertices that share no surface triangle with a vertex outside the mask."""
    idx = c.mask_indices(root, species, h)
    faces = c.surface(root, species, h)[1]
    in_mask = np.zeros(faces.max() + 1, bool)
    in_mask[idx] = True
    on_edge = np.zeros_like(in_mask)
    on_edge[faces[in_mask[faces].any(1) & ~in_mask[faces].all(1)].ravel()] = True
    return ~on_edge[idx]


def g1_composition(root, seg, vertex, labels):
    """Compare the spread and the tract composition of human and chimpanzee profiles along cross-species G1.

    G1 is oriented so that its dorsal end (high acoustic radiation share in the chimpanzee) is positive. Human
    centroids and chimpanzee vertices are used wherever G1 values are needed, as in g1_spread.csv; the entropy
    uses the vertex profiles of both species, so it does not depend on the k-means step.
    """
    sd = lambda x: x.std(ddof=1)
    iqr = lambda x: np.subtract(*np.percentile(x, [75, 25]))
    af = c.TRACTS.index('AF')
    no_af = [i for i in range(len(c.TRACTS)) if i != af]
    rows, shares = [], []
    for h in c.HEMIS:
        P = {'human': c.centroids(root, labels, h), 'chimpanzee': c.profiles(root, 'chimpanzee', h)}
        g = {sp: seg[(sp, h)][:, 0] for sp in c.SPECIES}
        sign = np.sign(stats.pearsonr(P['chimpanzee'][:, c.TRACTS.index('AR')], g['chimpanzee'])[0])
        g = {sp: sign * g[sp] for sp in c.SPECIES}
        gv = {sp: sign * vertex[(sp, h)][:, 0] for sp in c.SPECIES}
        keep = {sp: interior(root, sp, h) for sp in c.SPECIES}
        row = dict(hemisphere=h,
                   iqr_ratio=iqr(g['human']) / iqr(g['chimpanzee']),
                   sd_ratio_interior_vertices=sd(gv['human'][keep['human']]) / sd(gv['chimpanzee'][keep['chimpanzee']]),
                   interior_vertices_human=int(keep['human'].sum()), interior_vertices_chimpanzee=int(keep['chimpanzee'].sum()))
        for sp in c.SPECIES:
            row[f'entropy_{sp}'] = entropy(c.profiles(root, sp, h)).mean()
            dorsal = P[sp][g[sp] >= np.percentile(g[sp], 90)].mean(0)
            ventral = P[sp][g[sp] <= np.percentile(g[sp], 10)].mean(0)
            row[f'ends_eta2_{sp}'] = c.eta2(dorsal[None], ventral[None])[0, 0]
            d, v = dorsal[no_af] / dorsal[no_af].sum(), ventral[no_af] / ventral[no_af].sum()
            row[f'ends_eta2_without_AF_{sp}'] = c.eta2(d[None], v[None])[0, 0]
            fifth = np.digitize(g[sp], np.percentile(g[sp], [20, 40, 60, 80]))
            for f in range(5):
                shares += [dict(species=sp, hemisphere=h, g1_fifth=f + 1, tract=t, share=s)
                           for t, s in zip(c.TRACTS, P[sp][fifth == f].mean(0))]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(shares)


def main(root, out, n_surrogates):
    os.makedirs(out, exist_ok=True)
    ss = c.single_species(root)
    seg, vertex, labels = c.cross_species(root)
    summary = {}

    # 1. How strongly the species are linked in the cross-species kNN graph (with the k script 4 found), with the two
    # hemispheres of each species as a reference gap, and how close each profile's best match is in every group
    keys = [('human', 'L'), ('human', 'R'), ('chimpanzee', 'L'), ('chimpanzee', 'R')]
    X = np.vstack([c.centroids(root, labels, h) for h in c.HEMIS] + [c.profiles(root, 'chimpanzee', h) for h in c.HEMIS])
    sp, hemi = (np.repeat([k[i] for k in keys], [len(seg[k]) for k in keys]) for i in (0, 1))
    k = int(np.load(f'{root}/{c.CROSS}')['k'])
    S = c.similarity(X)
    e = np.column_stack(sparse.triu(c.knn_graph(S, c.nearest_neighbours(S, k), k), 1).nonzero())
    del S
    summary['knn_graph'] = dict(k=k, edges=len(e), share_between_species=np.mean(sp[e[:, 0]] != sp[e[:, 1]]))
    for s in c.SPECIES:
        w = e[(sp[e[:, 0]] == s) & (sp[e[:, 1]] == s)]
        summary['knn_graph'][f'share_between_hemispheres_{s}'] = np.mean(hemi[w[:, 0]] != hemi[w[:, 1]])
    print('cross-species kNN edges', summary['knn_graph'], flush=True)
    best_match(X, np.char.add(np.char.add(sp, ' '), hemi)).to_csv(f'{out}/best_match.csv')

    # 2. The 13 tests, at vertex level and on spatial parcels (k-means on the inflated surface coordinates)
    parcels = {n: {(sp, h): KMeans(n_clusters=n, n_init=10, random_state=0)
                   .fit_predict(c.surface(root, sp, h)[0][c.mask_indices(root, sp, h)])
                   for sp in c.SPECIES for h in c.HEMIS} for n in (50, 100)}
    rows, nulls = [], []
    for name, kind, k1, k2, g in the_13_tests():
        a, b = (ss[k1][:, g], ss[k2][:, g]) if kind == 'single' else (seg[k1][:, g], seg[k2][:, g])
        diff, p, null = permutation_test(a, b)
        nulls.append(null)
        row = dict(test=name, n1=len(a), n2=len(b), difference=diff, p=p, cohen_d=cohen_d(a, b))
        maps = ss if kind == 'single' else vertex
        for n in (50, 100):
            pa, pb = parcel_means(maps[k1][:, g], parcels[n][k1]), parcel_means(maps[k2][:, g], parcels[n][k2])
            row[f'p_{n}_parcels'] = permutation_test(pa, pb)[1]
        rows.append(row)
        print(f"{name:45s} diff={diff:+.5f} p={p:.4f} d={row['cohen_d']:+.2f} "
              f"p50={row['p_50_parcels']:.4f} p100={row['p_100_parcels']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(f'{out}/tests.csv', index=False)
    np.savez(f'{out}/permutation_nulls.npz', test=[t[0] for t in the_13_tests()], null=np.array(nulls))

    # 3. Spread of cross-species G1 (and the profile composition behind it), and the share of profile variance the k-means centroids keep
    sd = lambda x: x.std(ddof=1)
    rows = []
    for h in c.HEMIS:
        a, b = seg[('human', h)][:, 0], seg[('chimpanzee', h)][:, 0]
        p = permutation_test(a, b, stat=sd)[1]
        rows.append(dict(hemisphere=h, sd_human=sd(a), sd_chimpanzee=sd(b), ratio=sd(a) / sd(b), p=p))
    pd.DataFrame(rows).to_csv(f'{out}/g1_spread.csv', index=False)
    summary['kmeans_variance_kept'] = {}
    for h in c.HEMIS:
        P, C = c.profiles(root, 'human', h), c.centroids(root, labels, h)[labels[('human', h)]]
        summary['kmeans_variance_kept'][h] = 1 - ((P - C) ** 2).sum() / ((P - P.mean(0)) ** 2).sum()
    composition, tract_share = g1_composition(root, seg, vertex, labels)
    composition.to_csv(f'{out}/g1_composition.csv', index=False)
    tract_share.to_csv(f'{out}/g1_tract_share.csv', index=False)
    print(composition.T.to_string(), flush=True)

    # 4. Correspondence between single-species and cross-species gradients
    rows = [dict(species=sp, hemisphere=h, single=f'G{i + 1}', cross=f'G{j + 1}',
                 r=stats.pearsonr(ss[(sp, h)][:, i], vertex[(sp, h)][:, j])[0])
            for sp in c.SPECIES for h in c.HEMIS for i in range(4) for j in range(4)]
    pd.DataFrame(rows).to_csv(f'{out}/correspondence.csv', index=False)
    summary['cs_G2_on_human_G1_G3_R2'], summary['arch_R2'], summary['spatial_null_G3_vs_csG2'] = {}, {}, {}
    for h in c.HEMIS:
        X, y = np.column_stack([np.ones(len(ss[('human', h)])), ss[('human', h)][:, :3]]), vertex[('human', h)][:, 1]
        res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        summary['cs_G2_on_human_G1_G3_R2'][h] = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
        for sp in c.SPECIES:
            g1, g2 = seg[(sp, h)][:, 0], seg[(sp, h)][:, 1]
            Q = np.column_stack([np.ones_like(g1), g1, g1 ** 2])
            res = g2 - Q @ np.linalg.lstsq(Q, g2, rcond=None)[0]
            summary['arch_R2'][f'{sp}_{h}'] = 1 - res @ res / ((g2 - g2.mean()) @ (g2 - g2.mean()))
        if n_surrogates:
            D = geodesic_distances(root, 'human', h)
            summary['spatial_null_G3_vs_csG2'][h] = spatial_null_correlation(
                ss[('human', h)][:, 2], vertex[('human', h)][:, 1], D, n_surrogates)
            print('spatial null', h, summary['spatial_null_G3_vs_csG2'][h], flush=True)

    with open(f'{out}/summary.json', 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_surrogates', type=int, default=500, help='BrainSMASH surrogates per hemisphere (0 = skip)')
    args = parser.parse_args()
    main(c.ROOT, f'{c.ROOT}/results/5_supplementary_statistics', args.n_surrogates)
