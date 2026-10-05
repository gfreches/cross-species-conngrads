"""Figures for the paper on human-chimpanzee temporal lobe connectivity gradients.

Reads the outputs of scripts 2, 3, 6 and 10 and writes PNG (300 dpi) and PDF files to
results/11_figures/main and results/11_figures/supplementary:

  main/Figure01_dimensionality          reconstruction score vs number of gradients (chimpanzee, human, cross-species)
  main/Figure02_chimpanzee_gradients    chimpanzee G1 and G2 on the surfaces, and the profiles in G1-G2 space
  main/Figure03_chimpanzee_profiles     connectivity profiles at the chimpanzee locations A-D (left and right vertex)
  main/Figure04_chimpanzee_tests        permutation tests, chimpanzee left vs right (G1, G2)
  main/Figure05_human_gradients         human G1-G3 on the surfaces, and the profiles in G1-G2 and G1-G3 space
  main/Figure06_human_profiles          connectivity profiles at the human locations A-E
  main/Figure07_human_tests             permutation tests, human left vs right (G1-G3)
  main/Figure08_kmeans_check            eta2 between human vertices before and after the k-means step of script 5
  main/Figure09_cross_species           cross-species G1 and G2 on the surfaces, and all profiles in G1-G2 space
  main/Figure10_cross_species_profiles  profiles at the ends of cross-species G2 (A-C, chosen by rule, see below)
  main/Figure11_lateralization_tests    permutation tests, left vs right in the cross-species space
  main/Figure12_species_tests           permutation tests, human vs chimpanzee in the cross-species space
  supplementary/FigureS1-S3             single-species (human, chimpanzee) and cross-species gradients G1-G10
  supplementary/FigureS4                cross-species G1 against G2-G10
  supplementary/FigureS5                spread of the human and chimpanzee values along cross-species G1

Usage (from the project root, after script 10):
    python code/11_figures.py
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting

import common as c

# Species/hemisphere colours (checked for colour-blind separation) and the gradient colour map.
COLOR = {('human', 'L'): '#1f4e9c', ('human', 'R'): '#4f8fe6', ('chimpanzee', 'L'): '#b2182b', ('chimpanzee', 'R'): '#e8604f'}
NAME = {'human': 'Human', 'chimpanzee': 'Chimpanzee'}
CMAP = 'RdBu_r'  # same as the interactive website
INK, GRID = '#1a1a1a', '#e3e3e3'
KEYS = [('human', 'L'), ('human', 'R'), ('chimpanzee', 'L'), ('chimpanzee', 'R')]

# Locations shown in the profile figures (surface vertex indices, from the original Figures 3 and 6).
FIG3 = {'A': [('L', 13952), ('R', 13981)], 'B': [('L', 14924), ('R', 15182)],
        'C': [('R', 16699), ('L', 16702)], 'D': [('L', 6266), ('R', 5350)]}
FIG6 = {'A': [('R', 22600)], 'B': [('R', 31449)], 'C': [('L', 21558)], 'D': [('L', 9024)], 'E': [('R', 15595)]}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#5f6368', 'xtick.color': '#5f6368',
                     'ytick.color': '#5f6368', 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)
    print('saved', path, flush=True)


def letter(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=13, fontweight='bold', color=INK)


def brain(ax, root, sp, h, values=None, vmin=None, vmax=None, marker=None):
    """Lateral view of the inflated surface; temporal lobe coloured by `values` (or a flat tint), optional marker."""
    xyz, faces = c.surface(root, sp, h)
    data = np.full(len(xyz), np.nan)
    data[c.mask_indices(root, sp, h)] = 0.5 if values is None else values
    plotting.plot_surf((xyz, faces), data, hemi='left' if h == 'L' else 'right', view='lateral', axes=ax,
                       cmap='Greys' if values is None else CMAP, vmin=0 if values is None else vmin,
                       vmax=2.5 if values is None else vmax, engine='matplotlib', colorbar=False,
                       bg_map=np.r_[np.full(len(xyz) - 1, 0.25), 1.0])  # flat light grey for the rest of cortex
    for mesh in ax.collections:
        mesh.set_rasterized(True)  # keeps the PDFs small; text and axes stay vector
    if marker is not None:
        ax.computed_zorder = False  # draw the marker on top of the mesh
        p = xyz[marker] + np.array([-5.0 if h == 'L' else 5.0, 0, 0])  # nudge towards the viewer
        ax.scatter(*p, s=40, color='#ffd400', edgecolor=INK, linewidth=0.8, zorder=10, depthshade=False)


def polar_profile(ax, profiles, colors, labels):
    """Connectivity profile(s) over the 20 tracts as a closed polar line."""
    th = np.linspace(0, 2 * np.pi, len(c.TRACTS), endpoint=False)
    for prof, col, lab in zip(profiles, colors, labels):
        ax.plot(np.r_[th, th[0]], np.r_[prof, prof[0]], color=col, lw=1.6, label=lab)
        ax.fill(th, prof, color=col, alpha=0.18)
    ax.set_xticks(th)
    ax.set_xticklabels(c.TRACTS, fontsize=6)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    rmax = max(p.max() for p in profiles)
    ax.set_ylim(0, rmax * 1.05)
    ax.set_yticks(np.round(np.linspace(0, rmax, 4)[1:], 2))
    ax.tick_params(axis='y', labelsize=5.5, colors='#5f6368')
    ax.set_rlabel_position(99)
    ax.grid(color=GRID, lw=0.6)
    ax.spines['polar'].set_color(GRID)


def selected_dims(scores, min_gain=0.1):
    """The dimensionality rule of scripts 3 and 6: stop when the gain drops below min_gain or the score falls."""
    for d in range(1, len(scores)):
        if scores[d] < scores[d - 1] or scores[d] - scores[d - 1] < min_gain:
            return d
    return len(scores)


def figure01(root, out):
    rs = pd.read_csv(f'{root}/results/10_supplementary_statistics/reconstruction_scores.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, (emb, title), s in zip(axes, [('chimpanzee', 'Chimpanzee'), ('human', 'Human'), ('cross-species', 'Cross-species')], 'ABC'):
        y = rs.loc[rs.embedding == emb, 'reconstruction_r'].to_numpy()
        x, d = np.arange(1, len(y) + 1), selected_dims(y)
        ax.plot(x, y, color='#3b5b92', lw=2, marker='o', ms=5, mfc='white', mew=1.5)
        ax.plot(d, y[d - 1], 'o', ms=8, color='#3b5b92')
        ax.annotate(f'{d} gradients\nr = {y[d - 1]:.2f}', (d, y[d - 1]), xytext=(8, -26), textcoords='offset points')
        ax.set(xticks=x, ylim=(0, 1), xlabel='Number of gradients')
        ax.set_title(title, loc='left', fontsize=10)
        ax.grid(axis='y', color=GRID, lw=0.6)
        letter(ax, s)
    axes[0].set_ylabel('Reconstruction score (r)')
    save(fig, f'{out}/main/Figure01_dimensionality')


def profile_figure(root, out, name, species, locations):
    """Figures 3 and 6: one polar profile (all vertices of a location overlaid) and the brain location per letter."""
    P = {h: c.profiles(root, species, h) for h in c.HEMIS}
    n = len(locations)
    fig = plt.figure(figsize=(2.7 * n, 4.0))
    for i, (s, verts) in enumerate(locations.items()):
        ax = fig.add_axes([(i + 0.17) / n, 0.36, 0.66 / n, 0.58], projection='polar')
        polar_profile(ax, [P[h][c.vertex_row(root, species, h, v)] for h, v in verts],
                      [COLOR[(species, h)] for h, _ in verts], [f'{h} vertex {v}' for h, v in verts])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=6.5)
        letter(ax, s, -0.1, 1.08)
        for j, (h, v) in enumerate(verts):
            bx = fig.add_axes([i / n + j * 0.5 / n, 0.0, 0.5 / n if len(verts) > 1 else 1 / n, 0.27], projection='3d')
            brain(bx, root, species, h, marker=v)
    save(fig, f'{out}/main/{name}')


def species_figure(root, out, name, ss, sp, locations, panels):
    """Figures 2 and 5: single-species gradients on the surfaces (one colour scale per row, 1st-99th percentile) and
    the profiles in gradient space, with the locations of Figure 3 or 6 circled. panels: [(y gradient, letters)]."""
    keys, n = [(sp, 'L'), (sp, 'R')], max(g for g, _ in panels) + 1
    fig = plt.figure(figsize=(9, 2.3 * n))
    hh = 0.86 / n
    for g in range(n):
        vmin, vmax = np.percentile(np.concatenate([ss[k][:, g] for k in keys]), [1, 99])
        for j, k in enumerate(keys):
            ax = fig.add_axes([0.02 + j * 0.27, 0.93 - (g + 1) * hh, 0.31, hh * 1.12], projection='3d')
            brain(ax, root, *k, ss[k][:, g], vmin, vmax)
            if g == 0:
                ax.set_title(f'{NAME[sp]} {k[1]}', fontsize=8, y=0.92)
        fig.text(0.0, 0.95 - (g + 0.15) * hh, f'G{g + 1}', fontsize=9, fontweight='bold')
    cax = fig.add_axes([0.17, 0.06, 0.2, 0.025 * 2 / n])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), cax=cax, orientation='horizontal', ticks=[0, 1])
    cb.set_ticklabels(['min', 'max']); cb.outline.set_visible(False)
    ph = (0.84 - 0.1 * (len(panels) - 1)) / len(panels)
    for i, (gy, letters) in enumerate(panels):
        ax = fig.add_axes([0.63, 0.94 - (i + 1) * ph - i * 0.1, 0.35, ph])
        for k in keys:
            ax.scatter(ss[k][:, 0] * 1e3, ss[k][:, gy] * 1e3, s=3, lw=0, alpha=0.5, color=COLOR[k],
                       label=f'{NAME[sp]} {k[1]}', rasterized=True)
        for s in letters:
            xy = [ss[(sp, h)][c.vertex_row(root, sp, h, v), [0, gy]] * 1e3 for h, v in locations[s]]
            for j, (x, y) in enumerate(xy):
                ax.scatter([x], [y], s=120, facecolors='none', edgecolors=INK, lw=1.2, zorder=5)
                if j == 0 or np.hypot(*(xy[j] - xy[0])) > 1:  # label the left and right vertex once if they overlap
                    ax.annotate(s, (x, y), xytext=(7, 4), textcoords='offset points', fontsize=11, fontweight='bold')
        ax.set(xlabel='G1 (x 10$^{-3}$)', ylabel=f'G{gy + 1} (x 10$^{{-3}}$)')
        ax.axhline(0, color=GRID, lw=0.8, zorder=0); ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        if i == 0:
            ax.legend(frameon=False, markerscale=3, fontsize=7, loc='best')
    save(fig, f'{out}/main/{name}')


def null_figure(root, out, name, tests, ncols=2):
    """Figures 4, 7, 11 and 12: null distribution (10,000 label permutations, script 10) and observed difference."""
    z = np.load(f'{root}/results/10_supplementary_statistics/permutation_nulls.npz')
    T = pd.read_csv(f'{root}/results/10_supplementary_statistics/tests.csv').set_index('test')
    nrows = -(-len(tests) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows + 0.3), squeeze=False)
    for ax, s, (test, title) in zip(axes.flat, 'ABCDE', tests):
        null = z['null'][list(z['test']).index(test)] * 1e3
        obs, p = T.loc[test, 'difference'] * 1e3, T.loc[test, 'p']
        col = '#6b6b6b' if 'vs chimpanzee' in test else COLOR[(test.split()[0], 'L')]
        ax.hist(null, 50, histtype='stepfilled', color=col, alpha=0.25)
        ax.hist(null, 50, histtype='step', color=col, lw=1.2)
        ax.axvline(obs, color=INK, lw=1.4, ls='--')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)  # headroom so the p value never sits on the bars
        right = obs < np.median(null)  # p value on the side away from the observed line
        ax.text(0.97 if right else 0.03, 0.95, 'p < 0.0001' if p == 0 else f'p = {p:.2g}', transform=ax.transAxes,
                ha='right' if right else 'left', va='top')
        ax.set_title(title, loc='left', fontsize=9)
        ax.set_xlabel('Difference in means (x 10$^{-3}$)')
        ax.grid(axis='y', color=GRID, lw=0.6)
        letter(ax, s, -0.16, 1.06)
    for ax in axes[:, 0]:
        ax.set_ylabel('Permutations')
    handles = [matplotlib.patches.Patch(facecolor='#e6e6e6', edgecolor='#6b6b6b', label='Null distribution (10,000 permutations)'),
               matplotlib.lines.Line2D([], [], color=INK, lw=1.4, ls='--', label='Observed difference')]
    fig.tight_layout(rect=(0, 0.06 / nrows, 1, 1))
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=7.5)
    save(fig, f'{out}/main/{name}')


def figure08(root, out, labels):
    """eta2 between all pairs of human vertices: original profiles vs the profile of each vertex's k-means centroid."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, h, s in zip(axes, c.HEMIS, 'AB'):
        P, lab = c.profiles(root, 'human', h), labels[('human', h)]
        iu = np.triu_indices(len(P), 1)
        x, y = c.eta2(P, P)[iu], c.eta2(*[c.centroids(root, labels, h)[lab]] * 2)[iu]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#dce6f5', COLOR[('human', h)]])
        ax.hexbin(x, y, gridsize=150, bins='log', cmap=cmap, mincnt=1, linewidths=0, rasterized=True)
        ax.plot([0, 1], [0, 1], color='#5f6368', lw=0.8, ls='--', zorder=0)
        ax.text(0.04, 0.95, f'r = {np.corrcoef(x, y)[0, 1]:.4f}', transform=ax.transAxes, va='top')
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='eta$^2$, vertex profiles')
        ax.set_title(f'Human {h}: {len(P):,} vertices, {lab.max() + 1:,} centroids', loc='left', fontsize=9)
        letter(ax, s)
    axes[0].set_ylabel('eta$^2$, centroid profiles')
    save(fig, f'{out}/main/Figure08_kmeans_check')


def figure09(root, out, seg, vertex, picks):
    fig = plt.figure(figsize=(11, 4.6))
    for gi in range(2):
        vals = np.concatenate([vertex[k][:, gi] for k in KEYS])
        vmin, vmax = np.percentile(vals, [1, 99])
        for ki, k in enumerate(KEYS):
            ax = fig.add_axes([0.0 + ki * 0.135, 0.52 - gi * 0.47, 0.15, 0.42], projection='3d')
            brain(ax, root, *k, vertex[k][:, gi], vmin, vmax)
            if gi == 0:
                ax.set_title(f'{NAME[k[0]]} {k[1]}', fontsize=8, y=0.92)
        fig.text(0.0, 0.93 - gi * 0.47, f'Cross-species G{gi + 1}', fontsize=9, fontweight='bold')
    cax = fig.add_axes([0.2, 0.06, 0.15, 0.02])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), cax=cax, orientation='horizontal', ticks=[0, 1])
    cb.set_ticklabels(['min', 'max']); cb.outline.set_visible(False)
    ax = fig.add_axes([0.62, 0.1, 0.36, 0.82])
    for k in [('chimpanzee', 'L'), ('chimpanzee', 'R'), ('human', 'L'), ('human', 'R')]:
        ax.scatter(seg[k][:, 0] * 1e3, seg[k][:, 1] * 1e3, s=3, lw=0, alpha=0.5, color=COLOR[k],
                   label=f'{NAME[k[0]]} {k[1]}', rasterized=True)
    for s, (k, row) in picks.items():
        x, y = vertex[k][row, :2] * 1e3
        ax.scatter([x], [y], s=120, facecolors='none', edgecolors=INK, lw=1.2, zorder=5)
        ax.annotate(s, (x, y), xytext=(-14 if s == 'C' else 7, 4), textcoords='offset points', fontsize=11, fontweight='bold')
    ax.set(xlabel='Cross-species G1 (x 10$^{-3}$)', ylabel='Cross-species G2 (x 10$^{-3}$)')
    ax.axhline(0, color=GRID, lw=0.8, zorder=0); ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, markerscale=3, loc='lower left', fontsize=7)
    save(fig, f'{out}/main/Figure09_cross_species')


def figure10(root, out, labels, picks):
    """A: human centroid with the lowest cross-species G2 (shown at its most typical vertex),
    B: chimpanzee vertex with the lowest G2 (the one closest to the human end), C: chimpanzee vertex with the highest G2."""
    fig = plt.figure(figsize=(9, 4.4))
    for i, (s, (k, row)) in enumerate(picks.items()):
        prof = c.centroids(root, labels, k[1])[labels[k][row]] if k[0] == 'human' else c.profiles(root, *k)[row]
        v = c.mask_indices(root, *k)[row]
        ax = fig.add_axes([i / 3 + 0.03, 0.38, 0.27, 0.52], projection='polar')
        polar_profile(ax, [prof], [COLOR[k]], [None])
        ax.set_title(f'{NAME[k[0]]} {k[1]} (vertex {v})', fontsize=8, pad=18)
        letter(ax, s, -0.1, 1.1)
        brain(fig.add_axes([i / 3 + 0.04, 0.0, 0.27, 0.3], projection='3d'), root, *k, marker=v)
    save(fig, f'{out}/main/Figure10_cross_species_profiles')


def gradient_grid(root, out, name, maps, keys, n_kept, n=10):
    """Supplementary Figures S1-S3: rows = G1..Gn, columns = hemispheres (and species).
    Each row has one colour scale across its columns (1st-99th percentile); retained gradients are labelled in bold."""
    fig = plt.figure(figsize=(1.9 * len(keys), 1.25 * n))
    for g in range(n):
        vmin, vmax = np.percentile(np.concatenate([maps[k][:, g] for k in keys]), [1, 99])
        for j, k in enumerate(keys):
            ax = fig.add_axes([0.06 + j * 0.94 / len(keys), 1 - (g + 1) / n, 0.94 / len(keys), 1 / n], projection='3d')
            brain(ax, root, *k, maps[k][:, g], vmin, vmax)
            if g == 0:
                ax.set_title(f'{NAME[k[0]]} {k[1]}', fontsize=9, y=0.95)
        fig.text(0.0, 1 - (g + 0.5) / n, f'G{g + 1}', fontsize=10, va='center',
                 fontweight='bold' if g < n_kept else 'normal', color=INK if g < n_kept else '#5f6368')
    cax = fig.add_axes([0.35, -0.02, 0.3, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), cax=cax, orientation='horizontal', ticks=[0, 1])
    cb.set_ticklabels(['min', 'max']); cb.outline.set_visible(False)
    save(fig, f'{out}/supplementary/{name}')


def figure_s4(out, seg):
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.4))
    for g, ax in zip(range(1, 10), axes.flat):
        for k in [('chimpanzee', 'L'), ('chimpanzee', 'R'), ('human', 'L'), ('human', 'R')]:
            ax.scatter(seg[k][:, 0] * 1e3, seg[k][:, g] * 1e3, s=1.5, lw=0, alpha=0.5, color=COLOR[k],
                       label=f'{NAME[k[0]]} {k[1]}', rasterized=True)
        ax.set(xlabel='G1 (x 10$^{-3}$)', ylabel=f'G{g + 1} (x 10$^{{-3}}$)')
    axes[0, 0].legend(frameon=False, markerscale=4, fontsize=6.5)
    fig.tight_layout()
    save(fig, f'{out}/supplementary/FigureS4_cross_species_G1_vs_G2-G10')


def figure_s5(root, out, seg):
    st = pd.read_csv(f'{root}/results/10_supplementary_statistics/g1_spread.csv').set_index('hemisphere')
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    bins = np.linspace(-6, 9, 46)
    for ax, h in zip(axes, c.HEMIS):
        for sp in ['chimpanzee', 'human']:
            v = seg[(sp, h)][:, 0] * 1e3
            ax.hist(v, bins, histtype='stepfilled', color=COLOR[(sp, h)], alpha=0.25)
            ax.hist(v, bins, histtype='step', color=COLOR[(sp, h)], lw=1.6, label=f'{NAME[sp]} {h} (SD {v.std(ddof=1):.2f})')
        ax.legend(frameon=False, loc='upper right', fontsize=7.5,
                  title=f'Human SD = {100 * st.loc[h, "ratio"]:.0f}% of chimpanzee SD', title_fontsize=7.5)
        ax.set(xlabel='Cross-species G1 (x 10$^{-3}$)', ylim=(0, 560))
        ax.set_title('Left hemisphere' if h == 'L' else 'Right hemisphere', loc='left', fontsize=10)
        ax.grid(axis='y', color=GRID, lw=0.6)
    axes[0].set_ylabel('Number of profiles')
    save(fig, f'{out}/supplementary/FigureS5_G1_spread')


def main(root, out):
    ss = c.single_species(root)
    seg, vertex, labels = c.cross_species(root)

    # Locations at the ends of cross-species G2 (Figures 9 and 10), chosen by rule rather than by hand.
    hum = min(c.HEMIS, key=lambda h: vertex[('human', h)][:, 1].min())
    lab = labels[('human', hum)]
    centroid = lab[np.argmin(vertex[('human', hum)][:, 1])]
    members = np.flatnonzero(lab == centroid)
    C = c.centroids(root, labels, hum)[centroid]
    row_a = members[np.argmin(((c.profiles(root, 'human', hum)[members] - C) ** 2).sum(1))]
    lo = min(c.HEMIS, key=lambda h: vertex[('chimpanzee', h)][:, 1].min())
    hi = max(c.HEMIS, key=lambda h: vertex[('chimpanzee', h)][:, 1].max())
    picks = {'A': (('human', hum), row_a),
             'B': (('chimpanzee', lo), int(np.argmin(vertex[('chimpanzee', lo)][:, 1]))),
             'C': (('chimpanzee', hi), int(np.argmax(vertex[('chimpanzee', hi)][:, 1])))}

    lr = lambda sp, g, kind: (f'{sp} L vs R, {kind} G{g}', f'{NAME[sp]}, left vs right: {kind} G{g}')
    hc = lambda h, g: (f'human {h} vs chimpanzee {h}, cross-species G{g}',
                       f'Human vs chimpanzee, {"left" if h == "L" else "right"}: cross-species G{g}')

    figure01(root, out)
    species_figure(root, out, 'Figure02_chimpanzee_gradients', ss, 'chimpanzee', FIG3, [(1, 'ABCD')])
    profile_figure(root, out, 'Figure03_chimpanzee_profiles', 'chimpanzee', FIG3)
    null_figure(root, out, 'Figure04_chimpanzee_tests', [lr('chimpanzee', g, 'single-species') for g in (1, 2)])
    species_figure(root, out, 'Figure05_human_gradients', ss, 'human', FIG6, [(1, 'ABC'), (2, 'DE')])
    profile_figure(root, out, 'Figure06_human_profiles', 'human', FIG6)
    null_figure(root, out, 'Figure07_human_tests', [lr('human', g, 'single-species') for g in (1, 2, 3)], ncols=3)
    figure08(root, out, labels)
    figure09(root, out, seg, vertex, picks)
    figure10(root, out, labels, picks)
    null_figure(root, out, 'Figure11_lateralization_tests',
                [lr(sp, g, 'cross-species') for g in (1, 2) for sp in ('human', 'chimpanzee')])
    null_figure(root, out, 'Figure12_species_tests', [hc(h, g) for g in (1, 2) for h in c.HEMIS])
    rs = pd.read_csv(f'{root}/results/10_supplementary_statistics/reconstruction_scores.csv')
    kept = {e: selected_dims(rs.loc[rs.embedding == e, 'reconstruction_r'].to_numpy()) for e in rs.embedding.unique()}
    gradient_grid(root, out, 'FigureS1_human_gradients_G1-G10', ss, [('human', 'L'), ('human', 'R')], kept['human'])
    gradient_grid(root, out, 'FigureS2_chimpanzee_gradients_G1-G10', ss, [('chimpanzee', 'L'), ('chimpanzee', 'R')], kept['chimpanzee'])
    gradient_grid(root, out, 'FigureS3_cross_species_gradients_G1-G10', vertex, KEYS, kept['cross-species'])
    figure_s4(out, seg)
    figure_s5(root, out, seg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_root', default='.')
    args = parser.parse_args()
    main(args.project_root, os.path.join(args.project_root, 'results', '11_figures'))
