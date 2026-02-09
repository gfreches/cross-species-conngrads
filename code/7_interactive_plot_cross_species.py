#!/usr/bin/env python3
"""
Tabbed interactive Dash + Plotly application for visualizing brain connectivity
gradients across species.

Three tabs:
  Tab 1 - Individual Gradients:
      View per-species combined-hemisphere gradients painted on brain surfaces
      (L and R hemispheres side by side).  Select species and gradient number.

  Tab 2 - Cross-Species Gradients:
      View joint cross-species gradients on all species' surfaces simultaneously.
      Select a gradient number and see it rendered on every species / hemisphere.

  Tab 3 - Interactive Explorer:
      Scatter-plot exploration of gradient space with marginal histograms.
      Click a point to see connectivity profile and nearest neighbour.
      A gradient surface panel shows one of the selected gradients rendered
      on brain surfaces alongside the scatter plot.
      A data-source selector switches between chimpanzee-only, human-only, and
      cross-species gradient data.

To run the app, specify the parameters of the analysis run via command-line
arguments (same interface as the previous version of this script).
"""

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask
from sklearn.metrics.pairwise import euclidean_distances

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_N_TRACTS_EXPECTED = 20
DEFAULT_TRACT_NAMES = [
    "AC", "AF", "AR", "CBD", "CBP", "CBT", "CST", "FA",
    "FMI", "FMA", "FX", "IFOF", "ILF", "MDLF",
    "OR", "SLF I", "SLF II", "SLF III", "UF", "VOF",
]
DEFAULT_PLOT_CONFIGS_SCATTER = {
    "human_L":       {"color": "blue",         "label": "Human L"},
    "human_R":       {"color": "cornflowerblue","label": "Human R"},
    "chimpanzee_L":  {"color": "red",          "label": "Chimpanzee L"},
    "chimpanzee_R":  {"color": "lightcoral",   "label": "Chimpanzee R"},
}
DEFAULT_SPECIES_SYMBOLS = {"human": "circle", "chimpanzee": "circle"}

SURFACE_FILE_PATTERNS = {
    "human":      "Human32k.{hem}.inflated.surf.gii",
    "chimpanzee": "ChimpYerkes29.{hem}.inflated.20k_fs_LR.surf.gii",
}

COLORSCALE_OPTIONS = [
    {"label": "RdBu (reversed)", "value": "RdBu_r"},
    {"label": "Spectral (reversed)", "value": "Spectral_r"},
    {"label": "Viridis", "value": "Viridis"},
    {"label": "Plasma", "value": "Plasma"},
    {"label": "Turbo", "value": "Turbo"},
]

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------
SURFACE_DIR_GLOBAL = ""
INDIVIDUAL_GRAD_DIR_GLOBAL = ""
AVERAGE_BP_DIR_GLOBAL = ""
MASK_DIR_GLOBAL = ""

SURFACE_DATA_CACHE = {}
MASK_CACHE = {}
BLUEPRINT_CACHE = {}         # (species, hem) -> ndarray (vertices, tracts)

# Tab 1 -- individual gradients (combined only)
# Key: species  ->  {hem: np.ndarray (n_vertices, n_grads)}
INDIVIDUAL_GRADIENTS = {}
# Metadata list: [{species, hems: [str], n_grads: int}, ...]
INDIVIDUAL_GRADIENT_INFO = []

# Tab 2 -- cross-species gradient maps on surfaces
# Key: (species, hem)  ->  np.ndarray (n_vertices, n_grads)
CROSS_SPECIES_GRADIENT_MAPS = {}
N_CROSS_SPECIES_GRADIENTS = 0
CROSS_SPECIES_SPECIES_LIST = []

# Tab 3 -- explorer
# Cross-species scatter data
df_global = pd.DataFrame()
PLOT_CONFIGS_SCATTER_GLOBAL = {}
SPECIES_SYMBOLS_GLOBAL = {}
TRACT_NAMES_GLOBAL = []
N_TRACTS_EXPECTED_GLOBAL = 0
AVAILABLE_EXPLORER_GRADIENTS = []

# Per-species DataFrames for the explorer data-source selector
df_by_source = {}           # "cross_species" / "human" / "chimpanzee" -> DataFrame
grad_cols_by_source = {}    # same keys -> list of gradient column names

# Per-species individual gradient maps for Surface Plot mode
# Key: (species, hem) -> ndarray (n_vertices, n_grads) from combined analysis
INDIVIDUAL_GRADIENT_MAPS = {}
N_INDIVIDUAL_GRADS_BY_SPECIES = {}   # species -> int

EMPTY_SURFACE_FIG = None
EMPTY_SPIDER = None


# ===================================================================
#  Data loading helpers
# ===================================================================

def load_surface(species, hemisphere):
    """Load and cache brain-surface mesh from a GIFTI .surf.gii file."""
    cache_key = f"{species}_{hemisphere}"
    if cache_key in SURFACE_DATA_CACHE:
        return SURFACE_DATA_CACHE[cache_key]

    pattern = SURFACE_FILE_PATTERNS.get(species.lower())
    if not pattern:
        return None, None

    surf_file = pattern.format(hem=hemisphere)
    surf_path = os.path.join(SURFACE_DIR_GLOBAL, species, surf_file)
    if not os.path.exists(surf_path):
        return None, None

    try:
        img = nib.load(surf_path)
        vertices, faces = img.darrays[0].data, img.darrays[1].data
        SURFACE_DATA_CACHE[cache_key] = (vertices, faces)
        return vertices, faces
    except Exception:
        return None, None


def load_mask(species, hemisphere):
    """Load and cache a temporal-lobe mask (boolean array)."""
    cache_key = f"{species}_{hemisphere}"
    if cache_key in MASK_CACHE:
        return MASK_CACHE[cache_key]

    mask_file = f"{species}_{hemisphere}.func.gii"
    mask_path = os.path.join(MASK_DIR_GLOBAL, species, mask_file)
    if not os.path.exists(mask_path):
        return None

    try:
        img = nib.load(mask_path)
        mask = img.darrays[0].data > 0.5
        MASK_CACHE[cache_key] = mask
        return mask
    except Exception:
        return None


def load_individual_gradients():
    """Scan results/3_individual_species_gradients/ and load combined gradient GIFTIs."""
    global INDIVIDUAL_GRADIENTS, INDIVIDUAL_GRADIENT_INFO
    global INDIVIDUAL_GRADIENT_MAPS, N_INDIVIDUAL_GRADS_BY_SPECIES

    if not os.path.exists(INDIVIDUAL_GRAD_DIR_GLOBAL):
        print("Individual gradient directory not found -- Tab 1 will be empty.")
        return

    for species in sorted(os.listdir(INDIVIDUAL_GRAD_DIR_GLOBAL)):
        species_dir = os.path.join(INDIVIDUAL_GRAD_DIR_GLOBAL, species)
        if not os.path.isdir(species_dir):
            continue

        for hem in ("L", "R"):
            # Only load combined analysis
            pattern = f"all_computed_gradients_{species}_COMBINED_{hem}.func.gii"
            fpath = os.path.join(species_dir, pattern)
            if not os.path.exists(fpath):
                continue
            try:
                img = nib.load(fpath)
                n_grads = len(img.darrays)
                grad_data = np.column_stack([d.data for d in img.darrays])

                INDIVIDUAL_GRADIENTS.setdefault(species, {})[hem] = grad_data
                INDIVIDUAL_GRADIENT_MAPS[(species, hem)] = grad_data
                N_INDIVIDUAL_GRADS_BY_SPECIES[species] = n_grads

                existing = [
                    info for info in INDIVIDUAL_GRADIENT_INFO
                    if info["species"] == species
                ]
                if existing:
                    if hem not in existing[0]["hems"]:
                        existing[0]["hems"].append(hem)
                else:
                    INDIVIDUAL_GRADIENT_INFO.append({
                        "species": species,
                        "hems": [hem],
                        "n_grads": n_grads,
                    })
                print(f"  Loaded: {species} {hem} combined ({n_grads} gradients)")
            except Exception as e:
                print(f"  Error loading {fpath}: {e}")

    # Build per-species DataFrames for the explorer
    _build_per_species_explorer_data()


def _build_per_species_explorer_data():
    """Build DataFrames for per-species explorer from individual gradient maps."""
    global df_by_source, grad_cols_by_source

    for species, n_grads in N_INDIVIDUAL_GRADS_BY_SPECIES.items():
        records = []
        for hem in ("L", "R"):
            gmap = INDIVIDUAL_GRADIENT_MAPS.get((species, hem))
            if gmap is None:
                continue
            mask = load_mask(species, hem)
            if mask is None:
                mask = np.any(gmap != 0.0, axis=1) & np.all(np.isfinite(gmap), axis=1)
            tl_indices = np.where(mask)[0]
            for vtx_id in tl_indices:
                rec = {
                    "species": species, "hem": hem,
                    "species_hem": f"{species}_{hem}",
                    "orig_vtx_id": int(vtx_id),
                }
                for d in range(n_grads):
                    rec[f"g{d + 1}"] = float(gmap[vtx_id, d])
                records.append(rec)

        if records:
            df_sp = pd.DataFrame.from_records(records)
            df_sp["df_idx"] = df_sp.index
            df_by_source[species] = df_sp
            grad_cols_by_source[species] = [f"g{i + 1}" for i in range(n_grads)]
            print(f"  Explorer data for {species}: {len(df_sp)} vertices, "
                  f"{n_grads} gradients")


def load_cross_species_from_npz(npz_file_path):
    """
    Load cross-species data from the .npz produced by script 6.

    Populates:
      - CROSS_SPECIES_GRADIENT_MAPS  (Tab 2)
      - df_global / AVAILABLE_EXPLORER_GRADIENTS  (Tab 3)
    """
    global df_global, CROSS_SPECIES_GRADIENT_MAPS, N_CROSS_SPECIES_GRADIENTS
    global CROSS_SPECIES_SPECIES_LIST, AVAILABLE_EXPLORER_GRADIENTS

    if not os.path.exists(npz_file_path):
        print(f"NPZ file not found: {npz_file_path}")
        return False

    try:
        npz_data = np.load(npz_file_path, allow_pickle=True)
    except Exception as e:
        print(f"ERROR loading NPZ: {e}")
        return False

    required = ("cross_species_gradients", "segment_info_detailed_for_remapping")
    if not all(k in npz_data for k in required):
        print("ERROR: NPZ missing required keys.")
        return False

    grads = npz_data["cross_species_gradients"]
    segments = list(npz_data["segment_info_detailed_for_remapping"])
    n_dims = grads.shape[1]
    N_CROSS_SPECIES_GRADIENTS = n_dims

    # ------------------------------------------------------------------
    # Reconstruct per-vertex gradient maps (Tab 2)
    # ------------------------------------------------------------------
    species_seen = set()
    for seg in segments:
        s             = seg.get("species")
        h             = seg.get("hem")
        start_row     = seg.get("start_row_in_concat")
        end_row       = seg.get("end_row_in_concat")
        tl_indices    = seg.get("original_tl_indices")
        cluster_labs  = seg.get("cluster_labels_for_remapping")
        n_verts_total = seg.get("num_total_surface_verts")

        if any(v is None for v in (s, h, start_row, end_row, tl_indices, cluster_labs, n_verts_total)):
            continue

        species_seen.add(s)
        seg_grads = grads[start_row:end_row, :]

        full_maps = np.zeros((n_verts_total, n_dims), dtype=np.float32)
        for gi in range(n_dims):
            full_maps[tl_indices, gi] = seg_grads[:, gi][cluster_labs]

        CROSS_SPECIES_GRADIENT_MAPS[(s, h)] = full_maps

    CROSS_SPECIES_SPECIES_LIST = sorted(species_seen)

    # ------------------------------------------------------------------
    # Build DataFrame with all gradient dimensions (Tab 3)
    # ------------------------------------------------------------------
    records = []
    for seg in segments:
        s            = seg.get("species")
        h            = seg.get("hem")
        start_row    = seg.get("start_row_in_concat")
        end_row      = seg.get("end_row_in_concat")
        tl_indices   = seg.get("original_tl_indices")
        cluster_labs = seg.get("cluster_labels_for_remapping")

        if any(v is None for v in (s, h, start_row, end_row, tl_indices, cluster_labs)):
            continue

        for i, vtx_id in enumerate(tl_indices):
            gidx = start_row + cluster_labs[i]
            if not (0 <= gidx < grads.shape[0]):
                continue
            rec = {
                "species": s, "hem": h, "species_hem": f"{s}_{h}",
                "orig_vtx_id": int(vtx_id),
            }
            for d in range(n_dims):
                rec[f"g{d + 1}"] = grads[gidx, d]
            records.append(rec)

    df_global = pd.DataFrame.from_records(records)
    if not df_global.empty:
        df_global["df_idx"] = df_global.index
        AVAILABLE_EXPLORER_GRADIENTS = [f"g{i + 1}" for i in range(n_dims)]
        df_by_source["cross_species"] = df_global
        grad_cols_by_source["cross_species"] = list(AVAILABLE_EXPLORER_GRADIENTS)
    return True


def preload_masks():
    """Eagerly load all mask files into MASK_CACHE at startup."""
    if not os.path.isdir(MASK_DIR_GLOBAL):
        return
    for species in os.listdir(MASK_DIR_GLOBAL):
        species_dir = os.path.join(MASK_DIR_GLOBAL, species)
        if not os.path.isdir(species_dir):
            continue
        for hem in ("L", "R"):
            result = load_mask(species, hem)
            if result is not None:
                print(f"  Mask cached: {species} {hem} ({int(result.sum())} TL vertices)")


def setup_dynamic_plot_configs(df, defaults, default_symbols):
    """Assign colours / symbols for any species_hem combos found in *df*."""
    global PLOT_CONFIGS_SCATTER_GLOBAL, SPECIES_SYMBOLS_GLOBAL
    PLOT_CONFIGS_SCATTER_GLOBAL = defaults.copy()
    SPECIES_SYMBOLS_GLOBAL = default_symbols.copy()
    if df.empty:
        return

    for sh_key in df["species_hem"].unique():
        if sh_key not in PLOT_CONFIGS_SCATTER_GLOBAL:
            parts = sh_key.split("_")
            species = parts[0] if parts else sh_key
            hem = parts[1] if len(parts) > 1 else ""
            label = f"{species.capitalize()} {hem.upper()}".strip()
            idx = len(PLOT_CONFIGS_SCATTER_GLOBAL) % len(go.colors.qualitative.Plotly)
            PLOT_CONFIGS_SCATTER_GLOBAL[sh_key] = {
                "color": go.colors.qualitative.Plotly[idx],
                "label": label,
            }

    for spec in df["species"].unique():
        if spec not in SPECIES_SYMBOLS_GLOBAL:
            sym_pool = ["circle", "square", "diamond", "cross", "x", "star"]
            new_count = sum(1 for s in SPECIES_SYMBOLS_GLOBAL if s not in default_symbols)
            SPECIES_SYMBOLS_GLOBAL[spec] = sym_pool[new_count % len(sym_pool)]


# ===================================================================
#  Figure-generation utilities
# ===================================================================

def _empty_fig(msg="", height=450):
    return go.Figure(layout={"title_text": msg, "height": height})


def make_surface_with_gradient(
    species, hemisphere, gradient_values, title="",
    colorscale="RdBu_r", cmin=None, cmax=None,
    show_colorbar=True, height=450,
):
    """3D brain surface coloured by per-vertex gradient values."""
    vertices, faces = load_surface(species, hemisphere)
    if vertices is None:
        return _empty_fig(f"Surface not found: {species} {hemisphere}", height)
    if gradient_values is None or len(gradient_values) != len(vertices):
        return _empty_fig(f"Gradient data mismatch: {species} {hemisphere}", height)

    mask = load_mask(species, hemisphere)
    if mask is None:
        mask = (gradient_values != 0.0) & np.isfinite(gradient_values)

    if cmin is None or cmax is None:
        roi_vals = gradient_values[mask & np.isfinite(gradient_values)]
        if roi_vals.size > 0:
            cmin = float(np.nanmin(roi_vals))
            cmax = float(np.nanmax(roi_vals))
        else:
            cmin, cmax = -1.0, 1.0
        if cmin == cmax:
            cmin, cmax = cmin - 1.0, cmax + 1.0

    _lighting = dict(ambient=0.65, diffuse=0.7, specular=0.1)
    _lightpos = dict(x=100, y=200, z=300)

    tl_face_mask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
    tl_faces = faces[tl_face_mask]
    non_tl_faces = faces[~tl_face_mask]

    safe_grad = np.nan_to_num(gradient_values, nan=0.0, posinf=0.0, neginf=0.0)

    fig = go.Figure()

    if len(non_tl_faces) > 0:
        ntl_uv = np.unique(non_tl_faces)
        ntl_remap = np.empty(len(vertices), dtype=np.intp)
        ntl_remap[ntl_uv] = np.arange(len(ntl_uv))
        fig.add_trace(go.Mesh3d(
            x=vertices[ntl_uv, 0], y=vertices[ntl_uv, 1], z=vertices[ntl_uv, 2],
            i=ntl_remap[non_tl_faces[:, 0]],
            j=ntl_remap[non_tl_faces[:, 1]],
            k=ntl_remap[non_tl_faces[:, 2]],
            color="rgb(211,211,211)", opacity=1.0,
            hoverinfo="none",
            lighting=_lighting, lightposition=_lightpos,
        ))

    if len(tl_faces) > 0:
        tl_uv = np.unique(tl_faces)
        tl_remap = np.empty(len(vertices), dtype=np.intp)
        tl_remap[tl_uv] = np.arange(len(tl_uv))
        fig.add_trace(go.Mesh3d(
            x=vertices[tl_uv, 0], y=vertices[tl_uv, 1], z=vertices[tl_uv, 2],
            i=tl_remap[tl_faces[:, 0]],
            j=tl_remap[tl_faces[:, 1]],
            k=tl_remap[tl_faces[:, 2]],
            intensity=safe_grad[tl_uv],
            intensitymode="vertex",
            colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=show_colorbar,
            colorbar=dict(title="Value", len=0.6) if show_colorbar else None,
            opacity=1.0,
            hoverinfo="none",
            lighting=_lighting, lightposition=_lightpos,
        ))

    eye = dict(x=-1.7, y=0, z=0) if hemisphere.upper() == "L" else dict(x=1.7, y=0, z=0)
    fig.update_layout(
        title_text=title, title_font_size=12, height=height,
        margin=dict(l=5, r=60, t=40, b=5),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=1, z=1), aspectmode="data",
            camera=dict(eye=eye),
        ),
    )
    return fig


def make_surface_with_gradient_and_highlight(
    species, hemisphere, gradient_values, title="",
    colorscale="RdBu_r", cmin=None, cmax=None,
    show_colorbar=True, height=350, highlight_vtx_id=None,
):
    """3D brain surface with gradient colours AND an optional vertex highlight.

    Used by the Surface Plot mode in the explorer tab.
    """
    fig = make_surface_with_gradient(
        species, hemisphere, gradient_values, title,
        colorscale=colorscale, cmin=cmin, cmax=cmax,
        show_colorbar=show_colorbar, height=height,
    )

    if highlight_vtx_id is not None:
        vertices, _ = load_surface(species, hemisphere)
        if vertices is not None and 0 <= int(highlight_vtx_id) < len(vertices):
            vtx = vertices[int(highlight_vtx_id)]
            fig.add_trace(go.Scatter3d(
                x=[vtx[0]], y=[vtx[1]], z=[vtx[2]], mode="markers",
                marker=dict(size=8, color="yellow", line=dict(width=2, color="black")),
                hoverinfo="skip",
            ))

    return fig


def make_surface_plot_highlight(species, hemisphere, highlight_vtx_id=None, title="Surface"):
    """Grey brain surface with an optional highlighted vertex marker."""
    vertices, faces = load_surface(species, hemisphere)
    if vertices is None:
        return _empty_fig(f"Surface not found: {species} {hemisphere}", 350)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightgrey", opacity=1.0, hoverinfo="none",
    ))

    if highlight_vtx_id is not None and 0 <= int(highlight_vtx_id) < len(vertices):
        vtx = vertices[int(highlight_vtx_id)]
        fig.add_trace(go.Scatter3d(
            x=[vtx[0]], y=[vtx[1]], z=[vtx[2]], mode="markers",
            marker=dict(size=8, color="yellow", line=dict(width=2, color="black")),
            hoverinfo="skip",
        ))

    eye = dict(x=-1.7, y=0, z=0) if hemisphere.upper() == "L" else dict(x=1.7, y=0, z=0)
    fig.update_layout(
        title_text=title, title_font_size=12, height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=1, z=1), aspectmode="data",
            camera=dict(eye=eye),
        ),
    )
    return fig


def _load_blueprint(species, hemisphere):
    """Load and cache the masked-average blueprint for a species/hemisphere."""
    cache_key = (species, hemisphere)
    if cache_key in BLUEPRINT_CACHE:
        return BLUEPRINT_CACHE[cache_key]

    bp_file = f"average_{species}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii"
    bp_path = os.path.join(AVERAGE_BP_DIR_GLOBAL, species, bp_file)
    if not os.path.exists(bp_path):
        BLUEPRINT_CACHE[cache_key] = None
        return None
    try:
        img = nib.load(bp_path)
        data = np.array([d.data for d in img.darrays]).T  # (vertices, tracts)
        BLUEPRINT_CACHE[cache_key] = data
        return data
    except Exception:
        BLUEPRINT_CACHE[cache_key] = None
        return None


def get_vertex_profile(species, hemisphere, vertex_id):
    """Retrieve the connectivity profile for a single vertex."""
    if vertex_id == -1 or pd.isna(vertex_id):
        return None
    data = _load_blueprint(species, hemisphere)
    if data is None:
        return None
    vertex_id = int(vertex_id)
    if 0 <= vertex_id < data.shape[0]:
        return data[vertex_id, :]
    return None


def make_spider(profile, label, color):
    """Radar / spider plot for a connectivity profile."""
    if profile is None or profile.size == 0:
        return go.Figure(layout={"title_text": f"No data for {label}", "height": 400})

    values = np.concatenate((profile, [profile[0]]))
    categories = TRACT_NAMES_GLOBAL + [TRACT_NAMES_GLOBAL[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself", name=label, line_color=color,
    ))
    max_val = np.max(values) if values.size > 0 else 0.05
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max_val * 1.1, 0.05)]),
            angularaxis=dict(tickfont=dict(size=8)),
        ),
        showlegend=False, title_text=label, title_font_size=12,
        margin=dict(l=30, r=30, t=50, b=30), height=400,
    )
    return fig


def make_scatter(
    x_grad="g1", y_grad="g2",
    selected_df_idx=None, closest_df_idx=None,
    xaxis_range=None, yaxis_range=None,
    df_source=None, plot_configs=None, species_symbols=None,
):
    """Main scatter plot with marginal histograms for the explorer tab."""
    df = df_source if df_source is not None else df_global
    if plot_configs is None:
        plot_configs = PLOT_CONFIGS_SCATTER_GLOBAL
    if species_symbols is None:
        species_symbols = SPECIES_SYMBOLS_GLOBAL

    if df.empty or x_grad not in df.columns or y_grad not in df.columns:
        return _empty_fig("No data loaded.", 950)

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
        column_widths=[0.92, 0.08], row_heights=[0.08, 0.92],
        horizontal_spacing=0.01, vertical_spacing=0.01,
        specs=[[{"type": "histogram"}, {}],
               [{"type": "scattergl"}, {"type": "histogram"}]],
    )

    # Marginal histograms
    for sh_key, cfg in plot_configs.items():
        grp = df[df["species_hem"] == sh_key]
        if grp.empty:
            continue
        fig.add_trace(go.Histogram(x=grp[x_grad], marker_color=cfg["color"], opacity=0.8, showlegend=False), row=1, col=1)
        fig.add_trace(go.Histogram(y=grp[y_grad], marker_color=cfg["color"], opacity=0.8, showlegend=False), row=2, col=2)

    # Main scatter
    legend_added = set()
    for sh_key, grp in df.groupby("species_hem"):
        cfg = plot_configs.get(sh_key, {})
        label = cfg.get("label")
        show_leg = label and label not in legend_added
        if show_leg:
            legend_added.add(label)

        fig.add_trace(go.Scattergl(
            x=grp[x_grad], y=grp[y_grad], mode="markers",
            marker=dict(
                size=8, opacity=0.88,
                line=dict(width=1, color="black"),
                color=cfg.get("color"),
                symbol=species_symbols.get(grp["species"].iloc[0]),
            ),
            name=label, showlegend=bool(show_leg),
            customdata=np.stack(
                [grp["df_idx"], grp["species"], grp["hem"], grp["orig_vtx_id"]], axis=-1,
            ),
            hovertemplate=(
                f"{x_grad}: %{{x:.3f}}<br>{y_grad}: %{{y:.3f}}<br>"
                "species: %{customdata[1]}<br>hem: %{customdata[2]}<br>"
                "vtx: %{customdata[3]}<extra></extra>"
            ),
        ), row=2, col=1)

    # Highlight selected point
    if selected_df_idx is not None and selected_df_idx in df.index:
        row = df.loc[selected_df_idx]
        fig.add_trace(go.Scatter(
            x=[row[x_grad]], y=[row[y_grad]], mode="markers",
            marker=dict(size=19, color="black", line=dict(width=4, color="yellow"), symbol="x"),
            showlegend=False,
            customdata=[[selected_df_idx, row["species"], row["hem"], row["orig_vtx_id"]]],
            hovertemplate=(
                f"{x_grad}: %{{x:.3f}}<br>{y_grad}: %{{y:.3f}}<br>"
                "species: %{customdata[1]}<br>hem: %{customdata[2]}<br>"
                "vtx: %{customdata[3]}<extra></extra>"
            ),
        ), row=2, col=1)

    # Highlight closest-neighbour point
    if closest_df_idx is not None and closest_df_idx in df.index:
        row = df.loc[closest_df_idx]
        clr = plot_configs.get(f"{row['species']}_{row['hem']}", {}).get("color", "grey")
        fig.add_trace(go.Scatter(
            x=[row[x_grad]], y=[row[y_grad]], mode="markers",
            marker=dict(size=30, color=clr, line=dict(width=6, color="yellow"), symbol="star"),
            showlegend=False,
            customdata=[[closest_df_idx, row["species"], row["hem"], row["orig_vtx_id"]]],
            hovertemplate=(
                f"{x_grad}: %{{x:.3f}}<br>{y_grad}: %{{y:.3f}}<br>"
                "species: %{customdata[1]}<br>hem: %{customdata[2]}<br>"
                "vtx: %{customdata[3]}<extra></extra>"
            ),
        ), row=2, col=1)

    x_label = f"Gradient {x_grad[1:]}" if x_grad.startswith("g") else x_grad
    y_label = f"Gradient {y_grad[1:]}" if y_grad.startswith("g") else y_grad
    fig.update_layout(
        height=950,
        title_text=f"Temporal Lobe Embedding ({x_label} vs {y_label})",
        uirevision=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=x_label, row=2, col=1, range=xaxis_range)
    fig.update_yaxes(title_text=y_label, row=2, col=1, range=yaxis_range)
    return fig


# ===================================================================
#  Helper: resolve the right gradient maps for explorer surface mode
# ===================================================================

def _get_explorer_surface_data(data_source, grad_idx, colorscale):
    """Return a list of (species, hem, fig) for the selected gradient.

    For single-species sources returns 2 hemispheres.
    For cross-species returns 4 (chimp L, chimp R, human L, human R).
    """
    results = []

    if data_source == "cross_species":
        species_order = ["chimpanzee", "human"]
        maps = CROSS_SPECIES_GRADIENT_MAPS
        n_grads = N_CROSS_SPECIES_GRADIENTS
    else:
        species_order = [data_source]
        maps = {(s, h): v for (s, h), v in INDIVIDUAL_GRADIENT_MAPS.items()
                if s == data_source}
        n_grads = N_INDIVIDUAL_GRADS_BY_SPECIES.get(data_source, 0)

    if grad_idx is None or grad_idx >= n_grads:
        return results

    # Shared colour range
    extremes = []
    for (s, h), gmap in maps.items():
        if s not in species_order:
            continue
        if grad_idx < gmap.shape[1]:
            col = gmap[:, grad_idx]
            m = load_mask(s, h)
            if m is None:
                m = (col != 0.0) & np.isfinite(col)
            roi = col[m & np.isfinite(col)]
            if roi.size > 0:
                extremes.extend([float(roi.min()), float(roi.max())])

    shared_min = min(extremes) if extremes else -1.0
    shared_max = max(extremes) if extremes else 1.0

    for species in species_order:
        for hem in ("L", "R"):
            gmap = maps.get((species, hem))
            if gmap is not None and grad_idx < gmap.shape[1]:
                values = gmap[:, grad_idx]
                title = f"{species.capitalize()} {hem}"
                results.append((species, hem, values, title, shared_min, shared_max))

    return results


# ===================================================================
#  Tab layouts
# ===================================================================

def _no_data_message(text):
    return html.Div(
        html.P(text, style={"color": "grey", "fontStyle": "italic", "padding": "40px"}),
    )


def create_tab1_layout():
    """Tab 1 -- Individual Gradients on surfaces (combined only)."""
    if not INDIVIDUAL_GRADIENT_INFO:
        return _no_data_message(
            "No individual gradient data found.  Run Script 3 first to generate gradient files."
        )

    dataset_options = []
    for info in INDIVIDUAL_GRADIENT_INFO:
        label = f"{info['species'].capitalize()}"
        value = info['species']
        dataset_options.append({"label": label, "value": value})

    default_ds = dataset_options[0]["value"]
    max_grads = INDIVIDUAL_GRADIENT_INFO[0]["n_grads"]

    return html.Div([
        # Controls row
        html.Div([
            html.Div([
                html.Label("Species:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="tab1-dataset", options=dataset_options,
                    value=default_ds, clearable=False, style={"width": "220px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Label("Gradient:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="tab1-gradient-num",
                    options=[{"label": f"Gradient {i + 1}", "value": i} for i in range(max_grads)],
                    value=0, clearable=False, style={"width": "160px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Label("Colorscale:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="tab1-colorscale", options=COLORSCALE_OPTIONS,
                    value="RdBu_r", clearable=False, style={"width": "200px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "gap": "24px", "padding": "16px",
            "flexWrap": "wrap", "alignItems": "center",
        }),
        # Surface plots (L and R side-by-side)
        html.Div([
            dcc.Graph(id="tab1-surface-L", style={"flex": "1"}, config={"scrollZoom": False}),
            dcc.Graph(id="tab1-surface-R", style={"flex": "1"}, config={"scrollZoom": False}),
        ], style={"display": "flex", "gap": "10px"}),
    ])


def create_tab2_layout():
    """Tab 2 -- Cross-species gradients on all surfaces."""
    if N_CROSS_SPECIES_GRADIENTS == 0:
        return _no_data_message(
            "No cross-species gradient data found.  Run Scripts 5 & 6 first."
        )

    return html.Div([
        # Controls row
        html.Div([
            html.Div([
                html.Label("Gradient:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="tab2-gradient-num",
                    options=[{"label": f"Gradient {i + 1}", "value": i}
                             for i in range(N_CROSS_SPECIES_GRADIENTS)],
                    value=0, clearable=False, style={"width": "160px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Label("Colorscale:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="tab2-colorscale", options=COLORSCALE_OPTIONS,
                    value="RdBu_r", clearable=False, style={"width": "200px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "gap": "24px", "padding": "16px",
            "flexWrap": "wrap", "alignItems": "center",
        }),
        # Container for all species surfaces (filled by callback)
        html.Div(id="tab2-surfaces-container"),
    ])


def _build_data_source_options():
    """Build the dropdown options for the explorer data-source selector."""
    options = []
    if "cross_species" in df_by_source:
        options.append({"label": "Cross-Species", "value": "cross_species"})
    for species in sorted(N_INDIVIDUAL_GRADS_BY_SPECIES.keys()):
        if species in df_by_source:
            options.append({"label": species.capitalize(), "value": species})
    return options


def _build_grad_options(source_key):
    """Gradient dropdown options for the given data source."""
    cols = grad_cols_by_source.get(source_key, AVAILABLE_EXPLORER_GRADIENTS)
    return [{"label": f"Gradient {g[1:]}", "value": g} for g in cols]


def create_tab3_layout():
    """Tab 3 -- Interactive Explorer (scatter + gradient surface)."""
    global EMPTY_SPIDER, EMPTY_SURFACE_FIG
    EMPTY_SPIDER = go.Figure(
        go.Scatterpolar(r=[], theta=[]),
        layout={"height": 400},
    )
    EMPTY_SURFACE_FIG = go.Figure(layout={
        "height": 350,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    })

    has_any_data = bool(df_by_source)
    if not has_any_data:
        return _no_data_message(
            "No gradient data found.  Run Scripts 3, 5 & 6 to generate data."
        )

    data_source_options = _build_data_source_options()
    default_source = data_source_options[0]["value"] if data_source_options else "cross_species"
    grad_options = _build_grad_options(default_source)
    default_x = grad_options[0]["value"] if grad_options else "g1"
    default_y = grad_options[1]["value"] if len(grad_options) > 1 else default_x
    default_match = "different" if default_source == "cross_species" else "same"

    return html.Div([
        dcc.Store(id="zoom-state", data=None),
        dcc.Store(id="selected-idx", data=None),

        # ---- Top control bar ----
        html.Div([
            html.Div([
                html.Label("Data Source:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="explorer-data-source",
                    options=data_source_options,
                    value=default_source,
                    clearable=False,
                    style={"width": "200px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Label("X-Axis:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="explorer-x-grad", options=grad_options,
                    value=default_x, clearable=False, style={"width": "160px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Label("Y-Axis:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="explorer-y-grad", options=grad_options,
                    value=default_y, clearable=False, style={"width": "160px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Label("Show on Surface:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="explorer-surface-grad", options=grad_options,
                    value=default_x, clearable=False, style={"width": "160px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Label("Colorscale:", style={"fontWeight": "bold", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="explorer-colorscale", options=COLORSCALE_OPTIONS,
                    value="RdBu_r", clearable=False, style={"width": "200px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "gap": "24px", "padding": "16px",
            "alignItems": "center", "flexWrap": "wrap",
            "borderBottom": "1px solid #ddd", "marginBottom": "8px",
        }),

        # Main grid: scatter on left, detail panels on right
        html.Div([
            # Left column -- scatter plot
            dcc.Graph(id="scatter-g", figure=_empty_fig("Loading...", 950), config={"scrollZoom": False}),
            # Right column -- clicked point + neighbour details
            html.Div([
                html.Div([
                    html.H4("Clicked Point", style={"margin": "0 0 4px 0"}),
                    html.Div([
                        dcc.Graph(id="clicked-spider", figure=EMPTY_SPIDER, style={"width": "48%"}, config={"scrollZoom": False}),
                        dcc.Graph(id="clicked-surface", figure=EMPTY_SURFACE_FIG, style={"width": "48%"}, config={"scrollZoom": False}),
                    ], style={"display": "flex", "justifyContent": "space-between"}),
                ]),
                # Distance / match mode controls
                html.Div([
                    html.Div([
                        html.Label("Distance Mode:", style={"fontWeight": "bold", "marginRight": "8px"}),
                        dcc.Dropdown(
                            id="distance-mode",
                            options=[
                                {"label": "Euclidean", "value": "euclidean"},
                                {"label": "X-axis only", "value": "x_only"},
                                {"label": "Y-axis only", "value": "y_only"},
                            ],
                            value="euclidean", clearable=False, style={"width": "160px"},
                        ),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div([
                        html.Label("Match Mode:", style={"fontWeight": "bold", "marginRight": "8px"}),
                        dcc.Dropdown(
                            id="match-mode",
                            options=[
                                {"label": "Cross-Species", "value": "different"},
                                {"label": "Same Species", "value": "same"},
                            ],
                            value=default_match, clearable=False, style={"width": "160px"},
                        ),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={
                    "display": "grid", "grid-template-columns": "1fr 1fr",
                    "gap": "20px", "padding": "12px 0",
                }),
                html.Div([
                    html.H4("Closest Neighbor", style={"margin": "0 0 4px 0"}),
                    html.Div([
                        dcc.Graph(id="closest-spider", figure=EMPTY_SPIDER, style={"width": "48%"}, config={"scrollZoom": False}),
                        dcc.Graph(id="closest-surface", figure=EMPTY_SURFACE_FIG, style={"width": "48%"}, config={"scrollZoom": False}),
                    ], style={"display": "flex", "justifyContent": "space-between"}),
                ]),
            ], style={"display": "flex", "flexDirection": "column", "gap": "12px"}),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "minmax(700px, 1fr) 850px",
            "gap": "20px", "width": "100%",
        }),

        # ---- Gradient surface panel ----
        html.Div(id="explorer-surface-container", style={
            "padding": "16px 0", "borderTop": "1px solid #ddd", "marginTop": "16px",
        }),
    ])


# ===================================================================
#  Dash app & top-level layout
# ===================================================================
server_flask = Flask(__name__)
app = dash.Dash(__name__, server=server_flask, suppress_callback_exceptions=True)


def create_app_layout():
    return html.Div([
        html.H2("Cross-Species Connectivity Gradients", style={"padding": "0 16px"}),
        dcc.Tabs(id="main-tabs", value="tab-3", children=[
            dcc.Tab(label="Individual Gradients",    value="tab-1", children=[create_tab1_layout()]),
            dcc.Tab(label="Cross-Species Gradients", value="tab-2", children=[create_tab2_layout()]),
            dcc.Tab(label="Interactive Explorer",     value="tab-3", children=[create_tab3_layout()]),
        ]),
    ])


# ===================================================================
#  Callbacks -- Tab 1
# ===================================================================

@app.callback(
    Output("tab1-gradient-num", "options"),
    Output("tab1-gradient-num", "value"),
    Input("tab1-dataset", "value"),
    prevent_initial_call=True,
)
def update_tab1_gradient_options(species):
    """Refresh the gradient-number dropdown when the species selection changes."""
    if not species:
        return [], None
    info = next(
        (i for i in INDIVIDUAL_GRADIENT_INFO if i["species"] == species), None,
    )
    if not info:
        return [], None
    opts = [{"label": f"Gradient {i + 1}", "value": i} for i in range(info["n_grads"])]
    return opts, 0


@app.callback(
    Output("tab1-surface-L", "figure"),
    Output("tab1-surface-R", "figure"),
    Input("tab1-dataset", "value"),
    Input("tab1-gradient-num", "value"),
    Input("tab1-colorscale", "value"),
)
def update_tab1_surfaces(species, grad_idx, colorscale):
    """Render L and R hemisphere surfaces coloured by the selected gradient."""
    empty = _empty_fig("Select a species and gradient.", 500)
    if not species or grad_idx is None:
        return empty, empty

    vals_by_hem = {}
    for hem in ("L", "R"):
        gdata = INDIVIDUAL_GRADIENTS.get(species, {}).get(hem)
        if gdata is not None and grad_idx < gdata.shape[1]:
            vals_by_hem[hem] = gdata[:, grad_idx]

    # Combined analysis always uses shared colour range
    all_roi_vals = []
    for hem, v in vals_by_hem.items():
        m = load_mask(species, hem)
        if m is None:
            m = (v != 0.0) & np.isfinite(v)
        all_roi_vals.append(v[m & np.isfinite(v)])
    all_roi = np.concatenate(all_roi_vals) if all_roi_vals else np.array([])
    if all_roi.size > 0:
        shared_min = float(all_roi.min())
        shared_max = float(all_roi.max())
    else:
        shared_min, shared_max = -1.0, 1.0

    figs = []
    for hem in ("L", "R"):
        if hem in vals_by_hem:
            title = f"{species.capitalize()} {hem} -- Gradient {grad_idx + 1}"
            fig = make_surface_with_gradient(
                species, hem, vals_by_hem[hem], title,
                colorscale=colorscale,
                cmin=shared_min, cmax=shared_max,
                show_colorbar=True, height=500,
            )
        else:
            fig = _empty_fig(f"No data: {species} {hem}", 500)
        figs.append(fig)

    return figs[0], figs[1]


# ===================================================================
#  Callbacks -- Tab 2
# ===================================================================

@app.callback(
    Output("tab2-surfaces-container", "children"),
    Input("tab2-gradient-num", "value"),
    Input("tab2-colorscale", "value"),
)
def update_tab2_surfaces(grad_idx, colorscale):
    """Render all species/hemisphere surfaces for the selected gradient."""
    if grad_idx is None:
        return html.P("Select a gradient.", style={"color": "grey", "padding": "20px"})

    # Shared colour range across every species/hemisphere (finite masked vertices)
    extremes = []
    for (_s, _h), maps in CROSS_SPECIES_GRADIENT_MAPS.items():
        if grad_idx < maps.shape[1]:
            col = maps[:, grad_idx]
            m = load_mask(_s, _h)
            if m is None:
                m = (col != 0.0) & np.isfinite(col)
            roi = col[m & np.isfinite(col)]
            if roi.size > 0:
                extremes.extend([float(roi.min()), float(roi.max())])

    shared_min = min(extremes) if extremes else -1.0
    shared_max = max(extremes) if extremes else 1.0

    species_blocks = []
    for species in CROSS_SPECIES_SPECIES_LIST:
        hem_graphs = []
        for hem in ("L", "R"):
            maps = CROSS_SPECIES_GRADIENT_MAPS.get((species, hem))
            if maps is not None and grad_idx < maps.shape[1]:
                values = maps[:, grad_idx]
                title = f"{species.capitalize()} {hem}"
                fig = make_surface_with_gradient(
                    species, hem, values, title,
                    colorscale=colorscale,
                    cmin=shared_min, cmax=shared_max,
                    show_colorbar=(hem == "R"), height=450,
                )
            else:
                fig = _empty_fig(f"No data: {species} {hem}", 450)
            hem_graphs.append(dcc.Graph(figure=fig, style={"flex": "1"}, config={"scrollZoom": False}))

        species_blocks.append(html.Div([
            html.H4(
                f"{species.capitalize()}",
                style={"margin": "8px 0 4px 16px"},
            ),
            html.Div(hem_graphs, style={"display": "flex", "gap": "10px"}),
        ]))

    return html.Div(
        species_blocks,
        style={"display": "flex", "flexDirection": "column", "gap": "16px"},
    )


# ===================================================================
#  Callbacks -- Tab 3  (Interactive Explorer)
# ===================================================================

# --- Update gradient dropdowns + match mode when data source changes ---
@app.callback(
    Output("explorer-x-grad", "options"),
    Output("explorer-x-grad", "value"),
    Output("explorer-y-grad", "options"),
    Output("explorer-y-grad", "value"),
    Output("explorer-surface-grad", "options"),
    Output("explorer-surface-grad", "value"),
    Output("match-mode", "value"),
    Input("explorer-data-source", "value"),
    prevent_initial_call=True,
)
def update_explorer_grad_options(data_source):
    opts = _build_grad_options(data_source)
    val_x = opts[0]["value"] if opts else "g1"
    val_y = opts[1]["value"] if len(opts) > 1 else val_x
    match_val = "different" if data_source == "cross_species" else "same"
    return opts, val_x, opts, val_y, opts, val_x, match_val


# --- Zoom persistence ---
@app.callback(
    Output("zoom-state", "data"),
    Input("scatter-g", "relayoutData"),
    State("zoom-state", "data"),
    prevent_initial_call=True,
)
def save_zoom(relayoutData, old_zoom):
    if relayoutData is None:
        return dash.no_update
    new_zoom = old_zoom or {}
    if "xaxis.range[0]" in relayoutData:
        new_zoom["xaxis"] = [relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"]]
    elif "xaxis.autorange" in relayoutData:
        new_zoom["xaxis"] = None
    if "yaxis.range[0]" in relayoutData:
        new_zoom["yaxis"] = [relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"]]
    elif "yaxis.autorange" in relayoutData:
        new_zoom["yaxis"] = None
    return new_zoom


# --- Scatter: main interaction callback ---
@app.callback(
    Output("selected-idx", "data"),
    Output("clicked-spider", "figure"),
    Output("closest-spider", "figure"),
    Output("clicked-surface", "figure"),
    Output("closest-surface", "figure"),
    Output("scatter-g", "figure"),
    Input("scatter-g", "clickData"),
    Input("distance-mode", "value"),
    Input("match-mode", "value"),
    Input("explorer-x-grad", "value"),
    Input("explorer-y-grad", "value"),
    Input("explorer-data-source", "value"),
    State("zoom-state", "data"),
    State("selected-idx", "data"),
)
def handle_scatter_interactions(
    clickData, distance_mode, match_mode, x_grad, y_grad,
    data_source, zoom_state, current_idx,
):
    """Main callback for the scatter-plot explorer."""
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    df = df_by_source.get(data_source, df_global)
    if df.empty:
        empty_scatter = _empty_fig("No data for this source.", 950)
        return None, EMPTY_SPIDER, EMPTY_SPIDER, EMPTY_SURFACE_FIG, EMPTY_SURFACE_FIG, empty_scatter

    # On initial load or when axes / data source change, just render the scatter
    if not triggered_id or triggered_id in ("explorer-x-grad", "explorer-y-grad", "explorer-data-source"):
        scatter = make_scatter(x_grad, y_grad, df_source=df)
        return None, EMPTY_SPIDER, EMPTY_SPIDER, EMPTY_SURFACE_FIG, EMPTY_SURFACE_FIG, scatter

    # Toggle selection on click
    selected_idx = current_idx
    if triggered_id == "scatter-g" and clickData:
        pt = clickData["points"][0]
        cd = pt.get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            try:
                clicked_df_idx = int(cd[0])
                selected_idx = None if clicked_df_idx == current_idx else clicked_df_idx
            except (ValueError, TypeError):
                selected_idx = None

    zoom = zoom_state or {}

    # Nothing selected -> clear details
    if selected_idx is None or selected_idx not in df.index:
        scatter = make_scatter(
            x_grad, y_grad,
            xaxis_range=zoom.get("xaxis"), yaxis_range=zoom.get("yaxis"),
            df_source=df,
        )
        return None, EMPTY_SPIDER, EMPTY_SPIDER, EMPTY_SURFACE_FIG, EMPTY_SURFACE_FIG, scatter

    # ------ Clicked-point details ------
    sel = df.loc[selected_idx]
    s_sp, s_hem, s_vtx = sel["species"], sel["hem"], sel["orig_vtx_id"]
    s_label = f"{s_sp.capitalize()} {s_hem} (vtx {s_vtx})"
    s_color = PLOT_CONFIGS_SCATTER_GLOBAL.get(f"{s_sp}_{s_hem}", {}).get("color")
    clicked_spider  = make_spider(get_vertex_profile(s_sp, s_hem, s_vtx), s_label, s_color)
    clicked_surface = make_surface_plot_highlight(s_sp, s_hem, s_vtx, s_label)

    # ------ Closest neighbour ------
    if match_mode == "different":
        cand = df[df.species != s_sp]
    else:
        cand = df[(df.species == s_sp) & (df.df_idx != selected_idx)]

    closest_idx = None
    closest_spider  = EMPTY_SPIDER
    closest_surface = EMPTY_SURFACE_FIG

    if not cand.empty and x_grad in df.columns and y_grad in df.columns:
        sel_coords  = sel[[x_grad, y_grad]].values.reshape(1, -1)
        cand_coords = cand[[x_grad, y_grad]].values

        if distance_mode == "euclidean":
            dists = euclidean_distances(sel_coords, cand_coords)[0]
        elif distance_mode == "x_only":
            dists = np.abs(cand_coords[:, 0] - sel_coords[0, 0])
        else:  # y_only
            dists = np.abs(cand_coords[:, 1] - sel_coords[0, 1])

        cr = cand.iloc[np.argmin(dists)]
        closest_idx = int(cr.df_idx)
        c_sp, c_hem, c_vtx = cr["species"], cr["hem"], cr["orig_vtx_id"]
        c_label = f"Closest: {c_sp.capitalize()} {c_hem} (vtx {c_vtx})"
        c_color = PLOT_CONFIGS_SCATTER_GLOBAL.get(f"{c_sp}_{c_hem}", {}).get("color")
        closest_spider  = make_spider(get_vertex_profile(c_sp, c_hem, c_vtx), c_label, c_color)
        closest_surface = make_surface_plot_highlight(c_sp, c_hem, c_vtx, c_label)

    scatter = make_scatter(
        x_grad, y_grad, selected_idx, closest_idx,
        zoom.get("xaxis"), zoom.get("yaxis"),
        df_source=df,
    )
    return selected_idx, clicked_spider, closest_spider, clicked_surface, closest_surface, scatter


# --- Gradient surface panel ---
@app.callback(
    Output("explorer-surface-container", "children"),
    Input("explorer-surface-grad", "value"),
    Input("explorer-colorscale", "value"),
    Input("explorer-data-source", "value"),
)
def update_explorer_surfaces(surface_grad, colorscale, data_source):
    """Render brain surfaces showing the selected gradient."""
    def _grad_idx(grad_str):
        if grad_str and grad_str.startswith("g"):
            try:
                return int(grad_str[1:]) - 1
            except ValueError:
                pass
        return 0

    idx = _grad_idx(surface_grad)
    surface_data = _get_explorer_surface_data(data_source, idx, colorscale)

    if not surface_data:
        return html.P("No surface data available for this source.",
                       style={"color": "grey", "padding": "20px"})

    grad_label = (f"Gradient {surface_grad[1:]}"
                  if surface_grad and surface_grad.startswith("g")
                  else str(surface_grad))

    children = [
        html.H4(f"Surface: {grad_label}",
                 style={"margin": "0 0 8px 0", "textAlign": "center"}),
    ]

    graphs = []
    for i, (species, hem, values, title, cmin, cmax) in enumerate(surface_data):
        fig = make_surface_with_gradient(
            species, hem, values, title,
            colorscale=colorscale, cmin=cmin, cmax=cmax,
            show_colorbar=(i == len(surface_data) - 1),
            height=350,
        )
        graphs.append(
            dcc.Graph(figure=fig, config={"scrollZoom": False}, style={"flex": "1"})
        )

    if len(graphs) <= 2:
        children.append(
            html.Div(graphs, style={"display": "flex", "gap": "10px"})
        )
    else:
        # Cross-species: chimpanzee on top, human on bottom
        children.append(html.Div([
            html.Div(graphs[:2], style={"display": "flex", "gap": "10px"}),
            html.Div(graphs[2:], style={"display": "flex", "gap": "10px"}),
        ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}))

    return html.Div(children)


# ===================================================================
#  Main entry-point
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tabbed interactive viewer for cross-species gradient data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--species_list_for_run", type=str, required=True,
        help='Comma-separated species in the Script 6 run (e.g. "human,chimpanzee").',
    )
    parser.add_argument(
        "--target_k_species_for_run", type=str, required=True,
        help="Reference species used in Script 6.",
    )
    parser.add_argument(
        "--project_root", type=str, default=".",
        help="Project root directory (contains data/ and results/).",
    )
    parser.add_argument(
        "--n_tracts", type=int, default=DEFAULT_N_TRACTS_EXPECTED,
        help="Expected number of tracts / features.",
    )
    parser.add_argument(
        "--tract_names", type=str, default=",".join(DEFAULT_TRACT_NAMES),
        help="Comma-separated tract names for spider plots.",
    )
    parser.add_argument(
        "--surface_dir", type=str, default=None,
        help="Directory with species sub-folders of .surf.gii files.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # ---- Paths ----
    species_list = [s.strip().lower() for s in args.species_list_for_run.split(",")]
    run_id = f"{'_'.join(species_list)}_CrossSpecies_kRef_{args.target_k_species_for_run.strip().lower()}"
    npz_filename = f"cross_species_embedding_data_{run_id}.npz"
    npz_file_path = os.path.join(
        args.project_root, "results", "6_cross_species_gradients",
        "intermediates", run_id, npz_filename,
    )

    SURFACE_DIR_GLOBAL      = args.surface_dir or os.path.join(args.project_root, "data", "surfaces")
    INDIVIDUAL_GRAD_DIR_GLOBAL = os.path.join(args.project_root, "results", "3_individual_species_gradients")
    AVERAGE_BP_DIR_GLOBAL   = os.path.join(args.project_root, "results", "2_masked_average_blueprints")
    MASK_DIR_GLOBAL         = os.path.join(args.project_root, "data", "masks")
    N_TRACTS_EXPECTED_GLOBAL = args.n_tracts
    TRACT_NAMES_GLOBAL      = [n.strip() for n in args.tract_names.split(",")]

    # ---- Load data ----
    print("Pre-loading masks ...")
    preload_masks()

    print("Loading individual gradients (Tab 1) ...")
    load_individual_gradients()

    print("Loading cross-species embedding (Tabs 2 & 3) ...")
    if not load_cross_species_from_npz(npz_file_path):
        print("WARNING: Cross-species data unavailable -- Tabs 2 & 3 will show placeholders.")

    setup_dynamic_plot_configs(df_global, DEFAULT_PLOT_CONFIGS_SCATTER, DEFAULT_SPECIES_SYMBOLS)

    # ---- Launch ----
    app.layout = create_app_layout()
    print(f"\nStarting server at http://{args.host}:{args.port}")
    app.run(debug=args.debug, host=args.host, port=args.port)
