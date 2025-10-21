#!/usr/bin/env python3
"""
Interactive Dash + Plotly application for visualizing and exploring cross-species
neuroimaging gradient data.

This script loads the output from a specific analysis pipeline (a .npz file
containing gradient embeddings and vertex information) and generates an interactive
dashboard. The dashboard features:
  - A main scatter plot of the first two gradients with marginal histograms.
  - Selection of individual data points (vertices) to view details.
  - Visualization of the selected vertex's connectivity profile on a spider plot.
  - 3D rendering of the selected vertex on its native brain surface.
  - A tool to find and display the nearest neighbor in the gradient space, either
    within the same species or across different species.
  - A responsive layout that adapts to different screen sizes.

To run the app, specify the parameters of the analysis run, such as the species
list and the reference species, via command-line arguments.
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

# --- Default Configuration and Global State ---
# These constants provide default values if not specified via command line.
DEFAULT_N_TRACTS_EXPECTED = 20
DEFAULT_TRACT_NAMES = [
    "AC", "AF", "AR", "CBD", "CBP", "CBT", "CST", "FA",
    "FMI", "FMA", "FX", "IFOF", "ILF", "MDLF",
    "OR", "SLF I", "SLF II", "SLF III", "UF", "VOF"
]
DEFAULT_PLOT_CONFIGS_SCATTER = {
    "human_L": {"color": "blue", "label": "Human L"},
    "human_R": {"color": "cornflowerblue", "label": "Human R"},
    "chimpanzee_L": {"color": "red", "label": "Chimpanzee L"},
    "chimpanzee_R": {"color": "lightcoral", "label": "Chimpanzee R"}
}
DEFAULT_SPECIES_SYMBOLS = {"human": "circle", "chimpanzee": "circle"}

# Global variables to hold data and settings loaded at runtime.
df_global = pd.DataFrame()
SURFACE_DATA_CACHE = {}
PLOT_CONFIGS_SCATTER_GLOBAL = {}
SPECIES_SYMBOLS_GLOBAL = {}
TRACT_NAMES_GLOBAL = []
N_TRACTS_EXPECTED_GLOBAL = 0
AVERAGE_BP_DIR_GLOBAL = ""
EMPTY_SPIDER = None
EMPTY_SURFACE_FIG = None

# --- Data Loading ---
def load_data_from_npz(npz_file_path):
    """Loads and reshapes gradient data from the specified .npz file."""
    global df_global
    try:
        npz_data = np.load(npz_file_path, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Could not load NPZ file {npz_file_path}. Error: {e}")
        return False

    required_keys = ['cross_species_gradients', 'segment_info_detailed_for_remapping']
    if not all(key in npz_data for key in required_keys):
        print("ERROR: NPZ file is missing required keys.")
        return False

    grads = npz_data['cross_species_gradients']
    segments = list(npz_data['segment_info_detailed_for_remapping'])

    records = []
    for seg_info in segments:
        # Extract metadata for each species/hemisphere segment
        s, h = seg_info.get('species'), seg_info.get('hem')
        start_row, end_row = seg_info.get('start_row_in_concat'), seg_info.get('end_row_in_concat')
        original_tl_indices = seg_info.get('original_tl_indices')
        cluster_labels = seg_info.get('cluster_labels_for_remapping')

        if any(v is None for v in [s, h, start_row, end_row, original_tl_indices, cluster_labels]):
            continue

        # Map each original vertex to its representative gradient vector
        for i, vtx_id in enumerate(original_tl_indices):
            profile_idx = cluster_labels[i]
            global_grad_idx = start_row + profile_idx
            if not (0 <= global_grad_idx < grads.shape[0]):
                continue

            records.append({
                'species': s, 'hem': h, 'species_hem': f"{s}_{h}",
                'g1': grads[global_grad_idx, 0],
                'g2': grads[global_grad_idx, 1] if grads.shape[1] > 1 else 0,
                'orig_vtx_id': int(vtx_id)
            })

    df_global = pd.DataFrame.from_records(records)
    if not df_global.empty:
        df_global["df_idx"] = df_global.index
    return True

def setup_dynamic_plot_configs(df, defaults, default_symbols):
    """Generates plot colors and symbols if data contains new species."""
    global PLOT_CONFIGS_SCATTER_GLOBAL, SPECIES_SYMBOLS_GLOBAL
    PLOT_CONFIGS_SCATTER_GLOBAL = defaults.copy()
    SPECIES_SYMBOLS_GLOBAL = default_symbols.copy()
    if df.empty: return

    for sh_key in df["species_hem"].unique():
        if sh_key not in PLOT_CONFIGS_SCATTER_GLOBAL:
            species, hem = (sh_key.split('_') + [""])[:2]
            label = f"{species.capitalize()} {hem.upper()}".strip()
            color_idx = len(PLOT_CONFIGS_SCATTER_GLOBAL) % len(go.colors.qualitative.Plotly)
            PLOT_CONFIGS_SCATTER_GLOBAL[sh_key] = {"color": go.colors.qualitative.Plotly[color_idx], "label": label}

    for spec in df["species"].unique():
        if spec not in SPECIES_SYMBOLS_GLOBAL:
            symbols = ["circle", "square", "diamond", "cross", "x", "star"]
            new_count = sum(1 for s in SPECIES_SYMBOLS_GLOBAL if s not in default_symbols)
            SPECIES_SYMBOLS_GLOBAL[spec] = symbols[new_count % len(symbols)]

# --- Figure Generation Utilities ---
def load_surface(species, hemisphere):
    """Loads brain surface mesh data, using a cache to avoid re-reading files."""
    cache_key = f"{species}_{hemisphere}"
    if cache_key in SURFACE_DATA_CACHE:
        return SURFACE_DATA_CACHE[cache_key]

    surf_map = {
        'human': f"Human32k.{hemisphere}.inflated.surf.gii",
        'chimpanzee': f"ChimpYerkes29.{hemisphere}.inflated.20k_fs_LR.surf.gii"
    }
    surf_file = surf_map.get(species.lower())
    if not surf_file: return None, None

    surf_path = os.path.join(SURFACE_DIR_GLOBAL, species, surf_file)
    if not os.path.exists(surf_path): return None, None

    try:
        img = nib.load(surf_path)
        vertices, faces = img.darrays[0].data, img.darrays[1].data
        SURFACE_DATA_CACHE[cache_key] = (vertices, faces)
        return vertices, faces
    except Exception:
        return None, None

def make_surface_plot(species, hemisphere, highlight_vtx_id=None, title="Surface"):
    """Creates a 3D plot of a brain surface with an optional highlighted vertex."""
    vertices, faces = load_surface(species, hemisphere)
    if vertices is None:
        return go.Figure(layout={"title_text": f"Surface not found for {species} {hemisphere}", "height": 350})

    fig = go.Figure()
    # Plot the surface mesh using the vertices' native coordinate system
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightgrey', opacity=1.0, hoverinfo='none'
    ))

    if highlight_vtx_id is not None and 0 <= int(highlight_vtx_id) < len(vertices):
        vtx = vertices[int(highlight_vtx_id)]
        fig.add_trace(go.Scatter3d(
            x=[vtx[0]], y=[vtx[1]], z=[vtx[2]], mode='markers',
            marker=dict(size=8, color='yellow', line=dict(width=2, color='black')),
            hoverinfo='skip'
        ))

    camera_eye = dict(x=-1.7, y=0, z=0) if hemisphere.upper() == 'L' else dict(x=1.7, y=0, z=0)
    fig.update_layout(
        title_text=title, title_font_size=12, height=350, margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=1, z=1), aspectmode='data', camera=dict(eye=camera_eye)
        )
    )
    return fig

def get_vertex_profile(species, hemisphere, vertex_id):
    """Retrieves the connectivity profile for a single vertex from a .func.gii file."""
    if vertex_id == -1 or pd.isna(vertex_id): return None
    bp_file = f"average_{species}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii"
    bp_path = os.path.join(AVERAGE_BP_DIR_GLOBAL, species, bp_file)
    if not os.path.exists(bp_path): return None
    try:
        img = nib.load(bp_path)
        data = np.array([d.data for d in img.darrays]).T # (vertices, tracts)
        vertex_id = int(vertex_id)
        if 0 <= vertex_id < data.shape[0]:
            return data[vertex_id, :]
        return None
    except Exception:
        return None

def make_spider(profile, label, color):
    """Creates a spider (radar) plot for a connectivity profile."""
    if profile is None or profile.size == 0:
        return go.Figure(layout={"title_text": f"No data for {label}", "height": 400})

    # Close the loop for the spider plot
    values = np.concatenate((profile, [profile[0]]))
    categories = TRACT_NAMES_GLOBAL + [TRACT_NAMES_GLOBAL[0]]

    fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', name=label, line_color=color))
    max_val = np.max(values) if values.size > 0 else 0.05
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max_val * 1.1, 0.05)]),
                   angularaxis=dict(tickfont=dict(size=8))),
        showlegend=False, title_text=label, title_font_size=12,
        margin=dict(l=30, r=30, t=50, b=30), height=400
    )
    return fig

def make_scatter(selected_df_idx=None, closest_df_idx=None, xaxis_range=None, yaxis_range=None):
    """Creates the main scatter plot with marginal histograms and highlight markers."""
    if df_global.empty:
        return go.Figure(layout={"title_text": "No data loaded.", "height": 950})

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
        column_widths=[0.92, 0.08], row_heights=[0.08, 0.92],
        horizontal_spacing=0.01, vertical_spacing=0.01,
        specs=[[{"type": "histogram"}, {}], [{"type": "scattergl"}, {"type": "histogram"}]]
    )

    # Add marginal histograms
    for sh_key, config in PLOT_CONFIGS_SCATTER_GLOBAL.items():
        group = df_global[df_global["species_hem"] == sh_key]
        if not group.empty:
            fig.add_trace(go.Histogram(x=group["g1"], marker_color=config["color"], opacity=0.8, showlegend=False), row=1, col=1)
            fig.add_trace(go.Histogram(y=group["g2"], marker_color=config["color"], opacity=0.8, showlegend=False), row=2, col=2)

    # Add main scatter plot data
    legend_items_added = set()
    for sh_key, group in df_global.groupby("species_hem"):
        config = PLOT_CONFIGS_SCATTER_GLOBAL.get(sh_key, {})
        label = config.get('label')
        if label and label not in legend_items_added:
            show_leg = True
            legend_items_added.add(label)
        else:
            show_leg = False

        fig.add_trace(go.Scattergl(
            x=group["g1"], y=group["g2"], mode="markers",
            marker=dict(size=8, opacity=0.88, line=dict(width=1, color='black'),
                        color=config.get("color"), symbol=SPECIES_SYMBOLS_GLOBAL.get(group["species"].iloc[0])),
            name=label, showlegend=show_leg,
            customdata=np.stack([group["df_idx"], group["species"], group["hem"], group["orig_vtx_id"]], axis=-1),
            hovertemplate="g1: %{x:.3f}<br>g2: %{y:.3f}<br>species: %{customdata[1]}<br>hem: %{customdata[2]}<br>vtx: %{customdata[3]}<extra></extra>"
        ), row=2, col=1)

    # Add highlight marker for the selected point
    if selected_df_idx is not None:
        row = df_global.loc[selected_df_idx]
        fig.add_trace(go.Scatter(
            x=[row["g1"]], y=[row["g2"]], mode='markers',
            marker=dict(size=19, color='black', line=dict(width=4, color='yellow'), symbol="x"),
            showlegend=False,
            customdata=[[selected_df_idx, row['species'], row['hem'], row['orig_vtx_id']]],
            hovertemplate="g1: %{x:.3f}<br>g2: %{y:.3f}<br>species: %{customdata[1]}<br>hem: %{customdata[2]}<br>vtx: %{customdata[3]}<extra></extra>"
        ), row=2, col=1)

    # Add highlight marker for the closest point
    if closest_df_idx is not None:
        row = df_global.loc[closest_df_idx]
        color = PLOT_CONFIGS_SCATTER_GLOBAL.get(f"{row['species']}_{row['hem']}", {}).get("color", "grey")
        fig.add_trace(go.Scatter(
            x=[row["g1"]], y=[row["g2"]], mode='markers',
            marker=dict(size=30, color=color, line=dict(width=6, color='yellow'), symbol="star"),
            showlegend=False,
            customdata=[[closest_df_idx, row['species'], row['hem'], row['orig_vtx_id']]],
            hovertemplate="g1: %{x:.3f}<br>g2: %{y:.3f}<br>species: %{customdata[1]}<br>hem: %{customdata[2]}<br>vtx: %{customdata[3]}<extra></extra>"
        ), row=2, col=1)

    fig.update_layout(height=950, width=None, title_text="Cross-Species Temporal Lobe Embedding (G1 vs G2)",
                      uirevision=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Gradient 1", row=2, col=1, range=xaxis_range)
    fig.update_yaxes(title_text="Gradient 2", row=2, col=1, range=yaxis_range)
    return fig

# --- Dash App Definition ---
server_flask = Flask(__name__)
app = dash.Dash(__name__, server=server_flask)

def create_app_layout():
    """Defines the HTML structure of the Dash application."""
    global EMPTY_SPIDER, EMPTY_SURFACE_FIG
    EMPTY_SPIDER = go.Figure(go.Scatterpolar(r=[], theta=[]), layout={"height": 400})
    EMPTY_SURFACE_FIG = go.Figure(layout={"height": 350, "paper_bgcolor": 'rgba(0,0,0,0)', "plot_bgcolor": 'rgba(0,0,0,0)'})

    return html.Div([
        dcc.Store(id="zoom-state", data=None),
        dcc.Store(id="selected-idx", data=None),
        html.H2("Cross-Species Gradients: Interactive Exploration"),
        html.Div([
            # Left Column: Main scatter plot
            dcc.Graph(id="scatter-g", figure=make_scatter()),
            # Right Column: Details, controls, and neighbor plots
            html.Div([
                html.Div([
                    html.H4("Clicked Point", style={'margin': '0 0 4px 0'}),
                    html.Div([
                        dcc.Graph(id="clicked-spider", figure=EMPTY_SPIDER, style={'width': '48%'}),
                        dcc.Graph(id="clicked-surface", figure=EMPTY_SURFACE_FIG, style={'width': '48%'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                ]),
                html.Div([
                    html.Div([
                        html.Label("Distance Mode:", style={'fontWeight': 'bold', 'marginRight': '8px'}),
                        dcc.Dropdown(id='distance-mode', options=[{'label': 'Euclidean', 'value': 'euclidean'}, {'label': 'G1', 'value': 'g1'}, {'label': 'G2', 'value': 'g2'}], value='euclidean', clearable=False, style={'width': '140px'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                    html.Div([
                        html.Label("Match Mode:", style={'fontWeight': 'bold', 'marginRight': '8px'}),
                        dcc.Dropdown(id='match-mode', options=[{'label': 'Cross-Species', 'value': 'different'}, {'label': 'Same Species', 'value': 'same'}], value='different', clearable=False, style={'width': '140px'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ], style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'gap': '20px', 'padding': '12px 0'}),
                html.Div([
                    html.H4("Closest Neighbor", style={'margin': '0 0 4px 0'}),
                    html.Div([
                        dcc.Graph(id="closest-spider", figure=EMPTY_SPIDER, style={'width': '48%'}),
                        dcc.Graph(id="closest-surface", figure=EMPTY_SURFACE_FIG, style={'width': '48%'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                ]),
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '12px'}),
        ],
        className='main-container',  # Targeted by assets/style.css for responsiveness
        style={
            'display': 'grid',
            'gridTemplateColumns': 'minmax(700px, 1fr) 850px',
            'gap': '20px', 'width': '100%'}
        )
    ])

# --- Callbacks ---
@app.callback(
    Output("zoom-state", "data"),
    Input("scatter-g", "relayoutData"),
    State("zoom-state", "data"),
    prevent_initial_call=True
)
def save_zoom(relayoutData, old_zoom):
    """Saves the user's zoom/pan state for the main scatter plot."""
    if relayoutData is None: return dash.no_update
    new_zoom = old_zoom or {}
    if 'xaxis.range[0]' in relayoutData:
        new_zoom['xaxis'] = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
    elif 'xaxis.autorange' in relayoutData:
        new_zoom['xaxis'] = None
    if 'yaxis.range[0]' in relayoutData:
        new_zoom['yaxis'] = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']]
    elif 'yaxis.autorange' in relayoutData:
        new_zoom['yaxis'] = None
    return new_zoom

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
    State("zoom-state", "data"),
    State("selected-idx", "data"),
    prevent_initial_call=True
)
def handle_graph_interactions(clickData, distance_mode, match_mode, zoom_state, current_idx):
    """Main callback to handle user interactions like clicking and changing modes."""
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Toggle selection: select a new point or deselect the current one
    selected_idx = current_idx
    if triggered_id == "scatter-g" and clickData:
        clicked_df_idx = clickData["points"][0]["customdata"][0]
        selected_idx = None if clicked_df_idx == current_idx else clicked_df_idx

    # If no point is selected, clear detail plots and update scatter
    if selected_idx is None:
        scatter = make_scatter(xaxis_range=zoom_state.get('xaxis'), yaxis_range=zoom_state.get('yaxis'))
        return None, EMPTY_SPIDER, EMPTY_SPIDER, EMPTY_SURFACE_FIG, EMPTY_SURFACE_FIG, scatter

    # Generate plots for the selected point
    selected_row = df_global.loc[selected_idx]
    s_species, s_hem, s_vtx = selected_row["species"], selected_row["hem"], selected_row["orig_vtx_id"]
    s_label = f"{s_species.capitalize()} {s_hem} (vtx {s_vtx})"
    s_color = PLOT_CONFIGS_SCATTER_GLOBAL.get(f"{s_species}_{s_hem}", {}).get("color")
    clicked_spider = make_spider(get_vertex_profile(s_species, s_hem, s_vtx), s_label, s_color)
    clicked_surface = make_surface_plot(s_species, s_hem, s_vtx, s_label)

    # Find the closest neighbor based on the selected modes
    candidate_df = df_global[df_global.species != s_species if match_mode == 'different' else (df_global.species == s_species) & (df_global.df_idx != selected_idx)]
    closest_other_idx = None
    closest_spider, closest_surface = EMPTY_SPIDER, EMPTY_SURFACE_FIG

    if not candidate_df.empty:
        selected_coords = selected_row[['g1', 'g2']].values.reshape(1, -1)
        candidate_coords = candidate_df[['g1', 'g2']].values

        if distance_mode == 'euclidean':
            distances = euclidean_distances(selected_coords, candidate_coords)[0]
        else:  # 'g1' or 'g2'
            col_idx = 0 if distance_mode == 'g1' else 1
            distances = np.abs(candidate_coords[:, col_idx] - selected_coords[0, col_idx])

        closest_row = candidate_df.iloc[np.argmin(distances)]
        closest_other_idx = int(closest_row.df_idx)
        c_species, c_hem, c_vtx = closest_row["species"], closest_row["hem"], closest_row["orig_vtx_id"]
        c_label = f"Closest: {c_species.capitalize()} {c_hem} (vtx {c_vtx})"
        c_color = PLOT_CONFIGS_SCATTER_GLOBAL.get(f"{c_species}_{c_hem}", {}).get("color")
        closest_spider = make_spider(get_vertex_profile(c_species, c_hem, c_vtx), c_label, c_color)
        closest_surface = make_surface_plot(c_species, c_hem, c_vtx, c_label)

    scatter = make_scatter(selected_idx, closest_other_idx, zoom_state.get('xaxis'), zoom_state.get('yaxis'))
    return selected_idx, clicked_spider, closest_spider, clicked_surface, closest_surface, scatter


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Interactive Dash viewer for cross-species gradient data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Command-Line Arguments ---
    parser.add_argument('--species_list_for_run', type=str, required=True, help='Comma-separated list of species in the analysis run (e.g., "human,chimpanzee").')
    parser.add_argument('--target_k_species_for_run', type=str, required=True, help='The reference species used in the analysis run.')
    parser.add_argument('--project_root', type=str, default='.', help='Path to the project root directory containing data/ and results/.')
    parser.add_argument('--n_tracts', type=int, default=DEFAULT_N_TRACTS_EXPECTED, help="Expected number of tracts/features.")
    parser.add_argument('--tract_names', type=str, default=",".join(DEFAULT_TRACT_NAMES), help="Comma-separated list of tract names for spider plots.")
    parser.add_argument('--surface_dir', type=str, default=None, help='Directory with species subfolders containing .surf.gii files. Defaults to <project_root>/data/surfaces.')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Host address for the Dash app.")
    parser.add_argument('--port', type=int, default=8050, help="Port for the Dash app.")
    parser.add_argument('--debug', action='store_true', help="Enable Dash debug mode.")
    args = parser.parse_args()

    # --- Path and Parameter Setup ---
    # Construct paths based on project structure and run-specific arguments
    species_list = [s.strip().lower() for s in args.species_list_for_run.split(',')]
    run_identifier = f"{'_'.join(species_list)}_CrossSpecies_kRef_{args.target_k_species_for_run.strip().lower()}"
    npz_filename = f"cross_species_embedding_data_{run_identifier}.npz"
    npz_file_path = os.path.join(args.project_root, 'results', '6_cross_species_gradients', 'intermediates', run_identifier, npz_filename)

    AVERAGE_BP_DIR_GLOBAL = os.path.join(args.project_root, 'results', '2_masked_average_blueprints')
    SURFACE_DIR_GLOBAL = args.surface_dir or os.path.join(args.project_root, 'data', 'surfaces')
    N_TRACTS_EXPECTED_GLOBAL = args.n_tracts
    TRACT_NAMES_GLOBAL = [name.strip() for name in args.tract_names.split(',')]

    # --- App Initialization ---
    if not load_data_from_npz(npz_file_path):
        print("WARNING: Failed to load data. The application will run but may show no data points.")

    setup_dynamic_plot_configs(df_global, DEFAULT_PLOT_CONFIGS_SCATTER, DEFAULT_SPECIES_SYMBOLS)
    app.layout = create_app_layout()
    app.run(debug=args.debug, host=args.host, port=args.port)
