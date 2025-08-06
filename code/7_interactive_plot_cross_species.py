#!/usr/bin/env python3
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

# ---- Default Configuration Constants (from your script) ----
# These will be used if not overridden by command-line arguments.
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

# --- Global Variables ---
# These are intended to hold the state of the application after loading data and parsing args.
df_global = pd.DataFrame()
# For plot aesthetics, these will be initialized based on defaults and then potentially
# expanded by setup_dynamic_plot_configs if new species/hemispheres are found in the data.
PLOT_CONFIGS_SCATTER_GLOBAL = {}
SPECIES_SYMBOLS_GLOBAL = {}
# For spider plots and data validation
TRACT_NAMES_GLOBAL = []
N_TRACTS_EXPECTED_GLOBAL = 0
# For fetching original blueprint profiles
AVERAGE_BP_DIR_GLOBAL = ""
EMPTY_SPIDER = None


# ---- DATA LOADING (Adapted from your script, for Script 6 NPZ output) ----
def load_data_from_npz(npz_file_path):
    """
    Loads gradient data from the .npz file (output of Script 6) and populates df_global.
    Includes detailed print statements for debugging.

    Args:
        npz_file_path (str): Path to the .npz file.

    Returns:
        bool: True if data loading and basic processing was successful (df_global might still be empty),
              False if critical errors like file not found or essential keys missing occur.
    """
    global df_global 
    print(f"DEBUG load_data_from_npz: Attempting to load NPZ: {npz_file_path}")

    try:
        npz_data = np.load(npz_file_path, allow_pickle=True)
        print(f"DEBUG load_data_from_npz: Successfully opened NPZ file.")
    except FileNotFoundError:
        print(f"ERROR load_data_from_npz: NPZ file not found at {npz_file_path}")
        df_global = pd.DataFrame() 
        return False
    except Exception as e:
        print(f"ERROR load_data_from_npz: Could not load NPZ file {npz_file_path}. Error: {e}")
        df_global = pd.DataFrame()
        return False

    required_gradient_key = 'cross_species_gradients'
    segment_info_key_primary = 'segment_info_detailed_for_remapping'

    if required_gradient_key not in npz_data:
        print(f"ERROR load_data_from_npz: NPZ file is missing the required key: '{required_gradient_key}'.")
        df_global = pd.DataFrame()
        return False

    segments_data_array = None
    if segment_info_key_primary in npz_data:
        segments_data_array = npz_data[segment_info_key_primary]
        print(f"DEBUG load_data_from_npz: Found segment info using key: '{segment_info_key_primary}'")
    else:
        print(f"ERROR load_data_from_npz: NPZ file is missing expected segment information keys "
              f"('{segment_info_key_primary}').")
        df_global = pd.DataFrame()
        return False

    grads = npz_data[required_gradient_key]
    segments = list(segments_data_array) 
    
    print(f"DEBUG load_data_from_npz: NPZ Grads shape: {grads.shape}, Number of segments: {len(segments)}")
    if grads.ndim < 2 or grads.shape[1] < 2:
        print(f"ERROR load_data_from_npz: '{required_gradient_key}' in NPZ data must have at least 2 columns (for G1, G2). Found shape: {grads.shape}")
        df_global = pd.DataFrame()
        return False

    records = []
    for seg_idx, seg_info in enumerate(segments):
        print(f"\nDEBUG load_data_from_npz: Processing segment {seg_idx}...")
        s = seg_info.get('species')
        h = seg_info.get('hem')
        start_row = seg_info.get('start_row_in_concat')
        end_row = seg_info.get('end_row_in_concat')
        profiles_shape = seg_info.get('profiles_for_lle_shape')
        original_tl_indices = seg_info.get('original_tl_indices')
        cluster_labels = seg_info.get('cluster_labels_for_remapping')

        # --- VERBOSE NULL AND TYPE CHECKS ---
        if s is None: print(f"WARNING load_data_from_npz: Segment {seg_idx}: species is None. Skipping."); continue
        if h is None: print(f"WARNING load_data_from_npz: Segment {seg_idx} ({s}): hem is None. Skipping."); continue
        if start_row is None: print(f"WARNING load_data_from_npz: Segment {seg_idx} ({s} {h}): start_row is None. Skipping."); continue
        if end_row is None: print(f"WARNING load_data_from_npz: Segment {seg_idx} ({s} {h}): end_row is None. Skipping."); continue
        if profiles_shape is None or not isinstance(profiles_shape, (tuple, list)) or len(profiles_shape) != 2 :
            print(f"WARNING load_data_from_npz: Segment {seg_idx} ({s} {h}): profiles_shape invalid or None. Shape: {profiles_shape}. Skipping."); continue
        if original_tl_indices is None: print(f"WARNING load_data_from_npz: Segment {seg_idx} ({s} {h}): original_tl_indices is None. Skipping."); continue
        if cluster_labels is None: print(f"WARNING load_data_from_npz: Segment {seg_idx} ({s} {h}): cluster_labels is None. Skipping."); continue
        
        print(f"DEBUG load_data_from_npz: Segment {seg_idx} ({s} {h}): All basic fields present. profiles_shape: {profiles_shape}")

        num_profiles_in_segment = profiles_shape[0] # Number of representative profiles for this segment
        print(f"DEBUG load_data_from_npz: Segment {seg_idx} ({s} {h}): num_profiles_in_segment = {num_profiles_in_segment}")
        
        if num_profiles_in_segment != (end_row - start_row):
            print(f"WARNING load_data_from_npz: Profile count mismatch in segment info for {s} {h} (segment {seg_idx}). "
                  f"Shape indicates {num_profiles_in_segment}, range indicates {end_row - start_row}. Skipping segment.")
            continue
        
        current_cluster_labels = np.array(cluster_labels, copy=False) 
        current_original_tl_indices = np.array(original_tl_indices, copy=False)
        print(f"DEBUG load_data_from_npz: Segment {seg_idx} ({s} {h}): original_tl_indices shape: {current_original_tl_indices.shape}, cluster_labels shape: {current_cluster_labels.shape}")


        if len(current_cluster_labels) != len(current_original_tl_indices):
            print(f"WARNING load_data_from_npz: Cluster labels length ({len(current_cluster_labels)}) does not match original TL indices length "
                  f"({len(current_original_tl_indices)}) for {s} {h} (segment {seg_idx}). Skipping segment.")
            continue
        
        if current_original_tl_indices.size > 0: # If there are supposed to be vertices for this segment
            if current_cluster_labels.size == 0: # But no labels to map them
                print(f"WARNING load_data_from_npz: Original TL indices exist ({current_original_tl_indices.size}) but cluster labels array is empty for {s} {h} (segment {seg_idx}). Skipping.")
                continue
            # Check if cluster labels correctly index into the profiles of this segment
            if num_profiles_in_segment > 0: # Only makes sense if there are profiles to index into
                 max_label_val = np.max(current_cluster_labels)
                 if max_label_val >= num_profiles_in_segment:
                    print(f"WARNING load_data_from_npz: Max cluster label ({max_label_val}) is out of bounds for number of profiles "
                          f"({num_profiles_in_segment}) in segment {s} {h} (segment {seg_idx}). Skipping segment.")
                    continue
            elif current_cluster_labels.size > 0 and np.any(current_cluster_labels !=0 ): # Segment has 0 profiles, but labels are not all 0 (or empty)
                 print(f"WARNING load_data_from_npz: Segment {s} {h} has 0 profiles_for_lle, but cluster_labels are present and not all zero. Max label: {np.max(current_cluster_labels)}. Skipping.")
                 continue

        elif current_cluster_labels.size > 0: # original_tl_indices is empty but cluster_labels is not
             print(f"WARNING load_data_from_npz: Original TL indices array is empty, but cluster labels exist for {s} {h} (segment {seg_idx}). Skipping.")
             continue
        
        print(f"DEBUG load_data_from_npz: Segment {seg_idx} ({s} {h}): About to loop through {len(current_original_tl_indices)} original TL indices.")
        appended_count_this_segment = 0
        for i, orig_vtx_id_on_surface in enumerate(current_original_tl_indices):
            if current_cluster_labels.size == 0 and current_original_tl_indices.size > 0 : # Should have been caught
                print(f"CRITICAL DEBUG: Reached loop with empty cluster_labels but non-empty original_tl_indices for segment {seg_idx} ({s} {h})")
                break 
            
            # profile_index_in_segment should be a valid index into this segment's block of profiles
            # (which has size num_profiles_in_segment)
            profile_index_in_segment = current_cluster_labels[i] 
            
            if not (0 <= profile_index_in_segment < num_profiles_in_segment) and num_profiles_in_segment > 0 :
                print(f"WARNING load_data_from_npz: profile_index_in_segment ({profile_index_in_segment}) out of bounds "
                      f"[0, {num_profiles_in_segment-1}] for {s} {h} (orig_vtx: {orig_vtx_id_on_surface}). Skipping this vertex.")
                continue

            # The actual row in the *global* 'grads' array for this representative profile
            global_grad_vector_idx = start_row + profile_index_in_segment

            if not (0 <= global_grad_vector_idx < grads.shape[0]):
                print(f"WARNING load_data_from_npz: Calculated global_grad_vector_idx {global_grad_vector_idx} is out of bounds for grads array "
                      f"(shape {grads.shape[0]}) for {s} {h}, orig_vtx {orig_vtx_id_on_surface} (segment {seg_idx}). Skipping this vertex.")
                continue
                
            record = {
                'species': s, 'hem': h, 'species_hem': f"{s}_{h}",
                'g1': grads[global_grad_vector_idx, 0],
                'g2': grads[global_grad_vector_idx, 1] if grads.shape[1] > 1 else 0,
                'g3': grads[global_grad_vector_idx, 2] if grads.shape[1] > 2 else 0, # For potential 3D plots
                'orig_vtx_id': int(orig_vtx_id_on_surface), 
                'profile_idx_in_segment': int(profile_index_in_segment), 
                'global_grad_vector_idx': int(global_grad_vector_idx) 
            }
            records.append(record)
            appended_count_this_segment +=1
        
        print(f"DEBUG load_data_from_npz: Segment {seg_idx} ({s} {h}): Appended {appended_count_this_segment} records.")
            
    if not records:
        print("FINAL DEBUG load_data_from_npz: records list is empty before creating DataFrame.")
    
    df_global = pd.DataFrame.from_records(records) 
    if not df_global.empty:
        df_global["df_idx"] = df_global.index 
    else: 
        df_global["df_idx"] = pd.Series(dtype='int')
        print("WARNING load_data_from_npz: DataFrame is empty after processing all NPZ segments. No data points will be plotted.")

    print(f"DEBUG load_data_from_npz: DataFrame populated with {len(df_global)} rows.")
    return True

def setup_dynamic_plot_configs(df_loaded, default_configs, default_symbols):
    """Dynamically sets up global plot configurations based on loaded data and provided defaults."""
    global PLOT_CONFIGS_SCATTER_GLOBAL, SPECIES_SYMBOLS_GLOBAL
    
    PLOT_CONFIGS_SCATTER_GLOBAL = default_configs.copy()
    SPECIES_SYMBOLS_GLOBAL = default_symbols.copy()

    if df_loaded.empty:
        print("INFO: DataFrame for setup_dynamic_plot_configs is empty. Using only initial defaults.")
        return

    # Add/update PLOT_CONFIGS_SCATTER_GLOBAL for species_hem found in data
    unique_species_hem_data = df_loaded["species_hem"].unique()
    for sh_key in unique_species_hem_data:
        if sh_key not in PLOT_CONFIGS_SCATTER_GLOBAL:
            species_part = sh_key.split('_')[0] if '_' in sh_key else sh_key
            hem_part = sh_key.split('_')[1] if '_' in sh_key and len(sh_key.split('_')) > 1 else ""
            generic_label = f"{species_part.capitalize()} {hem_part.upper() if hem_part else ''}".strip()
            color_idx = len(PLOT_CONFIGS_SCATTER_GLOBAL) % len(go.colors.qualitative.Plotly)
            PLOT_CONFIGS_SCATTER_GLOBAL[sh_key] = {"color": go.colors.qualitative.Plotly[color_idx], "label": generic_label}

    # Add/update SPECIES_SYMBOLS_GLOBAL for species found in data
    unique_species_data = df_loaded["species"].unique()
    for spec in unique_species_data:
        if spec not in SPECIES_SYMBOLS_GLOBAL:
            symbol_options = ["circle", "square", "diamond", "cross", "x", "triangle-up", "star"]
            new_symbol_count = sum(1 for s_key in SPECIES_SYMBOLS_GLOBAL if s_key not in default_symbols)
            symbol_idx = new_symbol_count % len(symbol_options)
            SPECIES_SYMBOLS_GLOBAL[spec] = symbol_options[symbol_idx]

    print("Final Plot Configs used:", PLOT_CONFIGS_SCATTER_GLOBAL)
    print("Final Species Symbols used:", SPECIES_SYMBOLS_GLOBAL)


# ---- FIGURE GENERATION UTILITIES (Preserving your visual style) ----
def get_vertex_profile(species, hemisphere, vertex_id):
    global AVERAGE_BP_DIR_GLOBAL, N_TRACTS_EXPECTED_GLOBAL # Use global arguments

    if vertex_id == -1 or pd.isna(vertex_id):
        return None
        
    bp_dir = os.path.join(AVERAGE_BP_DIR_GLOBAL, species)
    # This assumes Script 2 output (masked average, not Script 5 downsampled)
    bp_file = f"average_{species}_blueprint.{hemisphere}_temporal_lobe_masked.func.gii"
    bp_path = os.path.join(bp_dir, bp_file)

    if not os.path.exists(bp_path):
        print(f"DEBUG get_vertex_profile: File not found {bp_path}")
        return None
    try:
        img = nib.load(bp_path)
        if not hasattr(img, 'darrays') or not img.darrays: # Check if darrays exist
            print(f"DEBUG get_vertex_profile: No darrays in {bp_path}")
            return None
        
        # Validate against N_TRACTS_EXPECTED_GLOBAL if it's crucial for TRACT_NAMES_GLOBAL alignment
        if len(img.darrays) != N_TRACTS_EXPECTED_GLOBAL:
             print(f"WARNING get_vertex_profile: Tract count in {bp_path} ({len(img.darrays)}) "
                   f"does not match N_TRACTS_EXPECTED_GLOBAL ({N_TRACTS_EXPECTED_GLOBAL}). "
                   "Spider plot may be affected if TRACT_NAMES_GLOBAL is misaligned.")
             # Proceeding with found tracts; make_spider will try to adapt.

        data_list = [d.data for d in img.darrays]
        # Removed checks for len(data_list) and all(len(d) == first_len) as they were redundant
        # with len(img.darrays) check if darrays are guaranteed to be same length by nibabel,
        # but it's safer to keep a check if darrays *could* have different lengths.
        # For .func.gii, each darray is typically a map over the same vertices.
        
        data = np.array(data_list).T # (vertices, tracts)
        vertex_id = int(vertex_id) # Ensure it's an int for indexing
        if not (0 <= vertex_id < data.shape[0]):
            print(f"DEBUG get_vertex_profile: vertex_id {vertex_id} out of bounds for {bp_path} (max: {data.shape[0]-1})")
            return None
        return data[vertex_id, :]
    except Exception as e:
        print(f"DEBUG get_vertex_profile: Exception loading {bp_path} for vtx {vertex_id}: {e}")
        return None

def make_spider(profile, label, color): # tract_labels will use global TRACT_NAMES_GLOBAL
    global TRACT_NAMES_GLOBAL
    
    tract_labels_to_use = TRACT_NAMES_GLOBAL[:] # Use a copy of the global

    # Adapt tract_labels_to_use if profile length doesn't match
    if profile is not None and len(profile) != len(tract_labels_to_use):
        print(f"  Spider plot for '{label}': Profile length {len(profile)} != Global TRACT_NAMES length {len(tract_labels_to_use)}. Adjusting.")
        if len(profile) < len(tract_labels_to_use):
            tract_labels_to_use = tract_labels_to_use[:len(profile)]
        else: # profile is longer
            profile = profile[:len(tract_labels_to_use)] # Truncate profile to match names

    if profile is None or profile.size == 0 or np.allclose(profile, 0):
        fig = go.Figure(layout={
            "title_text": f"No data for {label}", "height": 400, # Your height
            "margin": dict(l=30, r=30, t=50, b=30) # Your margins
        })
        fig.add_trace(go.Scatterpolar(r=[0]*len(tract_labels_to_use), theta=tract_labels_to_use, mode='markers', marker_opacity=0))
        fig.update_layout(polar=dict(angularaxis=dict(tickfont=dict(size=8)))) # Your tickfont size
        return fig

    values = np.concatenate((profile, [profile[0]]))
    categories = tract_labels_to_use + [tract_labels_to_use[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=label, marker_color=color, line_color=color))
    max_val = np.max(values) if values.size > 0 else 0.05
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max_val*1.1, 0.05)]), angularaxis=dict(tickfont=dict(size=8))), # Your settings
        showlegend=False, title_text=label, title_font_size=12, # Your title font size
        margin=dict(l=30, r=30, t=50, b=30), height=400 # Your margins & height
    )
    return fig

def blank_spider_fig():
    global TRACT_NAMES_GLOBAL
    current_tract_names = TRACT_NAMES_GLOBAL if TRACT_NAMES_GLOBAL else DEFAULT_TRACT_NAMES # Use default if global not set
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0]*len(current_tract_names), theta=current_tract_names, mode='markers', marker={'opacity': 0}, hoverinfo='none'))
    fig.update_layout(showlegend=False, polar=dict(radialaxis=dict(visible=False, range=[0,1]), angularaxis=dict(visible=False)),
                      margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400) # Your height
    return fig

def make_scatter(selected_df_idx=None, closest_df_idx=None, xaxis_range=None, yaxis_range=None):
    """Creates the main scatter plot with marginal histograms, matching your original visual style."""
    global df_global, PLOT_CONFIGS_SCATTER_GLOBAL, SPECIES_SYMBOLS_GLOBAL # Use global configs

    if df_global.empty:
        return go.Figure(layout={"title_text": "No data loaded. Check NPZ file path.", "height": 950})

    # This color_map is derived from PLOT_CONFIGS_SCATTER_GLOBAL for use in histograms
    color_map_for_hist = {cfg["label"]: cfg["color"] for cfg in PLOT_CONFIGS_SCATTER_GLOBAL.values() if "label" in cfg and "color" in cfg}


    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
        column_widths=[0.92, 0.08], row_heights=[0.08, 0.92], # Your original proportions
        horizontal_spacing=0.01, vertical_spacing=0.01,
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "scattergl"},    {"type": "histogram"}]] # Using scattergl as in your original
    )

    # Histograms
    # Your original script used df.groupby("label"). The 'label' column was created from PLOT_CONFIGS_SCATTER.
    # We'll achieve a similar grouping by iterating through PLOT_CONFIGS_SCATTER_GLOBAL.
    for sh_key, config in PLOT_CONFIGS_SCATTER_GLOBAL.items():
        group = df_global[df_global["species_hem"] == sh_key]
        if group.empty: continue
        marker_color = config.get("color", "grey")
        fig.add_trace(go.Histogram(x=group["g1"], marker_color=marker_color, opacity=0.8, showlegend=False, nbinsx=40), row=1, col=1)
        fig.add_trace(go.Histogram(y=group["g2"], marker_color=marker_color, opacity=0.8, showlegend=False, nbinsy=40), row=2, col=2)

    # Scatter
    legend_items_added = set()
    for sh_key, group in df_global.groupby("species_hem"):
        species = group["species"].iloc[0]
        config = PLOT_CONFIGS_SCATTER_GLOBAL.get(sh_key, {"color": "grey", "label": sh_key})
        marker_color = config["color"]
        marker_symbol = SPECIES_SYMBOLS_GLOBAL.get(species, "circle")
        
        show_leg = config['label'] not in legend_items_added
        if show_leg: legend_items_added.add(config['label'])

        fig.add_trace(go.Scattergl( # Your original uses Scattergl
            x=group["g1"], y=group["g2"], mode="markers",
            marker=dict(size=8, opacity=0.88, line=dict(width=1, color='black'), color=marker_color, symbol=marker_symbol), # Your style
            name=config["label"], showlegend=show_leg,
            customdata=np.stack([group["df_idx"], group["species"], group["hem"], group["orig_vtx_id"]], axis=-1),
            hovertemplate="g1: %{x:.3f}<br>g2: %{y:.3f}<br>species: %{customdata[1]}<br>hem: %{customdata[2]}<br>orig_vtx_id: %{customdata[3]}<extra></extra>" # Your hover
        ), row=2, col=1)

    # Axes formatting (your original)
    fig.update_xaxes(row=1, col=1, showticklabels=False)
    fig.update_yaxes(row=1, col=1, showticklabels=False) # Your yaxis for top-left hist was on col=1
    fig.update_xaxes(row=2, col=2, showticklabels=False) # X-axis for the Y-histogram
    fig.update_yaxes(row=2, col=2, showticklabels=False) # Y-axis for the Y-histogram
    fig.update_xaxes(title_text="Gradient 1", row=2, col=1)
    fig.update_yaxes(title_text="Gradient 2", row=2, col=1)

    if xaxis_range:
        fig.update_xaxes(range=xaxis_range, autorange=False, row=1, col=1)
        fig.update_xaxes(range=xaxis_range, autorange=False, row=2, col=1)
    if yaxis_range:
        fig.update_yaxes(range=yaxis_range, autorange=False, row=2, col=1)
        fig.update_yaxes(range=yaxis_range, autorange=False, row=2, col=2)

    if selected_df_idx is not None and selected_df_idx in df_global.index:
        fig.add_trace(go.Scatter( # Your original used Scatter for highlights
            x=[df_global.loc[selected_df_idx, "g1"]], y=[df_global.loc[selected_df_idx, "g2"]],
            mode='markers', marker=dict(size=19, color='black', line=dict(width=4, color='yellow'), symbol="x"), # Your style
            showlegend=False, hoverinfo='skip'), row=2, col=1)

    if closest_df_idx is not None and closest_df_idx in df_global.index:
        closest_row = df_global.loc[closest_df_idx]
        key = f"{closest_row['species']}_{closest_row['hem']}" # Use species_hem from df
        overlay_color = PLOT_CONFIGS_SCATTER_GLOBAL.get(key, {}).get("color", "grey") # Use global
        fig.add_trace(go.Scatter( # Your original used Scatter
            x=[closest_row["g1"]], y=[closest_row["g2"]],
            mode='markers', marker=dict(size=30, color=overlay_color, line=dict(width=6, color='yellow'), symbol="star", opacity=1.0), # Your style
            showlegend=False, hoverinfo='skip'), row=2, col=1)

    fig.update_layout( # Your layout settings
        height=950, width=900,
        margin=dict(t=50, r=40, b=70, l=80),
        title_text="Cross-Species Temporal Lobe Embedding (G1 vs G2)",
        legend=dict(orientation="h", yanchor="auto", y=0.90, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(0,0,0,0.13)", borderwidth=1,
                    font_size=13, itemclick="toggleothers"),
        bargap=0.1, barmode="group", hovermode="closest",
        uirevision=True # Your uirevision setting
    )
    return fig

# ---- FLASK + DASH ----
server_flask = Flask(__name__)
app = dash.Dash(__name__, server=server_flask, url_base_pathname='/', prevent_initial_callbacks=True)

# ---- APP LAYOUT (Your exact layout structure) ----
def create_app_layout():
    global EMPTY_SPIDER # Ensure it's initialized and accessible
    if EMPTY_SPIDER is None: EMPTY_SPIDER = blank_spider_fig() # Fallback initialization

    return html.Div([
        dcc.Store(id="zoom-state", data=None),
        dcc.Store(id="selected-idx", data=None),
        html.Div(id='dummy-input-for-initial-load', style={'display': 'none'}), # For initial callback

        html.H2("Cross-Species Gradients: Interactive Exploration"),

        html.Div([
            html.Div([
                dcc.Graph(id="scatter-g", figure={}, style={'height': '900px'}) # Initial empty figure
            ], style={'grid-column': '1', 'grid-row': '1 / span 3'}),

            html.Div([
                html.H4("Clicked Point: Tract Spider Plot", style={'margin': '0 0 4px 0'}),
                dcc.Graph(id="clicked-spider", figure=EMPTY_SPIDER, style={'height': '300px'})
            ], style={'grid-column': '2', 'grid-row': '1'}),

            html.Div([
                html.Div([
                    html.Label("Distance Mode:", style={'fontWeight': 'bold', 'marginRight': '8px'}),
                    dcc.Dropdown(id='distance-mode',
                                 options=[{'label': 'Euclidean',   'value': 'euclidean'},
                                          {'label': 'G1 Distance',  'value': 'g1'},
                                          {'label': 'G2 Distance',  'value': 'g2'}],
                                 value='euclidean', clearable=False, style={'width': '140px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'}),
                html.Div([
                    html.Label("Match Mode:", style={'fontWeight': 'bold', 'marginRight': '8px'}),
                    dcc.Dropdown(id='match-mode',
                                 options=[{'label': 'Cross-Species', 'value': 'different'},
                                          {'label': 'Same Species',     'value': 'same'}],
                                 value='different', clearable=False, style={'width': '140px'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'}),
            ], style={'grid-column': '2', 'grid-row': '2', 'display': 'grid', 
                      'grid-template-columns': '1fr 1fr', 'gap': '20px', 'padding': '12px 0'}),

            html.Div([
                html.H4("Closest Neighbor: Tract Spider Plot", style={'margin': '0 0 4px 0'}),
                dcc.Graph(id="closest-spider", figure=EMPTY_SPIDER, style={'height': '300px'})
            ], style={'grid-column': '2', 'grid-row': '3'}),
        ], style={
            'display': 'grid', 'grid-template-columns': 'minmax(700px,900px) 410px',
            'grid-template-rows':    '1fr auto 1fr', 'gap': '12px', 'width': '100%'
        })
    ])

# ---- CALLBACKS (Preserving your logic, adapting for global vars) ----
@app.callback(
    Output("zoom-state", "data"),
    Input("scatter-g", "relayoutData"),
    State("zoom-state", "data"),
    # prevent_initial_call=True is implicit from app config
)
def save_zoom(relayoutData, old_zoom): # Your exact save_zoom logic
    if relayoutData is None: return dash.no_update # Changed to dash.no_update from old_zoom
    
    new_zoom_data = old_zoom.copy() if old_zoom else {"xaxis": None, "yaxis": None}
    
    # Try to infer main x and y axes from relayoutData keys
    x_key_options = ["xaxis.range[0]", "xaxis2.range[0]"] # Add more if subplots change
    y_key_options = ["yaxis.range[0]", "yaxis2.range[0]"] # Add more if subplots change
    
    x_pref, y_pref = "xaxis", "yaxis" # Default for main scatter usually
    
    # Find which axis prefix is present in relayoutData for x
    for x_opt in x_key_options:
        if x_opt in relayoutData:
            x_pref = x_opt.split('.')[0]
            break
    # Find which axis prefix is present in relayoutData for y
    for y_opt in y_key_options:
        if y_opt in relayoutData:
            y_pref = y_opt.split('.')[0]
            break
            
    x_range_defined = f"{x_pref}.range[0]" in relayoutData and f"{x_pref}.range[1]" in relayoutData
    y_range_defined = f"{y_pref}.range[0]" in relayoutData and f"{y_pref}.range[1]" in relayoutData

    if x_range_defined:
        new_zoom_data["xaxis"] = [relayoutData[f"{x_pref}.range[0]"], relayoutData[f"{x_pref}.range[1]"]]
    elif f"{x_pref}.autorange" in relayoutData and relayoutData[f"{x_pref}.autorange"]:
        new_zoom_data["xaxis"] = None

    if y_range_defined:
        new_zoom_data["yaxis"] = [relayoutData[f"{y_pref}.range[0]"], relayoutData[f"{y_pref}.range[1]"]]
    elif f"{y_pref}.autorange" in relayoutData and relayoutData[f"{y_pref}.autorange"]:
        new_zoom_data["yaxis"] = None
    
    if new_zoom_data != old_zoom:
        return new_zoom_data
    return dash.no_update


@app.callback(
    [Output("selected-idx", "data"),
     Output("clicked-spider", "figure"),
     Output("closest-spider", "figure"),
     Output("scatter-g", "figure", allow_duplicate=True)], # allow_duplicate for updates
    [Input("scatter-g", "clickData"),
     Input("distance-mode", "value"),
     Input("match-mode", "value")],
    [State("zoom-state", "data"), # Listen to zoom state for redrawing
     State("selected-idx", "data")]
    # prevent_initial_call=True is implicit from app config
)
def handle_graph_interactions_and_updates(
    clickData, distance_mode, match_mode, zoom_state, current_selected_idx
):
    global df_global, PLOT_CONFIGS_SCATTER_GLOBAL, AVERAGE_BP_DIR_GLOBAL, N_TRACTS_EXPECTED_GLOBAL, TRACT_NAMES_GLOBAL, SPECIES_SYMBOLS_GLOBAL, EMPTY_SPIDER
    
    ctx = callback_context
    triggered_input_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered and ctx.triggered[0].get('value') is not None else None

    # Your logic for determining processed_selected_idx from clickData
    # This part needs careful restoration of your original curveNumber logic if it was essential
    # For now, a simplified version based on customdata:
    new_processed_selected_idx = current_selected_idx
    if triggered_input_id == "scatter-g" and clickData and clickData.get("points"):
        point_info = clickData["points"][0]
        if "customdata" in point_info and isinstance(point_info["customdata"], (list,tuple)) and len(point_info["customdata"]) > 0:
            try:
                clicked_df_idx = int(point_info["customdata"][0]) # df_idx is customdata[0]
                new_processed_selected_idx = None if clicked_df_idx == current_selected_idx else clicked_df_idx
            except (ValueError, TypeError):
                pass # Invalid customdata, selection doesn't change
    
    # Your original structure for updating plots:
    xaxis_range = zoom_state.get("xaxis") if zoom_state else None
    yaxis_range = zoom_state.get("yaxis") if zoom_state else None

    if new_processed_selected_idx is None or new_processed_selected_idx not in df_global.index:
        scatter_fig = make_scatter(None, None, xaxis_range, yaxis_range)
        return None, EMPTY_SPIDER, EMPTY_SPIDER, scatter_fig

    selected_row = df_global.loc[new_processed_selected_idx]
    s_species, s_hem, s_vtx = selected_row["species"], selected_row["hem"], selected_row["orig_vtx_id"]
    s_profile = get_vertex_profile(s_species, s_hem, s_vtx)
    s_label = f"{s_species.capitalize()} {s_hem} (vtx {s_vtx})"
    s_color_key = f"{s_species}_{s_hem}"
    s_color = PLOT_CONFIGS_SCATTER_GLOBAL.get(s_color_key, {}).get("color", "grey")
    clicked_spider_fig = make_spider(s_profile, s_label, s_color)

    if match_mode == 'same':
        candidate_df = df_global[(df_global.species == s_species) & (df_global.df_idx != new_processed_selected_idx)]
    else: # 'different'
        candidate_df = df_global[df_global.species != s_species]

    closest_other_idx = None
    closest_spider_fig = EMPTY_SPIDER
    if not candidate_df.empty:
        other_coords = candidate_df[['g1', 'g2']].values
        selected_coords = np.array([[selected_row.g1, selected_row.g2]])
        if distance_mode == 'euclidean': distances = euclidean_distances(selected_coords, other_coords)[0]
        elif distance_mode == 'g1': distances = np.abs(other_coords[:, 0] - selected_row.g1)
        else: distances = np.abs(other_coords[:, 1] - selected_row.g2) # g2
        if distances.size > 0:
            min_dist_local_idx = np.argmin(distances)
            closest_row = candidate_df.iloc[min_dist_local_idx]
            closest_other_idx = int(closest_row.df_idx)
            c_species, c_hem, c_vtx = closest_row["species"], closest_row["hem"], closest_row["orig_vtx_id"]
            c_profile = get_vertex_profile(c_species, c_hem, c_vtx)
            c_label = f"{'Same' if match_mode=='same' else 'Closest'}: {c_species.capitalize()} {c_hem} (vtx {c_vtx})"
            c_color_key = f"{c_species}_{c_hem}"
            c_color = PLOT_CONFIGS_SCATTER_GLOBAL.get(c_color_key, {}).get("color", "grey")
            closest_spider_fig = make_spider(c_profile, c_label, c_color)
            
    scatter_fig = make_scatter(new_processed_selected_idx, closest_other_idx, xaxis_range, yaxis_range)
    # The Output("scatter-g", "clickData") was removed as it's an anti-pattern. Dash clears clickData.
    return new_processed_selected_idx, clicked_spider_fig, closest_spider_fig, scatter_fig


@app.callback(
    Output('scatter-g', 'figure', allow_duplicate=True), # Your ID
    Input('dummy-input-for-initial-load', 'data'), 
    prevent_initial_call=False # This callback MUST run on initial load
)
def initial_scatter_load(_): 
    global df_global, EMPTY_SPIDER
    if EMPTY_SPIDER is None: EMPTY_SPIDER = blank_spider_fig()
    if df_global.empty:
        fig = go.Figure(layout={"title_text":"Data not loaded. Check NPZ path or console.", "height":900})
        fig.add_annotation(text="Ensure --npz_file points to valid Script 6 output.", showarrow=False)
        return fig
    return make_scatter() # Call with defaults from your original script (no highlights, no zoom)


# ---- MAIN EXECUTION ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive Dash viewer for cross-species gradient data.")
    
    # Required arguments to identify the specific Script 6 run
    parser.add_argument('--species_list_for_run', type=str, required=True,
                        help='Comma-separated list of the species included in the run (e.g., "human,chimpanzee").')
    parser.add_argument('--target_k_species_for_run', type=str, required=True,
                        help='The target_k_species used as the reference in the run.')
    
    # Optional arguments with sensible defaults
    parser.add_argument('--project_root', type=str, default='.',
                        help='Path to the project root directory containing data/ and results/.')
    parser.add_argument('--n_tracts', type=int, default=DEFAULT_N_TRACTS_EXPECTED,
                        help=f"Expected number of tracts/features. Default: {DEFAULT_N_TRACTS_EXPECTED}.")
    parser.add_argument('--tract_names', type=str, default=",".join(DEFAULT_TRACT_NAMES),
                        help="Comma-separated list of tract names for spider plots.")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Host address for the Dash app.")
    parser.add_argument('--port', type=int, default=8050, help="Port for the Dash app.")
    parser.add_argument('--debug', action='store_true', help="Enable Dash debug mode for development.")

    args = parser.parse_args()

    # Automatically determine paths based on project structure and run info ---
    try:
        # Reconstruct the run identifier to find the correct .npz file
        species_list = [s.strip().lower() for s in args.species_list_for_run.split(',')]
        species_str = "_".join(species_list) # Use the same non-sorted order as Script 6
        target_species_str = args.target_k_species_for_run.strip().lower()
        run_identifier = f"{species_str}_CrossSpecies_kRef_{target_species_str}"

        # Construct full paths to the required input files and directories
        npz_filename = f"cross_species_embedding_data_{run_identifier}.npz"
        npz_file_path = os.path.join(
            args.project_root, 'results', '6_cross_species_gradients', 
            'intermediates', run_identifier, npz_filename
        )
        average_bp_dir_path = os.path.join(args.project_root, 'results', '2_masked_average_blueprints')

    except Exception as e:
        print(f"Error constructing paths from species info: {e}")
        exit(1)

    # --- Populate Global Variables from constructed paths and args ---
    AVERAGE_BP_DIR_GLOBAL = average_bp_dir_path
    N_TRACTS_EXPECTED_GLOBAL = args.n_tracts
    TRACT_NAMES_GLOBAL = [name.strip() for name in args.tract_names.split(',')]

    if len(TRACT_NAMES_GLOBAL) != N_TRACTS_EXPECTED_GLOBAL:
        print(f"WARNING: Provided --tract_names count ({len(TRACT_NAMES_GLOBAL)}) "
              f"does not match --n_tracts ({N_TRACTS_EXPECTED_GLOBAL}). Adjusting tract names list.")
        # Adjust tract names list if there's a mismatch
        if len(TRACT_NAMES_GLOBAL) < N_TRACTS_EXPECTED_GLOBAL:
            TRACT_NAMES_GLOBAL.extend([f"Tract {i+1}" for i in range(len(TRACT_NAMES_GLOBAL), N_TRACTS_EXPECTED_GLOBAL)])
        else:
            TRACT_NAMES_GLOBAL = TRACT_NAMES_GLOBAL[:N_TRACTS_EXPECTED_GLOBAL]
    
    EMPTY_SPIDER = blank_spider_fig()
    
    # Load the main data from the automatically found .npz file
    data_load_was_successful = load_data_from_npz(npz_file_path)
    
    # Setup plot configurations based on the loaded data
    setup_dynamic_plot_configs(df_global, DEFAULT_PLOT_CONFIGS_SCATTER, DEFAULT_SPECIES_SYMBOLS)

    if not data_load_was_successful or df_global.empty:
        print("WARNING: Failed to load data or data is empty. The application will run but may show no data.")
    
    # Create and run the Dash app
    app.layout = create_app_layout()
    
    print(f"Attempting to start Dash server on http://{args.host}:{args.port}/")
    app.run(debug=args.debug, host=args.host, port=args.port)