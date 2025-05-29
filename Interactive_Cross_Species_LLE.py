import os
import numpy as np
import pandas as pd
import nibabel as nib
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

# --- Config ---
NPZ_FILE_PATH = "data/gradient_outputs/cross_species_embedding_data_2Species_LR_Combined.npz"
DOWNSAMPLED_BP_DIR = "data/downsampled_temporal_lobe_blueprints"
AVERAGE_BP_DIR = "data/temporal_lobe_average_blueprints"
N_TRACTS_EXPECTED = 20
TRACT_NAMES = [
    "AC", "AF", "AR", "CBD", "CBP", "CBT", "CST", "FA",
    "FMI", "FMA", "FX", "IFOF", "ILF", "MDLF",
    "OR", "SLF I", "SLF II", "SLF III", "UF", "VOF"
]
PLOT_CONFIGS_SCATTER = {
    "human_L": {"color": "blue", "label": "Human L"},
    "human_R": {"color": "cornflowerblue", "label": "Human R"},
    "chimpanzee_L": {"color": "red", "label": "Chimpanzee L"},
    "chimpanzee_R": {"color": "lightcoral", "label": "Chimpanzee R"}
}
SPECIES_SYMBOLS = {"human": "circle", "chimpanzee": "circle"}

def load_data():
    npz_data = np.load(NPZ_FILE_PATH, allow_pickle=True)
    grads = npz_data['cross_species_gradients']
    segments = list(npz_data['full_segment_info_for_remapping'])
    records = []
    for seg in segments:
        s = seg['species']
        h = seg['hem']
        start = seg['start_row_in_concat']
        end = seg['end_row_in_concat']
        original_tl_indices = seg['original_tl_indices']
        cluster_labels = seg['cluster_labels']
        k_val = end - start  # n centroids for left/right
        # Human: use downsampled blueprints for spider, Chimp: use average blueprints
        for i, grad_idx in enumerate(range(start, end)):
            vtx = -1
            if s == "human":
                if cluster_labels is not None:
                    idxs = np.where(cluster_labels == i)[0]
                    if len(idxs) > 0:
                        vtx = original_tl_indices[idxs[0]]
            else:
                vtx = original_tl_indices[i] if i < len(original_tl_indices) else -1
            records.append(dict(
                row_idx=grad_idx,
                species=s, hem=h,
                species_hem=f"{s}_{h}",
                g1=grads[grad_idx, 0],
                g2=grads[grad_idx, 1],
                orig_vtx_id=vtx,
                centroid_idx=i,
                k_val=k_val  # Needed for ds file names for human
            ))
    df = pd.DataFrame.from_records(records)
    df.reset_index(drop=True, inplace=True)
    df["df_idx"] = df.index
    return df

def get_vertex_profile(species, hemisphere, vertex_id, k_val=None):
    # Returns the tract profile for a given vertex_id (proper path for ds/original)
    if species == "human":
        # Downsampled blueprint path
        bp_file = f"average_human_blueprint_{hemisphere}_TL_dsViz_k{k_val}.func.gii"
        bp_path = os.path.join(DOWNSAMPLED_BP_DIR, bp_file)
    else:
        # Chimp uses original average blueprint
        bp_dir = os.path.join(AVERAGE_BP_DIR, species)
        bp_file = f"average_{species}_blueprint_{hemisphere}_temporal_lobe.func.gii"
        bp_path = os.path.join(bp_dir, bp_file)
    if not os.path.exists(bp_path):
        print(f"Blueprint not found: {bp_path}")
        return None
    img = nib.load(bp_path)
    if len(img.darrays) != N_TRACTS_EXPECTED:
        print("Tract count mismatch")
        return None
    data = np.array([d.data for d in img.darrays]).T
    if not (0 <= vertex_id < data.shape[0]):
        return None
    return data[vertex_id, :]

def make_spider_plot(profile, label, color, tract_labels=TRACT_NAMES):
    if profile is None or np.allclose(profile, 0):
        return go.Figure(layout={"title": f"No data for {label}"})
    values = np.concatenate((profile, [profile[0]]))
    categories = tract_labels + [tract_labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        name=label, marker_color=color, line=dict(color=color)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(0.05, np.max(values)*1.1)])),
        showlegend=False,
        title=label,
        margin=dict(l=30, r=30, t=50, b=30), height=400
    )
    return fig

df = load_data()
color_map = {cfg["label"]: cfg["color"] for cfg in PLOT_CONFIGS_SCATTER.values()}
df["label"] = df["species_hem"].map(lambda x: PLOT_CONFIGS_SCATTER[x]["label"])

def make_scatter(selected_idx=None, closest_idx=None, xaxis_range=None, yaxis_range=None):
    fig = px.scatter(
        df, x='g1', y='g2', color='label', symbol='species',
        color_discrete_map=color_map, symbol_map=SPECIES_SYMBOLS,
        custom_data=['df_idx', 'species', 'hem', 'orig_vtx_id', 'centroid_idx', 'k_val'],
        hover_data={'df_idx': True, 'species': True, 'hem': True, 'orig_vtx_id': True, 'centroid_idx': True, 'k_val': True},
        title="Cross-Species Temporal Lobe Embedding (G1 vs G2)"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.88, line=dict(width=0.8, color='Black')))

    layout_args = dict(
        xaxis_title="Gradient 1", yaxis_title="Gradient 2",
        height=900, width=900,
        legend_title_text=None,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.03,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0.13)", borderwidth=1,
            font=dict(size=13),
            itemclick="toggleothers"
        ),
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(t=65, r=20, l=80, b=90)
    )
    if xaxis_range is not None:
        layout_args["xaxis_range"] = xaxis_range
    if yaxis_range is not None:
        layout_args["yaxis_range"] = yaxis_range
    fig.update_layout(**layout_args)

    # Highlight: clicked point
    if selected_idx is not None:
        fig.add_trace(go.Scatter(
            x=[df.loc[selected_idx, "g1"]],
            y=[df.loc[selected_idx, "g2"]],
            mode='markers',
            marker=dict(size=19, color='black', line=dict(width=4, color='yellow'), symbol="x"),
            name="Clicked",
            showlegend=False
        ))
    # Highlight: closest other-species point
    if closest_idx is not None:
        color_key = f"{df.loc[closest_idx,'species']}_{df.loc[closest_idx,'hem']}"
        color = PLOT_CONFIGS_SCATTER[color_key]["color"]
        fig.add_trace(go.Scatter(
            x=[df.loc[closest_idx, "g1"]],
            y=[df.loc[closest_idx, "g2"]],
            mode='markers',
            marker=dict(
                size=30, color=color,
                line=dict(width=6, color='yellow'),
                symbol="star",
                opacity=1.0
            ),
            name="Closest",
            showlegend=False
        ))
    return fig

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Cross-Species Gradients: Interactive Exploration"),
    html.Div([
        html.Div([
            dcc.Graph(id="scatter-g", figure=make_scatter(), style={'height': '900px'})
        ], style={'grid-column': '1', 'grid-row': '1 / span 2'}),
        html.Div([
            html.H4("Clicked Point: Tract Spider Plot", style={'margin-bottom': '0'}),
            dcc.Graph(id="clicked-spider", figure=go.Figure(), style={'height': '420px'}),
        ], style={'grid-column': '2', 'grid-row': '1'}),
        html.Div([
            html.H4("Closest Other-Species Point: Tract Spider Plot", style={'margin-bottom': '0'}),
            dcc.Graph(id="closest-spider", figure=go.Figure(), style={'height': '420px'}),
        ], style={'grid-column': '2', 'grid-row': '2'}),
    ], style={
        'display': 'grid',
        'grid-template-columns': 'minmax(700px, 900px) 410px',
        'grid-template-rows': '430px 430px',
        'gap': '12px',
        'width': '100%',
        'align-items': 'start'
    })
], style={'max-width': '1450px', 'margin': '0 auto'})

@app.callback(
    [Output("clicked-spider", "figure"),
     Output("closest-spider", "figure"),
     Output("scatter-g", "figure")],
    [Input("scatter-g", "clickData"),
     Input("scatter-g", "relayoutData")]
)
def update_spiders(clickData, relayoutData):
    if not clickData or "points" not in clickData or not clickData["points"]:
        return go.Figure(), go.Figure(), make_scatter()
    point = clickData["points"][0]
    if "customdata" not in point:
        return go.Figure(), go.Figure(), make_scatter()
    custom = point["customdata"]
    df_idx = custom[0]
    species = custom[1]
    hem = custom[2]
    vtx_id = custom[3]
    centroid_idx = custom[4]
    k_val = custom[5]
    label = f"{species.capitalize()} {hem} (vtx {vtx_id})"
    color_key = f"{species}_{hem}"
    spider_color = PLOT_CONFIGS_SCATTER[color_key]["color"]

    profile = get_vertex_profile(species, hem, vtx_id, k_val=k_val)
    spider1 = make_spider_plot(profile, label, spider_color)

    clicked_g1, clicked_g2 = point["x"], point["y"]
    mask = (df['species'] != species)
    dists = euclidean_distances(
        np.array([[clicked_g1, clicked_g2]]),
        df.loc[mask, ['g1', 'g2']].values
    )[0]
    idx = np.argmin(dists)
    row2 = df.loc[mask].iloc[idx]
    df_idx2 = row2["df_idx"]
    vtx2 = row2["orig_vtx_id"]
    label2 = f"{row2['species'].capitalize()} {row2['hem']} (vtx {vtx2})"
    color_key2 = f"{row2['species']}_{row2['hem']}"
    spider_color2 = PLOT_CONFIGS_SCATTER[color_key2]["color"]
    k_val2 = row2["k_val"]
    profile2 = get_vertex_profile(row2["species"], row2["hem"], vtx2, k_val=k_val2)
    spider2 = make_spider_plot(profile2, label2, spider_color2)

    # Compute axes limits to fit both points only if needed (don't zoom in)
    xvals = [clicked_g1, row2['g1']]
    yvals = [clicked_g2, row2['g2']]
    min_x, max_x = min(xvals), max(xvals)
    min_y, max_y = min(yvals), max(yvals)
    pad_x = 0.08 * (max_x - min_x) if max_x != min_x else 0.001
    pad_y = 0.08 * (max_y - min_y) if max_y != min_y else 0.001

    # Default: preserve current zoom if it contains both points, else zoom out to include them both
    if relayoutData and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData and \
       'yaxis.range[0]' in relayoutData and 'yaxis.range[1]' in relayoutData:
        xmin_cur, xmax_cur = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        ymin_cur, ymax_cur = relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']
        need_expand_x = not (xmin_cur <= min_x <= xmax_cur and xmin_cur <= max_x <= xmax_cur)
        need_expand_y = not (ymin_cur <= min_y <= ymax_cur and ymin_cur <= max_y <= ymax_cur)
        if not (need_expand_x or need_expand_y):
            xaxis_range = [xmin_cur, xmax_cur]
            yaxis_range = [ymin_cur, ymax_cur]
        else:
            # Only expand if needed, never zoom in
            xaxis_range = [
                min(min_x - pad_x, xmin_cur) if min_x < xmin_cur else xmin_cur,
                max(max_x + pad_x, xmax_cur) if max_x > xmax_cur else xmax_cur
            ]
            yaxis_range = [
                min(min_y - pad_y, ymin_cur) if min_y < ymin_cur else ymin_cur,
                max(max_y + pad_y, ymax_cur) if max_y > ymax_cur else ymax_cur
            ]
    else:
        # Initial load or no zoom: just fit both points with a bit of padding
        xaxis_range = [min_x - pad_x, max_x + pad_x]
        yaxis_range = [min_y - pad_y, max_y + pad_y]

    return (
        spider1, spider2,
        make_scatter(
            selected_idx=df_idx, closest_idx=df_idx2,
            xaxis_range=xaxis_range, yaxis_range=yaxis_range
        )
    )
