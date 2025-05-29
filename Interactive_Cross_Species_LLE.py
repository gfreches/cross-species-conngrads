import os
import numpy as np
import pandas as pd
import nibabel as nib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances

# ---- CONFIG ----
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

# ---- DATA LOADING ----
@st.cache_data
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
        if s == "human":
            for i, grad_idx in enumerate(range(start, end)):
                vtx = -1
                if cluster_labels is not None:
                    idxs = np.where(cluster_labels == i)[0]
                    if len(idxs) > 0:
                        vtx = original_tl_indices[idxs[0]]
                records.append(dict(
                    row_idx=grad_idx,
                    species=s, hem=h,
                    species_hem=f"{s}_{h}",
                    g1=grads[grad_idx, 0],
                    g2=grads[grad_idx, 1],
                    orig_vtx_id=vtx,
                ))
        else:  # chimpanzee, NO deduplication
            for i, grad_idx in enumerate(range(start, end)):
                vtx = original_tl_indices[i] if i < len(original_tl_indices) else -1
                records.append(dict(
                    row_idx=grad_idx,
                    species=s, hem=h,
                    species_hem=f"{s}_{h}",
                    g1=grads[grad_idx, 0],
                    g2=grads[grad_idx, 1],
                    orig_vtx_id=vtx,
                ))
    df = pd.DataFrame.from_records(records)
    df.reset_index(drop=True, inplace=True)
    df["df_idx"] = df.index
    df["label"] = df["species_hem"].map(lambda x: PLOT_CONFIGS_SCATTER[x]["label"])
    return df

def get_vertex_profile(species, hemisphere, vertex_id):
    # For human, use downsampled blueprint; for chimp, use the average blueprint
    if species == "human":
        k_val = 2713 if hemisphere == "L" else 2950
        bp_file = f"average_human_blueprint_{hemisphere}_TL_dsViz_k{k_val}.func.gii"
        bp_path = os.path.join(DOWNSAMPLED_BP_DIR, bp_file)
    else:
        bp_dir = os.path.join(AVERAGE_BP_DIR, species)
        bp_file = f"average_{species}_blueprint_{hemisphere}_temporal_lobe.func.gii"
        bp_path = os.path.join(bp_dir, bp_file)
    if not os.path.exists(bp_path):
        return None
    img = nib.load(bp_path)
    if len(img.darrays) != N_TRACTS_EXPECTED:
        return None
    data = np.array([d.data for d in img.darrays]).T
    if not (0 <= vertex_id < data.shape[0]):
        return None
    return data[vertex_id, :]

def make_spider_plot(profile, label, color):
    if profile is None or np.allclose(profile, 0):
        return go.Figure(layout={"title": f"No data for {label}"})
    values = np.concatenate((profile, [profile[0]]))
    categories = TRACT_NAMES + [TRACT_NAMES[0]]
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

# ---- MAIN APP ----
st.set_page_config(layout="wide")
st.title("Cross-Species Gradients: Interactive Exploration")

df = load_data()
color_map = {cfg["label"]: cfg["color"] for cfg in PLOT_CONFIGS_SCATTER.values()}

# --- Interactive Scatter ---
scatter_fig = px.scatter(
    df, x='g1', y='g2', color='label', symbol='species',
    color_discrete_map=color_map, symbol_map=SPECIES_SYMBOLS,
    custom_data=['df_idx', 'species', 'hem', 'orig_vtx_id'],
    hover_data={'row_idx': True, 'species': True, 'hem': True, 'orig_vtx_id': True},
    title="Cross-Species Temporal Lobe Embedding (G1 vs G2)"
)
scatter_fig.update_traces(marker=dict(size=8, opacity=0.88, line=dict(width=0.8, color='Black')))
scatter_fig.update_layout(
    xaxis_title="Gradient 1", yaxis_title="Gradient 2",
    height=800, width=800,
    legend_title_text=None,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=0.01,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(0,0,0,0.13)", borderwidth=1,
        font=dict(size=13),
        itemclick="toggleothers"
    ),
    title=dict(y=0.98, x=0.5, xanchor="center", yanchor="top"),
    margin=dict(t=55, r=20, l=80, b=90)
)

click_data = st.plotly_chart(scatter_fig, use_container_width=True, height=800, width=800, key="main_scatter", **{"clickData": True})

if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None

if click_data and click_data["points"]:
    st.session_state.clicked_point = click_data["points"][0]
point = st.session_state.clicked_point if "clicked_point" in st.session_state else None

col1, col2 = st.columns(2)
with col1:
    if point is not None:
        custom = point["customdata"]
        df_idx = custom[0]
        species = custom[1]
        hem = custom[2]
        vtx_id = custom[3]
        label = f"{species.capitalize()} {hem} (vtx {vtx_id})"
        color_key = f"{species}_{hem}"
        spider_color = PLOT_CONFIGS_SCATTER[color_key]["color"]
        profile = get_vertex_profile(species, hem, vtx_id)
        spider1 = make_spider_plot(profile, label, spider_color)
        st.plotly_chart(spider1, use_container_width=True)
with col2:
    if point is not None:
        clicked_g1, clicked_g2 = point["x"], point["y"]
        mask = (df['species'] != species)
        dists = euclidean_distances(
            np.array([[clicked_g1, clicked_g2]]),
            df.loc[mask, ['g1', 'g2']].values
        )[0]
        idx = np.argmin(dists)
        row2 = df.loc[mask].iloc[idx]
        vtx2 = row2["orig_vtx_id"]
        label2 = f"{row2['species'].capitalize()} {row2['hem']} (vtx {vtx2})"
        color_key2 = f"{row2['species']}_{row2['hem']}"
        spider_color2 = PLOT_CONFIGS_SCATTER[color_key2]["color"]
        profile2 = get_vertex_profile(row2["species"], row2["hem"], vtx2)
        spider2 = make_spider_plot(profile2, label2, spider_color2)
        st.plotly_chart(spider2, use_container_width=True)
