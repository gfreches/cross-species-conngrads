import os
import numpy as np
import pandas as pd
import nibabel as nib
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

# --- Config ---
NPZ_FILE_PATH = "data/cross_species_embedding_data_2Species_LR_Combined.npz"
MASKED_BLUEPRINT_PARENT_DIR = "data/temporal_lobe_average_blueprints"
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

# --- Data Loading and Helper Functions ---
@st.cache_data(show_spinner=True)
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
        else:
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
    return df

def get_vertex_profile(species, hemisphere, vertex_id):
    bp_dir = os.path.join(MASKED_BLUEPRINT_PARENT_DIR, species)
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
        margin=dict(l=30, r=30, t=50, b=30), height=420
    )
    return fig

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Cross-Species Gradients Explorer")
st.title("Cross-Species Gradients: Interactive Exploration")

df = load_data()
color_map = {cfg["label"]: cfg["color"] for cfg in PLOT_CONFIGS_SCATTER.values()}
df["label"] = df["species_hem"].map(lambda x: PLOT_CONFIGS_SCATTER[x]["label"])

# Main scatter plot
st.subheader("Main Embedding (G1 vs G2)")
scatter_fig = go.Figure()
for k, group in df.groupby("label"):
    color = color_map[k]
    scatter_fig.add_trace(go.Scatter(
        x=group["g1"], y=group["g2"],
        mode="markers", marker=dict(size=8, color=color, line=dict(width=0.8, color='Black')),
        name=k, customdata=group[["df_idx", "species", "hem", "orig_vtx_id"]],
        hovertemplate="(%{x:.4f}, %{y:.4f})<br>species: %{customdata[1]}<br>hem: %{customdata[2]}<br>vtx: %{customdata[3]}<extra></extra>"
    ))
scatter_fig.update_layout(
    xaxis_title="Gradient 1", yaxis_title="Gradient 2",
    height=800, width=800,
    legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", yanchor="top")
)
selected = st.plotly_chart(scatter_fig, use_container_width=True, select_events=True, key="scatter")

# Streamlit selection model: use st.session_state or handle clicks with plotly_events
# Use st.plotly_events for selection if available
import streamlit_plotly_events
clicked = streamlit_plotly_events.plotly_events(
    scatter_fig, select_event=True, click_event=True, hover_event=False, key="select"
)

# Layout for spider plots
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Clicked Point: Tract Spider Plot")
with c2:
    st.subheader("Closest Other-Species Point: Tract Spider Plot")

# Handle click
if clicked:
    point = clicked[0]
    idx = int(point["customdata"][0])
    row = df.loc[idx]
    label = f"{row['species'].capitalize()} {row['hem']} (vtx {row['orig_vtx_id']})"
    color_key = f"{row['species']}_{row['hem']}"
    color = PLOT_CONFIGS_SCATTER[color_key]["color"]
    profile = get_vertex_profile(row['species'], row['hem'], row['orig_vtx_id'])
    spider1 = make_spider_plot(profile, label, color)
    with c1:
        st.plotly_chart(spider1, use_container_width=True)

    # Find closest other-species point
    clicked_g1, clicked_g2 = row['g1'], row['g2']
    mask = (df['species'] != row['species'])
    dists = euclidean_distances(
        np.array([[clicked_g1, clicked_g2]]),
        df.loc[mask, ['g1', 'g2']].values
    )[0]
    idx2 = df.loc[mask].index[np.argmin(dists)]
    row2 = df.loc[idx2]
    label2 = f"{row2['species'].capitalize()} {row2['hem']} (vtx {row2['orig_vtx_id']})"
    color_key2 = f"{row2['species']}_{row2['hem']}"
    color2 = PLOT_CONFIGS_SCATTER[color_key2]["color"]
    profile2 = get_vertex_profile(row2['species'], row2['hem'], row2['orig_vtx_id'])
    spider2 = make_spider_plot(profile2, label2, color2)
    with c2:
        st.plotly_chart(spider2, use_container_width=True)

# End
