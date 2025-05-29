import os
import numpy as np
import pandas as pd
import nibabel as nib
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

# --- Config ---
NPZ_FILE_PATH = "data/gradient_outputs_cross_species_embedding_data_2Species_LR_Combined.npz"
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

# --- Helpers ---

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
        for i, grad_idx in enumerate(range(start, end)):
            records.append(dict(
                row_idx=grad_idx,
                species=s, hem=h,
                species_hem=f"{s}_{h}",
                g1=grads[grad_idx, 0],
                g2=grads[grad_idx, 1],
                centroid_idx=i
            ))
    df = pd.DataFrame.from_records(records)
    df.reset_index(drop=True, inplace=True)
    return df

def get_centroid_profile(species, hemisphere, centroid_idx):
    if species == "human":
        # Downsampled: use k value for correct file (L=2713, R=2950)
        k = 2713 if hemisphere == 'L' else 2950
        bp_file = f"average_human_blueprint_{hemisphere}_TL_dsViz_k{k}.func.gii"
        bp_path = os.path.join(DOWNSAMPLED_BP_DIR, bp_file)
    else:  # chimpanzee: use original average blueprint
        bp_path = os.path.join(
            AVERAGE_BP_DIR, "chimpanzee",
            f"average_chimpanzee_blueprint_{hemisphere}_temporal_lobe.func.gii"
        )
    if not os.path.exists(bp_path):
        st.warning(f"Blueprint not found: {bp_path}")
        return None
    img = nib.load(bp_path)
    if len(img.darrays) != N_TRACTS_EXPECTED:
        st.warning(f"Tract count mismatch for {bp_path}")
        return None
    data = np.array([d.data for d in img.darrays]).T
    if not (0 <= centroid_idx < data.shape[0]):
        st.warning(f"Centroid idx {centroid_idx} out of bounds in {bp_path}")
        return None
    return data[centroid_idx, :]

def make_spider_plot(profile, label, color, tract_labels=TRACT_NAMES):
    if profile is None or np.allclose(profile, 0):
        fig = go.Figure()
        fig.update_layout(title=f"No data for {label}")
        return fig
    values = np.concatenate((profile, [profile[0]]))
    categories = tract_labels + [tract_labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        name=label, marker_color=color, line=dict(color=color, width=3)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(0.05, np.max(values)*1.1)])),
        showlegend=False,
        title=label,
        margin=dict(l=30, r=30, t=60, b=30), height=400
    )
    return fig

def plot_embedding(df, clicked_idx=None, closest_idx=None):
    color_map = {cfg["label"]: cfg["color"] for cfg in PLOT_CONFIGS_SCATTER.values()}
    df["label"] = df["species_hem"].map(lambda x: PLOT_CONFIGS_SCATTER[x]["label"])
    fig = go.Figure()
    # Main points
    for species_hem, group in df.groupby('species_hem'):
        label = PLOT_CONFIGS_SCATTER[species_hem]["label"]
        color = PLOT_CONFIGS_SCATTER[species_hem]["color"]
        fig.add_trace(go.Scatter(
            x=group['g1'], y=group['g2'],
            mode='markers', marker=dict(size=8, color=color, line=dict(width=0.8, color='Black'), opacity=0.85),
            name=label, legendgroup=label, hoverinfo="text",
            customdata=np.stack([group.index, group['species'], group['hem'], group['centroid_idx']], axis=1),
            text=[f"{row['species'].capitalize()} {row['hem']}<br>Centroid {row['centroid_idx']}" for idx, row in group.iterrows()],
            showlegend=True
        ))
    # Highlight clicked
    if clicked_idx is not None:
        row = df.loc[clicked_idx]
        fig.add_trace(go.Scatter(
            x=[row.g1], y=[row.g2],
            mode='markers',
            marker=dict(size=22, color='black', line=dict(width=4, color='yellow'), symbol="x"),
            name="Clicked", showlegend=False
        ))
    # Highlight closest
    if closest_idx is not None:
        row = df.loc[closest_idx]
        color = PLOT_CONFIGS_SCATTER[f"{row['species']}_{row['hem']}"]["color"]
        fig.add_trace(go.Scatter(
            x=[row.g1], y=[row.g2],
            mode='markers',
            marker=dict(size=32, color=color, line=dict(width=7, color='yellow'), symbol="star"),
            name="Closest", showlegend=False
        ))
    fig.update_layout(
        xaxis_title="Gradient 1", yaxis_title="Gradient 2",
        height=850, width=830,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.025,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0.13)", borderwidth=1,
            font=dict(size=13),
            itemclick="toggleothers"
        ),
        title=dict(text="Cross-Species Temporal Lobe Embedding (G1 vs G2)", y=0.96, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(t=65, r=20, l=80, b=90)
    )
    return fig

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Cross-Species Gradients: Interactive Explorer (Downsampled & Average Blueprints)")

df = load_data()

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("**Click a point on the scatter to see tract profiles for that centroid/region.**")
    embedding_fig = plot_embedding(df)
    click = st.plotly_chart(embedding_fig, use_container_width=True, click_data=None, key="mainplot", config={'displayModeBar': True, 'scrollZoom': True})

with col2:
    clicked_idx = None
    closest_idx = None
    spider1 = spider2 = go.Figure()
    click_event = st.session_state.get('mainplot_click_data') or st.session_state.get('mainplot') or st.experimental_get_query_params().get('mainplot')
    if 'mainplot_click_data' in st.session_state:
        click_event = st.session_state['mainplot_click_data']
    elif click is not None and hasattr(click, 'clickData'):
        click_event = click.clickData
    if click_event and 'points' in click_event and click_event['points']:
        point = click_event['points'][0]
        custom = point["customdata"]
        clicked_idx = int(custom[0])
        clicked_row = df.loc[clicked_idx]
        c_species, c_hem, c_centroid = clicked_row['species'], clicked_row['hem'], clicked_row['centroid_idx']
        label1 = f"{c_species.capitalize()} {c_hem} (centroid {c_centroid})"
        color1 = PLOT_CONFIGS_SCATTER[f"{c_species}_{c_hem}"]["color"]
        profile1 = get_centroid_profile(c_species, c_hem, c_centroid)
        spider1 = make_spider_plot(profile1, label1, color1)
        # Find closest other-species point
        mask = (df['species'] != c_species)
        dists = euclidean_distances(
            np.array([[clicked_row['g1'], clicked_row['g2']]]),
            df.loc[mask, ['g1', 'g2']].values
        )[0]
        idx = np.argmin(dists)
        closest_idx = df.loc[mask].index[idx]
        closest_row = df.loc[closest_idx]
        label2 = f"{closest_row['species'].capitalize()} {closest_row['hem']} (centroid {closest_row['centroid_idx']})"
        color2 = PLOT_CONFIGS_SCATTER[f"{closest_row['species']}_{closest_row['hem']}"]["color"]
        profile2 = get_centroid_profile(closest_row['species'], closest_row['hem'], closest_row['centroid_idx'])
        spider2 = make_spider_plot(profile2, label2, color2)
        # Replot embedding with highlights
        embedding_fig = plot_embedding(df, clicked_idx=clicked_idx, closest_idx=closest_idx)
        st.plotly_chart(embedding_fig, use_container_width=True, key="mainplot-replot")
    st.markdown("#### Clicked Point: Tract Spider Plot")
    st.plotly_chart(spider1, use_container_width=True)
    st.markdown("#### Closest Other-Species Point: Tract Spider Plot")
    st.plotly_chart(spider2, use_container_width=True)
