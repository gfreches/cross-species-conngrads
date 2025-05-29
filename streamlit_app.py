import streamlit as st
from streamlit_dash import st_dash

# Import the Dash app object from your script
from Interactive_Cross_Species_LLE import app as dash_app

st.set_page_config(
    page_title="Cross-Species Gradients Explorer",
    layout="wide",
)

st.title("Cross-Species Temporal Lobe Gradients Explorer")
st.markdown("""
Interactively explore cross-species connectome gradients.
- **Click** on a point to view its tract profile ("spider plot").
- The closest point from the other species will be highlighted and plotted for comparison.
- Zoom and pan as needed.
""")

# Embed the Dash app below
st_dash(dash_app, width=1600, height=1200)
