import streamlit as st
from streamlit_dash import st_dash
from Interactive_Cross_Species_LLE import app

st.title("Testing Dash in Streamlit")
st_dash(app, mode='external')
