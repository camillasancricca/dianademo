from sqlite3 import connect

import pandas as pd
import streamlit as st
from PIL import Image
import requests



from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title="DIANA", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>h1{color: black; font-family: 'Roboto', sans-serif;}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;  /* Cambia il colore dello sfondo qui */
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("Welcome to the DIANA tool!")
st.write("**DIANA** is a **data-centric AI-based tool** able to support and guide users in selecting and validating the data preparation tasks to perform in a data analysis pipeline.")
st.write("DIANA adopts a human-centered approach: involving the users in all stages, supporting their decisions, and leaving them in control of the process.")

st.write("DIANA's main functionalities are:")
st.write("1) **exploration**, **profiling**, and **data quality assessment** functionalities to make the users aware of the characteristics and anomalies of the data")
st.write("2) **recommendations** on the best sequence of data preparation actions that best-fit users’ analysis purposes")
st.write("3) **explainability** to enable also non-expert practitioners to be involved in the pipeline phases")
st.write("4) **sliding autonomy** that is the system’s ability to incorporate human intervention when needed, e.g., increasing/decreasing the system support based on the user needs, skills and expertise.")

start_button = st.button("Let's start the pipeline!")
st.session_state['x'] = 0
if start_button:
    switch_page("upload")

