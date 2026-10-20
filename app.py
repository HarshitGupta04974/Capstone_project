"""
Step 3: Local Streamlit Autism Screening & Model Evaluation Dashboard
=====================================================================
Run locally on your PC:
    streamlit run 3_app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import streamlit as st

st.set_page_config(
    page_title="ASD Screening & ML Benchmark (Local)",
    page_icon="🧠",
    layout="wide",
)

ITEMS = [f"A{i}" for i in range(1, 11)]
ITEM_DESCRIPTIONS = {
    "A1": "Notices small sounds when others do not",
    "A2": "Concentrates more on the whole picture rather than small details",
    "A3": "Finds it easy to do more than one thing at once",
    "A4": "If there is an interruption, can switch back to what they were doing very quickly",
    "A5": "Finds it easy to 'read between the lines' when someone is talking",
    "A6": "Knows how to tell if someone listening is getting bored",
    "A7": "Finds it difficult to work out characters' intentions in a story",
    "A8": "Likes to collect information about categories of things (e.g., types of cars, birds, trains)",
    "A9": "Finds it easy to work out what someone is thinking or feeling just by looking at their face",
    "A10": "Finds it difficult to work out people's intentions",
}


def get_age_group(age):
    if age <= 10:
        return "child"
    elif age <= 15:
        return "adolescent"
    elif age <= 24:
        return "young_adult"
    return "adult"


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@st.cache_resource
def load_artifacts():
    model_path = os.path.join("results", "xgb_platt.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_tables_and_data():
    data = {}

    # Check both the current directory and ./data/ for the harmonized CSV
    if os.path.exists("harmonized_asd_3450.csv"):
        data["df"] = pd.read_csv("harmonized_asd_3450.csv")
    elif os.path.exists(os.path.join("data", "harmonized_asd_3450.csv")):
        data["df"] = pd.read_csv(os.path.join("data", "harmonized_asd_3450.csv"))

    if os.path.exists("results/table1_holdout.csv"):
        data["t1"] = pd.read_csv("results/table1_holdout.csv", index_col=0)

    if os.path.exists("results/table2_calibration.csv"):
        data["t2"] = pd.read_csv("results/table2_calibration.csv", index_col=0)

    if os.path.exists("results/significance.csv"):
        data["sig"] = pd.read_csv("results/significance.csv")

    if os.path.exists("results/summary.json"):
        with open("results/summary.json", "r") as f:
            data["summary"] = json.load(f)

    return data


artifacts = load_artifacts()
saved_data = load_tables_and_data()

st.title("🧠 Autism Spectrum Disorder (ASD) Screening — Local ML Dashboard")
st.caption("Running 100% locally on your machine | Powered by Platt-Calibrated XGBoost & AQ-10 / Q-CHAT-10 Harmonized Schema")

if artifacts is None:
    st.error(
        "Model artifacts not found in `results/xgb_platt.joblib`! "
        "Please run the dataset builder and model training scripts first."
    )
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🩺 Interactive ASD Screening",
    "📊 Hold-Out Benchmarks & Calibration",
    "🗂️ Harmonized Dataset Explorer (3,450 Rows)",
])

# =========================================================================== #
# TAB 1: INTERACTIVE SCREENING PREDICTION
# =========================================================================== #
with tab1:
    col_form, col_res = st.columns([1.35, 1.0], gap="large")

    with col_form:
        st.subheader("1. Demographic & Clinical Profile")
        d1, d2, d3 = st.columns(3)
        with d1:
            age = st.number_input("Age (Years)", min_value=1, max_value=100, value=21, step=1)
            sex = st.selectbox("Sex", options=["m", "f"], format_func=lambda x: "Male (m)" if x == "m" else "Female (f)")
        with d2:
            jaundice = st.selectbox("Born with Jaundice?", options=["no", "yes"])
            family_history = st.selectbox("Family Member with ASD?", options=["no", "yes"])
        with d3:
            ethnicity = st.selectbox
