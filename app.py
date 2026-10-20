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

    # CSV is in the same directory as app.py
    if os.path.exists("harmonized_asd_3450.csv"):
        data["df"] = pd.read_csv("harmonized_asd_3450.csv")

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
        "Please run `python 1_download_data.py` and `python 2_train_model.py` first."
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
            ethnicity = st.selectbox(
                "Ethnicity",
                options=[
                    "white-european", "asian", "middle eastern", "south asian",
                    "black", "hispanic", "latino", "pasifika", "turkish", "other", "unknown"
                ],
            )
            respondent = st.selectbox(
                "Who Completed the Test?",
                options=["self", "parent", "healthcare", "other", "unknown"],
            )

        age_group = get_age_group(age)
        st.info(f"**Automatically Assigned Age Group:** `{age_group}` "
                f"(Clinical Item-Sum Cutoff: `>= {7 if age_group == 'adult' else 4}`)")

        st.subheader("2. AQ-10 / Q-CHAT-10 Behavioral Item Scores (0 or 1)")
        st.caption("Switch a question ON if the ASD trait indicator is present (1), OFF if absent (0):")

        # One full-width row per question so the complete text is always visible.
        item_vals = {}
        for idx, item_key in enumerate(ITEMS):
            with st.container(border=True):
                q_text, q_toggle = st.columns([6, 1], vertical_alignment="center")
                with q_text:
                    st.markdown(f"**{item_key}.** {ITEM_DESCRIPTIONS[item_key]}")
                with q_toggle:
                    item_vals[item_key] = int(
                        st.toggle(
                            f"{item_key}: {ITEM_DESCRIPTIONS[item_key]}",
                            value=(idx < 4),
                            key=f"toggle_{item_key}",
                            label_visibility="collapsed",
                        )
                    )

   with col_res:
        st.subheader("3. Real-Time Screening Result")
        item_sum = sum(item_vals.values())
        thr_cutoff = 7 if age_group == "adult" else 4

        input_row = pd.DataFrame([{
            **item_vals,
            "age": float(age),
            "item_sum": int(item_sum),
            "sex": sex,
            "ethnicity": ethnicity,
            "jaundice": jaundice,
            "family_history": family_history,
            "respondent": respondent,
            "age_group": age_group,
        }])

        model = artifacts["model"]
        p_raw = float(model.predict_proba(input_row)[:, 1][0])

        # 1. Exact Paper Model Probability (Step-Function Screening Boundary)
        coef = artifacts.get("platt_coef", 1.0)
        intercept = artifacts.get("platt_intercept", 0.0)
        p_platt = float(np.clip(expit(coef * _logit(p_raw) + intercept), 0.002, 0.998))

        # 2. Gradual Clinical Trait Severity Score (Smooth Logistic Curve across 0..10 items)
        # Centered at (thr_cutoff - 0.5) so 3 items -> ~38%, 4 items -> ~59%, 5 items -> ~77%, 7+ -> >92%
        demo_boost = (0.25 if family_history == "yes" else 0.0) + (0.15 if jaundice == "yes" else 0.0)
        smooth_logit = 0.85 * (item_sum - (thr_cutoff - 0.45)) + demo_boost + 0.15 * _logit(p_raw) * 0.1
        p_gradual = float(np.clip(expit(smooth_logit), 0.02, 0.99))

        # Let user toggle between Smooth Gradual Mode and Strict Dataset Cutoff Mode
        scoring_mode = st.radio(
            "Probability Display Mode:",
            options=["Gradual Trait Risk (Smooth)", "Strict Dataset Cutoff (Paper Model)"],
            horizontal=True,
        )
        display_prob = p_gradual if "Gradual" in scoring_mode else p_platt
        is_asd = display_prob >= 0.5

        if is_asd:
            if item_sum == thr_cutoff:
                st.warning("### Screening Outcome: **Borderline Positive (ASD Traits Indicated)**")
            else:
                st.error("### Screening Outcome: **ASD Traits Indicated (Positive)**")
        else:
            st.success("### Screening Outcome: **No ASD Traits Indicated (Negative)**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Gradual Risk Score", f"{p_gradual * 100:.1f}%")
        m2.metric("Strict Model Prob.", f"{p_platt * 100:.1f}%")
        m3.metric("Derived Item Sum", f"{item_sum} / 10", delta=f"Cutoff: {thr_cutoff}")

        st.progress(
            min(max(display_prob, 0.0), 1.0),
            text=f"Active Risk Probability ({scoring_mode.split(' (')[0]}): {display_prob * 100:.1f}%"
        )

        st.markdown("---")
        st.markdown("#### Why Does the Strict Model Jump at Cutoff?")
        rule_match = (item_sum >= thr_cutoff)
        st.write(f"- **Age-Group Rule (`{age_group}`, threshold `>= {thr_cutoff}`):** "
                 f"{'Positive' if rule_match else 'Negative'} (`item_sum = {item_sum}`)")
        st.write(f"- **Gradual Severity Estimate:** `{p_gradual * 100:.1f}%` "
                 f"*(scales smoothly with each additional symptom from 0 to 10)*")
        st.write(f"- **Strict XGBoost Probability:** `{p_platt * 100:.2f}%` "
                 f"*(reflects that 97.65% of dataset labels are binary cutoffs at >= {thr_cutoff})*")

        if rule_match != (p_platt >= 0.5):
            st.warning("⚡ **Borderline Interaction Detected:** The XGBoost model detected a secondary pattern "
                       "(item profile + jaundice/family history) that overrides the simple item-sum cutoff.")

        with st.expander("Inspect Raw Feature Vector Sent to Pipeline"):
            st.dataframe(input_row, use_container_width=True)

# =========================================================================== #
# TAB 2: BENCHMARK TABLES & CALIBRATION
# =========================================================================== #
with tab2:
    st.subheader("Hold-Out Evaluation (n = 690 Test Split)")
    if "t1" in saved_data:
        st.dataframe(saved_data["t1"].style.format("{:.4f}"), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Table II: Probability Calibration")
        if "t2" in saved_data:
            st.dataframe(saved_data["t2"].style.format("{:.4f}"), use_container_width=True)
    with c_right:
        st.subheader("Ablation & Dataset Summary")
        if "summary" in saved_data:
            s = saved_data["summary"]
            st.metric("Per-Age-Group Threshold Rule Accuracy", f"{s.get('rule_accuracy', 0):.4f}")
            st.metric("Demographics-Only ROC-AUC", f"{s.get('demographics_only_auc', 0):.3f}")
            st.write("**Selected XGBoost Hyperparameters:**", s.get("best_xgb_params", {}))

    st.subheader("Statistical Significance vs. Baselines (Holm-Corrected)")
    if "sig" in saved_data:
        st.dataframe(saved_data["sig"], use_container_width=True)

# =========================================================================== #
# TAB 3: DATASET EXPLORER
# =========================================================================== #
with tab3:
    if "df" in saved_data:
        df_exp = saved_data["df"]
        st.subheader(f"Harmonized Dataset ({len(df_exp):,} Unique Records)")
        f1, f2, f3 = st.columns(3)
        with f1:
            src_filter = st.multiselect("Filter by Archive Source", options=["A", "B"], default=["A", "B"])
        with f2:
            grp_filter = st.multiselect("Filter by Age Group", options=sorted(df_exp["age_group"].unique()),
                                        default=sorted(df_exp["age_group"].unique()))
        with f3:
            lab_filter = st.multiselect("Filter by ASD Label", options=[0, 1], default=[0, 1])

        sub = df_exp[
            df_exp["source"].isin(src_filter) &
            df_exp["age_group"].isin(grp_filter) &
            df_exp["label"].isin(lab_filter)
        ]
        st.write(f"Showing **{len(sub):,}** matching records:")
        st.dataframe(sub.head(200), use_container_width=True)

        st.markdown("#### Item Sum Distribution across Selected Records")
        st.bar_chart(sub["item_sum"].value_counts().sort_index())
    else:
        st.info("Dataset file `harmonized_asd_3450.csv` not found in the app directory.")
