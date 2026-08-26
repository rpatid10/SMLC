"""
CardioGuard - Streamlit production UI
--------------------------------------
Loads the trained pipeline exported by the notebook (`cardioguard_pipeline.pkl` and
`cardioguard_explainer.pkl`, both saved by Section 10 of the notebook) and serves a
browser-based clinical risk calculator.

Run with:
    streamlit run app.py

Requires `cardioguard_pipeline.pkl` and `cardioguard_explainer.pkl` to be in the same folder
(generate them by running the notebook once, or re-run `python train.py` if you exported one).
"""
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

st.set_page_config(page_title="CardioGuard Risk Calculator", page_icon="\u2764\ufe0f", layout="centered")

FEATURE_ORDER = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("cardioguard_pipeline.pkl")
    explainer_assets = joblib.load("cardioguard_explainer.pkl")
    return pipeline, explainer_assets


try:
    pipeline, assets = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found."
    )
    st.stop()

rf_model = assets["rf_model"]
rf_preprocess = assets["rf_preprocess"]
feature_names_out = assets["feature_names_out"]
tree_explainer = shap.TreeExplainer(rf_model)

st.title("\u2764\ufe0f CardioGuard Risk Calculator")
st.caption(
    "Developed By **Rahul Patidar**  "
    "PES Univercity - **Mtech DSAI**  "
    "Supervised classification - Mini Project  "
    "trained on 302 patients from the UCI Cleveland Heart Disease dataset. "
    "**Not a medical device - for educational demonstration only.**"
)

with st.form("patient_form"):
    st.subheader("Patient parameters")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 90, 55)
        sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
        chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 240)
        thalach = st.slider("Max heart rate achieved", 60, 210, 150)
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)

    with col2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        exang = st.selectbox("Exercise-induced angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        ca = st.selectbox("Major vessels colored by fluoroscopy", options=list(range(4)))
        cp = st.selectbox("Chest pain code (cp)", options=[0, 1, 2, 3])
        restecg = st.selectbox("Resting ECG code (restecg)", options=[0, 1, 2])
        slope = st.selectbox("ST slope code (slope)", options=[0, 1, 2])
        thal = st.selectbox("Thalassemia code (thal)", options=[1, 2, 3])

   

    submitted = st.form_submit_button("Predict Risk", type="primary")

if submitted:
    patient = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal,
    }])[FEATURE_ORDER]

    risk_prob = pipeline.predict_proba(patient)[0, 1]
    is_high_risk = risk_prob >= 0.5

    st.divider()
    if is_high_risk:
        st.error(f"### HIGH RISK - predicted probability: {risk_prob:.1%}")
    else:
        st.success(f"### LOW RISK - predicted probability: {risk_prob:.1%}")
    st.progress(min(max(risk_prob, 0.0), 1.0))

    # Local explanation via the fast, exact tree explainer (see notebook Section 9)
    transformed = rf_preprocess.transform(patient)
    sv = tree_explainer.shap_values(transformed)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]
    contributions = pd.Series(sv[0], index=feature_names_out).sort_values(key=np.abs, ascending=False).head(5)

    st.subheader("Top factors driving this prediction")
    st.caption("SHAP values from the Random Forest component")
    for feat, val in contributions.items():
        direction = "\u2b06\ufe0f pushes risk UP" if val > 0 else "\u2b07\ufe0f pushes risk DOWN"
        st.write(f"**{feat}** - {direction}  (SHAP = {val:+.3f})")

st.divider()
st.caption(
    "Model: tuned Stacking Ensemble." 
    "Dataset: UCI Cleveland Heart Disease"
    "(302 unique patients)."
)
