# ❤️ CardioGuard — Explainable Heart Disease Risk Classification

**A hyperparameter-tuned Stacking Ensemble with SHAP-based explainability, deployed as an interactive clinical risk calculator.**

SMLC (UX25CS635A) — Supervised Machine Learning Classification Mini Project
**Rahul Patidar** · SRN: PES1PGE25DS049 · 1st Semester M.Tech (Data Science & AI) · PES University
*Under the guidance of Dr. Shylaja S. Sharath, HOD, Dept. of CSE, PES University*

> ⚠️ **Not a medical device.** This project is an educational/academic demonstration only and must not be used for real clinical decision-making.

---

## 📌 What this project does

CardioGuard predicts a patient's risk of heart disease from 13 standard clinical measurements, using a rigorously validated, leak-proof machine learning pipeline — and, just as importantly, **explains why** it made that prediction instead of returning a black-box number.

- **Dataset:** UCI Cleveland Heart Disease dataset (302 unique patients after de-duplication)
- **Final model:** Stacking Ensemble — Logistic Regression + Random Forest + Gradient Boosting (tuned base learners) with a Logistic Regression meta-learner
- **Performance:** 0.898 cross-validated ROC-AUC · 0.881 held-out test ROC-AUC · 77.0% test accuracy (vs. 54.1% majority-class baseline)
- **Explainability:** Global permutation importance + per-patient SHAP values
- **Deployment:** Streamlit web app + in-notebook ipywidgets dashboard, both sharing one saved pipeline

---

## 📂 Repository contents

| File | Description |
|---|---|
| `CardioGuard_Heart_Disease_Classification.py` | Full end-to-end notebook/script: data loading, preprocessing, model tuning, stacking ensemble, evaluation, explainability, and artifact export |
| `app.py` | Streamlit web application — the clinical risk calculator UI |
| `cardioguard_pipeline.pkl` | Saved, fitted scikit-learn `Pipeline` (preprocessing + final stacking classifier) |
| `cardioguard_explainer.pkl` | Saved assets for explainability (tuned Random Forest model, preprocessor, output feature names) used by SHAP |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 🩺 Dataset

Source: **UCI Machine Learning Repository — Heart Disease (Cleveland)**
🔗 https://archive.ics.uci.edu/dataset/45/heart+disease

- 303 patients originally → **302 unique patients** after removing one exact duplicate
- **13 clinical features** + 1 binary target (disease present / absent)
- Near-balanced classes: 164 disease-positive (54.3%), 138 disease-negative (45.7%)
- Originally collected at the Cleveland Clinic Foundation and published by Detrano et al., 1989, *American Journal of Cardiology*

### Feature reference

| Code | Full Name | Meaning | Type |
|---|---|---|---|
| `age` | Age | Age in years | Continuous |
| `sex` | Sex | 0 = Female, 1 = Male | Binary |
| `cp` | Chest Pain Type | Clinical category of chest pain (0–3) | Categorical |
| `trestbps` | Resting Blood Pressure | mm Hg, measured at rest | Continuous |
| `chol` | Serum Cholesterol | mg/dl | Continuous |
| `fbs` | Fasting Blood Sugar | 1 if > 120 mg/dl, else 0 | Binary |
| `restecg` | Resting ECG Result | Category of resting heart electrical pattern (0–2) | Categorical |
| `thalach` | Max Heart Rate Achieved | Highest heart rate during exercise stress test | Continuous |
| `exang` | Exercise-Induced Angina | 1 if exercise triggered chest pain, else 0 | Binary |
| `oldpeak` | ST Depression | ECG ST-segment dip, exercise vs. rest | Continuous |
| `slope` | ST Segment Slope | Shape of ST segment at peak exercise (0–2) | Categorical |
| `ca` | Major Vessels Colored | Vessels (0–3) highlighted on fluoroscopy | Continuous |
| `thal` | Thalassemia Result | Blood-disorder-linked heart test result (1–3) | Categorical |

> **Data-quality note:** `cp`, `restecg`, `slope`, and `thal` numeric codes are inconsistently documented across public mirrors of this dataset. They are therefore modelled as **unordered categorical** variables (one-hot encoded), with no assumed ordinal relationship.

---

## ⚙️ Methodology

1. **Preprocessing** — a `ColumnTransformer` (StandardScaler for continuous features, passthrough for binary features, one-hot encoding for nominal categoricals) is wrapped inside a scikit-learn `Pipeline` together with the model, so it is fit **only on the training fold** of every split — no data leakage.
2. **Base learner tuning** — six classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, SVM (RBF), Gradient Boosting) are each tuned with `GridSearchCV` over 5-fold stratified cross-validation, optimizing ROC-AUC.
3. **Stacking ensemble** — candidate combinations of tuned base learners (top-3 / top-4 / all six) are compared purely by cross-validated ROC-AUC on the training set; the **top-3 combination (Logistic Regression + Random Forest + Gradient Boosting)** with a Logistic Regression meta-learner is selected, without ever consulting the test set.
4. **Evaluation** — final performance is reported on a held-out, stratified 80/20 split (241 train / 61 test patients).
5. **Explainability** — permutation importance (global) and `shap.TreeExplainer` on the Random Forest component (per-patient, local).
6. **Deployment** — the fitted pipeline is serialized once with `joblib` and shared by both front-ends.

---

## 📊 Results

| Model | CV ROC-AUC | Test Acc. | Test Prec. | Test Rec. | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.896 | 0.787 | 0.778 | 0.848 | 0.812 | 0.872 |
| K-Nearest Neighbors | 0.889 | 0.803 | 0.800 | 0.848 | 0.824 | 0.892 |
| Decision Tree | 0.826 | 0.721 | 0.690 | 0.879 | 0.773 | 0.708 |
| Random Forest | 0.894 | 0.754 | 0.750 | 0.818 | 0.783 | 0.886 |
| SVM (RBF) | 0.883 | 0.836 | 0.811 | 0.909 | 0.857 | 0.918 |
| Gradient Boosting | 0.890 | 0.787 | 0.778 | 0.848 | 0.812 | 0.874 |
| **Stacking Ensemble (proposed)** | **0.898** | **0.770** | **0.757** | **0.848** | **0.800** | **0.881** |

**Confusion matrix (test set, n=61):**

| | Predicted: No Disease | Predicted: Disease |
|---|---|---|
| **Actual: No Disease** | 19 | 9 |
| **Actual: Disease** | 5 | 28 |

**Sanity check:** majority-class baseline accuracy = 54.1% → model test accuracy = 77.0%.

**Top features by permutation importance:** `ca` (ΔROC-AUC = 0.071), `thal` (0.038), `thalach` (0.017), `oldpeak` (0.016) — consistent with established cardiology risk markers.

---

## 🚀 Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce the analysis

Run the full pipeline script (data loading → preprocessing → tuning → stacking → evaluation → explainability → artifact export):

```bash
python CardioGuard_Heart_Disease_Classification.py
```

This regenerates `cardioguard_pipeline.pkl` and `cardioguard_explainer.pkl`.

### 3. Launch the risk calculator app

```bash
streamlit run app.py
```

Make sure `cardioguard_pipeline.pkl` and `cardioguard_explainer.pkl` are in the same folder as `app.py` (either run step 2 first, or use the ones already provided).

The app will open in your browser. Enter patient details (age, blood pressure, cholesterol, chest pain type, etc.) and click **Predict Risk** to see:
- A risk probability (LOW / HIGH RISK)
- The top clinical factors driving that specific prediction (SHAP values)

---

## 🧠 Explainability at a glance

- **Global (permutation importance):** shuffles one feature at a time across the test set and measures the resulting drop in ROC-AUC — a bigger drop means the model relies more heavily on that feature.
- **Local (SHAP values):** for one specific patient, `shap.TreeExplainer` computes exactly how much each feature pushed their individual prediction up or down from a baseline — e.g. *"thal pushes this patient's risk UP by 0.18."*

---

## ⚠️ Limitations & future work

- Dataset size is modest (302 patients) — a larger, more diverse cohort would strengthen confidence in the reported test-set numbers.
- SHAP currently explains the Random Forest component (fast, exact) rather than the full three-model stack; a `KernelExplainer` on the ensemble would be exact but computationally heavier.
- No probability calibration or external clinical-site validation has been performed — required before any real-world use.

---

## 📚 References

1. R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease," *American Journal of Cardiology*, vol. 64, no. 5, pp. 304–310, 1989.
2. S. Mohan, C. Thirumalai, and G. Srivastava, "Effective heart disease prediction using hybrid machine learning techniques," *IEEE Access*, vol. 7, pp. 81542–81554, 2019.
3. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017, pp. 4765–4774.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
5. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

---

## 👤 Author

**Rahul Patidar**
SRN: PES1PGE25DS049 · 1st Semester, M.Tech (Data Science & Artificial Intelligence) · PES University
Under the guidance of **Dr. Shylaja S. Sharath**, HOD, Dept. of CSE, PES University
