"""
Spam Sentinel - Streamlit front-end for ML Assignment 2 (UCI Spambase).

The app only *loads* the pre-trained pipelines from model/ and scores the
uploaded CSV with them. Nothing is re-trained here.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
CLASS_NAMES = ["Not spam (0)", "Spam (1)"]

st.set_page_config(page_title="Spam Sentinel | ML Assignment 2", page_icon="✉️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .hero {background: linear-gradient(90deg,#14365d 0%,#1f6f8b 100%);
             color:#fff; padding:1.1rem 1.4rem; border-radius:12px; margin-bottom:1rem;}
      .hero h1 {font-size:1.7rem; margin:0; color:#fff;}
      .hero p {margin:.3rem 0 0 0; opacity:.9;}
      div[data-testid="stMetric"] {background:#f3f7fb; border:1px solid #dbe6f0;
             border-radius:10px; padding:.6rem .8rem;}
    </style>
    <div class="hero">
      <h1>✉️ Spam Sentinel</h1>
      <p>Five classifiers trained on the UCI Spambase data. Upload a labelled test CSV,
      pick a model and see how it performs - no retraining happens in this app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}
    ref = pd.read_csv(MODEL_DIR / "metrics.csv")
    params = json.loads((MODEL_DIR / "best_params.json").read_text())
    return meta, models, ref, params


def score(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob) if y.nunique() == 2 else np.nan
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_prob


try:
    META, MODELS, REF_METRICS, BEST_PARAMS = load_artifacts()
except Exception as exc:  # pragma: no cover - shown in the UI
    st.error(f"Could not load the saved models from `model/`: {exc}")
    st.stop()

FEATURES, TARGET = META["features"], META["target"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + target)", type=["csv"])
    st.caption(f"Needs all {len(FEATURES)} feature columns and the target column "
               f"`{TARGET}` (0 = not spam, 1 = spam).")
    if SAMPLE_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
        use_sample = st.checkbox("No file? Use the bundled test_data.csv", value=False,
                                 disabled=uploaded is not None)
    else:
        use_sample = False

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODELS.keys()),
                              index=list(MODELS.keys()).index("Support Vector Machine"))
    threshold = st.slider("Spam decision threshold (probability)", 0.05, 0.95, 0.50, 0.05,
                          help="0.50 reproduces the README numbers. Move it to trade "
                               "precision against recall.")
    st.divider()
    st.caption(f"Models trained with scikit-learn {META['scikit_learn']} · seed {META['seed']}")

# ---------------------------------------------------------------- data
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    df = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("👈 Upload a test CSV in the sidebar (or tick *Use the bundled test_data.csv*) to begin.")
    st.subheader("Reference results on the held-out test split")
    st.dataframe(REF_METRICS.set_index("ML Model Name").style.format("{:.4f}")
                 .highlight_max(axis=0, color="#cfe8d5"), width="stretch")
    st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing or TARGET not in df.columns:
    st.error("The uploaded file does not match the training schema.")
    if TARGET not in df.columns:
        st.write(f"Missing target column: `{TARGET}`")
    if missing:
        st.write(f"Missing {len(missing)} feature column(s):", missing[:20])
    st.stop()

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES].astype(float)
y = df[TARGET].astype(int)

c1, c2, c3 = st.columns(3)
c1.metric("Rows scored", f"{len(df):,}")
c2.metric("Spam share", f"{y.mean():.1%}")
c3.metric("Data source", source)

tab_model, tab_compare, tab_data, tab_about = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔎 Data & predictions", "ℹ️ About"])

# ---------------------------------------------------------------- selected model
model = MODELS[model_name]
y_prob = model.predict_proba(X)[:, 1]
# strict ">" so 0.50 reproduces model.predict() (ties go to class 0)
y_pred = (y_prob > threshold).astype(int)
sel = {
    "Accuracy": accuracy_score(y, y_pred),
    "AUC": roc_auc_score(y, y_prob) if y.nunique() == 2 else np.nan,
    "Precision": precision_score(y, y_pred, zero_division=0),
    "Recall": recall_score(y, y_pred, zero_division=0),
    "F1": f1_score(y, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y, y_pred),
}

with tab_model:
    st.subheader(f"{model_name} — evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, sel.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")
    st.caption("Tuned hyper-parameters: " +
               ", ".join(f"`{k}={v}`" for k, v in BEST_PARAMS[model_name].items()))

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"{fp} genuine e-mails flagged as spam · {fn} spam e-mails missed")
    with right:
        st.markdown("**ROC curve**")
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, y_prob)
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            ax.plot(fpr, tpr, color="#1f6f8b", lw=2, label=f"AUC = {sel['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig, width="stretch")
        else:
            st.info("ROC needs both classes in the uploaded data.")

    st.markdown("**Classification report**")
    rep = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.style.format("{:.4f}"), width="stretch")

# ---------------------------------------------------------------- comparison
with tab_compare:
    st.subheader("All five models on the uploaded data (threshold 0.50)")
    rows = []
    for name, m in MODELS.items():
        metrics, _, _ = score(m, X, y)
        rows.append({"ML Model Name": name, **metrics})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8d5"),
                 width="stretch")
    best = comp["MCC"].idxmax()
    st.success(f"Best model on this file by MCC: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")

    long = comp.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="crest", ax=ax)
    ax.set_ylim(max(0, long["Score"].min() - 0.05), 1)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    st.pyplot(fig, width="stretch")
    st.download_button("⬇️ Download comparison table", comp.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# ---------------------------------------------------------------- data
with tab_data:
    st.subheader(f"Predictions from {model_name}")
    out = df.copy()
    out["predicted"] = y_pred
    out["spam_probability"] = y_prob.round(4)
    out["correct"] = out["predicted"] == out[TARGET]
    only_wrong = st.toggle("Show misclassified rows only", value=False)
    view = out[~out["correct"]] if only_wrong else out
    st.dataframe(view[[TARGET, "predicted", "spam_probability", "correct"] + FEATURES].head(500),
                 width="stretch", height=360)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        f"""
**Dataset:** UCI Spambase (id 94) — 4,601 e-mails, 57 numeric features
(48 word frequencies, 6 character frequencies, 3 capital-letter run-length statistics).
Exact duplicates were removed ({META['n_rows_after_dedup']:,} rows kept) before a stratified
80/20 split.

**Pre-processing:** `log(1+x)` + standard scaling inside each pipeline for Logistic
Regression, KNN, Naive Bayes and SVM; the Decision Tree uses raw features.
Hyper-parameters were picked by 5-fold stratified CV (F1) on the training split only.

**Positive class** for Precision / Recall / F1 is *spam* (1).
"""
    )
    st.markdown("**Reference metrics on the held-out test split (from training):**")
    st.dataframe(REF_METRICS.set_index("ML Model Name").style.format("{:.4f}"),
                 width="stretch")
