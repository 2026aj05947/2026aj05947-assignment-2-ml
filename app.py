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
CLASS_NAMES = ["Not spam", "Spam"]
ACCENT = "#2f4a6d"

st.set_page_config(page_title="Spam Sentinel | ML Assignment 2", layout="wide",
                   initial_sidebar_state="collapsed")
sns.set_theme(style="whitegrid", font_scale=0.8)

st.markdown(
    """
    <style>
      header[data-testid="stHeader"] {height: 0; background: transparent;}
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px;}
      div[data-testid="stVerticalBlock"] {gap: .6rem;}
      h1 {font-size: 1.9rem !important; padding: 0 !important; margin: 0 !important;}
      .label {font-size:.92rem; font-weight:600; color:#1f2937; margin:.3rem 0 .1rem 0;}
      .meta-row {display:flex; flex-wrap:wrap; gap:.5rem 2.4rem; margin:.9rem 0 0 0;
                 font-size:.82rem; color:#4b5563;}
      .meta-row div span {display:block; font-size:.64rem; letter-spacing:.06em;
                 text-transform:uppercase; color:#9ca3af; margin-bottom:.15rem;}
      .meta-row div b {font-weight:600; color:#1f2937;}
      .subtle {font-size:.85rem !important; color:#6b7280; margin:.15rem 0 .9rem 0; line-height:1.4;}
      .panel {background:#f6f7f9; border:1px solid #e5e7eb; border-radius:10px;
              padding:.25rem .9rem .1rem .9rem;}
      .strip {display:grid; grid-template-columns: repeat(6, 1fr); border:1px solid #e5e7eb;
              border-radius:10px; overflow:hidden; margin:.2rem 0 .4rem 0;}
      .strip div {padding:.55rem .8rem; border-right:1px solid #e5e7eb;}
      .strip div:last-child {border-right:none;}
      .strip span {display:block; font-size:.68rem; letter-spacing:.05em;
                   text-transform:uppercase; color:#6b7280;}
      .strip b {font-size:1.25rem; font-weight:600; color:#1f2937;}
      .status {font-size:.82rem; color:#4b5563; margin:.1rem 0 .3rem 0;}
      .status b {color:#1f2937;}
      div[data-testid="stFileUploader"] section {padding: .5rem .7rem;}
      button[data-baseweb="tab"] {padding-top: .3rem; padding-bottom: .3rem;}
      .st-key-header_box, .st-key-controls_box {border: 1.5px solid #111827 !important;
                          border-radius: 12px; padding: 1rem 1.4rem .9rem 1.4rem;}
      .st-key-header_box {padding-bottom: 1.25rem !important;}
      .st-key-header_box .subtle {margin: .55rem 0 .35rem 0 !important; padding-top: .6rem;
                                  border-top: 1px solid #e5e7eb;}
      .st-key-controls_box {margin-bottom: .4rem;}
    </style>
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


def metrics_at(y, y_prob, threshold=0.5):
    # strict ">" so 0.50 reproduces model.predict() (ties go to class 0)
    y_pred = (y_prob > threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if y.nunique() == 2 else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


try:
    META, MODELS, REF_METRICS, BEST_PARAMS = load_artifacts()
except Exception as exc:  # pragma: no cover - shown in the UI
    st.error(f"Could not load the saved models from `model/`: {exc}")
    st.stop()

FEATURES, TARGET = META["features"], META["target"]

# ---------------------------------------------------------------- header
with st.container(key="header_box"):
    st.title("Spam Sentinel")
    st.markdown(
        """
        <div class="meta-row">
          <div><span>Course</span><b>AIMLZG565</b> · Machine Learning</div>
          <div><span>Assignment</span><b>Assignment 2</b></div>
          <div><span>Name</span><b>Raghu Prashanka</b></div>
          <div><span>Student ID</span><b>2026AJ05947</b></div>
        </div>
        <div class="subtle">Spam e-mail classification on the UCI Spambase dataset with five pre-trained
        models. Upload a labelled test CSV to evaluate them &mdash; nothing is retrained here.</div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- controls
with st.container(key="controls_box"):
    c_data, c_model, c_thr = st.columns([1.5, 1, 1], gap="medium")
    with c_data:
        uploaded = st.file_uploader("1. Test data (CSV with 57 features + `is_spam`)", type=["csv"])
        s1, s2 = st.columns([1.15, 1])
        use_sample = s1.toggle("Use bundled test_data.csv", value=False,
                               disabled=uploaded is not None or not SAMPLE_CSV.exists())
        if SAMPLE_CSV.exists():
            s2.download_button("Download sample CSV", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv", width="stretch")
    with c_model:
        model_name = st.selectbox("2. Model", list(MODELS.keys()),
                                  index=list(MODELS.keys()).index("Support Vector Machine"))
        st.caption("Tuned: " + ", ".join(f"{k} = {v}" for k, v in BEST_PARAMS[model_name].items()))
    with c_thr:
        threshold = st.slider("3. Spam threshold", 0.05, 0.95, 0.50, 0.05,
                              help="0.50 reproduces the README numbers. Lower it to catch more "
                                   "spam, raise it to flag fewer genuine e-mails.")
        st.caption(f"Models trained with scikit-learn {META['scikit_learn']} · seed {META['seed']}")

# ---------------------------------------------------------------- data
if uploaded is not None:
    df, source = pd.read_csv(uploaded), uploaded.name
elif use_sample:
    df, source = pd.read_csv(SAMPLE_CSV), "bundled test_data.csv"
else:
    st.info("Upload a test CSV or switch on *Use bundled test_data.csv* to see the evaluation. "
            "Reference results from training are shown below.")
    st.dataframe(REF_METRICS.set_index("ML Model Name").style.format("{:.4f}")
                 .highlight_max(axis=0, color="#e5e7eb"), width="stretch")
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
X, y = df[FEATURES].astype(float), df[TARGET].astype(int)

st.markdown(
    f'<p class="status">Evaluating <b>{len(df):,}</b> rows from <b>{source}</b> · '
    f'<b>{y.mean():.1%}</b> spam · threshold <b>{threshold:.2f}</b></p>',
    unsafe_allow_html=True,
)

tab_model, tab_compare, tab_data, tab_about = st.tabs(
    ["Selected model", "Compare all models", "Predictions", "About"])

# ---------------------------------------------------------------- selected model
y_prob = MODELS[model_name].predict_proba(X)[:, 1]
sel, y_pred = metrics_at(y, y_prob, threshold)

with tab_model:
    st.markdown(
        '<div class="strip">' + "".join(
            f'<div><span>{k}</span><b>{"n/a" if pd.isna(v) else f"{v:.4f}"}</b></div>'
            for k, v in sel.items()) + "</div>",
        unsafe_allow_html=True,
    )

    g1, g2, g3 = st.columns([1, 1, 1.15], gap="medium")
    with g1:
        st.markdown('<div class="label">Confusion matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", cbar=False, linewidths=0.5,
                    linecolor="#d1d5db", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, annot_kws={"size": 11})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"{fp} genuine flagged as spam · {fn} spam missed")
    with g2:
        st.markdown('<div class="label">ROC curve</div>', unsafe_allow_html=True)
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, y_prob)
            fig, ax = plt.subplots(figsize=(3.4, 2.9))
            ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {sel['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
        else:
            st.info("ROC needs both classes in the uploaded data.")
    with g3:
        st.markdown('<div class="label">Classification report</div>', unsafe_allow_html=True)
        rep = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).T.drop(index="accuracy", errors="ignore")
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                          "f1-score": "{:.3f}", "support": "{:d}"}),
                     width="stretch", height=178)

# ---------------------------------------------------------------- comparison
with tab_compare:
    rows = []
    for name, m in MODELS.items():
        scores, _ = metrics_at(y, m.predict_proba(X)[:, 1], 0.5)
        rows.append({"ML Model Name": name, **scores})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    best = comp["MCC"].idxmax()

    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        st.markdown(f'<div class="label">All models at threshold 0.50 &nbsp;·&nbsp; '
                    f'<span style="font-weight:400;color:#4b5563">best by MCC:</span> {best}</div>',
                    unsafe_allow_html=True)
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#e5e7eb"),
                     width="stretch")
        st.download_button("Download comparison table", comp.round(4).to_csv().encode(),
                           file_name="model_comparison.csv", mime="text/csv")
    with right:
        long = comp.reset_index().melt(id_vars="ML Model Name", var_name="Metric",
                                       value_name="Score")
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name",
                    palette=sns.light_palette(ACCENT, n_colors=6)[1:], ax=ax)
        ax.set_ylim(max(0, long["Score"].min() - 0.05), 1)
        ax.set_xlabel("")
        ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22),
                  frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

# ---------------------------------------------------------------- predictions
with tab_data:
    out = df.copy()
    out["predicted"] = y_pred
    out["spam_probability"] = y_prob.round(4)
    out["correct"] = out["predicted"] == out[TARGET]
    p1, p2 = st.columns([1, 1])
    only_wrong = p1.toggle(f"Show misclassified only ({int((~out['correct']).sum())} rows)")
    p2.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")
    view = out[~out["correct"]] if only_wrong else out
    st.dataframe(view[[TARGET, "predicted", "spam_probability", "correct"] + FEATURES].head(500),
                 width="stretch", height=340)

# ---------------------------------------------------------------- about
with tab_about:
    a1, a2 = st.columns([1, 1], gap="medium")
    with a1:
        st.markdown(
            f"""
**Dataset** — UCI Spambase (id 94): 4,601 e-mails, 57 numeric features
(48 word frequencies, 6 character frequencies, 3 capital-letter run-length statistics).
Exact duplicates removed ({META['n_rows_after_dedup']:,} rows kept), then a stratified 80/20 split.

**Pre-processing** — `log(1+x)` + standard scaling inside the pipelines for Logistic Regression,
KNN, Naive Bayes and SVM; the Decision Tree uses raw features. Hyper-parameters chosen by
5-fold stratified CV (F1) on the training split only.

**Positive class** for Precision / Recall / F1 is *spam* (1).
"""
        )
    with a2:
        st.markdown("**Reference metrics on the held-out test split (from training)**")
        st.dataframe(REF_METRICS.set_index("ML Model Name").style.format("{:.4f}"),
                     width="stretch")
