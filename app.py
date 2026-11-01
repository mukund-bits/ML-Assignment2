import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.stApp, header[data-testid="stHeader"] {background:#F5F5F0;}
.stApp p, .stApp label, .stApp h4 {color:#2b2118; font-family:Georgia, serif;}
[data-testid="stFileUploaderDropzone"], [data-testid="stSelectbox"] div {background-color:#f3e3cc !important;}
.stApp button {border-color:#c2410c;}
.topbox {background:#D96868; color:#fdf6ec; padding:22px 26px; border-radius:4px;
         border-left:10px solid #f59e0b; margin-bottom:18px;}
.topbox h1 {color:#1C352D; margin:0; font-size:30px;}
.topbox p {margin:4px 0 0 0; font-size:15px; opacity:0.9;}
.card {background:#65D0F4; border:2px dashed #c2410c; border-radius:4px;
       padding:12px 6px; text-align:center; margin-bottom:10px;}
.card .lbl {font-size:13px; letter-spacing:1px; text-transform:uppercase;}
.card .val {font-size:26px; font-weight:bold; color:#7c2d12;}
.stTabs [data-baseweb="tab-list"] {background:#FFF8B0; padding:4px 8px; border-radius:4px;}
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p {color:#dc2626 !important; font-family:'Georgia', serif;}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p {color:#dc2626 !important; font-weight:bold;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#FFF8B0;}
</style>
<div class="topbox">
  <h1>🩺 Breast Cancer Classifier</h1>
  <p>Upload test data &rarr; pick a model &rarr; see how it does (M = 1, B = 0)</p>
</div>
""", unsafe_allow_html=True)

mfiles = {
    "Logistic Regression": "model/logreg.pkl",
    "Decision Tree": "model/dtree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/nbayes.pkl",
    "SVM": "model/svm.pkl",
}


def get_scores(m, xs, yt):
    pr = m.predict(xs)
    pb = m.predict_proba(xs)[:, 1]
    sc = {
        "Accuracy": accuracy_score(yt, pr),
        "AUC": roc_auc_score(yt, pb),
        "Precision": precision_score(yt, pr),
        "Recall": recall_score(yt, pr),
        "F1": f1_score(yt, pr),
        "MCC": matthews_corrcoef(yt, pr),
    }
    return sc, pr


st.markdown("#### 1️⃣ Upload test CSV")
upf = st.file_uploader("CSV with all 30 features and the 'diagnosis' column", type=["csv"])

if upf is not None:
    df = pd.read_csv(upf)
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    yt = df["diagnosis"]
    xt = df.drop(columns=["diagnosis"])

    scl = joblib.load("model/scaler.pkl")
    xs = scl.transform(xt)

    st.caption(f"Loaded {df.shape[0]} rows")

    t1, t2 = st.tabs(["🔍 Single Model", "📊 All Models"])

    with t1:
        st.markdown("#### 2️⃣ Choose a model")
        pick = st.selectbox("Model", list(mfiles.keys()), label_visibility="collapsed")
        mm = joblib.load(mfiles[pick])
        sc, pr = get_scores(mm, xs, yt)

        st.markdown("#### 3️⃣ Evaluation metrics")
        cc = st.columns(3)
        for i, (k, v) in enumerate(sc.items()):
            cc[i % 3].markdown(f"<div class='card'><div class='lbl'>{k}</div>"
                               f"<div class='val'>{v:.4f}</div></div>", unsafe_allow_html=True)

        st.markdown("#### 4️⃣ Confusion matrix")
        cm = confusion_matrix(yt, pr)
        fig, ax = plt.subplots(figsize=(4, 3.2))
        fig.patch.set_facecolor("#fdf6ec")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("#### 5️⃣ Classification report")
        rep = classification_report(yt, pr, target_names=["Benign", "Malignant"], output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

    with t2:
        st.markdown("#### All 5 models on this test data")
        allr = []
        for nm, fp in mfiles.items():
            s, _ = get_scores(joblib.load(fp), xs, yt)
            allr.append({"Model": nm, **s})
        st.dataframe(pd.DataFrame(allr).set_index("Model").round(4), width="stretch")
else:
    st.info("👆 Upload test_data.csv to get started")
