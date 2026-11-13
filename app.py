"""Heart Disease Predictive Analytics - Streamlit app.

Run:  streamlit run app.py
Needs the `artifacts/` folder produced by the notebook (unzip artifacts.zip next to this file).
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

ART = "artifacts"
st.set_page_config(page_title="Heart Disease Predictive Analytics", page_icon="❤️", layout="wide")


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ART, "best_heart_disease_model.joblib"))


@st.cache_data
def load_json(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)


@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(ART, name))


def show_image(name, caption=None):
    path = os.path.join(ART, name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"{name} not found in the artifacts folder.")


if not os.path.isdir(ART) or not os.path.exists(os.path.join(ART, "feature_info.json")):
    st.error("The `artifacts/` folder was not found. Run the notebook, download `artifacts.zip`, "
             "unzip it next to `app.py`, then restart the app.")
    st.stop()

info = load_json("feature_info.json")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Heart Disease Analytics")
page = st.sidebar.radio("Go to", [
    "Predict",
    "Model comparison",
    "Regression & SVM kernels",
    "EDA",
    "PCA & clustering",
    "About",
])
st.sidebar.caption("Synthetic Kaggle data - for academic use only, not medical advice.")

# ---------------------------------------------------------------- Predict
if page == "Predict":
    st.title("Heart disease risk prediction")
    st.caption(f"Model: **{info['best_model']}** (chosen by cross-validation)")

    values = {}
    cols = st.columns(3)
    i = 0
    for c in info["columns"]:
        with cols[i % 3]:
            if c in info["numeric"]:
                m = info["numeric"][c]
                is_int = float(m["min"]).is_integer() and float(m["max"]).is_integer()
                if is_int:
                    values[c] = st.number_input(c, min_value=int(m["min"]), max_value=int(m["max"]),
                                                value=int(m["median"]), step=1)
                else:
                    values[c] = st.number_input(c, min_value=float(m["min"]), max_value=float(m["max"]),
                                                value=float(m["median"]), step=0.1)
            else:
                opts = info["categorical"][c]["values"]
                mode = info["categorical"][c]["mode"]
                default = opts.index(mode) if mode in opts else 0
                values[c] = st.selectbox(c, opts, index=default)
        i += 1

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])[info["columns"]]
        model = load_model()
        pred = int(model.predict(row)[0])
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(row)[0, 1])
            st.metric("Probability of heart disease (Presence)", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
        if pred == 1:
            st.error("Prediction: **Presence** of heart disease")
        else:
            st.success("Prediction: **Absence** of heart disease")
        st.caption("Educational demo built on synthetic data. Not a diagnosis.")

# ---------------------------------------------------------------- Model comparison
elif page == "Model comparison":
    st.title("Supervised & ensemble model comparison")
    res = load_csv("model_comparison_results.csv")
    tab1, tab2, tab3 = st.tabs(["Full test split", "Sampled models (KNN / Linear SVM)", "Cross-validation"])
    with tab1:
        full = res[res["Group"] == "Full"].sort_values("F1", ascending=False)
        st.dataframe(full.drop(columns=["Group"]), use_container_width=True)
        st.bar_chart(full.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]])
        c1, c2 = st.columns(2)
        with c1:
            show_image("best_confusion_matrix.png", "Confusion matrix - best model")
        with c2:
            show_image("best_roc_curve.png", "ROC curve - best model")
        show_image("feature_importance.png")
    with tab2:
        st.warning("These models were trained/tested on samples, so do not rank them directly "
                   "against the full-split models.")
        st.dataframe(res[res["Group"] == "Sampled"].drop(columns=["Group"]), use_container_width=True)
    with tab3:
        st.write("Best model was chosen using 3-fold cross-validated ROC-AUC on training data.")
        st.dataframe(load_csv("cv_results.csv"), use_container_width=True)

# ---------------------------------------------------------------- Regression & SVM
elif page == "Regression & SVM kernels":
    st.title("Regression baselines and SVM kernels")
    st.subheader("Simple, Multiple, Lasso and Ridge regression")
    st.caption("The target is binary, so the 0/1 outcome is treated as numeric and thresholded at 0.5. "
               "These are educational baselines, not the preferred classifiers.")
    st.dataframe(load_csv("regression_results.csv"), use_container_width=True)
    st.subheader("SVM kernels (sampled)")
    kdf = load_csv("svm_kernel_results.csv")
    st.dataframe(kdf, use_container_width=True)
    st.bar_chart(kdf.set_index("Model")[["Accuracy", "F1", "ROC_AUC"]])

# ---------------------------------------------------------------- EDA
elif page == "EDA":
    st.title("Exploratory data analysis")
    for title, f in [("Target distribution", "eda_target.png"),
                     ("Predictor distributions", "eda_distributions.png"),
                     ("Correlation heatmap", "eda_correlation.png"),
                     ("Boxplots", "eda_boxplots.png")]:
        st.subheader(title)
        show_image(f)
    sv = os.path.join(ART, "sweetviz_report.html")
    if os.path.exists(sv):
        with open(sv, "rb") as fh:
            st.download_button("Download Sweetviz report", fh, file_name="sweetviz_report.html")

# ---------------------------------------------------------------- PCA & clustering
elif page == "PCA & clustering":
    st.title("PCA, K-Means and DBSCAN")
    st.subheader("PCA - explained variance")
    pv = load_csv("pca_variance.csv")
    st.line_chart(pv.set_index("Component")["Cumulative"])
    st.caption(f"{len(pv)} components retain at least 95% of the variance.")

    st.subheader("K-Means")
    ks = load_csv("kmeans_scores.csv")
    c1, c2 = st.columns(2)
    c1.line_chart(ks.set_index("k")["inertia"])
    c2.line_chart(ks.set_index("k")["silhouette"])
    show_image("kmeans_clusters.png")

    st.subheader("DBSCAN")
    show_image("dbscan_clusters.png")
    pts = load_csv("cluster_points.csv")
    n_noise = int((pts["DBSCAN"] == -1).sum())
    n_clu = pts.loc[pts["DBSCAN"] != -1, "DBSCAN"].nunique()
    st.write(f"Clusters found: **{n_clu}** | noise points: **{n_noise}** ({n_noise / len(pts):.1%})")
    st.dataframe(pd.crosstab(pts["KMeans"], pts["Actual_target"]), use_container_width=True)

# ---------------------------------------------------------------- About
else:
    st.title("About this project")
    st.markdown("""
**Methods covered**

| Group | Methods |
|---|---|
| Supervised | Simple & Multiple Linear Regression, Lasso, Logistic Regression, KNN, SVM, Decision Tree |
| Ensemble | Bagging, Random Forest, Boosting |
| Unsupervised | K-Means, DBSCAN |
| Dimensionality reduction | PCA |

**Data:** Kaggle Playground Series S6E2 (synthetic heart-disease data).
**Disclaimer:** results are for learning only and are not clinical advice.
""")
    st.caption(f"Model trained with scikit-learn {info.get('sklearn_version', '?')}")
