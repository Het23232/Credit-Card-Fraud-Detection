"""
Credit Card Fraud Detection - Streamlit app
Run:  streamlit run app.py

Dataset: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud)
Columns: Time, V1..V28 (PCA features), Amount, Class (1 = fraud)
Place creditcard.csv next to this file, or upload it from the sidebar.
No dataset? Use the built-in synthetic demo data.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

LOCAL_FILE = Path(__file__).parent / "creditcard.csv"
SEED = 42

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading data...")
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data(show_spinner="Generating demo data...")
def make_demo(n: int = 60000, fraud_rate: float = 0.004, seed: int = SEED) -> pd.DataFrame:
    """Synthetic data shaped like the Kaggle dataset (for demo only)."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n,
        n_features=28,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[1 - fraud_rate],
        flip_y=0.002,
        class_sep=1.2,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    amt = rng.lognormal(3.2, 1.2, n)
    amt[y == 1] = rng.lognormal(3.8, 1.5, int((y == 1).sum()))
    df["Amount"] = amt.round(2)
    df["Class"] = y
    return df


def prep(df: pd.DataFrame):
    """Feature engineering: Time -> Hour of day, Amount -> log(1+Amount)."""
    X = df.drop(columns=["Class"], errors="ignore").copy()
    X["Hour"] = (X["Time"] // 3600) % 24
    X["Amount_log"] = np.log1p(X["Amount"])
    X = X.drop(columns=["Time", "Amount"])
    y = df["Class"].astype(int) if "Class" in df.columns else None
    return X, y


def stratified_fraction(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    parts = [g.sample(frac=frac, random_state=SEED) for _, g in df.groupby("Class")]
    return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------


def apply_imbalance(strategy, X, y):
    if strategy == "Random undersampling (10:1)":
        rng = np.random.default_rng(SEED)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        keep_neg = rng.choice(neg, size=min(len(neg), len(pos) * 10), replace=False)
        idx = np.concatenate([pos, keep_neg])
        rng.shuffle(idx)
        return X.iloc[idx], y.iloc[idx]
    if strategy == "SMOTE oversampling":
        return SMOTE(random_state=SEED).fit_resample(X, y)
    return X, y


def build_model(name, strategy, n_trees, max_depth, contamination):
    weighted = strategy == "Class weights"
    if name == "Logistic Regression":
        return "supervised", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced" if weighted else None),
        )
    if name == "Random Forest":
        return "supervised", RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=SEED,
            class_weight="balanced_subsample" if weighted else None,
        )
    return "anomaly", make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=n_trees, contamination=contamination, n_jobs=-1, random_state=SEED
        ),
    )


def score_samples(res, X):
    X = X.reindex(columns=res["features"])
    if res["kind"] == "anomaly":
        s = -res["model"].decision_function(X)
        return np.clip((s - res["lo"]) / (res["hi"] - res["lo"] + 1e-12), 0, 1)
    return res["model"].predict_proba(X)[:, 1]


def train_model(df, name, strategy, test_size, n_trees, max_depth):
    X, y = prep(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    amount_te = df.loc[X_te.index, "Amount"].to_numpy()
    kind, model = build_model(name, strategy, n_trees, max_depth, max(float(y.mean()), 1e-4))

    t0 = time.time()
    if kind == "anomaly":
        model.fit(X_tr)  # unsupervised: labels are NOT used for training
        tr_scores = -model.decision_function(X_tr)
        lo, hi = float(tr_scores.min()), float(tr_scores.max())
    else:
        Xf, yf = apply_imbalance(strategy, X_tr, y_tr)
        model.fit(Xf, yf)
        lo = hi = None
    res = dict(
        model=model, kind=kind, name=name, strategy=strategy, features=list(X.columns),
        lo=lo, hi=hi, train_time=time.time() - t0, X_test=X_te, y_test=y_te.to_numpy(),
        amount_test=amount_te, test_index=X_te.index.to_numpy(),
    )
    res["scores"] = score_samples(res, X_te)
    res["run_id"] = time.time_ns()
    return res


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(np.clip(t[np.argmax(f1)], 0.01, 0.99))


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    return dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        mcc=matthews_corrcoef(y, pred),
        cm=confusion_matrix(y, pred, labels=[0, 1]),
        pred=pred,
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("💳 Fraud Detection")
st.sidebar.header("1. Data")
options = ["Upload creditcard.csv", "Synthetic demo data"]
if LOCAL_FILE.exists():
    options.insert(0, "Local creditcard.csv")
source = st.sidebar.radio("Data source", options)

df = None
if source == "Local creditcard.csv":
    df = load_csv(LOCAL_FILE)
elif source == "Upload creditcard.csv":
    up = st.sidebar.file_uploader("Kaggle creditcard.csv", type="csv")
    if up is not None:
        df = load_csv(up)
else:
    df = make_demo()
    st.sidebar.caption("Synthetic data - for demonstrating the app only.")

if df is not None:
    missing = {"Time", "Amount", "Class"} - set(df.columns)
    if missing:
        st.error(f"Dataset is missing required columns: {sorted(missing)}")
        st.stop()
    frac = st.sidebar.slider("Use fraction of data (faster runs)", 0.1, 1.0, 1.0, 0.1)
    df = stratified_fraction(df, frac)

st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox(
    "Algorithm", ["Random Forest", "Logistic Regression", "Isolation Forest (anomaly detection)"]
)
model_name = model_name.replace(" (anomaly detection)", "")
is_anomaly = model_name == "Isolation Forest"

strategies = ["Class weights", "Random undersampling (10:1)", "None"]
if HAS_SMOTE:
    strategies.insert(2, "SMOTE oversampling")
strategy = st.sidebar.selectbox(
    "Imbalance handling", strategies, disabled=is_anomaly,
    help="Isolation Forest is unsupervised, so this does not apply to it.",
)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.25, 0.05)
n_trees, max_depth = 100, 12
if model_name != "Logistic Regression":
    n_trees = st.sidebar.slider("Number of trees", 50, 300, 100, 50)
if model_name == "Random Forest":
    max_depth = st.sidebar.slider("Max depth", 4, 30, 12)

train_clicked = st.sidebar.button("🚀 Train model", type="primary", disabled=df is None)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

st.title("Credit Card Fraud Detection")
st.caption("Extreme class imbalance · precision-recall optimisation · anomaly detection")

if df is None:
    st.info("👈 Upload the Kaggle `creditcard.csv` in the sidebar, or choose the synthetic demo data.")
    st.stop()

if train_clicked:
    with st.spinner(f"Training {model_name}..."):
        st.session_state["res"] = train_model(
            df, model_name, "None" if is_anomaly else strategy, test_size, n_trees, max_depth
        )

tab_eda, tab_eval, tab_cost, tab_try = st.tabs(
    ["📊 Data & imbalance", "🎯 Model evaluation", "💰 Business cost", "🔍 Try transactions"]
)

# ---- EDA ---------------------------------------------------------------------
with tab_eda:
    n_fraud = int(df["Class"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud cases", f"{n_fraud:,}")
    c3.metric("Fraud rate", f"{n_fraud / len(df):.3%}")
    c4.metric("Imbalance ratio", f"1 : {int((len(df) - n_fraud) / max(n_fraud, 1)):,}")

    st.warning(
        f"**Accuracy paradox:** a model that labels *every* transaction as legitimate is "
        f"**{1 - n_fraud / len(df):.2%} accurate** yet catches zero fraud. That is why this project "
        "uses precision, recall, PR-AUC and MCC instead of accuracy."
    )

    left, right = st.columns(2)
    counts = df["Class"].map({0: "Legitimate", 1: "Fraud"}).value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    fig = px.bar(counts, x="Class", y="Count", log_y=True, color="Class", text="Count",
                 title="Class distribution (log scale)",
                 color_discrete_map={"Legitimate": "#2a9d8f", "Fraud": "#e63946"})
    left.plotly_chart(fig, width="stretch")

    d = df.copy()
    d["Label"] = d["Class"].map({0: "Legitimate", 1: "Fraud"})
    fig = px.box(d, x="Label", y="Amount", color="Label", log_y=True, points=False,
                 title="Transaction amount by class (log scale)",
                 color_discrete_map={"Legitimate": "#2a9d8f", "Fraud": "#e63946"})
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    d["Hour"] = ((d["Time"] // 3600) % 24).astype(int)
    hourly = d.groupby("Hour")["Class"].agg(["mean", "sum"]).reset_index()
    fig = px.bar(hourly, x="Hour", y="mean", title="Fraud rate by hour of day",
                 labels={"mean": "Fraud rate"}, hover_data=["sum"])
    left.plotly_chart(fig, width="stretch")

    vcols = [c for c in df.columns if c.startswith("V")]
    corr = df[vcols + ["Amount"]].corrwith(df["Class"]).sort_values(key=abs, ascending=False).head(12)
    fig = px.bar(corr[::-1], orientation="h", title="Top features correlated with fraud",
                 labels={"value": "Correlation with Class", "index": "Feature"})
    fig.update_layout(showlegend=False)
    right.plotly_chart(fig, width="stretch")

    with st.expander("Preview data"):
        st.dataframe(df.head(100), width="stretch")

# ---- Evaluation --------------------------------------------------------------
res = st.session_state.get("res")

with tab_eval:
    if res is None:
        st.info("Pick a model in the sidebar and click **Train model**.")
    else:
        y_te, s = res["y_test"], res["scores"]
        ap, auc = average_precision_score(y_te, s), roc_auc_score(y_te, s)
        label = res["name"] + ("" if res["kind"] == "anomaly" else f" · {res['strategy']}")
        st.subheader(label)
        st.caption(f"Trained in {res['train_time']:.1f}s · test set: {len(y_te):,} transactions, "
                   f"{int(y_te.sum())} frauds"
                   + (" · trained WITHOUT labels (unsupervised)" if res["kind"] == "anomaly" else ""))

        default_thr = best_f1_threshold(y_te, s)
        thr = st.slider("Decision threshold", 0.01, 0.99, default_thr, 0.01, key=f"thr_{res['run_id']}",
                        help="Default = threshold that maximises F1. Lower it to catch more fraud.")
        m = metrics_at(y_te, s, thr)

        c = st.columns(6)
        c[0].metric("PR-AUC", f"{ap:.3f}", help="Average precision - the headline metric for imbalanced data")
        c[1].metric("ROC-AUC", f"{auc:.3f}")
        c[2].metric("Precision", f"{m['precision']:.3f}")
        c[3].metric("Recall", f"{m['recall']:.3f}")
        c[4].metric("F1", f"{m['f1']:.3f}")
        c[5].metric("MCC", f"{m['mcc']:.3f}")

        left, right = st.columns(2)
        p, r, t = precision_recall_curve(y_te, s)
        fig = go.Figure()
        fig.add_scatter(x=r, y=p, mode="lines", name="PR curve", line=dict(color="#e63946"))
        fig.add_scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="Current threshold",
                        marker=dict(size=12, color="black"))
        fig.add_hline(y=y_te.mean(), line_dash="dot", annotation_text="No-skill baseline")
        fig.update_layout(title=f"Precision-Recall curve (AP = {ap:.3f})",
                          xaxis_title="Recall", yaxis_title="Precision")
        left.plotly_chart(fig, width="stretch")

        fpr, tpr, _ = roc_curve(y_te, s)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#457b9d"))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dot"))
        fig.update_layout(title=f"ROC curve (AUC = {auc:.3f})",
                          xaxis_title="False positive rate", yaxis_title="True positive rate")
        right.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        cm = m["cm"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"],
                        title="Confusion matrix")
        left.plotly_chart(fig, width="stretch")
        with right:
            st.markdown(
                f"**Reading the matrix at threshold {thr:.2f}**\n\n"
                f"- ✅ Frauds caught: **{cm[1,1]}** of {cm[1].sum()}\n"
                f"- ❌ Frauds missed (false negatives): **{cm[1,0]}**\n"
                f"- ⚠️ Legit flagged as fraud (false positives): **{cm[0,1]}**\n"
                f"- Analysts must review **{cm[0,1] + cm[1,1]}** alerts to catch {cm[1,1]} frauds."
            )

        # score distribution
        fig = px.histogram(x=s, color=np.where(y_te == 1, "Fraud", "Legitimate"), nbins=60, log_y=True,
                           barmode="overlay", opacity=0.7, title="Score distribution by true class",
                           labels={"x": "Fraud score", "color": "Class"},
                           color_discrete_map={"Legitimate": "#2a9d8f", "Fraud": "#e63946"})
        fig.add_vline(x=thr, line_dash="dash")
        st.plotly_chart(fig, width="stretch")

        # feature importance
        mdl = res["model"]
        imp = None
        if isinstance(mdl, RandomForestClassifier):
            imp = pd.Series(mdl.feature_importances_, index=res["features"])
        elif res["name"] == "Logistic Regression":
            imp = pd.Series(np.abs(mdl[-1].coef_[0]), index=res["features"])
        if imp is not None:
            with st.expander("Feature importance"):
                top = imp.sort_values().tail(15)
                st.plotly_chart(px.bar(top, orientation="h", labels={"value": "Importance", "index": ""}),
                                width="stretch")

# ---- Business cost -----------------------------------------------------------
with tab_cost:
    if res is None:
        st.info("Train a model first.")
    else:
        st.markdown(
            "In real fraud systems the two errors do not cost the same. "
            "A **missed fraud** costs the transaction amount; a **false alarm** costs analyst time / customer friction."
        )
        fp_cost = st.number_input("Cost of one false alarm (review / customer friction)", 0.0, 1000.0, 5.0, 1.0)
        y_te, s, amt = res["y_test"], res["scores"], res["amount_test"]
        grid = np.linspace(0.01, 0.99, 99)
        costs = []
        for t_ in grid:
            pred = s >= t_
            fn_loss = amt[(y_te == 1) & ~pred].sum()
            fp_loss = ((y_te == 0) & pred).sum() * fp_cost
            costs.append(fn_loss + fp_loss)
        costs = np.array(costs)
        best_i = int(np.argmin(costs))
        no_model = amt[y_te == 1].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Cost with no model", f"{no_model:,.0f}")
        c2.metric("Cost at best threshold", f"{costs[best_i]:,.0f}", f"-{(1 - costs[best_i] / no_model):.1%}")
        c3.metric("Cost-optimal threshold", f"{grid[best_i]:.2f}")
        fig = px.line(x=grid, y=costs, labels={"x": "Threshold", "y": "Total cost"},
                      title="Total cost vs decision threshold (test set)")
        fig.add_vline(x=grid[best_i], line_dash="dash", annotation_text="optimal")
        st.plotly_chart(fig, width="stretch")
        st.caption("The cost-optimal threshold is usually very different from the default 0.5 - "
                   "a key insight for imbalanced problems.")

# ---- Try transactions --------------------------------------------------------
with tab_try:
    if res is None:
        st.info("Train a model first.")
    else:
        st.subheader("Score a transaction from the test set")
        pick = st.radio("Pick a random…", ["Fraud transaction", "Legitimate transaction", "Any"], horizontal=True)
        if st.button("🎲 Pick random transaction"):
            y_te = res["y_test"]
            pool = np.where(y_te == 1)[0] if pick == "Fraud transaction" else (
                np.where(y_te == 0)[0] if pick == "Legitimate transaction" else np.arange(len(y_te)))
            st.session_state["pick_i"] = int(np.random.choice(pool))
        i = st.session_state.get("pick_i")
        if i is not None and i < len(res["y_test"]):
            row = df.loc[res["test_index"][i]]
            sc = float(res["scores"][i])
            thr_used = st.session_state.get(f"thr_{res['run_id']}", 0.5)
            flagged = sc >= thr_used
            a, b = st.columns(2)
            a.metric("Fraud score", f"{sc:.3f}", help=f"Threshold in use: {thr_used:.2f}")
            a.markdown(f"### {'🚨 FLAGGED as fraud' if flagged else '✅ Looks legitimate'}")
            b.markdown(f"**Actual label:** {'Fraud' if res['y_test'][i] == 1 else 'Legitimate'}  \n"
                       f"**Amount:** {row['Amount']:,.2f}  \n"
                       f"**Hour of day:** {int((row['Time'] // 3600) % 24)}")
            correct = flagged == bool(res["y_test"][i])
            (st.success if correct else st.error)("Model is correct ✔" if correct else "Model is wrong ✘")
            with st.expander("All features"):
                st.dataframe(row.to_frame("value"))

        st.divider()
        st.subheader("Batch-score your own file")
        up2 = st.file_uploader("CSV with the same columns (Class optional)", type="csv", key="batch")
        if up2 is not None:
            new = pd.read_csv(up2)
            if {"Time", "Amount"} - set(new.columns):
                st.error("File needs at least Time, V1..V28 and Amount columns.")
            else:
                Xn, _ = prep(new)
                new["fraud_score"] = score_samples(res, Xn)
                new["flagged"] = new["fraud_score"] >= st.session_state.get(f"thr_{res['run_id']}", 0.5)
                st.write(f"Flagged **{int(new['flagged'].sum())}** of {len(new)} transactions.")
                st.dataframe(new.sort_values("fraud_score", ascending=False).head(200), width="stretch")
                st.download_button("Download scored CSV", new.to_csv(index=False), "scored_transactions.csv")
