import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bank Marketing Dashboard", layout="wide")

# ===== Sidebar Menu =====
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Overview", "Clustering", "Model Performance", "Feature Importance", "Customer Targeting"]
)

# Load Data
df_clustered = pd.read_csv("data/dataset_clustered.csv")
cluster_summary = pd.read_csv("data/cluster_summary.csv")
model_perf = pd.read_csv("data/model_performance.csv")
feat_imp = pd.read_csv("data/feature_importance.csv")
df_pred = pd.read_csv("data/predicted_customers.csv")

# ==========================================================
# PAGE 1 — OVERVIEW
# ==========================================================
if menu == "Overview":
    st.title(" Overview Dataset")
    st.write("Tampilan awal dari dataset yang sudah dibersihkan.")
    st.dataframe(df_clustered.head())

    st.subheader("Distribusi Target (Yes/No)")
    fig = px.histogram(df_pred, x="predicted", color="predicted")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGE 2 — CLUSTERING
# ==========================================================
elif menu == "Clustering":
    st.title("🔵 Customer Segmentation (Clustering)")

    st.subheader("Profil Tiap Cluster")
    st.dataframe(cluster_summary)

    st.subheader("Visualisasi PCA 2D (Cluster)")
    fig2 = px.scatter(df_clustered, x="PC1", y="PC2", color="cluster")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# PAGE 3 — MODEL PERFORMANCE
# ==========================================================
elif menu == "Model Performance":
    st.title("📊 Perbandingan Performa Model")

    st.dataframe(model_perf)

    fig = px.bar(model_perf, x="model", y="f1_yes", title="F1 Score per Model")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGE 4 — FEATURE IMPORTANCE
# ==========================================================
elif menu == "Feature Importance":
    st.title("⭐ Feature Importance")

    fig = px.bar(feat_imp.head(20), x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGE 5 — CUSTOMER TARGETING
# ==========================================================
elif menu == "Customer Targeting":
    st.title("🎯 Customer Targeting")

    threshold = st.slider("Minimum Probabilitas YES", 0.0, 1.0, 0.5, 0.01)

    targeted = df_pred[df_pred['prob_yes'] >= threshold]

    st.write(f"Jumlah nasabah potensial: {len(targeted)}")
    st.dataframe(targeted[['age', 'job', 'balance', 'cluster', 'predicted', 'prob_yes']])
