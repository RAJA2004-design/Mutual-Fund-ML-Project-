"""
Master Execution Script: Machine Learning-Based Risk-Return Profiling and Clustering
Orchestrates data inspection, EDA, clustering (K-Means, Hierarchical, DBSCAN, PCA),
evaluation, profiling, figure generation, and table exports.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from ml_pipeline.data_loader import MutualFundDataLoader, CORE_FEATURES, EXTENDED_FEATURES
from ml_pipeline.eda import ExploratoryDataAnalysis
from ml_pipeline.clustering import ClusterEngine
from ml_pipeline.profiling import ClusterProfiler
from ml_pipeline.visualization import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("MLAnalysisRunner")

# Directory setup
RESULTS_DIR = os.path.join(BASE_DIR, "data", "ml_results")
TABLES_DIR = os.path.join(BASE_DIR, "reports", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    logger.info("=================================================================")
    logger.info("STARTING ML RISK-RETURN PROFILING & CLUSTERING PIPELINE")
    logger.info("=================================================================")

    # -------------------------------------------------------------------------
    # STEP 1: DATA INSPECTION, PREPROCESSING & STANDARDIZATION
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 1: DATA INSPECTION & PREPROCESSING ---")
    loader = MutualFundDataLoader()
    clean_df = loader.load_data()
    
    # Audit data quality
    quality_audit = loader.audit_data_quality()
    with open(os.path.join(TABLES_DIR, "data_quality_audit.json"), "w") as f:
        json.dump(quality_audit, f, indent=2)

    # Detect outliers using IQR and Z-Score
    outlier_summary_df, outlier_flags_df = loader.detect_outliers(CORE_FEATURES)
    outlier_summary_df.to_csv(os.path.join(TABLES_DIR, "outliers_summary.csv"), index=False)
    logger.info(f"Outlier summary exported to {os.path.join(TABLES_DIR, 'outliers_summary.csv')}")

    # Standardize features
    unscaled_df, X_scaled, scaled_df = loader.get_feature_matrix(features=CORE_FEATURES, scale_method="standard")

    # -------------------------------------------------------------------------
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & CORRELATION ANALYSIS
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 2: EXPLORATORY DATA ANALYSIS ---")
    eda = ExploratoryDataAnalysis(clean_df, feature_cols=CORE_FEATURES)
    desc_stats_df = eda.compute_descriptive_statistics()
    desc_stats_df.to_csv(os.path.join(TABLES_DIR, "descriptive_statistics.csv"), index=False)

    pearson_corr, spearman_corr = eda.compute_correlations()
    pearson_corr.to_csv(os.path.join(TABLES_DIR, "correlation_pearson.csv"))
    spearman_corr.to_csv(os.path.join(TABLES_DIR, "correlation_spearman.csv"))

    cat_profiles_df = eda.compute_category_profiles()
    cat_profiles_df.to_csv(os.path.join(TABLES_DIR, "category_profiles_summary.csv"), index=False)

    top_sharpe, bottom_sharpe = eda.get_top_and_bottom_funds(metric="sharpe_ratio", top_n=5)
    top_sharpe.to_csv(os.path.join(TABLES_DIR, "top_5_sharpe_funds.csv"), index=False)
    bottom_sharpe.to_csv(os.path.join(TABLES_DIR, "bottom_5_sharpe_funds.csv"), index=False)

    # -------------------------------------------------------------------------
    # STEP 3: MACHINE LEARNING CLUSTERING & EVALUATION
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 3: MACHINE LEARNING CLUSTERING (K-MEANS, HIERARCHICAL, DBSCAN, PCA) ---")
    engine = ClusterEngine(X_scaled=X_scaled, feature_names=CORE_FEATURES, random_state=42)

    # 3.1 K-Means Evaluation for K = 2 to 8
    kmeans_eval_df = engine.evaluate_kmeans_range(k_min=2, k_max=8)
    kmeans_eval_df.to_csv(os.path.join(RESULTS_DIR, "kmeans_evaluation_metrics.csv"), index=False)
    kmeans_eval_df.to_csv(os.path.join(TABLES_DIR, "kmeans_evaluation_metrics.csv"), index=False)
    logger.info("K-Means evaluation metrics (K=2 to 8):\n" + kmeans_eval_df.to_string(index=False))

    # Optimal K selection: K = 4 provides clean financial taxonomy and strong validation
    OPTIMAL_K = 4
    kmeans_labels, centroids, kmeans_metrics = engine.fit_kmeans(optimal_k=OPTIMAL_K)

    # 3.2 Hierarchical Agglomerative Clustering
    linkage_matrix = engine.compute_hierarchical_linkage(method="ward")
    hier_labels, hier_metrics = engine.fit_hierarchical(n_clusters=OPTIMAL_K, linkage_method="ward")

    # 3.3 DBSCAN Clustering
    k_distances = engine.compute_k_distances(k_neighbors=4)
    # Epsilon = 1.45 based on 4-NN elbow inspection
    dbscan_labels, dbscan_metrics = engine.fit_dbscan(eps=1.45, min_samples=4)

    # 3.4 Principal Component Analysis (PCA)
    pca_model, X_pca, loadings_df, variance_df = engine.fit_pca(n_components=len(CORE_FEATURES))
    loadings_df.to_csv(os.path.join(RESULTS_DIR, "pca_loadings.csv"))
    loadings_df.to_csv(os.path.join(TABLES_DIR, "pca_loadings.csv"))
    variance_df.to_csv(os.path.join(RESULTS_DIR, "pca_explained_variance.csv"), index=False)
    variance_df.to_csv(os.path.join(TABLES_DIR, "pca_explained_variance.csv"), index=False)
    logger.info("PCA Explained Variance Summary:\n" + variance_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 4: CLUSTER PROFILING & FINANCIAL CHARACTERIZATION
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 4: CLUSTER PROFILING & FINANCIAL TAXONOMY ---")
    
    # Consolidate complete fund-level dataset with all cluster assignments and PCA coordinates
    clustered_df = clean_df.copy()
    clustered_df["kmeans_cluster"] = kmeans_labels
    clustered_df["hierarchical_cluster"] = hier_labels
    clustered_df["dbscan_cluster"] = dbscan_labels
    clustered_df["PC1"] = np.round(X_pca[:, 0], 4)
    clustered_df["PC2"] = np.round(X_pca[:, 1], 4)
    clustered_df["PC3"] = np.round(X_pca[:, 2], 4)

    profiler = ClusterProfiler(clustered_df, cluster_col="kmeans_cluster")
    cluster_labels_map, cluster_desc_map, profile_series = profiler.assign_financial_taxonomy()
    clustered_df["cluster_profile_label"] = profile_series

    # Save complete fund-level dataset
    clustered_csv_path = os.path.join(RESULTS_DIR, "clustered_funds_complete.csv")
    clustered_df.to_csv(clustered_csv_path, index=False)
    logger.info(f"Complete clustered dataset exported to {clustered_csv_path}")

    # Generate cluster summaries
    cluster_summary_df = profiler.generate_cluster_summary(numeric_features=CORE_FEATURES + ["sortino_ratio", "calmar_ratio"])
    cluster_summary_df["profile_label"] = cluster_summary_df["cluster"].map(cluster_labels_map)
    cluster_summary_df.to_csv(os.path.join(RESULTS_DIR, "cluster_profiles_summary.csv"), index=False)
    cluster_summary_df.to_csv(os.path.join(TABLES_DIR, "cluster_profiles_summary.csv"), index=False)

    category_crosstab_df = profiler.analyze_category_composition()
    category_crosstab_df.to_csv(os.path.join(TABLES_DIR, "cluster_category_crosstab.csv"))

    representatives = profiler.get_cluster_representatives(n_top=3)
    rep_records = []
    for c, df_rep in representatives.items():
        df_rep["cluster"] = c
        df_rep["cluster_label"] = cluster_labels_map.get(c, "")
        rep_records.append(df_rep)
    rep_df = pd.concat(rep_records, ignore_index=True)
    rep_df.to_csv(os.path.join(TABLES_DIR, "cluster_representative_funds.csv"), index=False)

    logger.info("Financial Taxonomy Assigned to Clusters:")
    for c, lbl in cluster_labels_map.items():
        logger.info(f"  Cluster {c} ({lbl}): {cluster_desc_map[c]}")

    # -------------------------------------------------------------------------
    # STEP 5: VISUALIZATIONS GENERATION (PUBLICATION QUALITY 300 DPI)
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 5: GENERATING HIGH-RESOLUTION CHARTS ---")
    viz = Visualizer(output_dir=FIGURES_DIR)

    # EDA Charts
    viz.plot_correlation_heatmap(pearson_corr, filename="eda_correlation_heatmap.png")
    viz.plot_feature_distributions(clean_df, CORE_FEATURES, filename="eda_feature_distributions.png")
    viz.plot_risk_return_scatter(clean_df, filename="eda_risk_return_scatter.png")
    viz.plot_category_boxplots(clean_df, filename="eda_boxplots_by_category.png")

    # Clustering Diagnostics & Evaluation Charts
    viz.plot_kmeans_evaluation(kmeans_eval_df, optimal_k=OPTIMAL_K, filename="kmeans_elbow_and_metrics.png")
    viz.plot_pca_scree(variance_df, filename="pca_scree_plot.png")
    viz.plot_pca_biplot(X_pca, loadings_df, labels=kmeans_labels, filename="pca_biplot.png")
    viz.plot_clusters_in_pca(X_pca, kmeans_labels, title="K-Means Clusters in Principal Component Space (PC1 vs. PC2)", filename="kmeans_pca_clusters_2d.png")
    viz.plot_hierarchical_dendrogram(linkage_matrix, n_clusters=OPTIMAL_K, filename="hierarchical_dendrogram.png")
    viz.plot_clusters_in_pca(X_pca, hier_labels, title="Hierarchical Agglomerative Clusters (Ward Linkage, n=4)", filename="hierarchical_clusters_pca.png")
    viz.plot_dbscan_k_distance(k_distances, chosen_eps=1.45, filename="dbscan_k_distance.png")
    viz.plot_clusters_in_pca(X_pca, dbscan_labels, title="DBSCAN Clusters & Noise Points in PCA Space", filename="dbscan_clusters_pca.png")

    # Cluster Profile & Radar Charts
    viz.plot_cluster_radar(clustered_df, cluster_col="kmeans_cluster", features=CORE_FEATURES, filename="cluster_radar_profile.png")
    viz.plot_category_cluster_distribution(category_crosstab_df, filename="cluster_category_distribution.png")

    logger.info("=================================================================")
    logger.info("ML ANALYSIS AND EXPORT PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Results Directory: {RESULTS_DIR}")
    logger.info(f"Tables Directory:  {TABLES_DIR}")
    logger.info(f"Figures Directory: {FIGURES_DIR}")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()
