"""
Visualization Module for Machine Learning Risk-Return Profiling
Generates publication-quality static charts (300 DPI) for exploratory data analysis,
clustering diagnostics, PCA factor decompositions, and cluster profiles.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.autolayout": True
})

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    """Produces professional static figures for EDA and ML clustering."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, "reports", "figures")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.colors = ["#1B365D", "#2E8B57", "#D9534F", "#F0AD4E", "#5BC0DE", "#8E44AD", "#34495E"]

    def _save_fig(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved figure: {path}")

    # =========================================================================
    # 1. EXPLORATORY DATA ANALYSIS (EDA) CHARTS
    # =========================================================================
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, filename: str = "eda_correlation_heatmap.png"):
        """Plots annotated Pearson correlation heatmap with diverging colormap."""
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
        )
        plt.title("Correlation Heatmap of Key Risk-Return Metrics (132 Schemes)", fontsize=14, fontweight="bold", pad=15)
        self._save_fig(filename)

    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str], filename: str = "eda_feature_distributions.png"):
        """Plots multi-panel histograms with KDE curves."""
        n_feats = len(features)
        cols = 3
        rows = (n_feats + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        for idx, feat in enumerate(features):
            ax = axes[idx]
            s = df[feat].dropna()
            sns.histplot(s, kde=True, ax=ax, color="#1B365D", bins=20, edgecolor="white", alpha=0.7)
            mean_val = s.mean()
            median_val = s.median()
            ax.axvline(mean_val, color="#D9534F", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="#2E8B57", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f}")
            ax.set_title(feat.replace("_", " ").title(), fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9, loc="upper right")

        # Hide unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Empirical Distributions of Core Financial Features", fontsize=16, fontweight="bold", y=1.02)
        self._save_fig(filename)

    def plot_risk_return_scatter(self, df: pd.DataFrame, filename: str = "eda_risk_return_scatter.png"):
        """Plots Volatility vs CAGR scatter plot colored by Sharpe ratio."""
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            df["annualized_volatility_pct"],
            df["cagr_5y_pct"],
            c=df["sharpe_ratio"],
            cmap="viridis",
            s=60,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.85
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Sharpe Ratio (Rf = 6.50%)", fontsize=11, fontweight="bold")
        
        # Add benchmark references if available
        plt.axhline(6.5, color="#D9534F", linestyle="--", alpha=0.7, label="Risk-Free Rate (6.50%)")
        plt.title("Risk vs. Return: Annualized Volatility vs. 5-Year CAGR (132 Funds)", fontsize=14, fontweight="bold")
        plt.xlabel("Annualized Volatility (%)", fontsize=12, fontweight="bold")
        plt.ylabel("5-Year CAGR (%)", fontsize=12, fontweight="bold")
        plt.legend(loc="upper left")
        self._save_fig(filename)

    def plot_category_boxplots(self, df: pd.DataFrame, filename: str = "eda_boxplots_by_category.png"):
        """Plots comparative boxplots for CAGR, Volatility, Sharpe, and Beta across categories."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ("cagr_5y_pct", "5-Year CAGR (%)", axes[0, 0]),
            ("annualized_volatility_pct", "Annualized Volatility (%)", axes[0, 1]),
            ("sharpe_ratio", "Sharpe Ratio", axes[1, 0]),
            ("beta_nifty500", "Market Beta (vs NIFTY 500)", axes[1, 1])
        ]

        order = df.groupby("category")["sharpe_ratio"].median().sort_values(ascending=False).index

        for col, label, ax in metrics:
            sns.boxplot(data=df, y="category", x=col, order=order, ax=ax, palette="Blues_r")
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_xlabel(label)
            ax.set_ylabel("")

        plt.suptitle("Cross-Sectional Category Comparison across 12 SEBI Equity Categories", fontsize=16, fontweight="bold", y=1.02)
        self._save_fig(filename)

    # =========================================================================
    # 2. CLUSTERING EVALUATION & DIAGNOSTICS CHARTS
    # =========================================================================
    def plot_kmeans_evaluation(self, eval_df: pd.DataFrame, optimal_k: int = 4, filename: str = "kmeans_elbow_and_metrics.png"):
        """Plots 4-panel evaluation: Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        k_vals = eval_df["k"]

        # Panel 1: Inertia (Elbow)
        axes[0, 0].plot(k_vals, eval_df["inertia"], marker="o", color="#1B365D", linewidth=2)
        axes[0, 0].axvline(optimal_k, color="#D9534F", linestyle="--", label=f"Optimal K = {optimal_k}")
        axes[0, 0].set_title("Within-Cluster Sum of Squares (Inertia)", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Number of Clusters (K)")
        axes[0, 0].set_ylabel("Inertia")
        axes[0, 0].legend()

        # Panel 2: Silhouette Score
        axes[0, 1].plot(k_vals, eval_df["silhouette_score"], marker="s", color="#2E8B57", linewidth=2)
        axes[0, 1].axvline(optimal_k, color="#D9534F", linestyle="--", label=f"Optimal K = {optimal_k}")
        axes[0, 1].set_title("Silhouette Score (Higher is Better)", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Number of Clusters (K)")
        axes[0, 1].set_ylabel("Silhouette Score")
        axes[0, 1].legend()

        # Panel 3: Davies-Bouldin Index
        axes[1, 0].plot(k_vals, eval_df["davies_bouldin_index"], marker="^", color="#D9534F", linewidth=2)
        axes[1, 0].axvline(optimal_k, color="#1B365D", linestyle="--", label=f"Optimal K = {optimal_k}")
        axes[1, 0].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Number of Clusters (K)")
        axes[1, 0].set_ylabel("Davies-Bouldin Index")
        axes[1, 0].legend()

        # Panel 4: Calinski-Harabasz Score
        axes[1, 1].plot(k_vals, eval_df["calinski_harabasz_score"], marker="d", color="#8E44AD", linewidth=2)
        axes[1, 1].axvline(optimal_k, color="#D9534F", linestyle="--", label=f"Optimal K = {optimal_k}")
        axes[1, 1].set_title("Calinski-Harabasz Score (Higher is Better)", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Number of Clusters (K)")
        axes[1, 1].set_ylabel("Variance Ratio")
        axes[1, 1].legend()

        plt.suptitle("K-Means Multi-Metric Clustering Validation (K = 2 to 8)", fontsize=16, fontweight="bold", y=1.02)
        self._save_fig(filename)

    def plot_pca_scree(self, variance_df: pd.DataFrame, filename: str = "pca_scree_plot.png"):
        """Plots scree bar chart of individual variance explained and cumulative line."""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        comps = variance_df["component"]
        ev_pct = variance_df["explained_variance_pct"]
        cum_pct = variance_df["cumulative_variance_pct"]

        bars = ax1.bar(comps, ev_pct, color="#1B365D", alpha=0.8, label="Individual Variance (%)")
        ax1.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Explained Variance (%)", color="#1B365D", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, max(ev_pct) * 1.15)

        # Label on bars
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=10)

        ax2 = ax1.twinx()
        ax2.plot(comps, cum_pct, color="#D9534F", marker="o", linewidth=2.5, label="Cumulative Variance (%)")
        ax2.set_ylabel("Cumulative Variance (%)", color="#D9534F", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 105)
        ax2.axhline(80, color="gray", linestyle=":", label="80% Threshold")

        plt.title("PCA Scree Plot & Cumulative Explained Variance", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save_fig(filename)

    def plot_pca_biplot(self, X_pca: np.ndarray, loadings_df: pd.DataFrame, labels: np.ndarray = None, filename: str = "pca_biplot.png"):
        """Plots PCA biplot showing observations and feature loading vectors (arrows)."""
        plt.figure(figsize=(11, 8))
        
        # Plot observations
        if labels is not None:
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=40, alpha=0.6, edgecolors="white")
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], color="#1B365D", s=40, alpha=0.6)

        # Plot loading arrows
        scale = 3.5  # visual scaling factor for arrows
        for feat in loadings_df.index:
            pc1_load = loadings_df.loc[feat, "PC1"] * scale
            pc2_load = loadings_df.loc[feat, "PC2"] * scale
            plt.arrow(0, 0, pc1_load, pc2_load, color="#D9534F", alpha=0.8, width=0.03, head_width=0.15)
            plt.text(pc1_load * 1.12, pc2_load * 1.12, feat.replace("_pct", ""), color="#8B0000", fontsize=9, fontweight="bold")

        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.title("PCA Biplot: Observations & Feature Vectors (PC1 vs. PC2)", fontsize=14, fontweight="bold")
        plt.xlabel("Principal Component 1 (Risk & Market Factor)", fontsize=12, fontweight="bold")
        plt.ylabel("Principal Component 2 (Quality & Efficiency Factor)", fontsize=12, fontweight="bold")
        self._save_fig(filename)

    def plot_clusters_in_pca(self, X_pca: np.ndarray, labels: np.ndarray, title: str, filename: str):
        """Plots 2D PCA cluster scatter with centroids and cluster color coding."""
        plt.figure(figsize=(10, 7))
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("tab10", len(unique_labels))

        for idx, lbl in enumerate(unique_labels):
            mask = (labels == lbl)
            lbl_name = f"Cluster {lbl}" if lbl != -1 else "Noise / Outlier"
            color = palette[idx] if lbl != -1 else "black"
            marker = "o" if lbl != -1 else "x"
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                color=color,
                marker=marker,
                s=65 if lbl != -1 else 40,
                label=lbl_name,
                alpha=0.8,
                edgecolors="black" if lbl != -1 else None,
                linewidth=0.5
            )

        # Plot centroids for positive clusters
        pos_clusters = [l for l in unique_labels if l != -1]
        for c in pos_clusters:
            c_center = X_pca[labels == c].mean(axis=0)
            plt.scatter(c_center[0], c_center[1], c="black", marker="*", s=250, edgecolors="white", linewidth=1.5, zorder=10)

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Principal Component 1", fontsize=12, fontweight="bold")
        plt.ylabel("Principal Component 2", fontsize=12, fontweight="bold")
        plt.legend(loc="best", frameon=True)
        self._save_fig(filename)

    def plot_hierarchical_dendrogram(self, Z: np.ndarray, n_clusters: int = 4, filename: str = "hierarchical_dendrogram.png"):
        """Plots hierarchical clustering dendrogram with threshold cut line."""
        plt.figure(figsize=(12, 7))
        # Cut height is the distance at which n_clusters are formed
        cut_height = (Z[-(n_clusters - 1), 2] + Z[-n_clusters, 2]) / 2.0
        
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_contracted=True,
            color_threshold=cut_height
        )
        plt.axhline(cut_height, color="#D9534F", linestyle="--", linewidth=1.5, label=f"Cut Height = {cut_height:.2f} (n={n_clusters})")
        plt.title("Hierarchical Agglomerative Clustering Dendrogram (Ward Linkage)", fontsize=14, fontweight="bold")
        plt.xlabel("Sample Index or (Cluster Size)", fontsize=11)
        plt.ylabel("Euclidean Distance", fontsize=11)
        plt.legend()
        self._save_fig(filename)

    def plot_dbscan_k_distance(self, k_distances: np.ndarray, chosen_eps: float = 1.45, filename: str = "dbscan_k_distance.png"):
        """Plots k-distance graph for selecting DBSCAN epsilon parameter."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_distances)), k_distances, color="#1B365D", linewidth=2)
        plt.axhline(chosen_eps, color="#D9534F", linestyle="--", label=f"Selected eps = {chosen_eps}")
        plt.title("DBSCAN Epsilon Determination: Sorted 4-Nearest-Neighbor Distance", fontsize=14, fontweight="bold")
        plt.xlabel("Data Points Sorted by Distance", fontsize=12)
        plt.ylabel("4th Nearest Neighbor Distance", fontsize=12)
        plt.legend()
        self._save_fig(filename)

    def plot_cluster_radar(self, df_with_clusters: pd.DataFrame, cluster_col: str, features: List[str], filename: str = "cluster_radar_profile.png"):
        """Plots radar / spider chart comparing mean standardized metrics across clusters."""
        clusters = sorted(df_with_clusters[cluster_col].unique())
        if -1 in clusters:
            clusters.remove(-1)  # omit noise if present
            
        # Standardize features for radar comparison
        std_df = df_with_clusters.copy()
        for f in features:
            mu = std_df[f].mean()
            sig = std_df[f].std()
            std_df[f] = (std_df[f] - mu) / sig if sig > 0 else 0

        means = std_df.groupby(cluster_col)[features].mean()
        
        # Angles for radar
        num_vars = len(features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # close the circle

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        palette = sns.color_palette("tab10", len(clusters))

        for idx, c in enumerate(clusters):
            values = means.loc[c].tolist()
            values += values[:1]
            ax.plot(angles, values, color=palette[idx], linewidth=2, label=f"Cluster {c}")
            ax.fill(angles, values, color=palette[idx], alpha=0.15)

        clean_labels = [f.replace("_5y_pct", "").replace("_pct", "").replace("_rf", "").replace("_", " ").title() for f in features]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(clean_labels, fontsize=10, fontweight="bold")
        ax.set_title("Standardized Cluster Risk-Return Signatures (Radar Profile)", fontsize=14, fontweight="bold", y=1.08)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        self._save_fig(filename)

    def plot_category_cluster_distribution(self, crosstab_df: pd.DataFrame, filename: str = "cluster_category_distribution.png"):
        """Plots stacked bar chart showing SEBI category breakdown per cluster."""
        ct = crosstab_df.drop(index="Total", columns="Total", errors="ignore")
        plt.figure(figsize=(12, 7))
        ct.plot(kind="bar", stacked=True, colormap="tab10", edgecolor="black", linewidth=0.5, ax=plt.gca())
        plt.title("Distribution of SEBI Mutual Fund Categories Across Discovered Clusters", fontsize=14, fontweight="bold")
        plt.xlabel("SEBI Equity Category", fontsize=11, fontweight="bold")
        plt.ylabel("Fund Count", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        self._save_fig(filename)
