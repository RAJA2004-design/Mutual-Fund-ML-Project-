"""
Clustering and Dimensionality Reduction Module
Implements K-Means (K=2-8 evaluation), Hierarchical Agglomerative Clustering,
DBSCAN Density-Based Clustering, and Principal Component Analysis (PCA).
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ClusterEngine:
    """Coordinates unsupervised machine learning models and clustering evaluations."""

    def __init__(self, X_scaled: np.ndarray, feature_names: List[str], random_state: int = 42):
        self.X_scaled = X_scaled
        self.feature_names = feature_names
        self.random_state = random_state

    # =========================================================================
    # 1. K-MEANS CLUSTERING & EVALUATION (K = 2 to 8)
    # =========================================================================
    def evaluate_kmeans_range(self, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
        """
        Evaluates K-Means clustering across K = 2 to 8 using four canonical validation metrics:
        1. Inertia / Within-Cluster Sum of Squares (Elbow Method)
        2. Silhouette Score (Separation & Cohesion, -1 to +1, higher is better)
        3. Davies-Bouldin Index (Ratio of within-cluster distance to between-cluster distance, lower is better)
        4. Calinski-Harabasz Score (Variance Ratio Criterion, higher is better)
        """
        eval_records = []
        logger.info(f"Evaluating K-Means models from K={k_min} to K={k_max}...")

        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=25, max_iter=300)
            labels = km.fit_predict(self.X_scaled)

            inertia = km.inertia_
            sil = silhouette_score(self.X_scaled, labels)
            db = davies_bouldin_score(self.X_scaled, labels)
            ch = calinski_harabasz_score(self.X_scaled, labels)

            eval_records.append({
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(db, 4),
                "calinski_harabasz_score": round(ch, 2)
            })

        eval_df = pd.DataFrame(eval_records)
        return eval_df

    def fit_kmeans(self, optimal_k: int = 4) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Fits optimal K-Means model, returns cluster labels, centroids, and evaluation metrics."""
        km = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=30, max_iter=500)
        labels = km.fit_predict(self.X_scaled)
        centroids = km.cluster_centers_

        metrics = {
            "k": optimal_k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(self.X_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(self.X_scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(self.X_scaled, labels))
        }
        logger.info(f"Fitted K-Means (K={optimal_k}): Silhouette={metrics['silhouette']:.3f}, DB={metrics['davies_bouldin']:.3f}, CH={metrics['calinski_harabasz']:.1f}")
        return labels, centroids, metrics

    # =========================================================================
    # 2. HIERARCHICAL AGGLOMERATIVE CLUSTERING
    # =========================================================================
    def compute_hierarchical_linkage(self, method: str = "ward") -> np.ndarray:
        """Computes linkage matrix for dendrogram visualization using scipy."""
        logger.info(f"Computing hierarchical linkage matrix using '{method}' criterion...")
        Z = linkage(self.X_scaled, method=method, metric="euclidean")
        return Z

    def fit_hierarchical(self, n_clusters: int = 4, linkage_method: str = "ward") -> Tuple[np.ndarray, Dict[str, float]]:
        """Fits Agglomerative Hierarchical clustering model."""
        agg = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage=linkage_method)
        labels = agg.fit_predict(self.X_scaled)

        metrics = {
            "n_clusters": n_clusters,
            "linkage": linkage_method,
            "silhouette": float(silhouette_score(self.X_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(self.X_scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(self.X_scaled, labels))
        }
        logger.info(f"Fitted Hierarchical (n={n_clusters}, {linkage_method}): Silhouette={metrics['silhouette']:.3f}, DB={metrics['davies_bouldin']:.3f}")
        return labels, metrics

    # =========================================================================
    # 3. DBSCAN (DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS WITH NOISE)
    # =========================================================================
    def compute_k_distances(self, k_neighbors: int = 4) -> np.ndarray:
        """Computes sorted k-nearest neighbor distances to determine the optimal epsilon."""
        neigh = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs = neigh.fit(self.X_scaled)
        distances, _ = nbrs.kneighbors(self.X_scaled)
        k_distances = np.sort(distances[:, k_neighbors - 1])
        return k_distances

    def fit_dbscan(self, eps: float = 1.45, min_samples: int = 4) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fits DBSCAN to detect dense clusters and identify anomalous funds (noise = -1).
        """
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(self.X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        metrics = {
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio_pct": round((n_noise / len(labels)) * 100.0, 2)
        }

        # Calculate metrics only if at least 2 non-noise clusters exist
        if n_clusters >= 2:
            non_noise_mask = (labels != -1)
            if non_noise_mask.sum() > n_clusters:
                metrics["silhouette_core"] = float(silhouette_score(self.X_scaled[non_noise_mask], labels[non_noise_mask]))
                metrics["davies_bouldin_core"] = float(davies_bouldin_score(self.X_scaled[non_noise_mask], labels[non_noise_mask]))
            else:
                metrics["silhouette_core"] = np.nan
                metrics["davies_bouldin_core"] = np.nan
        else:
            metrics["silhouette_core"] = np.nan
            metrics["davies_bouldin_core"] = np.nan

        logger.info(f"Fitted DBSCAN (eps={eps}, min_samples={min_samples}): Found {n_clusters} clusters, {n_noise} noise points ({metrics['noise_ratio_pct']}%)")
        return labels, metrics

    # =========================================================================
    # 4. PRINCIPAL COMPONENT ANALYSIS (PCA)
    # =========================================================================
    def fit_pca(self, n_components: int = None) -> Tuple[PCA, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Fits PCA on standardized features.
        Returns:
            pca_model: Fitted PCA estimator
            X_pca: Projected coordinates (samples x components)
            loadings_df: Component loadings (features x components)
            variance_df: Explained variance per component
        """
        if n_components is None:
            n_components = min(self.X_scaled.shape)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(self.X_scaled)

        # Variance table
        evr = pca.explained_variance_ratio_
        cumulative_evr = np.cumsum(evr)
        eigenvalues = pca.explained_variance_

        var_records = []
        for i in range(len(evr)):
            var_records.append({
                "component": f"PC{i+1}",
                "eigenvalue": round(float(eigenvalues[i]), 4),
                "explained_variance_ratio": round(float(evr[i]), 4),
                "explained_variance_pct": round(float(evr[i] * 100.0), 2),
                "cumulative_variance_pct": round(float(cumulative_evr[i] * 100.0), 2)
            })
        variance_df = pd.DataFrame(var_records)

        # Loadings matrix (Correlation between original features and PCs)
        # Loadings = eigenvectors * sqrt(eigenvalues)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(n_components)]
        ).round(4)

        logger.info(f"PCA fitted: PC1 explains {variance_df.iloc[0]['explained_variance_pct']}%, PC2 explains {variance_df.iloc[1]['explained_variance_pct']}% (Cumulative 2D: {variance_df.iloc[1]['cumulative_variance_pct']}%)")
        return pca, X_pca, loadings_df, variance_df
