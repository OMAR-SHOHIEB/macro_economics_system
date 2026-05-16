from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd


class Make_Transform:
    # """
    # This class applies different transformations on datasets (train, val, test)
    # in a safe machine learning pipeline manner.

    # Key rule:
    # - Always FIT on train only
    # - Then TRANSFORM on validation and test sets
    # - Avoid data leakage

    # ignore_cols:
    # Columns that should NOT be transformed (e.g., categorical like Country)
    # """

    def __init__(self, ignore_cols=['Country']):
        self.ignore_cols = ignore_cols
        self.scaler = None

# =========================
# ____ Standard Scaler ____
# =========================
    def standard_scaler(self, train, val, test):
        # """
        # StandardScaler (Z-score normalization)

        # Formula:
        #     X_scaled = (X - mean) / std

        # Theory:
        # - Centers data around zero (mean = 0)
        # - Scales variance to 1 (std = 1)
        # - Assumes data is roughly normally distributed

        # When to use:
        # - Linear models
        # - Logistic Regression
        # - Neural Networks

        # Notes:
        # - Sensitive to outliers
        # - Not suitable for heavily skewed data
        # """

        cols = [c for c in train.columns if c not in self.ignore_cols]

        self.scaler = StandardScaler()

        train[cols] = self.scaler.fit_transform(train[cols])
        val[cols]   = self.scaler.transform(val[cols])
        test[cols]  = self.scaler.transform(test[cols])

        return train, val, test

# =========================
# _____ Robust Scaler _____
# =========================
    def robust_scaler(self, train, val, test):
        # """
        # RobustScaler

        # Formula:
        #     X_scaled = (X - median) / IQR
        # where:
        #     IQR = Q3 - Q1 (Interquartile Range)

        # Theory:
        # - Uses median instead of mean
        # - Resistant to outliers
        # - More stable for real-world noisy data

        # When to use:
        # - Data with outliers
        # - Economic data (GDP, Inflation, etc.)

        # Advantage:
        # - Not affected by extreme values

        # Recommended for your project:
        # - Yes (very suitable for economic indicators)
        # """

        cols = [c for c in train.columns if c not in self.ignore_cols]

        self.scaler = RobustScaler()

        train[cols] = self.scaler.fit_transform(train[cols])
        val[cols]   = self.scaler.transform(val[cols])
        test[cols]  = self.scaler.transform(test[cols])

        return train, val, test

# =========================
# ______ Log Transform ____
# =========================
    def log(self, train, val, test):
        # """
        # Log Transformation

        # Formula:
        #     X_new = log(X)

        # Theory:
        # - Reduces skewness in data
        # - Compresses large values
        # - Helps make distribution more normal-like

        # When to use:
        # - Highly skewed features
        # - Data with large ranges (e.g., GDP)

        # Limitations:
        # - Cannot handle:
        #     - Zero values
        #     - Negative values
        # - Needs preprocessing before applying

        # Warning:
        # - Using log on invalid values will produce NaNs
        # """

        cols = [c for c in train.columns if c not in self.ignore_cols]

        for c in cols:
            train[c] = np.log(train[c].replace(0, np.nan))
            val[c]   = np.log(val[c].replace(0, np.nan))
            test[c]  = np.log(test[c].replace(0, np.nan))

        return train, val, test

# =========================
# ______ Log1p Transform __
# =========================
    def log1p(self, train, val, test):
        """
        Log1p Transformation
        Handles zero and negative values by applying sign(x) * log(1 + abs(x)).
        """
        cols = [c for c in train.columns if c not in self.ignore_cols]

        for c in cols:
            train[c] = np.sign(train[c]) * np.log1p(np.abs(train[c]))
            val[c]   = np.sign(val[c])   * np.log1p(np.abs(val[c]))
            test[c]  = np.sign(test[c])  * np.log1p(np.abs(test[c]))

        return train, val, test

# =========================
# ______ PCA Cluster ______
# =========================
    def pca_cluster_transformation(self, train, val, test, correlation_threshold=0.90):
        """
        Finds clusters of highly correlated features and compresses each cluster 
        into a single Principal Component (PCA).
        """
        import networkx as nx
        from sklearn.decomposition import PCA

        cols = [c for c in train.columns if c not in self.ignore_cols]
        corr_matrix = train[cols].corr().abs()
        
        # Build a graph to find clusters of highly correlated features
        edges = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    edges.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        G = nx.Graph()
        G.add_nodes_from(cols)
        G.add_edges_from(edges)
        
        # Extract clusters (connected components with >1 feature)
        clusters = [list(c) for c in nx.connected_components(G) if len(c) > 1]
        
        for idx, cluster in enumerate(clusters):
            print(f"Applying PCA to highly correlated cluster: {cluster}")
            # Compress the cluster's shared variance into a single Principal Component
            pca = PCA(n_components=1)
            
            train_pca = pca.fit_transform(train[cluster])
            val_pca   = pca.transform(val[cluster])
            test_pca  = pca.transform(test[cluster])
            
            pc_name = f'PCA_Cluster_{idx}'
            train[pc_name] = train_pca
            val[pc_name]   = val_pca
            test[pc_name]  = test_pca
            
            # Drop the original redundant columns
            train.drop(columns=cluster, inplace=True)
            val.drop(columns=cluster, inplace=True)
            test.drop(columns=cluster, inplace=True)
            
        return train, val, test