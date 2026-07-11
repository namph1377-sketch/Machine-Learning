import numpy as np
import time

class Node:
    def __init__(self, feature_index: int = None, threshold: float = None, left=None, right=None, value: float = None):
        self.feature_index = feature_index  
        self.threshold = threshold          
        self.left = left    
        self.right = right  
        self.value = value  

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressorScratch:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, max_features=None, random_state: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  
        self.random_state = random_state
        self.root = None                  
        self._rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorScratch":
        n_features = X.shape[1]
        self._n_features_to_sample = self._resolve_max_features(n_features)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = len(y)
        max_depth_reached = self.max_depth is not None and depth >= self.max_depth
        too_few_samples   = n_samples < self.min_samples_split
        pure_node         = np.var(y) == 0.0

        if max_depth_reached or too_few_samples or pure_node:
            return Node(value=float(np.mean(y)))

        best = self._best_split(X, y)
        if best is None:
            return Node(value=float(np.mean(y)))

        feat_idx, threshold = best
        left_mask  = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        left_child  = self._grow_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=feat_idx, threshold=threshold, left=left_child, right=right_child)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        parent_var = float(np.var(y))

        candidate_features = self._rng.choice(n_features, size=self._n_features_to_sample, replace=False)
        best_gain      = -np.inf
        best_feat_idx  = None
        best_threshold = None

        for feat_idx in candidate_features:
            col = X[:, feat_idx]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask  = col <= threshold
                right_mask = ~left_mask
                n_left  = left_mask.sum()
                n_right = right_mask.sum()

                if n_left == 0 or n_right == 0:
                    continue

                weighted_child_var = ((n_left / n_samples) * np.var(y[left_mask]) + 
                                      (n_right / n_samples) * np.var(y[right_mask]))
                gain = parent_var - weighted_child_var

                if gain > best_gain:
                    best_gain      = gain
                    best_feat_idx  = feat_idx
                    best_threshold = threshold

        if best_feat_idx is None:
            return None
        return best_feat_idx, best_threshold

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        if node.is_leaf:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None: return n_features
        if mf == "sqrt": return max(1, int(np.floor(np.sqrt(n_features))))
        if mf == "log2": return max(1, int(np.floor(np.log2(n_features))))
        if isinstance(mf, (int, np.integer)): return max(1, min(int(mf), n_features))
        if isinstance(mf, float) and 0.0 < mf <= 1.0: return max(1, int(np.floor(mf * n_features)))
        raise ValueError(f"Unsupported max_features value: {mf!r}")


class RandomForestRegressorScratch:
    def __init__(self, n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2, max_features="sqrt", random_state: int = None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state
        self.trees_ = []
        self._master_rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        n_samples = X.shape[0]
        self.trees_ = []

        print(f"\n  Training {self.n_estimators} trees ", end="", flush=True)
        t0 = time.time()

        for i in range(self.n_estimators):
            tree_seed = self._master_rng.randint(0, 2**31 - 1)
            rng_tree  = np.random.RandomState(tree_seed)

            indices   = rng_tree.choice(n_samples, size=n_samples, replace=True)
            X_boot    = X[indices]
            y_boot    = y[indices]

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

            if (i + 1) % max(1, self.n_estimators // 10) == 0:
                print("▓", end="", flush=True)

        print(f"  done ({time.time() - t0:.1f}s)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        tree_predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(tree_predictions, axis=0)
