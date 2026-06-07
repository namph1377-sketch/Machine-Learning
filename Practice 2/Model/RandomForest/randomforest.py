
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time

warnings.filterwarnings("ignore")


# =============================================================================
#  PART 1 — DECISION TREE REGRESSOR FROM SCRATCH
# =============================================================================

class Node:
    """
    Represents a single node in a Decision Tree.

    A node is either:
      - An internal (split) node: stores the feature index + threshold
        used to route samples left (feature <= threshold) or right.
      - A leaf node: stores the predicted value = mean(y) for the
        samples that reached this node during training.
    """

    def __init__(
        self,
        feature_index: int = None,
        threshold: float = None,
        left=None,
        right=None,
        value: float = None
    ):
        # Split node attributes
        self.feature_index = feature_index  # which feature to split on
        self.threshold = threshold          # split boundary: X[:, j] <= threshold → left

        # Child nodes
        self.left = left    # Node for samples where feature <= threshold
        self.right = right  # Node for samples where feature > threshold

        # Leaf node attribute
        self.value = value  # prediction = mean(y) at this leaf; None if internal node

    @property
    def is_leaf(self) -> bool:
        """True when this node has a prediction (no children)."""
        return self.value is not None


class DecisionTreeRegressorScratch:
    """
    A single Decision Tree Regressor built entirely without scikit-learn.

    Splitting criterion: Variance Reduction (= MSE Reduction)
    ─────────────────────────────────────────────────────────
    At every candidate split we compute:

        Gain = Var(parent) - [ (n_L/n)*Var(left) + (n_R/n)*Var(right) ]

    The split maximising Gain is chosen.  This is mathematically equivalent
    to minimising the weighted MSE of the two child nodes.

    Feature sampling (max_features):
    ─────────────────────────────────
    Before evaluating splits at each node, we randomly draw max_features
    candidate features.  This is the key source of decorrelation across
    trees inside a Random Forest.

    Leaf prediction:
    ─────────────────
    leaf_value = mean(y_leaf)   ← optimal constant under MSE loss
    """

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features=None,
        random_state: int = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # int | "sqrt" | "log2" | None (= all)
        self.random_state = random_state
        self.root = None                  # root Node, set after fit()
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorScratch":
        """Grow the full tree on training data (X, y)."""
        n_features = X.shape[1]
        self._n_features_to_sample = self._resolve_max_features(n_features)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a predicted value for each row in X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # ------------------------------------------------------------------
    # Tree growing
    # ------------------------------------------------------------------

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively grow the tree using a depth-first strategy.

        Stopping conditions (→ create leaf):
          1. max_depth reached
          2. n_samples < min_samples_split
          3. All target values are identical (zero variance → no gain possible)
          4. No valid split found (all features constant)
        """
        n_samples = len(y)

        # --- Stopping conditions ---
        max_depth_reached = self.max_depth is not None and depth >= self.max_depth
        too_few_samples   = n_samples < self.min_samples_split
        pure_node         = np.var(y) == 0.0

        if max_depth_reached or too_few_samples or pure_node:
            return Node(value=float(np.mean(y)))

        # --- Find best binary split ---
        best = self._best_split(X, y)

        if best is None:          # no informative split found → leaf
            return Node(value=float(np.mean(y)))

        feat_idx, threshold = best

        # Partition samples
        left_mask  = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        # Recurse
        left_child  = self._grow_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=feat_idx,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    # ------------------------------------------------------------------
    # Split search
    # ------------------------------------------------------------------

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Search over a random subset of features and all valid thresholds
        for the split that maximises Variance Reduction.

        Returns (feature_index, threshold) or None if no valid split exists.
        """
        n_samples, n_features = X.shape
        parent_var = self._variance(y)

        # Randomly sample candidate features (decorrelation mechanism)
        candidate_features = self._rng.choice(
            n_features,
            size=self._n_features_to_sample,
            replace=False
        )

        best_gain      = -np.inf
        best_feat_idx  = None
        best_threshold = None

        for feat_idx in candidate_features:
            col = X[:, feat_idx]
            # Candidate thresholds = unique sorted values
            # We use midpoints between consecutive unique values for cleaner splits
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue  # constant feature → can't split

            # Midpoints as thresholds (avoids boundary issues)
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask  = col <= threshold
                right_mask = ~left_mask

                n_left  = left_mask.sum()
                n_right = right_mask.sum()

                # Skip degenerate splits
                if n_left == 0 or n_right == 0:
                    continue

                gain = self._variance_reduction(
                    y, y[left_mask], y[right_mask],
                    parent_var, n_samples, n_left, n_right
                )

                if gain > best_gain:
                    best_gain      = gain
                    best_feat_idx  = feat_idx
                    best_threshold = threshold

        if best_feat_idx is None:
            return None

        return best_feat_idx, best_threshold

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _variance(y: np.ndarray) -> float:
        """
        Population variance: Var(y) = E[(y - ȳ)²]

        Using np.var() which is O(n) and numerically stable.
        This is the same as MSE of the constant prediction ȳ.
        """
        return float(np.var(y))

    @staticmethod
    def _variance_reduction(
        y_parent, y_left, y_right,
        parent_var, n, n_left, n_right
    ) -> float:
        """
        Variance Reduction (Information Gain under MSE):

            Gain = Var(parent)
                   - [ (n_L / n) * Var(left) + (n_R / n) * Var(right) ]

        A positive gain means the split reduces the average unexplained
        variance — i.e. the split is informative.
        """
        weighted_child_var = (
            (n_left  / n) * np.var(y_left) +
            (n_right / n) * np.var(y_right)
        )
        return parent_var - weighted_child_var

    # ------------------------------------------------------------------
    # Prediction traversal
    # ------------------------------------------------------------------

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        """
        Walk a single sample x down the tree until a leaf is reached.
        At each internal node: go left if x[feature] <= threshold, else right.
        """
        if node.is_leaf:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _resolve_max_features(self, n_features: int) -> int:
        """
        Translate the max_features hyperparameter into an integer count.

        "sqrt"  → floor(√p)          ← default for Random Forests (regression)
        "log2"  → floor(log₂ p)
        int     → used directly
        None    → all features (= a plain CART tree)
        """
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.floor(np.sqrt(n_features))))
        if mf == "log2":
            return max(1, int(np.floor(np.log2(n_features))))
        if isinstance(mf, (int, np.integer)):
            return max(1, min(int(mf), n_features))
        if isinstance(mf, float) and 0.0 < mf <= 1.0:
            return max(1, int(np.floor(mf * n_features)))
        raise ValueError(f"Unsupported max_features value: {mf!r}")


# =============================================================================
#  PART 2 — RANDOM FOREST REGRESSOR FROM SCRATCH
# =============================================================================

class RandomForestRegressorScratch:
    """
    Ensemble of DecisionTreeRegressorScratch models trained on bootstrap
    samples of the data.

    How it reduces variance (bias-variance trade-off):
    ───────────────────────────────────────────────────
    Each tree Ti is trained on a bootstrap sample of size n drawn *with
    replacement* from the original n observations.  On average each
    bootstrap sample contains ~63.2 % unique rows; the rest are
    duplicates, giving each tree a slightly different view of the data.

    Additionally, at every split each tree considers only a random subset
    of max_features features (handled inside DecisionTreeRegressorScratch).
    This *decorrelates* the trees: even if one feature is very dominant,
    it won't dominate every tree.

    Final prediction = arithmetic mean of all tree predictions:

        ŷ(x) = (1/T) Σᵢ Tᵢ(x)

    Averaging T independent (and identically distributed) predictions
    reduces the variance by a factor of T without increasing bias —
    this is the core mathematical motivation for bagging.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features="sqrt",
        random_state: int = None
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state

        self.trees_ = []   # list of fitted DecisionTreeRegressorScratch

        # Master RNG; each tree gets its own seed derived from this
        self._master_rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        """
        Fit n_estimators trees, each on a bootstrap sample of (X, y).

        Bootstrap sampling:
            indices = np.random.choice(n_samples, n_samples, replace=True)
        """
        n_samples = X.shape[0]
        self.trees_ = []

        print(f"\n  Training {self.n_estimators} trees ", end="", flush=True)
        t0 = time.time()

        for i in range(self.n_estimators):
            # --- Bootstrap sample ---
            # Each call to _master_rng.randint gives a reproducible but
            # different seed for the i-th tree.
            tree_seed = self._master_rng.randint(0, 2**31 - 1)
            rng_tree  = np.random.RandomState(tree_seed)

            indices  = rng_tree.choice(n_samples, size=n_samples, replace=True)
            X_boot   = X[indices]
            y_boot   = y[indices]

            # --- Train one tree ---
            tree = DecisionTreeRegressorScratch(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features      = self.max_features,
                random_state      = tree_seed
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

            # Progress indicator
            if (i + 1) % max(1, self.n_estimators // 10) == 0:
                print("▓", end="", flush=True)

        elapsed = time.time() - t0
        print(f"  done ({elapsed:.1f}s)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions from all trees via averaging.

        tree_predictions shape: (n_estimators, n_samples)
        final prediction shape: (n_samples,)
        """
        tree_predictions = np.array(
            [tree.predict(X) for tree in self.trees_]
        )                                       # shape: (T, n_samples)
        return np.mean(tree_predictions, axis=0)  # shape: (n_samples,)


# =============================================================================
#  PART 3 — SYNTHETIC E-COMMERCE DATASET
# =============================================================================

def generate_ecommerce_dataset(n_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic e-commerce product sales dataset.

    Target construction (noisy non-linear function):
    ─────────────────────────────────────────────────
        base_sales  = 200 - 1.5 * price
                    + 4.0 * discount_pct
                    + 0.05 * ad_spend
                    + 8.0 * previous_purchases
                    + 50.0 * holiday_flag
                    + category_effect
        sales = max(0, base_sales + N(0, 30))
    """
    rng = np.random.RandomState(random_state)

    # --- Numerical features ---
    price               = rng.uniform(5, 300, n_rows)
    discount_pct        = rng.uniform(0, 50, n_rows)
    customer_age        = rng.randint(18, 70, n_rows).astype(float)
    previous_purchases  = rng.poisson(5, n_rows).astype(float)
    purchase_frequency  = rng.uniform(0.1, 5.0, n_rows)   # purchases / month
    ad_spend            = rng.uniform(0, 2000, n_rows)
    holiday_flag        = rng.choice([0, 1], n_rows, p=[0.75, 0.25]).astype(float)

    # --- Categorical features ---
    categories     = ["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"]
    category_raw   = rng.choice(categories, n_rows)

    genders        = ["Male", "Female", "Other"]
    gender_raw     = rng.choice(genders, n_rows, p=[0.45, 0.50, 0.05])

    locations      = ["Urban", "Suburban", "Rural"]
    location_raw   = rng.choice(locations, n_rows, p=[0.50, 0.35, 0.15])

    # --- Text feature: product_description ---
    words = [
        "premium", "budget", "exclusive", "trending", "bestseller",
        "limited", "seasonal", "new", "classic", "eco-friendly",
        "luxury", "value", "top-rated", "popular", "stylish"
    ]
    def random_description(seed_idx: int) -> str:
        n_words = rng.randint(5, 20)
        return " ".join(rng.choice(words, n_words))

    product_description = [random_description(i) for i in range(n_rows)]

    # --- Target: sales (non-linear + noisy) ---
    category_effect_map = {
        "Electronics": 80, "Clothing": 40, "Books": 10,
        "Home": 30, "Sports": 50, "Beauty": 60
    }
    category_effect = np.array([category_effect_map[c] for c in category_raw])

    base_sales = (
        200.0
        - 1.5  * price
        + 4.0  * discount_pct
        + 0.05 * ad_spend
        + 8.0  * previous_purchases
        + 50.0 * holiday_flag
        + category_effect
        + 2.0  * purchase_frequency * 10   # interaction
    )

    noise  = rng.normal(0, 30, n_rows)
    sales  = np.maximum(0, base_sales + noise)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "price":               price,
        "discount_pct":        discount_pct,
        "customer_age":        customer_age,
        "previous_purchases":  previous_purchases,
        "purchase_frequency":  purchase_frequency,
        "ad_spend":            ad_spend,
        "holiday_flag":        holiday_flag,
        "category":            category_raw,
        "gender":              gender_raw,
        "location":            location_raw,
        "product_description": product_description,
        "sales":               np.round(sales, 2),
    })

    return df


# =============================================================================
#  PART 4 — PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Feature engineering + encoding pipeline.

    Text features (handcrafted, no NLP library):
      - description_length : character count
      - word_count         : space-separated token count

    Categorical features:
      - One-hot encoded via pandas.get_dummies()

    Returns (X: np.ndarray, y: np.ndarray, feature_names: list[str])
    """
    df = df.copy()

    # --- Text features ---
    df["description_length"] = df["product_description"].str.len()
    df["word_count"]         = df["product_description"].str.split().str.len()

    # Drop raw text column (not useful directly)
    df.drop(columns=["product_description"], inplace=True)

    # --- Target ---
    y = df.pop("sales").to_numpy(dtype=np.float64)

    # --- One-hot encode categoricals ---
    df = pd.get_dummies(df, columns=["category", "gender", "location"], drop_first=False)

    # Ensure all columns are numeric (bool → float)
    df = df.astype(np.float64)

    feature_names = df.columns.tolist()
    X = df.to_numpy(dtype=np.float64)

    return X, y, feature_names


# =============================================================================
#  PART 5 — TRAINING & EVALUATION
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, split: str = "Test") -> dict:
    """
    Compute and print regression metrics using sklearn's metric functions.
    Returns a dict of {metric_name: value}.
    """
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    metrics = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R²": r2}

    print(f"\n  ── {split} Set Metrics ──────────────────────────")
    for name, val in metrics.items():
        print(f"    {name:<6}: {val:>10.4f}")
    print("  ────────────────────────────────────────────────")

    return metrics


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main():

    print("=" * 60)
    print(" RANDOM FOREST REGRESSOR — FROM SCRATCH")
    print(" E-Commerce Product Sales Prediction")
    print("=" * 60)

    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    print("\n[1] Loading dataset ...")

    train_df = pd.read_csv("retail_train_80.csv")
    test_df = pd.read_csv("retail_test_20.csv")

    TARGET = "sales_amount_log"

    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values

    X_test = test_df.drop(columns=[TARGET]).values
    y_test = test_df[TARGET].values

    print(f"Train samples : {X_train.shape[0]}")
    print(f"Test samples  : {X_test.shape[0]}")
    print(f"Features      : {X_train.shape[1]}")

    # =====================================================
    # 2. MODEL CONFIG
    # =====================================================

    print("\n[2] Creating Random Forest ...")

    model = RandomForestRegressorScratch(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        max_features=10,
        random_state=42
    )

    print(f"n_estimators      = {model.n_estimators}")
    print(f"max_depth         = {model.max_depth}")
    print(f"min_samples_split = {model.min_samples_split}")
    print(f"max_features      = {model.max_features}")

    # =====================================================
    # 3. TRAIN
    # =====================================================

    print("\n[3] Training ...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # =====================================================
    # 4. PREDICT
    # =====================================================

    print("\n[4] Predicting ...")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # =====================================================
    # 5. EVALUATE
    # =====================================================

    print("\n[5] Evaluation")

    train_metrics = evaluate(
        y_train,
        y_pred_train,
        split="Train"
    )

    test_metrics = evaluate(
        y_test,
        y_pred_test,
        split="Test"
    )
# =====================================================
    # [MỚI] 5b. LƯU KẾT QUẢ DỰ ĐOÁN VÀO FILE CSV
    # =====================================================
    print("\n[5b] Saving predictions to retail_test_20.csv ...")
    
    # Đọc lại file test gốc để giữ nguyên các cột ban đầu
    test_output_df = pd.read_csv("retail_test_20.csv")
    
    # Ghi giá trị dự đoán vào cột cuối cùng
    test_output_df["y_predict"] = y_pred_test
    
    # Lưu đè vào file test hoặc lưu thành file mới tùy bạn (ở đây lưu thành test_data.csv theo ý bạn)
    test_output_df.to_csv("test_data.csv", index=False)
    
    print("-> Đã lưu thành công vào file 'test_data.csv' (cột cuối: y_predict)")
    # =====================================================
    # 6. SAMPLE PREDICTIONS
    # =====================================================

    print("\n[6] Sample Predictions")

    print(f"{'Actual':>12} {'Predicted':>12} {'Error':>12}")
    print("-" * 40)

    for actual, pred in zip(
        y_test[:10],
        y_pred_test[:10]
    ):

        error = actual - pred

        print(
            f"{actual:12.2f}"
            f"{pred:12.2f}"
            f"{error:12.2f}"
        )

    # =====================================================
    # 7. FOREST SUMMARY
    # =====================================================

    print("\n[7] Forest Summary")

    print(f"Trees trained : {len(model.trees_)}")

    depths = []

    def tree_depth(node):

        if node is None or node.is_leaf:
            return 0

        return 1 + max(
            tree_depth(node.left),
            tree_depth(node.right)
        )

    for tree in model.trees_:
        depths.append(
            tree_depth(tree.root)
        )

    print(
        f"Average depth : {np.mean(depths):.2f}"
    )

    print(
        f"Max depth     : {max(depths)}"
    )

    print(
        f"Min depth     : {min(depths)}"
    )

    # =====================================================
    # DONE
    # =====================================================

    print("\n" + "=" * 60)
    print(" Pipeline Complete")
    print("=" * 60)

    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main()