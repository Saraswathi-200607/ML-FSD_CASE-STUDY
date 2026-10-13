"""Pipeline construction: preprocessing + classifier in ONE scikit-learn object.

Because the preprocessing lives inside the Pipeline, it is fitted on training
data only (inside .fit) and is applied identically at prediction time.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ALGORITHMS = ["logistic_regression", "random_forest", "gradient_boosting"]
# "none"     = train on the data as it is
# "balanced" = give the rare positive class a larger weight during training
#              (cost-sensitive learning; no synthetic rows are created, so no
#              leakage into validation folds is possible)
STRATEGIES = ["none", "balanced"]


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_pipeline(algorithm, strategy, seed=42):
    balanced = strategy == "balanced"
    if algorithm == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced" if balanced else None
        )
    elif algorithm == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=20, n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample" if balanced else None,
        )
    elif algorithm == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", clf)])


def fit_pipeline(pipe, X, y, algorithm, strategy):
    """Fit; GradientBoosting has no class_weight option, so use sample weights."""
    if algorithm == "gradient_boosting" and strategy == "balanced":
        pipe.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe
