# ==========================================================================
# INSURANCE CLAIM-RISK ANALYTICS, ANOMALY SCREENING AND PREDICTOR - ENHANCED
# Single-cell Jupyter version. Paste into ONE notebook cell and run.
# Edit the CONFIG block in PART 9 (DATA_PATH etc.).
# Needs: numpy pandas scikit-learn scipy joblib matplotlib (+ ipywidgets for
# the predictor form). Tested with Python 3.13, scikit-learn 1.9.1, pandas 3.0.
#
# Educational prototype trained on French MTPL data (freMTPL2freq). It is NOT
# validated for India / ICICI Lombard and is not an underwriting, pricing or
# fraud system.
# ==========================================================================
"""What is new compared with Insurance_Claim_Risk_Analytics_FIXED.py
-------------------------------------------------------------------
MODEL
 1. HistGradientBoosting classifier (native categorical handling, early
    stopping, all 406,807 training rows) added to the candidates; a small
    hyper-parameter grid is searched on VALIDATION data only.
 2. Optional isotonic recalibration, accepted only if a cross-fitted check on
    the validation set shows a lower Brier score.
 3. NEW actuarial track: Poisson claim-FREQUENCY model with exposure
    (HistGradientBoosting, loss="poisson", weights = exposure) benchmarked
    against a Poisson GLM. It gives an annual claim frequency (claims per
    policy-year) - the quantity actuaries price - and P(>=1 claim in t years)
    = 1 - exp(-lambda * t).
EVALUATION
 4. Bootstrap 95% confidence intervals for test ROC-AUC, AP and Brier.
 5. Lift / gains table by decile and a review-capacity table (top k%).
 6. Threshold options: F1 (default), cost-based and capacity-based - all
    chosen on validation, compared side by side.
 7. Stability check by Geographic area (AUC and calibration per segment).
 8. Partial-dependence plots for the most important numeric features.
ANOMALY SCREENING
 9. Two separate screens: (A) PROSPECTIVE - pre-outcome fields only, usable
    at underwriting for data-quality checks; (B) RETROSPECTIVE - adds claims,
    exposure, frequency and a model-based "claims surprise" (Poisson tail
    probability of seeing this many claims given the policy's risk).
10. Every flagged record gets its top-3 reasons (robust z-scores) and a
    review priority; flag stability is measured across 5 random seeds.
PREDICTOR
11. Explanations use partial dependence (portfolio-average effect of each
    input value) instead of one-at-a-time changes from a default policy,
    which could be misleading when inputs interact.
12. Shows the annual claim frequency and a policy-term slider; every saved
    prediction is appended to a log with the model version id.
13. Batch scoring of a CSV of new policies: score_csv().
ENGINEERING
14. EDA uses value bands (rank-based deciles were distorted by file order),
    a model card JSON is exported, and all choices are logged.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import chi2_contingency, poisson
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              IsolationForest, RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, mean_poisson_deviance, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger("claim_risk")

RANDOM_STATE = 42
TARGET = "Claim_Flag"
ID_COLUMN = "Policy ID"
CLAIMS_COLUMN = "Number of claims"
EXPOSURE_COLUMN = "Exposure period"
DENSITY_COLUMN = "Population density"
NUMERIC_FEATURES: List[str] = ["Vehicle power", "Vehicle age", "Driver age", "Bonus-Malus score", DENSITY_COLUMN]
CATEGORICAL_FEATURES: List[str] = ["Vehicle brand", "Fuel type", "Geographic area", "Geographic region"]
FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS: List[str] = [ID_COLUMN, CLAIMS_COLUMN, EXPOSURE_COLUMN] + FEATURES
MIN_CLASS_COUNT = 30
BUNDLE_VERSION = 2
# Actuarial convention for freMTPL2freq (Noll et al., 2018; Wuthrich & Merz, 2023)
FREQ_CLAIMS_CAP = 4
FREQ_EXPOSURE_CAP = 1.0

BAND_DEFINITION = (
    "Portfolio-relative bands from the VALIDATION-set predicted probabilities of the selected model: "
    "Low = at or below the 75th percentile, Medium = above the 75th and at or below the 95th percentile, "
    "High = above the 95th percentile. These are rank bands, not absolute risk levels.")


class DataValidationError(ValueError):
    """The data cannot be used (empty, unusable)."""


class SchemaError(DataValidationError):
    """Required columns are missing."""


class InvalidInputError(ValueError):
    """A prediction input is impossible (e.g. negative age)."""


class TrainingError(RuntimeError):
    """Training could not complete."""


# ==========================================================================
# PART 1 - Data validation / cleaning (unchanged logic from the FIXED version)
# ==========================================================================
@dataclass
class DataQualityReport:
    n_raw: int
    n_clean: int
    n_quarantined: int
    quarantine_reasons: Dict[str, int]
    missing_after_blank_handling: Dict[str, int]
    non_numeric_text_coerced_to_missing: Dict[str, int]
    duplicate_full_rows: int
    duplicate_policy_ids: int
    review_flags: Dict[str, int]
    claim_prevalence: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _py(self.__dict__)


def _py(o: Any) -> Any:
    """Convert numpy / pandas objects to JSON-safe python objects."""
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.ndarray):
        return [_py(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _py(o.to_dict(orient="records"))
    return o


def _blank_to_nan(v: Any) -> Any:
    if v is None or v is pd.NA:
        return np.nan
    if isinstance(v, float) and np.isnan(v):
        return np.nan
    if isinstance(v, str):
        v = v.strip()
        return np.nan if v == "" else v
    return v


def _strip_blank(s: pd.Series) -> pd.Series:
    return s.astype(object).map(_blank_to_nan)


def _clean_categorical(s: pd.Series) -> pd.Series:
    def f(v: Any) -> Any:
        v = _blank_to_nan(v)
        if isinstance(v, float) and np.isnan(v):
            return np.nan
        return v if isinstance(v, str) else str(v)
    return s.astype(object).map(f)


def validate_and_clean(df: pd.DataFrame, drop_exact_duplicates: bool = False
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """Validate the raw frame; nothing is dropped silently (see FIXED version docs)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError("Missing required column(s): " + ", ".join(missing))
    if len(df) == 0:
        raise DataValidationError("The file contains no rows.")
    work = df[REQUIRED_COLUMNS].copy()
    coerced: Dict[str, int] = {}
    for c in [ID_COLUMN, CLAIMS_COLUMN, EXPOSURE_COLUMN] + NUMERIC_FEATURES:
        stripped = _strip_blank(work[c])
        num = pd.to_numeric(stripped, errors="coerce")
        coerced[c] = int((stripped.notna() & num.isna()).sum())
        work[c] = num.astype(float)
    for c in CATEGORICAL_FEATURES:
        work[c] = _clean_categorical(work[c])
    claims, expo = work[CLAIMS_COLUMN], work[EXPOSURE_COLUMN]
    checks: Dict[str, pd.Series] = {
        "missing_or_non_numeric_claim_count": claims.isna(),
        "negative_claim_count": claims < 0,
        "non_integer_claim_count": claims.notna() & ((claims % 1) != 0),
        "missing_or_non_numeric_exposure": expo.isna(),
        "non_positive_exposure": expo <= 0,
    }
    for c in NUMERIC_FEATURES:
        checks[f"negative_{c}"] = work[c] < 0
    flags = pd.DataFrame(checks)
    bad = flags.any(axis=1)
    quarantined = work[bad].copy()
    if len(quarantined):
        cols = np.array(flags.columns)
        quarantined["quarantine_reason"] = [";".join(cols[row]) for row in flags[bad].to_numpy()]
    else:
        quarantined["quarantine_reason"] = pd.Series(dtype=object)
    reasons = {k: int(v) for k, v in flags.sum().items() if v > 0}
    clean = work[~bad].copy()
    dup_rows = int(clean.duplicated().sum())
    dup_ids = int(clean[ID_COLUMN].dropna().duplicated().sum())
    notes: List[str] = []
    if drop_exact_duplicates and dup_rows:
        dup_mask = clean.duplicated()
        extra = clean[dup_mask].copy()
        extra["quarantine_reason"] = "exact_duplicate_row"
        quarantined = pd.concat([quarantined, extra])
        clean = clean[~dup_mask]
        reasons["exact_duplicate_row"] = dup_rows
        notes.append(f"{dup_rows:,} exact duplicate rows were quarantined (first copy kept).")
    elif dup_rows:
        notes.append(f"{dup_rows:,} exact duplicate rows were KEPT (option off).")
    if dup_ids:
        notes.append(f"{dup_ids:,} repeated Policy IDs found; the split is group-aware.")
    if clean.empty:
        raise DataValidationError("No valid rows remain after validation.")
    clean[TARGET] = (clean[CLAIMS_COLUMN] > 0).astype(int)
    review = {
        "driver_age_below_18": int((clean["Driver age"] < 18).sum()),
        "vehicle_age_above_50": int((clean["Vehicle age"] > 50).sum()),
        "exposure_above_1_year": int((clean[EXPOSURE_COLUMN] > 1).sum()),
        "claim_count_above_5": int((clean[CLAIMS_COLUMN] > 5).sum()),
    }
    report = DataQualityReport(
        n_raw=len(df), n_clean=len(clean), n_quarantined=len(quarantined), quarantine_reasons=reasons,
        missing_after_blank_handling={c: int(clean[c].isna().sum()) for c in FEATURES if clean[c].isna().any()},
        non_numeric_text_coerced_to_missing={k: v for k, v in coerced.items() if v},
        duplicate_full_rows=dup_rows, duplicate_policy_ids=dup_ids, review_flags=review,
        claim_prevalence=float(clean[TARGET].mean()), notes=notes)
    return clean, quarantined, report


def read_csv_path(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return pd.read_csv(p, low_memory=False)


# ==========================================================================
# PART 2 - Splitting
# ==========================================================================
@dataclass
class TrainConfig:
    seed: int = RANDOM_STATE
    test_size: float = 0.20
    val_size: float = 0.20
    max_rows: Optional[int] = None
    rf_n_estimators: int = 100
    rf_min_samples_leaf: int = 50
    rf_max_train_rows: Optional[int] = 100_000
    hgb_grid: Tuple[Dict[str, Any], ...] = (
        dict(learning_rate=0.10, max_leaf_nodes=31, min_samples_leaf=200),
        dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200),
        dict(learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=500),
    )
    threshold_method: str = "f1"        # "f1", "cost" or "capacity"
    cost_fp: float = 1.0                # cost of reviewing a policy that does not claim
    cost_fn: float = 5.0                # cost of missing a policy that claims
    review_capacity: float = 0.10       # share of policies the team can review
    calibrate: bool = True              # isotonic, only if the validation check improves Brier
    perm_sample_rows: int = 5_000
    perm_repeats: int = 3
    bootstrap_reps: int = 200
    n_jobs: int = -1


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    idx_train: pd.Index
    idx_val: pd.Index
    idx_test: pd.Index
    group_aware: bool


def stratified_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    keep, _ = train_test_split(np.arange(len(df)), train_size=max_rows, stratify=df[TARGET], random_state=seed)
    return df.iloc[np.sort(keep)]


def split_data(clean: pd.DataFrame, cfg: TrainConfig) -> Splits:
    y = clean[TARGET].reset_index(drop=True)
    ids = clean[ID_COLUMN].to_numpy()
    pos = np.arange(len(clean))
    groups = np.where(np.isnan(ids), -(pos + 1), ids)
    use_groups = bool(pd.Series(groups).duplicated().any())

    def holdout(idx: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if not use_groups:
                return train_test_split(idx, test_size=frac, stratify=y.iloc[idx], random_state=cfg.seed)
            n_splits = max(2, int(round(1 / frac)))
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
            a, b = next(sgkf.split(idx, y.iloc[idx], groups[idx]))
            return idx[a], idx[b]
        except ValueError as exc:
            raise TrainingError(f"Could not create a stratified split: {exc}") from exc

    train_val, test = holdout(pos, cfg.test_size)
    train, val = holdout(train_val, cfg.val_size / (1 - cfg.test_size))
    X = clean[FEATURES]
    tr, va, te = np.sort(train), np.sort(val), np.sort(test)
    return Splits(X.iloc[tr], clean[TARGET].iloc[tr], X.iloc[va], clean[TARGET].iloc[va],
                  X.iloc[te], clean[TARGET].iloc[te], clean.index[tr], clean.index[va], clean.index[te], use_groups)


# ==========================================================================
# PART 3 - Models
# ==========================================================================
def _log1p(x: Any) -> Any:
    return np.log1p(np.clip(x, 0, None))


class TreePrep(BaseEstimator, TransformerMixin):
    """Preparation for HistGradientBoosting: log1p(density) and pandas categories
    whose levels are frozen at fit time (unseen levels become missing)."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TreePrep":
        X = pd.DataFrame(X)
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)[FEATURES].copy()
        for c in NUMERIC_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        X[DENSITY_COLUMN] = np.log1p(X[DENSITY_COLUMN].clip(lower=0))
        for c in CATEGORICAL_FEATURES:
            vals = X[c].astype(object).where(X[c].notna(), None)
            X[c] = pd.Categorical([None if v is None else str(v) for v in vals], categories=self.categories_[c])
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(FEATURES, dtype=object)


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """One-hot preprocessing for LR / RF (fitted only inside Pipeline.fit)."""
    plain = [c for c in NUMERIC_FEATURES if c != DENSITY_COLUMN]
    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    dens_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median")),
                                         ("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
        dens_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), plain),
        ("dens", Pipeline(dens_steps), [DENSITY_COLUMN]),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ], remainder="drop")


def build_pipeline(kind: str, cfg: TrainConfig, params: Optional[Dict[str, Any]] = None) -> Pipeline:
    params = params or {}
    if kind == "dummy":
        return Pipeline([("preprocessor", build_preprocessor(False)), ("model", DummyClassifier(strategy="prior"))])
    if kind in ("logreg", "logreg_balanced"):
        est = LogisticRegression(max_iter=1000, random_state=cfg.seed,
                                 class_weight="balanced" if kind == "logreg_balanced" else None)
        return Pipeline([("preprocessor", build_preprocessor(True)), ("model", est)])
    if kind == "random_forest":
        est = RandomForestClassifier(n_estimators=cfg.rf_n_estimators, min_samples_leaf=cfg.rf_min_samples_leaf,
                                     max_features="sqrt", n_jobs=cfg.n_jobs, random_state=cfg.seed)
        return Pipeline([("preprocessor", build_preprocessor(False)), ("model", est)])
    if kind == "hgb":
        est = HistGradientBoostingClassifier(max_iter=1500, early_stopping=True, validation_fraction=0.1,
                                             n_iter_no_change=50, categorical_features="from_dtype",
                                             l2_regularization=1.0, random_state=cfg.seed, **params)
        return Pipeline([("preprocessor", TreePrep()), ("model", est)])
    raise ValueError(f"Unknown model kind: {kind}")


def candidate_specs(cfg: TrainConfig) -> List[Dict[str, Any]]:
    specs = [
        dict(name="Dummy (prior)", kind="dummy", eligible=False, note="Baseline: training prevalence for everyone."),
        dict(name="Logistic Regression (unweighted)", kind="logreg", eligible=True, note="Interpretable baseline."),
        dict(name="Logistic Regression (balanced weights)", kind="logreg_balanced", eligible=False,
             note="Class weights inflate probabilities; reported, not eligible."),
        dict(name="Random Forest", kind="random_forest", eligible=True, note="FIXED-version model (100k rows)."),
    ]
    for p in cfg.hgb_grid:
        specs.append(dict(name=f"HistGradientBoosting (lr={p['learning_rate']}, leaves={p['max_leaf_nodes']}, "
                               f"min_leaf={p['min_samples_leaf']})", kind="hgb", params=p, eligible=True,
                          note="Native categoricals; early stopping on 10% of training rows."))
    return specs


def _fit_quietly(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, **fit_params: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X, y, **fit_params)
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            logger.warning("Convergence warning during fit: %s", w.message)


class CalibratedModel(ClassifierMixin, BaseEstimator):
    """A fitted pipeline followed by an isotonic map fitted on validation predictions."""

    def __init__(self, pipeline: Pipeline, iso: Optional[IsotonicRegression] = None):
        self.pipeline = pipeline
        self.iso = iso

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.pipeline.predict_proba(X)[:, 1]
        if self.iso is not None:
            p = np.clip(self.iso.predict(p), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])

    def fit(self, X: Any, y: Any) -> "CalibratedModel":   # already fitted; present for sklearn API checks
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True


def calibration_check(y_val: Any, p_val: np.ndarray, seed: int) -> Tuple[Optional[IsotonicRegression], Dict[str, float]]:
    """Cross-fitted isotonic check on the validation set: fit on one half, score the other.
    Returns a map fitted on the whole validation set only if Brier improves in both halves."""
    y = np.asarray(y_val).astype(int)
    a, b = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
    gains = []
    for fit_idx, ev_idx in ((a, b), (b, a)):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val[fit_idx], y[fit_idx])
        gains.append(brier_score_loss(y[ev_idx], p_val[ev_idx]) -
                     brier_score_loss(y[ev_idx], np.clip(iso.predict(p_val[ev_idx]), 1e-6, 1 - 1e-6)))
    info = {"brier_gain_half_1": float(gains[0]), "brier_gain_half_2": float(gains[1])}
    if min(gains) > 0:
        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y), info
    return None, info


# ==========================================================================
# PART 4 - Metrics, thresholds, business tables
# ==========================================================================
def compute_metrics(y: Any, p: Any, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prev = float(y.mean())
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if both else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "brier_prevalence_baseline": prev * (1 - prev),
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "prevalence": prev, "mean_predicted_probability": float(p.mean()),
        "flagged_share": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y)),
    }


def choose_threshold(y: Any, p: Any, method: str = "f1", cost_fp: float = 1.0, cost_fn: float = 5.0,
                     capacity: float = 0.10) -> float:
    """Pick a threshold on VALIDATION data. f1: max F1; cost: min cost_fp*FP + cost_fn*FN;
    capacity: flag the top `capacity` share of policies."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if method == "capacity":
        return float(np.quantile(p, 1 - capacity))
    if method == "cost":
        grid = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
        costs = [cost_fp * ((p >= t) & (y == 0)).sum() + cost_fn * ((p < t) & (y == 1)).sum() for t in grid]
        return float(grid[int(np.argmin(costs))])
    if method != "f1":
        raise ValueError("method must be 'f1', 'cost' or 'capacity'")
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def threshold_table(y: Any, p: Any, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for t in sorted(set(float(x) for x in thresholds)):
        m = compute_metrics(y, p, t)
        rows.append({"Threshold": t, "Flagged share": m["flagged_share"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"]})
    return pd.DataFrame(rows)


def calibration_table(y: Any, p: Any, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), q=min(n_bins, len(d)), labels=False)
    return d.groupby("bin").agg(policies=("y", "size"), mean_predicted=("p", "mean"),
                                observed_rate=("y", "mean")).reset_index(drop=True)


def lift_table(y: Any, p: Any, n_bins: int = 10) -> pd.DataFrame:
    """Deciles from highest to lowest predicted risk: claim rate, lift and cumulative capture."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)}).sort_values("p", ascending=False)
    d["decile"] = np.arange(len(d)) * n_bins // len(d) + 1
    t = d.groupby("decile").agg(policies=("y", "size"), claims=("y", "sum"), mean_predicted=("p", "mean"))
    t["claim_rate"] = t["claims"] / t["policies"]
    t["lift"] = t["claim_rate"] / d["y"].mean()
    t["cumulative_share_of_claims"] = t["claims"].cumsum() / d["y"].sum()
    t["cumulative_share_of_policies"] = t["policies"].cumsum() / len(d)
    return t.reset_index()


def capacity_table(y: Any, p: Any, shares: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(p, dtype=float))
    rows = []
    for s in shares:
        k = max(1, int(round(s * len(y))))
        caught = y[order[:k]].sum()
        rows.append({"Review top": s, "Policies reviewed": k, "Claims captured": int(caught),
                     "Share of all claims": caught / y.sum(), "Precision": caught / k, "Lift": (caught / k) / y.mean()})
    return pd.DataFrame(rows)


def bootstrap_ci(y: Any, p: Any, reps: int, seed: int) -> pd.DataFrame:
    """Percentile bootstrap 95% intervals (stratified resampling of the test set)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stats = {"ROC-AUC": [], "Average Precision": [], "Brier": []}
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats["ROC-AUC"].append(roc_auc_score(y[idx], p[idx]))
        stats["Average Precision"].append(average_precision_score(y[idx], p[idx]))
        stats["Brier"].append(brier_score_loss(y[idx], p[idx]))
    point = {"ROC-AUC": roc_auc_score(y, p), "Average Precision": average_precision_score(y, p),
             "Brier": brier_score_loss(y, p)}
    return pd.DataFrame([{"Metric": k, "Estimate": point[k], "CI 2.5%": np.percentile(v, 2.5),
                          "CI 97.5%": np.percentile(v, 97.5)} for k, v in stats.items()])


def segment_table(X: pd.DataFrame, y: Any, p: Any, col: str) -> pd.DataFrame:
    d = pd.DataFrame({"seg": X[col].fillna("Missing").to_numpy(), "y": np.asarray(y), "p": np.asarray(p)})
    rows = []
    for seg, g in d.groupby("seg"):
        rows.append({col: seg, "Policies": len(g), "Observed rate": g.y.mean(), "Mean predicted": g.p.mean(),
                     "Predicted / observed": g.p.mean() / g.y.mean() if g.y.mean() > 0 else np.nan,
                     "ROC-AUC": roc_auc_score(g.y, g.p) if g.y.nunique() == 2 else np.nan})
    return pd.DataFrame(rows)


def assign_band(p: Any, bands: Dict[str, float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p <= bands["low_max"], "Low", np.where(p <= bands["medium_max"], "Medium", "High"))


def band_table(y: Any, p: Any, bands: Dict[str, float]) -> pd.DataFrame:
    d = pd.DataFrame({"band": assign_band(p, bands), "y": np.asarray(y), "p": np.asarray(p)})
    out = d.groupby("band").agg(policies=("y", "size"), mean_predicted=("p", "mean"), observed_claim_rate=("y", "mean"))
    out = out.reindex(["Low", "Medium", "High"]).dropna(how="all").reset_index()
    out["share_of_policies"] = out["policies"] / len(d)
    return out


# ==========================================================================
# PART 5 - Claim-occurrence training (classification track)
# ==========================================================================
@dataclass
class TrainingResult:
    selected_name: str
    model: Any
    threshold: float
    threshold_rule: str
    val_table: pd.DataFrame
    threshold_options: pd.DataFrame
    test_metrics: Dict[str, float]
    dummy_test_metrics: Dict[str, float]
    baseline_test_metrics: Dict[str, float]
    test_threshold_table: pd.DataFrame
    calibration: pd.DataFrame
    calibration_info: Dict[str, Any]
    risk_bands: Dict[str, Any]
    band_test_table: pd.DataFrame
    lift: pd.DataFrame
    capacity: pd.DataFrame
    ci: pd.DataFrame
    segments: pd.DataFrame
    importance: pd.DataFrame
    split_info: Dict[str, Any]
    y_test: np.ndarray
    p_test: np.ndarray
    p_val: np.ndarray
    splits: Splits
    notes: List[str]


def train_classifier(clean: pd.DataFrame, cfg: TrainConfig,
                     progress: Optional[Callable[[float, str], None]] = None) -> TrainingResult:
    prog = progress or (lambda f, m: None)
    counts = clean[TARGET].value_counts()
    if counts.get(1, 0) < MIN_CLASS_COUNT or counts.get(0, 0) < MIN_CLASS_COUNT:
        raise TrainingError(f"Need at least {MIN_CLASS_COUNT} policies in each class; found {counts.to_dict()}.")
    notes: List[str] = []
    data = clean
    if cfg.max_rows and len(data) > cfg.max_rows:
        data = stratified_sample(data, cfg.max_rows, cfg.seed)
        notes.append(f"Stratified sample of {len(data):,} rows drawn before splitting.")
    sp = split_data(data, cfg)
    specs = candidate_specs(cfg)
    fitted, val_prob, rows = {}, {}, []
    for i, spec in enumerate(specs):
        prog(0.02 + 0.55 * i / len(specs), f"Fitting {spec['name']}…")
        pipe = build_pipeline(spec["kind"], cfg, spec.get("params"))
        Xtr, ytr = sp.X_train, sp.y_train
        if spec["kind"] == "random_forest" and cfg.rf_max_train_rows and len(Xtr) > cfg.rf_max_train_rows:
            Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=cfg.rf_max_train_rows, stratify=ytr,
                                              random_state=cfg.seed)
        t0 = time.time()
        _fit_quietly(pipe, Xtr, ytr)
        secs = time.time() - t0
        p = pipe.predict_proba(sp.X_val)[:, 1]
        m = compute_metrics(sp.y_val, p, choose_threshold(sp.y_val, p, "f1"))
        fitted[spec["name"]], val_prob[spec["name"]] = pipe, p
        extra = f"; {pipe.named_steps['model'].n_iter_} boosting iterations" if spec["kind"] == "hgb" else ""
        rows.append({"Model": spec["name"], "Eligible": spec["eligible"], "Val ROC-AUC": m["roc_auc"],
                     "Val Average Precision": m["average_precision"], "Val Brier": m["brier"],
                     "Val Log loss": m["log_loss"], "Val F1 (val-optimal threshold)": m["f1"],
                     "Training rows": len(Xtr), "Fit seconds": round(secs, 1), "Note": spec["note"] + extra})
    val_table = pd.DataFrame(rows)
    elig = val_table[val_table["Eligible"]]
    selected = str(elig.sort_values("Val Average Precision", ascending=False).iloc[0]["Model"])
    pipe = fitted[selected]
    p_val_raw = val_prob[selected]

    # ---- calibration (validation only) ------------------------------------
    iso, cal_info = (None, {"skipped": True})
    if cfg.calibrate:
        iso, cal_info = calibration_check(sp.y_val, p_val_raw, cfg.seed)
    cal_info["applied"] = iso is not None
    model = CalibratedModel(pipe, iso)
    p_val = model.predict_proba(sp.X_val)[:, 1]

    # ---- threshold options (validation only) ------------------------------
    opts = []
    for meth in ("f1", "cost", "capacity"):
        t = choose_threshold(sp.y_val, p_val, meth, cfg.cost_fp, cfg.cost_fn, cfg.review_capacity)
        m = compute_metrics(sp.y_val, p_val, t)
        opts.append({"Rule": meth, "Threshold": t, "Flagged share": m["flagged_share"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"],
                     "Cost (FP×c_fp + FN×c_fn)": cfg.cost_fp * m["fp"] + cfg.cost_fn * m["fn"]})
    threshold_options = pd.DataFrame(opts)
    threshold = float(threshold_options.loc[threshold_options.Rule == cfg.threshold_method, "Threshold"].iloc[0])
    rule = {"f1": "Maximises F1 on the validation set",
            "cost": f"Minimises {cfg.cost_fp}*FP + {cfg.cost_fn}*FN on the validation set",
            "capacity": f"Flags the top {cfg.review_capacity:.0%} of validation predictions"}[cfg.threshold_method]
    bands = {"low_max": float(np.quantile(p_val, 0.75)), "medium_max": float(np.quantile(p_val, 0.95)),
             "definition": BAND_DEFINITION}

    # ---- one-time test evaluation -----------------------------------------
    prog(0.62, "Evaluating the selected model once on the held-out test set…")
    p_test = model.predict_proba(sp.X_test)[:, 1]
    test_metrics = compute_metrics(sp.y_test, p_test, threshold)
    dummy_metrics = compute_metrics(sp.y_test, fitted["Dummy (prior)"].predict_proba(sp.X_test)[:, 1], threshold)
    rf_p = fitted["Random Forest"].predict_proba(sp.X_test)[:, 1]
    baseline_metrics = compute_metrics(sp.y_test, rf_p, choose_threshold(sp.y_val, val_prob["Random Forest"], "f1"))
    grid = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, threshold]
    prog(0.68, "Bootstrap confidence intervals…")
    ci = bootstrap_ci(sp.y_test, p_test, cfg.bootstrap_reps, cfg.seed)
    prog(0.75, "Permutation importance on validation data…")
    n_perm = min(cfg.perm_sample_rows, len(sp.X_val))
    Xp, _, yp, _ = train_test_split(sp.X_val, sp.y_val, train_size=n_perm, stratify=sp.y_val, random_state=cfg.seed)
    perm = permutation_importance(model, Xp, yp, scoring="average_precision", n_repeats=cfg.perm_repeats,
                                  random_state=cfg.seed, n_jobs=1)
    importance = pd.DataFrame({"Feature": FEATURES, "Mean drop in Average Precision": perm.importances_mean,
                               "Std": perm.importances_std}).sort_values("Mean drop in Average Precision",
                                                                          ascending=False).reset_index(drop=True)
    if iso is None:
        notes.append("Isotonic recalibration was NOT applied: the cross-fitted validation check did not improve Brier in both halves.")
    else:
        notes.append("Isotonic recalibration applied (fitted on validation predictions; improved Brier in both cross-fitted halves).")
    split_info = {"design": "Stratified random 60/20/20 (no policy date => no out-of-time split)",
                  "group_aware_by_policy_id": sp.group_aware, "seed": cfg.seed,
                  "train_rows": len(sp.X_train), "validation_rows": len(sp.X_val), "test_rows": len(sp.X_test),
                  "train_prevalence": float(sp.y_train.mean()), "validation_prevalence": float(sp.y_val.mean()),
                  "test_prevalence": float(sp.y_test.mean())}
    prog(0.85, "Classification track done.")
    return TrainingResult(selected, model, threshold, rule, val_table, threshold_options, test_metrics, dummy_metrics,
                          baseline_metrics, threshold_table(sp.y_test, p_test, grid), calibration_table(sp.y_test, p_test),
                          cal_info, bands, band_table(sp.y_test, p_test, bands), lift_table(sp.y_test, p_test),
                          capacity_table(sp.y_test, p_test), ci, segment_table(sp.X_test, sp.y_test, p_test, "Geographic area"),
                          importance, split_info, np.asarray(sp.y_test), p_test, p_val, sp, notes)


# ==========================================================================
# PART 6 - Claim-frequency track (actuarial, exposure-aware)
# ==========================================================================
@dataclass
class FrequencyResult:
    selected_name: str
    model: Pipeline
    val_table: pd.DataFrame
    test_metrics: Dict[str, float]
    calibration: pd.DataFrame
    lam_test: np.ndarray


def _freq_target(clean: pd.DataFrame, idx: pd.Index) -> Tuple[pd.Series, pd.Series, pd.Series]:
    n = clean.loc[idx, CLAIMS_COLUMN].clip(upper=FREQ_CLAIMS_CAP)
    e = clean.loc[idx, EXPOSURE_COLUMN].clip(upper=FREQ_EXPOSURE_CAP)
    return n / e, e, n


def train_frequency(clean: pd.DataFrame, sp: Splits, cfg: TrainConfig) -> FrequencyResult:
    """Poisson frequency models: y = claims / exposure, sample_weight = exposure (equivalent to a
    log-exposure offset). Selected on validation Poisson deviance; test scored once."""
    ytr, wtr, _ = _freq_target(clean, sp.idx_train)
    yv, wv, _ = _freq_target(clean, sp.idx_val)
    null = float(np.average(ytr, weights=wtr))
    cands = {
        "Poisson GLM (one-hot, benchmark)": Pipeline([("preprocessor", build_preprocessor(True)),
                                                      ("model", PoissonRegressor(alpha=1e-4, max_iter=300))]),
        "Poisson HistGradientBoosting (lr=0.10)": Pipeline([("preprocessor", TreePrep()), ("model", HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.10, max_leaf_nodes=31, min_samples_leaf=200, max_iter=1500,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, categorical_features="from_dtype",
            l2_regularization=1.0, random_state=cfg.seed))]),
        "Poisson HistGradientBoosting (lr=0.05)": Pipeline([("preprocessor", TreePrep()), ("model", HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, max_iter=1500,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, categorical_features="from_dtype",
            l2_regularization=1.0, random_state=cfg.seed))]),
    }
    null_dev = mean_poisson_deviance(yv, np.full(len(yv), null), sample_weight=wv)
    rows, fitted = [], {}
    for name, pipe in cands.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(sp.X_train, ytr, model__sample_weight=wtr.to_numpy())
        lam = np.clip(pipe.predict(sp.X_val), 1e-6, None)
        dev = mean_poisson_deviance(yv, lam, sample_weight=wv)
        pe = 1 - np.exp(-lam * wv.to_numpy())
        rows.append({"Model": name, "Val Poisson deviance": dev, "Val deviance explained (D²)": 1 - dev / null_dev,
                     "Val AUC of P(claim | own exposure)": roc_auc_score(sp.y_val, pe),
                     "Val AUC of annual frequency only": roc_auc_score(sp.y_val, lam),
                     "Fit seconds": round(time.time() - t0, 1)})
        fitted[name] = pipe
    vt = pd.DataFrame(rows)
    best = str(vt.sort_values("Val Poisson deviance").iloc[0]["Model"])
    model = fitted[best]
    yt, wt, nt = _freq_target(clean, sp.idx_test)
    lam_t = np.clip(model.predict(sp.X_test), 1e-6, None)
    null_dev_t = mean_poisson_deviance(yt, np.full(len(yt), null), sample_weight=wt)
    dev_t = mean_poisson_deviance(yt, lam_t, sample_weight=wt)
    pe_t = 1 - np.exp(-lam_t * wt.to_numpy())
    test = {"poisson_deviance": dev_t, "null_deviance": null_dev_t, "deviance_explained_D2": 1 - dev_t / null_dev_t,
            "auc_P_claim_given_exposure": roc_auc_score(sp.y_test, pe_t),
            "ap_P_claim_given_exposure": average_precision_score(sp.y_test, pe_t),
            "auc_annual_frequency_only": roc_auc_score(sp.y_test, lam_t),
            "predicted_claims": float((lam_t * wt).sum()), "observed_claims_capped": float(nt.sum()),
            "portfolio_annual_frequency": float(nt.sum() / wt.sum())}
    d = pd.DataFrame({"lam": lam_t, "e": wt.to_numpy(), "n": nt.to_numpy()})
    d["bin"] = pd.qcut(d["lam"].rank(method="first"), 10, labels=False)
    cal = d.groupby("bin").apply(lambda g: pd.Series({
        "exposure_years": g.e.sum(), "predicted_frequency": np.average(g.lam, weights=g.e),
        "observed_frequency": g.n.sum() / g.e.sum()}), include_groups=False).reset_index(drop=True)
    return FrequencyResult(best, model, vt, test, cal, lam_t)


# ==========================================================================
# PART 7 - Anomaly screening (prospective + retrospective)
# ==========================================================================
ANOMALY_CAVEAT = ("Screening only. An anomaly is a statistical oddity, NOT a fraud finding. "
                  "Screen A uses pre-outcome fields (usable at underwriting for data checks); "
                  "Screen B uses claims and exposure (post-outcome, retrospective review only).")


def _anomaly_matrix(clean: pd.DataFrame, screen: str, lam: Optional[np.ndarray] = None) -> pd.DataFrame:
    m = pd.DataFrame(index=clean.index)
    for c in ["Vehicle power", "Vehicle age", "Driver age", "Bonus-Malus score"]:
        m[c] = clean[c]
    m["log Population density"] = np.log1p(clean[DENSITY_COLUMN])
    if screen == "B":
        e = clean[EXPOSURE_COLUMN]
        m["log Exposure"] = np.log(e.clip(lower=1e-3))
        m["Number of claims"] = clean[CLAIMS_COLUMN]
        m["log1p Claim frequency"] = np.log1p(clean[CLAIMS_COLUMN] / e.clip(lower=1e-3))
        if lam is not None:
            mu = lam * e.clip(upper=FREQ_EXPOSURE_CAP).to_numpy()
            # -log10 of P(N >= n | mu): large when more claims than the risk profile predicts
            sf = poisson.sf(clean[CLAIMS_COLUMN].to_numpy() - 1, mu)
            m["Claims surprise (-log10 p)"] = -np.log10(np.clip(sf, 1e-300, 1))
    return m.fillna(m.median())


def _reasons(m: pd.DataFrame, rows: pd.Index, k: int = 3) -> List[str]:
    med = m.median()
    iqr = (m.quantile(0.75) - m.quantile(0.25)).replace(0, np.nan)
    iqr = iqr.fillna(m.std().replace(0, 1))
    z = ((m.loc[rows] - med) / iqr)
    out = []
    for _, r in z.iterrows():
        top = r.abs().sort_values(ascending=False).head(k)
        out.append("; ".join(f"{f} {'high' if r[f] > 0 else 'low'} (z={r[f]:+.1f})" for f in top.index))
    return out


def run_anomaly_screen(clean: pd.DataFrame, screen: str, contamination: float = 0.01, seed: int = RANDOM_STATE,
                       max_fit_rows: int = 100_000, n_estimators: int = 200, lam: Optional[np.ndarray] = None
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = _anomaly_matrix(clean, screen, lam)
    rng = np.random.RandomState(seed)
    fit_idx = rng.choice(len(m), size=min(max_fit_rows, len(m)), replace=False)
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=seed, n_jobs=-1)
    iso.fit(m.iloc[fit_idx].to_numpy())
    scores = np.empty(len(m))
    for s in range(0, len(m), 50_000):
        scores[s:s + 50_000] = -iso.decision_function(m.iloc[s:s + 50_000].to_numpy())
    out = clean[[ID_COLUMN, CLAIMS_COLUMN, EXPOSURE_COLUMN] + NUMERIC_FEATURES].copy()
    out["Anomaly_Score"] = scores
    out["Anomaly_Flag"] = (scores > 0).astype(int)
    if "Claims surprise (-log10 p)" in m:
        out["Claims_Surprise"] = m["Claims surprise (-log10 p)"].to_numpy()
        # review priority: average of the two percentile ranks (isolation and claims surprise)
        out["Review_Priority"] = (pd.Series(scores).rank(pct=True).to_numpy() +
                                  out["Claims_Surprise"].rank(pct=True).to_numpy()) / 2
    else:
        out["Review_Priority"] = pd.Series(scores).rank(pct=True).to_numpy()
    flagged = out.index[out["Anomaly_Flag"] == 1]
    out["Reasons"] = ""
    out.loc[flagged, "Reasons"] = _reasons(m, flagged)
    # rule-based data-quality checks (independent of the forest)
    dq = pd.Series("", index=out.index)
    dq[clean["Vehicle age"] > 50] += "vehicle age > 50; "
    dq[clean[EXPOSURE_COLUMN] > 1] += "exposure > 1 year; "
    dq[clean[CLAIMS_COLUMN] > 5] += "more than 5 claims; "
    out["Data_Quality_Rule"] = dq.str.rstrip("; ")
    return out.sort_values("Review_Priority", ascending=False), m


def anomaly_stability(m: pd.DataFrame, contamination: float = 0.01, seeds: Tuple[int, ...] = (1, 2, 3, 4, 5),
                      max_fit_rows: int = 100_000, eval_rows: int = 200_000) -> pd.DataFrame:
    """Jaccard overlap of the flagged sets produced with different random seeds."""
    rng = np.random.RandomState(0)
    ev = m.iloc[rng.choice(len(m), size=min(eval_rows, len(m)), replace=False)].to_numpy()
    sets = []
    for s in seeds:
        r = np.random.RandomState(s)
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=s, n_jobs=-1)
        iso.fit(m.iloc[r.choice(len(m), size=min(max_fit_rows, len(m)), replace=False)].to_numpy())
        sets.append(set(np.where(iso.decision_function(ev) < 0)[0]))
    rows = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            rows.append({"Seed A": seeds[i], "Seed B": seeds[j],
                         "Jaccard overlap": len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))})
    return pd.DataFrame(rows)


def run_association_tests(clean: pd.DataFrame) -> pd.DataFrame:
    """Chi-square tests; numeric predictors grouped into value-based quantile bins (ties kept together)."""
    rows = []
    for feature in FEATURES:
        vals = clean[feature]
        if feature in NUMERIC_FEATURES:
            try:
                group = pd.qcut(vals, q=10, duplicates="drop").astype(str)
            except ValueError:
                continue
            grouping = "Up to 10 value-based quantile bins"
        else:
            group = vals.astype(object).where(vals.notna(), "Missing").astype(str)
            grouping = "Original categories"
        table = pd.crosstab(group, clean[TARGET])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = chi2_contingency(table)
        n = int(table.to_numpy().sum())
        rows.append({"Feature": feature, "Chi-square": float(chi2), "Degrees of freedom": int(dof),
                     "P-value": float(p_value), "Cramer's V": float(np.sqrt(chi2 / (n * max(1, min(table.shape) - 1)))),
                     "Grouping": grouping})
    res = pd.DataFrame(rows)
    res["P-value (Bonferroni)"] = np.minimum(res["P-value"] * len(res), 1.0)
    return res.sort_values("Cramer's V", ascending=False).reset_index(drop=True)


# ==========================================================================
# PART 8 - Bundle, prediction, explanations, batch scoring
# ==========================================================================
def build_schema(X_train: pd.DataFrame) -> Dict[str, Any]:
    num = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()), "median": float(X_train[c].median()),
               "p01": float(X_train[c].quantile(0.01)), "p99": float(X_train[c].quantile(0.99))} for c in NUMERIC_FEATURES}
    cat = {}
    for c in CATEGORICAL_FEATURES:
        vc = X_train[c].dropna().value_counts()
        cat[c] = {"categories": sorted(vc.index.tolist()), "mode": str(vc.index[0])}
    return {"numeric": num, "categorical": cat, "features": FEATURES}


def _versions() -> Dict[str, str]:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__}


def make_bundle(res: TrainingResult, freq: Optional[FrequencyResult], quality: DataQualityReport,
                cfg: TrainConfig) -> Dict[str, Any]:
    trained = datetime.now(timezone.utc).isoformat(timespec="seconds")
    vid = hashlib.sha1(f"{trained}|{res.selected_name}|{res.threshold:.8f}".encode()).hexdigest()[:10]
    background = res.splits.X_train.sample(min(300, len(res.splits.X_train)), random_state=cfg.seed)
    meta = {"model_version_id": vid, "trained_at_utc": trained, "model_name": res.selected_name,
            "frequency_model_name": freq.selected_name if freq else None, "seed": cfg.seed, "target": TARGET,
            "target_definition": "1 if Number of claims > 0 else 0", "split": res.split_info,
            "config": _py({k: v for k, v in cfg.__dict__.items() if k != "hgb_grid"}), "versions": _versions(),
            "selection_rule": "Highest validation Average Precision among eligible models",
            "threshold_rule": res.threshold_rule, "calibration": res.calibration_info,
            "excluded_from_predictors": [ID_COLUMN, CLAIMS_COLUMN, EXPOSURE_COLUMN],
            "data_quality_counts": quality.to_dict(), "notes": res.notes,
            "intended_use": "Educational decision-support prototype; not an underwriting, pricing or fraud system.",
            "validity_warning": "Trained on French MTPL data; not validated for any Indian insurer."}
    return {"bundle_version": BUNDLE_VERSION, "model": res.model, "freq_model": freq.model if freq else None,
            "features": FEATURES, "schema": build_schema(res.splits.X_train), "threshold": res.threshold,
            "threshold_rule": res.threshold_rule, "risk_bands": res.risk_bands, "background": background,
            "base_rate": res.test_metrics["prevalence"],
            "base_frequency": freq.test_metrics["portfolio_annual_frequency"] if freq else None,
            "metrics": _py({"validation": res.val_table, "test_selected_model": res.test_metrics,
                            "test_dummy": res.dummy_test_metrics, "test_fixed_version_rf": res.baseline_test_metrics,
                            "test_bootstrap_ci": res.ci,
                            "frequency_test": freq.test_metrics if freq else None}),
            "metadata": _py(meta)}


def save_bundle(bundle: Dict[str, Any], dest: Union[str, Path]) -> None:
    joblib.dump(bundle, dest, compress=3)


def load_bundle(src: Union[str, Path]) -> Dict[str, Any]:
    """joblib uses pickle: only load files you trust, in an environment where this code is defined."""
    obj = joblib.load(src)
    needed = {"model", "features", "schema", "threshold", "risk_bands", "metadata"}
    if not isinstance(obj, dict) or not needed.issubset(obj):
        raise ValueError("File is not a valid claim-risk model bundle (version 2).")
    saved = obj["metadata"].get("versions", {}).get("scikit-learn")
    if saved and saved != sklearn.__version__:
        logger.warning("Bundle trained with scikit-learn %s, running %s.", saved, sklearn.__version__)
    return obj


def prepare_inputs(df: pd.DataFrame, bundle: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise SchemaError("Missing input field(s): " + ", ".join(missing))
    X = df[bundle["features"]].copy()
    warn: List[str] = []
    schema = bundle["schema"]
    for c in NUMERIC_FEATURES:
        stripped = _strip_blank(X[c])
        num = pd.to_numeric(stripped, errors="coerce").astype(float)
        if (stripped.notna() & num.isna()).any():
            warn.append(f"{c}: non-numeric value treated as missing.")
        if num.isna().any():
            warn.append(f"{c}: missing; the model's own missing-value handling is used.")
        if (num < 0).any():
            raise InvalidInputError(f"{c} cannot be negative.")
        lo, hi = schema["numeric"][c]["min"], schema["numeric"][c]["max"]
        if ((num < lo) | (num > hi)).any():
            warn.append(f"{c}: outside the training range ({lo:g}-{hi:g}); the prediction is an extrapolation.")
        X[c] = num
    for c in CATEGORICAL_FEATURES:
        X[c] = _clean_categorical(X[c])
        if X[c].isna().any():
            warn.append(f"{c}: missing; treated as unknown.")
        unseen = set(X[c].dropna()) - set(schema["categorical"][c]["categories"])
        if unseen:
            warn.append(f"{c}: unseen category {sorted(unseen)}; the estimate is less reliable.")
    return X, list(dict.fromkeys(warn))


def score_input(bundle: Dict[str, Any], row: Dict[str, Any], threshold: Optional[float] = None,
                term_years: float = 1.0) -> Dict[str, Any]:
    X, warn = prepare_inputs(pd.DataFrame([row]), bundle)
    p = float(bundle["model"].predict_proba(X)[:, 1][0])
    if not np.isfinite(p) or not (0 <= p <= 1):
        raise RuntimeError("Model returned an invalid probability.")
    thr = bundle["threshold"] if threshold is None else float(threshold)
    out = {"probability": p, "score_0_100": round(100 * p, 1), "risk_band": str(assign_band([p], bundle["risk_bands"])[0]),
           "threshold": thr, "above_threshold": bool(p >= thr), "warnings": warn,
           "model_version_id": bundle["metadata"]["model_version_id"]}
    if bundle.get("freq_model") is not None:
        lam = float(np.clip(bundle["freq_model"].predict(X)[0], 1e-6, None))
        out.update({"annual_frequency": lam, "term_years": term_years,
                    "p_claim_in_term": float(1 - np.exp(-lam * term_years))})
    return out


def explain_pd(bundle: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, float]:
    """Partial-dependence explanation: for each input, the average change in predicted probability
    across a 300-policy background sample when that input is set to this policy's value.
    Portfolio-average effect of the value; not causal; contributions do not sum to the prediction."""
    X, _ = prepare_inputs(pd.DataFrame([row]), bundle)
    bg = bundle["background"]
    base = bundle["model"].predict_proba(bg)[:, 1].mean()
    frames = []
    for f in FEATURES:
        b = bg.copy()
        b[f] = X[f].iloc[0]
        frames.append(b)
    p = bundle["model"].predict_proba(pd.concat(frames, ignore_index=True))[:, 1].reshape(len(FEATURES), len(bg))
    return {f: float(p[i].mean() - base) for i, f in enumerate(FEATURES)}


def score_csv(bundle: Dict[str, Any], path_in: Union[str, Path], path_out: Union[str, Path],
              term_years: float = 1.0) -> pd.DataFrame:
    """Batch-score new policies (CSV with the 9 feature columns). Rows with impossible values are
    reported, not silently dropped."""
    df = pd.read_csv(path_in, low_memory=False)
    X, warn = prepare_inputs(df, bundle)
    out = df.copy()
    out["predicted_probability"] = bundle["model"].predict_proba(X)[:, 1]
    out["risk_band"] = assign_band(out["predicted_probability"], bundle["risk_bands"])
    out["above_threshold"] = out["predicted_probability"] >= bundle["threshold"]
    if bundle.get("freq_model") is not None:
        lam = np.clip(bundle["freq_model"].predict(X), 1e-6, None)
        out["annual_frequency"] = lam
        out[f"p_claim_in_{term_years:g}_years"] = 1 - np.exp(-lam * term_years)
    out["model_version_id"] = bundle["metadata"]["model_version_id"]
    out.to_csv(path_out, index=False)
    for w in warn:
        logger.warning("Batch input: %s", w)
    return out


# ==========================================================================
# PART 9 - RUN EVERYTHING  (edit the CONFIG block, then run the cell)
# ==========================================================================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

try:
    from IPython.display import Markdown, display
except ImportError:  # plain python
    def display(x: Any) -> None:
        print(x)

    def Markdown(x: str) -> str:
        return x

# ---------------------------- CONFIG ---------------------------------------
DATA_PATH = os.environ.get("INSURANCE_DATA_PATH", "data/freMTPL2freq.csv")
OUTPUT_DIR = Path(os.environ.get("INSURANCE_OUTPUT_DIR", "insurance_risk_outputs_enhanced"))
SEED = 42
MAX_ROWS = None              # e.g. 100_000 for a quick trial (sample drawn BEFORE splitting)
THRESHOLD_METHOD = "f1"      # "f1", "cost" or "capacity"
COST_FP, COST_FN = 1.0, 5.0  # set from the business: cost of a needless review vs a missed claim
REVIEW_CAPACITY = 0.10       # share of policies the team can review (capacity rule)
RUN_FREQUENCY = True
RUN_ANOMALY = True
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1") == "1"
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def _show(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=200, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)


# ---- 1. Load + validate ---------------------------------------------------
display(Markdown("## 1. Dataset and data quality"))
raw = read_csv_path(DATA_PATH)
clean, quarantined, report = validate_and_clean(raw)
display(pd.DataFrame({"Item": ["Rows", "Usable rows", "Quarantined", "Claim prevalence", "Duplicate rows", "Repeated IDs"],
                      "Value": [f"{report.n_raw:,}", f"{report.n_clean:,}", f"{report.n_quarantined:,}",
                                f"{report.claim_prevalence:.2%}", report.duplicate_full_rows, report.duplicate_policy_ids]}))
print("Missing predictors:", report.missing_after_blank_handling or "none", "| Review flags:", report.review_flags)

# ---- 1b. EDA with VALUE bands (rank deciles were distorted by file order) ----
bands_def = {"Bonus-Malus score": [49, 50, 60, 70, 80, 90, 100, 230],
             "Vehicle age": [-1, 0, 1, 2, 4, 6, 8, 10, 15, 20, 100],
             "Driver age": [17, 21, 25, 30, 40, 50, 60, 70, 100]}
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), ["Bonus-Malus score", "Vehicle age", "Driver age", "Vehicle brand", "Fuel type", "Geographic area"]):
    if col in bands_def:
        t = clean.groupby(pd.cut(clean[col], bands_def[col]), observed=True)[TARGET].mean()
        t.index = [f"{int(i.left) + 1}-{int(i.right)}" if i.right - i.left > 1 else f"{int(i.right)}" for i in t.index]
    else:
        t = clean.groupby(col)[TARGET].mean().sort_values()
    ax.bar(t.index.astype(str), t.values * 100, color="#1f4e79")
    ax.axhline(clean[TARGET].mean() * 100, ls="--", color="grey")
    ax.set_title(f"Claim rate (%) by {col}")
    ax.tick_params(axis="x", rotation=60)
_show(fig, "claim_rate_eda_value_bands.png")
assoc = run_association_tests(clean)
display(assoc)

# ---- 2. Classification track ----------------------------------------------
display(Markdown("## 2. Claim-occurrence model (classification track, exposure not used)"))
cfg = TrainConfig(seed=SEED, max_rows=MAX_ROWS, threshold_method=THRESHOLD_METHOD, cost_fp=COST_FP,
                  cost_fn=COST_FN, review_capacity=REVIEW_CAPACITY)
res = train_classifier(clean, cfg, progress=lambda f, m: print(f"[{f:4.0%}] {m}", flush=True))
display(res.val_table.drop(columns=["Note"]).style.format({c: "{:.4f}" for c in res.val_table.columns if c.startswith("Val")}))
print(f"Selected: {res.selected_name} | calibration: {res.calibration_info} | threshold {res.threshold:.4f} ({res.threshold_rule})")
display(Markdown("### Threshold options (validation)"))
display(res.threshold_options.style.format({"Threshold": "{:.4f}", "Flagged share": "{:.2%}", "Precision": "{:.3f}",
                                            "Recall": "{:.3f}", "F1": "{:.3f}", "Cost (FP×c_fp + FN×c_fn)": "{:,.0f}"}))

display(Markdown("### Final TEST results (evaluated once)"))
t, b, rf = res.test_metrics, res.dummy_test_metrics, res.baseline_test_metrics
cmp = pd.DataFrame({"Selected model": t, "FIXED-version Random Forest": rf, "Dummy": b}).drop(index=["tn", "fp", "fn", "tp"]).T
display(cmp.style.format("{:.4f}"))
display(res.ci.style.format({"Estimate": "{:.4f}", "CI 2.5%": "{:.4f}", "CI 97.5%": "{:.4f}"}))
display(pd.DataFrame([[t["tn"], t["fp"]], [t["fn"], t["tp"]]], index=["Actual no claim", "Actual claim"],
                     columns=["Predicted no claim", "Predicted claim"]))
print(BAND_DEFINITION)
display(res.band_test_table)
display(Markdown("### Lift / gains and review capacity (test)"))
display(res.lift.style.format({"mean_predicted": "{:.2%}", "claim_rate": "{:.2%}", "lift": "{:.2f}",
                               "cumulative_share_of_claims": "{:.1%}", "cumulative_share_of_policies": "{:.0%}"}))
display(res.capacity.style.format({"Review top": "{:.0%}", "Share of all claims": "{:.1%}", "Precision": "{:.2%}", "Lift": "{:.2f}"}))
display(Markdown("### Stability by Geographic area (test)"))
display(res.segments.style.format({"Observed rate": "{:.2%}", "Mean predicted": "{:.2%}",
                                   "Predicted / observed": "{:.2f}", "ROC-AUC": "{:.3f}"}))
for n in res.notes:
    print("NOTE:", n)

fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
fpr, tpr, _ = roc_curve(res.y_test, res.p_test)
ax[0].plot(fpr, tpr, label=f"Selected (AUC {t['roc_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "--", c="grey"); ax[0].set_title("ROC (test)"); ax[0].legend()
pr, rc, _ = precision_recall_curve(res.y_test, res.p_test)
ax[1].plot(rc, pr); ax[1].axhline(t["prevalence"], ls="--", c="grey"); ax[1].set_title(f"Precision-Recall (test, AP {t['average_precision']:.3f})")
cal = res.calibration; top = max(cal.mean_predicted.max(), cal.observed_rate.max()) * 1.1
ax[2].plot(cal.mean_predicted, cal.observed_rate, "o-"); ax[2].plot([0, top], [0, top], "--", c="grey")
ax[2].set_title("Calibration (test deciles)"); ax[2].set_xlabel("Predicted"); ax[2].set_ylabel("Observed")
ax[3].plot(np.r_[0, res.lift.cumulative_share_of_policies], np.r_[0, res.lift.cumulative_share_of_claims], "o-", label="Model")
ax[3].plot([0, 1], [0, 1], "--", c="grey", label="Random"); ax[3].set_title("Cumulative gains (test)"); ax[3].legend()
_show(fig, "evaluation_plots.png")

imp = res.importance.iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
ax[0].barh(imp.Feature, imp["Mean drop in Average Precision"], xerr=imp.Std); ax[0].set_title("Permutation importance (validation)")
top_num = [f for f in res.importance.Feature if f in NUMERIC_FEATURES][:3]
Xpd = res.splits.X_val.sample(3000, random_state=SEED)
for f in top_num:
    pdp = partial_dependence(res.model, Xpd, [f], kind="average", grid_resolution=30, response_method="predict_proba")
    ax[1].plot(pdp["grid_values"][0], pdp["average"][0] * 100, label=f)
ax[1].set_title("Partial dependence (average predicted %, validation sample)"); ax[1].legend()
_show(fig, "importance_and_partial_dependence.png")
print("Importance and partial dependence describe the model, not causal effects.")

# ---- 3. Frequency track -----------------------------------------------------
freq = None
if RUN_FREQUENCY:
    display(Markdown("## 3. Claim-frequency model (actuarial track, exposure-aware)"))
    freq = train_frequency(clean, res.splits, cfg)
    display(freq.val_table.style.format({c: "{:.4f}" for c in freq.val_table.columns if c.startswith("Val")}))
    print("Selected frequency model:", freq.selected_name)
    display(pd.Series(freq.test_metrics).to_frame("Test").style.format("{:.4f}"))
    print("AUC of P(claim | own exposure) uses each policy's exposure, so it is NOT comparable with the "
          "classification AUC above; deviance explained is the primary frequency metric.")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    c = freq.calibration
    ax.plot(c.predicted_frequency, c.observed_frequency, "o-"); m_ = max(c.max()[["predicted_frequency", "observed_frequency"]]) * 1.1
    ax.plot([0, m_], [0, m_], "--", c="grey"); ax.set_xlabel("Predicted claims per policy-year"); ax.set_ylabel("Observed")
    ax.set_title("Frequency calibration (test deciles)")
    _show(fig, "frequency_calibration.png")

# ---- 4. Anomaly screening ---------------------------------------------------
if RUN_ANOMALY:
    display(Markdown("## 4. Anomaly screening (separate from the predictor)"))
    print(ANOMALY_CAVEAT)
    lam_all = np.clip(freq.model.predict(clean[FEATURES]), 1e-6, None) if freq else None
    screen_a, mat_a = run_anomaly_screen(clean, "A", seed=SEED)
    screen_b, mat_b = run_anomaly_screen(clean, "B", seed=SEED, lam=lam_all)
    summary = pd.DataFrame([
        {"Screen": "A - prospective (pre-outcome fields)", "Flagged": int(screen_a.Anomaly_Flag.sum()),
         "Share": screen_a.Anomaly_Flag.mean(), "Flagged with ≥1 claim": screen_a.loc[screen_a.Anomaly_Flag == 1, CLAIMS_COLUMN].gt(0).mean()},
        {"Screen": "B - retrospective (claims, exposure, surprise)", "Flagged": int(screen_b.Anomaly_Flag.sum()),
         "Share": screen_b.Anomaly_Flag.mean(), "Flagged with ≥1 claim": screen_b.loc[screen_b.Anomaly_Flag == 1, CLAIMS_COLUMN].gt(0).mean()}])
    display(summary.style.format({"Share": "{:.2%}", "Flagged with ≥1 claim": "{:.1%}"}))
    stab_a, stab_b = anomaly_stability(mat_a), anomaly_stability(mat_b)
    print(f"Flag stability across 5 seeds (mean Jaccard): screen A {stab_a['Jaccard overlap'].mean():.2f}, "
          f"screen B {stab_b['Jaccard overlap'].mean():.2f}")
    overlap = len(set(screen_a.index[screen_a.Anomaly_Flag == 1]) & set(screen_b.index[screen_b.Anomaly_Flag == 1]))
    print(f"Records flagged by both screens: {overlap:,}")
    display(screen_b.head(15)[[ID_COLUMN, CLAIMS_COLUMN, EXPOSURE_COLUMN, "Claims_Surprise", "Anomaly_Score",
                               "Review_Priority", "Reasons", "Data_Quality_Rule"]])
    screen_a.to_csv(OUTPUT_DIR / "anomaly_screen_A_prospective.csv", index=False)
    screen_b.to_csv(OUTPUT_DIR / "anomaly_screen_B_retrospective.csv", index=False)
    pd.concat([stab_a.assign(Screen="A"), stab_b.assign(Screen="B")]).to_csv(OUTPUT_DIR / "anomaly_stability.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "anomaly_summary.csv", index=False)

# ---- 5. Export --------------------------------------------------------------
display(Markdown("## 5. Export"))
bundle = make_bundle(res, freq, report, cfg)
res.val_table.to_csv(OUTPUT_DIR / "model_comparison_validation.csv", index=False)
res.threshold_options.to_csv(OUTPUT_DIR / "threshold_options_validation.csv", index=False)
res.calibration.to_csv(OUTPUT_DIR / "calibration_test.csv", index=False)
res.band_test_table.to_csv(OUTPUT_DIR / "risk_band_test.csv", index=False)
res.lift.to_csv(OUTPUT_DIR / "lift_table_test.csv", index=False)
res.capacity.to_csv(OUTPUT_DIR / "capacity_table_test.csv", index=False)
res.ci.to_csv(OUTPUT_DIR / "bootstrap_ci_test.csv", index=False)
res.segments.to_csv(OUTPUT_DIR / "segment_stability_test.csv", index=False)
res.importance.to_csv(OUTPUT_DIR / "permutation_importance_validation.csv", index=False)
assoc.to_csv(OUTPUT_DIR / "chi_square_association_tests.csv", index=False)
if freq:
    freq.val_table.to_csv(OUTPUT_DIR / "frequency_model_comparison_validation.csv", index=False)
    freq.calibration.to_csv(OUTPUT_DIR / "frequency_calibration_test.csv", index=False)
(OUTPUT_DIR / "data_quality.json").write_text(json.dumps(report.to_dict(), indent=2))
(OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(_py({
    "model": res.selected_name, "threshold": res.threshold, "test": res.test_metrics,
    "fixed_version_random_forest": res.baseline_test_metrics, "dummy": res.dummy_test_metrics,
    "bootstrap_ci": res.ci, "frequency": freq.test_metrics if freq else None}), indent=2))
(OUTPUT_DIR / "model_card.json").write_text(json.dumps(bundle["metadata"], indent=2))
save_bundle(bundle, OUTPUT_DIR / "claim_model_bundle_v2.joblib")
print("Saved outputs to", OUTPUT_DIR.resolve(), "| model version", bundle["metadata"]["model_version_id"])

# ---- 6. Policy Claim Probability Predictor ---------------------------------
display(Markdown("## 6. Policy Claim Probability Predictor"))
_CSS = """<style>
.cr-wrap{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:inherit;max-width:980px}
.cr-ptitle{font-weight:700;font-size:15px;margin:0 0 6px 0;opacity:.9}
.cr-hero{display:flex;flex-wrap:wrap;gap:14px;align-items:stretch;margin:10px 0}
.cr-card{border:1px solid rgba(128,128,128,.35);background:rgba(128,128,128,.10);border-radius:14px;padding:16px 20px}
.cr-score{flex:1 1 260px;text-align:center}.cr-big{font-size:64px;font-weight:800;line-height:1}
.cr-sub{font-size:13px;opacity:.75;margin-top:4px}
.cr-badge{display:inline-block;color:#fff;font-weight:700;padding:5px 16px;border-radius:999px;font-size:15px;margin:8px 4px 0 4px}
.cr-kpis{flex:2 1 380px;display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.cr-kpi .v{font-size:24px;font-weight:700}.cr-kpi .l{font-size:11px;opacity:.7;text-transform:uppercase;letter-spacing:.05em}
.cr-ruler{position:relative;height:18px;border-radius:9px;overflow:hidden;display:flex;margin:26px 0 4px 0}
.cr-mark{position:absolute;top:-6px;width:4px;height:30px;background:currentColor;border-radius:2px}
.cr-tick{position:absolute;top:-4px;width:0;height:26px;border-left:2px dashed currentColor;opacity:.8}
.cr-lbl{display:flex;justify-content:space-between;font-size:11px;opacity:.7}
.cr-drv{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px}
.cr-drv .n{flex:0 0 38%}.cr-drv .t{position:relative;flex:1;height:12px;background:rgba(128,128,128,.18);border-radius:6px}
.cr-drv .t:after{content:"";position:absolute;left:50%;top:-2px;height:16px;border-left:1px solid rgba(128,128,128,.7)}
.cr-drv .b{position:absolute;top:0;height:12px}.cr-drv .d{flex:0 0 70px;text-align:right;font-weight:600}
.cr-warn{border-left:4px solid #e0a100;background:rgba(224,161,0,.14);padding:8px 12px;border-radius:6px;margin:6px 0;font-size:13px}
.cr-india{border-left:4px solid #d64545;background:rgba(214,69,69,.12);padding:8px 12px;border-radius:6px;margin:8px 0;font-size:13px}
.cr-fine{font-size:12px;opacity:.75;margin-top:8px;line-height:1.5}
</style>"""


def launch_predictor(bundle: Dict[str, Any], out_dir: Union[str, Path]) -> Dict[str, Any]:
    import html as _h
    import ipywidgets as w
    from IPython.display import HTML, display as _display

    sch, bands, base_rate = bundle["schema"], bundle["risk_bands"], bundle["base_rate"]
    n_, c_ = sch["numeric"], sch["categorical"]
    COL = {"Low": "#2e9e5b", "Medium": "#e0a100", "High": "#d64545"}
    iv = lambda f, k: int(round(n_[f][k]))  # noqa: E731
    sty, lay = {"description_width": "125px"}, w.Layout(width="95%")

    def isl(f: str, label: str) -> Any:
        return w.IntSlider(value=iv(f, "median"), min=iv(f, "min"), max=iv(f, "max"), description=label,
                           continuous_update=False, layout=lay, style=sty)

    def dd(f: str, label: str) -> Any:
        return w.Dropdown(options=c_[f]["categories"], value=c_[f]["mode"], description=label, layout=lay, style=sty)

    dmin = max(n_[DENSITY_COLUMN]["min"], 1.0)
    wd = {"Driver age": isl("Driver age", "Driver age"), "Vehicle age": isl("Vehicle age", "Vehicle age (yrs)"),
          "Bonus-Malus score": isl("Bonus-Malus score", "Bonus-Malus"), "Vehicle power": isl("Vehicle power", "Vehicle power"),
          DENSITY_COLUMN: w.FloatLogSlider(value=min(max(n_[DENSITY_COLUMN]["median"], dmin), n_[DENSITY_COLUMN]["max"]),
                                           base=10, min=np.log10(dmin), max=np.log10(max(n_[DENSITY_COLUMN]["max"], dmin * 1.01)),
                                           step=0.02, readout_format=".0f", description="Pop. density /km²",
                                           continuous_update=False, layout=lay, style=sty),
          "Vehicle brand": dd("Vehicle brand", "Vehicle brand"), "Fuel type": dd("Fuel type", "Fuel type"),
          "Geographic area": dd("Geographic area", "Geographic area"), "Geographic region": dd("Geographic region", "Region")}
    term = w.FloatSlider(value=1.0, min=0.1, max=1.0, step=0.05, description="Policy term (yrs)",
                         readout_format=".2f", continuous_update=False, layout=lay, style=sty)
    thr = w.FloatSlider(value=float(min(max(bundle["threshold"], 0.01), 0.6)), min=0.01, max=0.6, step=0.005,
                        description="Alert threshold", readout_format=".3f", layout=w.Layout(width="520px"),
                        style={"description_width": "125px"})
    btn = w.Button(description="Predict and log", button_style="primary", icon="bolt", layout=w.Layout(width="200px", height="40px"))
    reset = w.Button(description="Reset", icon="undo", layout=w.Layout(width="120px", height="40px"))
    result = w.HTML()
    defaults = {k: v.value for k, v in wd.items()}
    state: Dict[str, Any] = {}

    def render(_: Any = None) -> None:
        s = state.get("last")
        if s is None:
            return
        if "error" in s:
            result.value = f"<div class='cr-wrap'><div class='cr-india'>{_h.escape(s['error'])}</div></div>"
            return
        p, eff, row = s["p"], s["eff"], s["row"]
        pr, band, tv = p["probability"], p["risk_band"], thr.value
        col, above = COL[band], pr >= tv
        scale = max(bands["medium_max"] * 2.2, pr * 1.2, tv * 1.2, base_rate * 2)
        lw, mw = 100 * bands["low_max"] / scale, 100 * (bands["medium_max"] - bands["low_max"]) / scale
        ruler = (f"<div class='cr-ruler'><div style='width:{lw:.2f}%;background:{COL['Low']}'></div>"
                 f"<div style='width:{mw:.2f}%;background:{COL['Medium']}'></div>"
                 f"<div style='width:{max(0, 100 - lw - mw):.2f}%;background:{COL['High']}'></div>"
                 f"<div class='cr-mark' style='left:{min(100, 100 * pr / scale):.2f}%'></div>"
                 f"<div class='cr-tick' style='left:{min(100, 100 * tv / scale):.2f}%'></div></div>"
                 f"<div class='cr-lbl'><span>0%</span><span>Low ≤ {bands['low_max']:.1%} | Medium ≤ {bands['medium_max']:.1%} | High above</span>"
                 f"<span>{scale:.0%}</span></div><div class='cr-lbl'><span>▮ this policy  ┆ alert threshold</span><span></span></div>")
        mx = max([abs(v) for v in eff.values()] + [1e-9])
        drv = []
        for f, d in sorted(eff.items(), key=lambda kv: -abs(kv[1]))[:6]:
            val = f"{row[f]:g}" if isinstance(row[f], (int, float)) else str(row[f])
            wp = 50 * abs(d) / mx
            side = f"left:50%;width:{wp:.1f}%;background:#d64545" if d > 0 else f"right:50%;width:{wp:.1f}%;background:#2e9e5b"
            drv.append(f"<div class='cr-drv'><div class='n'>{_h.escape(f)} = <b>{_h.escape(val)}</b></div>"
                       f"<div class='t'><div class='b' style='{side}'></div></div><div class='d'>{100 * d:+.2f} pp</div></div>")
        warns = "".join(f"<div class='cr-warn'>⚠ {_h.escape(m)}</div>" for m in p["warnings"])
        freq_kpis = ""
        if "annual_frequency" in p:
            bf = bundle.get("base_frequency") or np.nan
            freq_kpis = (f"<div class='cr-card cr-kpi'><div class='l'>Claims per policy-year (portfolio {bf:.3f})</div>"
                         f"<div class='v'>{p['annual_frequency']:.3f} <span style='font-size:14px'>({p['annual_frequency'] / bf:.2f}×)</span></div></div>"
                         f"<div class='cr-card cr-kpi'><div class='l'>P(≥1 claim in {p['term_years']:.2f} yr)</div><div class='v'>{p['p_claim_in_term']:.2%}</div></div>")
        saved = f" Logged to <code>{_h.escape(str(state['saved']))}</code>." if state.get("saved") else ""
        result.value = (
            "<div class='cr-wrap'><div class='cr-hero'>"
            f"<div class='cr-card cr-score' style='border-color:{col}'><div class='cr-sub'>CLAIM RISK SCORE (0-100)</div>"
            f"<div class='cr-big' style='color:{col}'>{p['score_0_100']:.1f}</div>"
            f"<div><span class='cr-badge' style='background:{col}'>{band.upper()} RISK</span>"
            f"<span class='cr-badge' style='background:{'#d64545' if above else '#2e9e5b'}'>{'ABOVE' if above else 'Below'} alert threshold</span></div></div>"
            "<div class='cr-kpis'>"
            f"<div class='cr-card cr-kpi'><div class='l'>P(≥1 claim), classifier</div><div class='v' style='color:{col}'>{pr:.2%}</div></div>"
            f"<div class='cr-card cr-kpi'><div class='l'>vs portfolio ({base_rate:.1%})</div><div class='v'>{pr / base_rate:.2f}×</div></div>"
            f"<div class='cr-card cr-kpi'><div class='l'>Alert threshold</div><div class='v'>{tv:.1%}</div></div>"
            f"{freq_kpis}"
            f"<div class='cr-card cr-kpi'><div class='l'>Model version</div><div class='v' style='font-size:15px'>{_h.escape(p['model_version_id'])}</div></div>"
            "</div></div>"
            f"<div class='cr-card'><div class='cr-ptitle'>Where this policy sits</div>{ruler}</div>"
            "<div class='cr-card' style='margin-top:12px'><div class='cr-ptitle'>Average effect of each input value across the portfolio (partial dependence)</div>"
            f"{''.join(drv)}<div class='cr-lbl' style='margin-top:4px'><span>◀ lowers average risk</span><span>raises average risk ▶</span></div>"
            "<div class='cr-fine'>pp = percentage points versus the portfolio-average prediction. Each bar shows what this value does "
            "on average across 300 reference policies - it is model behaviour, not a causal effect, and bars do not add up to the score.</div></div>"
            f"{warns}<div class='cr-india'><b>Not validated for India.</b> Trained on French motor third-party-liability data; "
            "nothing here establishes validity for ICICI Lombard or any Indian portfolio.</div>"
            "<div class='cr-fine'><b>Two estimates, two questions.</b> The classifier gives the chance of at least one claim within the "
            "period such policies are typically observed for (about half a year on average), so it is affected by how long similar policies "
            "stay on cover. The frequency model removes that effect: it gives claims per policy-year and converts it to the chosen term. "
            "For underwriting or pricing discussions use the frequency estimate; use the classifier score for ranking and review queues. "
            f"Bands are portfolio-relative. Not a premium, fraud finding or underwriting decision.{saved}</div></div>")

    def get_row() -> Dict[str, Any]:
        return {k: float(v.value) if k == DENSITY_COLUMN else v.value for k, v in wd.items()}

    def compute(save: bool = False) -> None:
        row = get_row()
        try:
            p = score_input(bundle, row, thr.value, term.value)
            state["last"] = {"p": p, "eff": explain_pd(bundle, row), "row": row}
            if save:
                path = Path(out_dir) / "prediction_log.csv"
                rec = pd.DataFrame([{**row, "predicted_probability": p["probability"], "risk_band": p["risk_band"],
                                     "alert_threshold": thr.value, "annual_frequency": p.get("annual_frequency"),
                                     "term_years": term.value, "p_claim_in_term": p.get("p_claim_in_term"),
                                     "model_version_id": p["model_version_id"],
                                     "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "disclaimer": "Educational; French data; not validated for India"}])
                rec.to_csv(path, mode="a", header=not path.exists(), index=False)
                state["saved"] = path
        except (InvalidInputError, SchemaError) as exc:
            state["last"] = {"error": f"Invalid input: {exc}"}
        except Exception as exc:  # noqa: BLE001
            state["last"] = {"error": f"Prediction failed safely: {exc}"}
        render()

    def on_reset(_: Any) -> None:
        for k, v in wd.items():
            v.value = defaults[k]
        compute()

    for x in list(wd.values()) + [term]:
        x.observe(lambda _: compute(), names="value")
    thr.observe(render, names="value")
    btn.on_click(lambda _: compute(save=True))
    reset.on_click(on_reset)

    def panel(title: str, kids: List[Any]) -> Any:
        return w.VBox([w.HTML(f"<div class='cr-wrap'><div class='cr-ptitle'>{title}</div></div>")] + kids,
                      layout=w.Layout(border="1px solid rgba(128,128,128,.4)", padding="10px 12px",
                                      margin="0 10px 10px 0", flex="1 1 400px"))
    left = panel("🚗 Driver &amp; vehicle", [wd["Driver age"], wd["Vehicle age"], wd["Bonus-Malus score"], wd["Vehicle power"], term])
    right = panel("📍 Location &amp; brand", [wd["Vehicle brand"], wd["Fuel type"], wd["Geographic region"],
                                              wd["Geographic area"], wd[DENSITY_COLUMN]])
    _display(HTML(_CSS))
    _display(w.VBox([w.HBox([left, right], layout=w.Layout(flex_flow="row wrap")),
                     w.HBox([btn, reset], layout=w.Layout(align_items="center", flex_flow="row wrap")), thr, result]))
    compute()
    return {"btn": btn, "result": result, "widgets": wd, "thr": thr, "term": term, "compute": compute}


try:
    PREDICTOR = launch_predictor(bundle, OUTPUT_DIR)
    print("Change any control to update; 'Predict and log' appends to prediction_log.csv with the model version id.")
except ImportError:
    print("ipywidgets not installed (pip install ipywidgets); default-policy example instead:")
    ex = {**{f: bundle["schema"]["numeric"][f]["median"] for f in NUMERIC_FEATURES},
          **{f: bundle["schema"]["categorical"][f]["mode"] for f in CATEGORICAL_FEATURES}}
    print(score_input(bundle, ex))
# Batch scoring example:  score_csv(bundle, "new_policies.csv", OUTPUT_DIR / "scored_policies.csv")
