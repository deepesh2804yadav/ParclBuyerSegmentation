#!/usr/bin/env python3
"""
model_pipeline.py
══════════════════════════════════════════════════════════════════════════════
Data Science & Machine Learning Pipeline
Project : Machine Learning-Based Buyer Segmentation and Investment Profiling
Client  : Parcl Co. Limited
══════════════════════════════════════════════════════════════════════════════

Pipeline stages
───────────────
  Stage 1  Data Cleaning
           ├─ Two-pass deduplication (full-row + composite-key)
           ├─ Type-aware missing-value imputation
           │    gender        : Corporate → 'Not Applicable' (business rule)
           │                    Individual → global mode
           │    other columns : global mode imputation
           └─ Derive integer `age` from date_of_birth (reference: 2024-01-01)

  Stage 2  Feature Engineering & Encoding
           ├─ OneHotEncoder  (drop='if_binary') on all categoricals
           │    Binary cols  → client_type, acquisition_purpose, loan_applied
           │                   (2 categories → 1 column each)
           │    OHE cols     → gender, country, region, referral_channel
           └─ StandardScaler on numeric cols: age, satisfaction_score

  Stage 3  Clustering Optimisation
           ├─ Silhouette Score  (primary criterion — maximise)
           ├─ Inertia / Elbow  (secondary — second-difference curvature)
           └─ Ambiguity resolution; auto-selects k or falls back to DEFAULT_K=4

  Stage 4  Model Training
           ├─ KMeans (n_clusters=optimal_k)          — production model
           └─ AgglomerativeClustering / Ward linkage  — comparative model

  Stage 5  Artifact Export
           ├─ models/kmeans_model.joblib
           ├─ models/agglomerative_model.joblib
           ├─ models/scaler.joblib
           ├─ models/encoder.joblib
           └─ data/processed/segmented_buyers.csv

REQUIREMENTS
────────────
  Python ≥ 3.9
  pip install numpy pandas scikit-learn joblib

USAGE
─────
  python model_pipeline.py
  Reads  : data/raw/parcl_synthetic_data.csv
  Writes : data/processed/segmented_buyers.csv  +  models/*.joblib
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

# ── I/O Paths ──────────────────────────────────────────────────────────────────
INPUT_FILE    = os.path.join("data", "raw",       "parcl_synthetic_data.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR     = "models"
OUTPUT_CSV    = os.path.join(PROCESSED_DIR, "segmented_buyers.csv")

KMEANS_PATH   = os.path.join(MODEL_DIR, "kmeans_model.joblib")
AGGLO_PATH    = os.path.join(MODEL_DIR, "agglomerative_model.joblib")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH  = os.path.join(MODEL_DIR, "encoder.joblib")

# ── Clustering Parameters ──────────────────────────────────────────────────────
K_RANGE              = range(2, 9)   # evaluate k = 2 … 8
DEFAULT_K: int       = 4            # fallback when selection is ambiguous
AMBIGUITY_GAP: float = 0.02         # minimum silhouette gap to trust auto-selection

# ── Feature Groups ─────────────────────────────────────────────────────────────
NUMERICAL_FEATURES: list[str] = ["age", "satisfaction_score"]

# All categorical features fed into OneHotEncoder(drop='if_binary').
# drop='if_binary' means:
#   • 2-category cols (client_type, acquisition_purpose, loan_applied)
#     produce ONE column each — equivalent to LabelEncoder behaviour,
#     with no information loss.
#   • Multi-category cols (gender, country, region, referral_channel)
#     receive full OHE treatment — one column per category.
# This single encoder object covers the role of both OneHotEncoder and
# LabelEncoder as specified in the project requirements.
OHE_FEATURES: list[str] = [
    "client_type",         # 2-cat → 1 col  (Individual / Corporate)
    "acquisition_purpose", # 2-cat → 1 col  (Investment / Personal Use)
    "loan_applied",        # 2-cat → 1 col  (Yes / No)
    "gender",              # 4-cat → 4 cols (Male / Female / Other / Not Applicable)
    "country",             # 13-cat → OHE
    "region",              # ~50-cat → OHE  (nested under country; kept per requirements)
    "referral_channel",    # 6-cat → OHE
]

# Fixed age reference — ensures results are identical regardless of run date
REFERENCE_DATE = pd.Timestamp("2024-01-01")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw CSV and assert schema integrity before any transformation.

    Raises
    ──────
    FileNotFoundError : if the CSV does not exist at `path`
    ValueError        : if expected columns are absent
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            "Run  data_generation.py  first."
        )

    df = pd.read_csv(path, low_memory=False)

    required_cols = {
        "client_id", "cohort_label", "client_type", "gender",
        "country", "region", "date_of_birth", "acquisition_purpose",
        "loan_applied", "referral_channel", "satisfaction_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing expected columns: {missing}")

    print(f"    ✓ {len(df):,} rows × {len(df.columns)} columns loaded from '{path}'")

    null_s = df.isna().sum()
    null_s = null_s[null_s > 0]
    if not null_s.empty:
        print("    Null inventory (pre-cleaning):")
        for col, n in null_s.items():
            print(f"      {col:<28s} {n:>4}  ({n / len(df) * 100:.1f} %)")

    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-pass deduplication.

    Pass 1 — exact full-row duplicates via pandas drop_duplicates()
    Pass 2 — composite-key (client_id + date_of_birth) duplicates,
              keeping the first occurrence.  This catches ETL re-run artefacts
              where non-key columns may differ slightly between copies.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["client_id", "date_of_birth"], keep="first")
    n_removed = n_before - len(df)
    print(f"    ✓ Deduplication: removed {n_removed} rows  "
          f"({n_before:,} → {len(df):,})")
    return df   # index NOT reset here — caller re-aligns cohort_labels first


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Type-aware missing-value imputation.

    gender
      Rule 1 → Corporate records with null gender  : 'Not Applicable'
               (corporate entities carry no individual gender attribute)
      Rule 2 → Individual records with null gender : mode of non-null
               Individual gender values

    All other columns (region, referral_channel, acquisition_purpose,
    loan_applied)
      Global mode imputation — replaces NaN with the most frequently
      observed value in that column.

    Asserts zero residual nulls on exit.
    """
    df = df.copy()

    # ── gender : rule-based + mode ────────────────────────────────────────────
    null_gender  = df["gender"].isna()
    corp_null    = null_gender & (df["client_type"] == "Corporate")
    indiv_null   = null_gender & (df["client_type"] == "Individual")

    df.loc[corp_null, "gender"] = "Not Applicable"

    if indiv_null.any():
        indiv_mode = (
            df.loc[~null_gender & (df["client_type"] == "Individual"), "gender"]
            .mode()[0]
        )
        df.loc[indiv_null, "gender"] = indiv_mode

    print(f"    ✓ {'gender':<26s} {null_gender.sum():>4} nulls → "
          f"rule-based (Corporate) + mode (Individual)")

    # ── remaining categorical columns : global mode ───────────────────────────
    mode_cols = ["region", "referral_channel", "acquisition_purpose", "loan_applied"]
    for col in mode_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            print(f"    ✓ {col:<26s} {n_null:>4} nulls → '{mode_val}' (mode)")

    residual = df.isna().sum().sum()
    assert residual == 0, (
        f"Unexpected residual nulls after imputation: {residual}\n"
        f"{df.isna().sum()[df.isna().sum() > 0]}"
    )
    print("    ✓ Zero residual nulls confirmed")
    return df


def _derive_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate integer age (whole years) from date_of_birth.
    Age is relative to REFERENCE_DATE = 2024-01-01.
    date_of_birth is dropped post-derivation (raw DOB is not a useful
    clustering feature once age is captured numerically).
    """
    df       = df.copy()
    dob      = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"]= ((REFERENCE_DATE - dob).dt.days / 365.25).astype(int)
    df       = df.drop(columns=["date_of_birth"])
    print(
        f"    ✓ 'age' derived  |  "
        f"range [{df['age'].min()}, {df['age'].max()}] yrs  |  "
        f"mean {df['age'].mean():.1f} yrs  |  "
        f"std {df['age'].std():.1f} yrs"
    )
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Master cleaning orchestrator.

    IMPORTANT — cohort_label separation:
        cohort_label is the Phase 1 ground-truth injection tag.  It is
        separated from the DataFrame immediately and BEFORE deduplication
        so its index remains aligned with df.  The index is re-synchronised
        after row-dropping, then reset to 0…n-1 for both objects.
        cohort_label is never seen by the feature engineering or clustering
        stages; it is reattached only in the final output CSV for validation.

    Returns
    ───────
    df_clean      : fully cleaned DataFrame (no cohort_label, no date_of_birth)
    cohort_labels : pd.Series, index = 0…n-1, aligned with df_clean
    """
    print("\n── Stage 1: Data Cleaning ──────────────────────────────────────────────")

    # ── Isolate ground-truth before any row filtering ─────────────────────────
    cohort_labels = df["cohort_label"].copy()          # preserves original index
    df = df.drop(columns=["cohort_label"])

    # ── Deduplicate (row count changes; re-align immediately) ─────────────────
    df = _drop_duplicates(df)                          # index NOT reset yet
    cohort_labels = cohort_labels.loc[df.index]        # label-align to survivor rows
    df = df.reset_index(drop=True)
    cohort_labels = cohort_labels.reset_index(drop=True)

    # ── Impute & engineer base features ──────────────────────────────────────
    df = _impute_missing(df)
    df = _derive_age(df)

    assert len(df) == len(cohort_labels), (
        f"Index mis-alignment: df={len(df)}, cohort_labels={len(cohort_labels)}"
    )
    print(f"\n    ✓ Clean shape: {len(df):,} rows × {len(df.columns)} columns")
    return df, cohort_labels


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FEATURE ENGINEERING & ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler, OneHotEncoder, list[str]]:
    """
    Build the fully-numeric feature matrix X for clustering.

    Encoding strategy
    ─────────────────
    OneHotEncoder (drop='if_binary', sparse_output=False)
        Handles ALL categorical columns with a single fitted object.
        For 2-category columns (client_type, acquisition_purpose, loan_applied),
        drop='if_binary' produces one 0/1 column — behaviourally identical to
        sklearn's LabelEncoder but unified under a single reusable artifact.
        For multi-category columns, all categories are fully OHE-expanded.

    StandardScaler (zero-mean, unit-variance)
        Applied to: age, satisfaction_score.

    Excluded from feature matrix
        client_id : unique identifier — no discriminating signal for clustering.

    Returns
    ───────
    X          : np.ndarray (n_samples × n_features)
    scaler     : fitted StandardScaler
    encoder    : fitted OneHotEncoder
    feat_names : list[str] — human-readable column names for X
    """
    print("\n── Stage 2: Feature Engineering & Encoding ─────────────────────────────")

    # client_id is a unique key with no discriminating signal — drop from X only.
    # region is included in OHE_FEATURES per project requirements; it is kept here.
    working = df.drop(columns=["client_id"]).copy()

    # ── OneHotEncoder on all categorical features ─────────────────────────────
    encoder = OneHotEncoder(
        drop="if_binary",        # binary cols → 1 col; multi-cat → full OHE
        sparse_output=False,     # return dense numpy array directly
        handle_unknown="ignore", # zero-vector for unseen categories at inference
    )
    ohe_matrix = encoder.fit_transform(working[OHE_FEATURES])
    ohe_names  = encoder.get_feature_names_out(OHE_FEATURES).tolist()

    print(f"    OneHotEncoder fitted on {len(OHE_FEATURES)} columns:")
    for col, cats in zip(OHE_FEATURES, encoder.categories_):
        n_out = sum(1 for n in ohe_names if n.startswith(col + "_"))
        print(f"      {col:<26s} {len(cats):>2} categories  →  {n_out} output cols")
    print(f"    Total OHE output columns: {ohe_matrix.shape[1]}")

    # ── StandardScaler on numerical features ──────────────────────────────────
    scaler        = StandardScaler()
    scaled_matrix = scaler.fit_transform(working[NUMERICAL_FEATURES])

    print(f"\n    StandardScaler fitted on: {NUMERICAL_FEATURES}")
    for i, col in enumerate(NUMERICAL_FEATURES):
        print(f"      {col:<20s}  μ = {scaler.mean_[i]:>7.3f}   σ = {scaler.scale_[i]:>6.3f}")

    # ── Assemble final feature matrix ─────────────────────────────────────────
    X          = np.hstack([scaled_matrix, ohe_matrix])
    feat_names = NUMERICAL_FEATURES + ohe_names

    print(
        f"\n    ✓ Feature matrix X: {X.shape[0]:,} × {X.shape[1]}  "
        f"({len(NUMERICAL_FEATURES)} scaled-numeric  |  "
        f"{ohe_matrix.shape[1]} OHE)"
    )
    return X, scaler, encoder, feat_names


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CLUSTERING OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def _find_elbow_k(k_vals: list[int], inertias: list[float]) -> int:
    """
    Locate the elbow (knee) of the inertia curve using the second-difference
    (curvature) method.

    The inertia series is normalised by its first value to make the metric
    scale-invariant.  The elbow corresponds to the k where the rate of
    inertia decrease slows the most — i.e., the point of maximum curvature.

    Returns DEFAULT_K if fewer than 4 k values are provided (not enough
    points to compute a meaningful second derivative).
    """
    if len(k_vals) < 4:
        return DEFAULT_K

    norm_inertia = np.array(inertias, dtype=float) / inertias[0]
    first_diff   = np.diff(norm_inertia)          # rate of decrease
    second_diff  = np.diff(first_diff)             # rate of change of decrease

    # Elbow index in second_diff corresponds to k_vals index offset by 2
    elbow_pos = int(np.argmax(second_diff)) + 2
    elbow_pos = min(elbow_pos, len(k_vals) - 1)   # guard against out-of-bounds
    return k_vals[elbow_pos]


def find_optimal_k(
    X: np.ndarray,
    k_range: range = K_RANGE,
    default_k: int = DEFAULT_K,
    ambiguity_gap: float = AMBIGUITY_GAP,
) -> tuple[int, dict[int, dict]]:
    """
    Evaluate clustering quality across k_range and select the optimal k.

    Metrics computed per k
    ──────────────────────
    Silhouette Score
        Mean ratio of intra-cluster cohesion to nearest-cluster separation.
        Range [-1, 1]; higher is better.  This is the primary decision criterion.

    Inertia (WCSS)
        Within-Cluster Sum of Squared distances to centroids.
        Lower is tighter; used as input to the Elbow method.

    k-selection logic
    ─────────────────
    1. Primary  : argmax(silhouette_score)
    2. Secondary: elbow point from second-difference of normalised inertia
    3. Conflict resolution:
         · If silhouette gap between top-2 candidates < ambiguity_gap → default_k
         · If silhouette winner ≠ elbow → silhouette winner takes priority
         · If both agree → confirmed selection

    Parameters
    ──────────
    X            : scaled feature matrix
    k_range      : range of k values to evaluate
    default_k    : fallback k when selection is ambiguous
    ambiguity_gap: minimum silhouette margin to trust auto-selection

    Returns
    ───────
    optimal_k : int
    results   : dict  {k: {'silhouette': float, 'inertia': float}}
    """
    print("\n── Stage 3: Clustering Optimisation ────────────────────────────────────")
    print(f"    Evaluating k ∈ {{{', '.join(map(str, k_range))}}} …\n")
    print(f"    {'k':>3}  {'Silhouette':>11}  {'Inertia':>14}  Visual")
    print(f"    {'─'*3}  {'─'*11}  {'─'*14}  {'─'*28}")

    results: dict[int, dict] = {}
    k_vals_list, inertia_list = [], []

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
            max_iter=300,
            algorithm="lloyd",
        )
        labels  = km.fit_predict(X)
        sil     = silhouette_score(X, labels, random_state=RANDOM_SEED)
        inertia = float(km.inertia_)

        results[k]       = {"silhouette": sil, "inertia": inertia}
        k_vals_list.append(k)
        inertia_list.append(inertia)

        bar = "▓" * int(sil * 40)
        print(f"    k={k}  {sil:>11.6f}  {inertia:>14,.1f}  {bar}")

    # ── Silhouette-based ranking ───────────────────────────────────────────────
    ranked_by_sil = sorted(results, key=lambda k: results[k]["silhouette"], reverse=True)
    sil_winner    = ranked_by_sil[0]
    sil_runner_up = ranked_by_sil[1]
    sil_gap       = (results[sil_winner]["silhouette"]
                     - results[sil_runner_up]["silhouette"])

    # ── Elbow-based suggestion ─────────────────────────────────────────────────
    elbow_k = _find_elbow_k(k_vals_list, inertia_list)

    print(f"\n    Silhouette winner : k={sil_winner}  "
          f"(score={results[sil_winner]['silhouette']:.4f}, "
          f"gap vs runner-up={sil_gap:.4f})")
    print(f"    Elbow suggestion  : k={elbow_k}")

    # ── Resolution ────────────────────────────────────────────────────────────
    if sil_gap < ambiguity_gap:
        optimal_k = default_k
        decision  = (f"ambiguous (gap {sil_gap:.4f} < threshold {ambiguity_gap:.2f}) "
                     f"→ default k={default_k}")
    elif sil_winner == elbow_k:
        optimal_k = sil_winner
        decision  = "silhouette and elbow agree"
    else:
        optimal_k = sil_winner
        decision  = (f"silhouette winner (k={sil_winner}) takes priority "
                     f"over elbow (k={elbow_k})")

    print(f"\n    ✓ Optimal k = {optimal_k}  [{decision}]")
    return optimal_k, results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    """
    Fit a KMeans model using the optimal cluster count.

    KMeans is the primary production model because:
      • It exposes a predict() method for scoring unseen records
      • Cluster centroids are interpretable in the original feature space
      • Fully deterministic with fixed random_state
      • Efficient at inference time (O(k × d) per new record)

    n_init=20 (>= default 10) runs 20 independent initialisations and
    keeps the best, reducing sensitivity to centroid initialisation.

    Returns fitted KMeans and integer cluster label array (0-indexed).
    """
    print(f"\n── Stage 4a: KMeans  (k={k}) ─────────────────────────────────────────────")
    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=20,
        max_iter=500,
        algorithm="lloyd",
    )
    labels  = km.fit_predict(X)
    sil     = silhouette_score(X, labels, random_state=RANDOM_SEED)
    n_iters = km.n_iter_

    print(f"    ✓ Converged in {n_iters} iterations")
    print(f"    ✓ Final inertia    : {km.inertia_:,.2f}")
    print(f"    ✓ Silhouette score : {sil:.6f}")
    counts = {c: int(np.sum(labels == c)) for c in range(k)}
    print(f"    Cluster sizes      : "
          + "  ".join(f"C{c}={n}" for c, n in counts.items()))
    return km, labels


def train_agglomerative(X: np.ndarray, k: int) -> tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fit a Ward-linkage Agglomerative Hierarchical Clustering model.

    Ward linkage minimises the total within-cluster variance at each merge
    step — the same objective as KMeans WCSS minimisation.  This makes the
    two models directly comparable and provides a meaningful cross-validation
    of the segmentation.

    IMPORTANT — production inference:
        AgglomerativeClustering has NO predict() method; it is a transductive
        model that assigns labels only to the training set.  For scoring new
        records in production, always use the KMeans model.  This model is
        saved for comparative analysis and dendrogram visualisation in Phase 3.

    Returns fitted AgglomerativeClustering and integer cluster label array.
    """
    print(f"\n── Stage 4b: AgglomerativeClustering  (k={k}, linkage=ward) ─────────────")
    agg = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",      # Ward = minimise within-cluster variance (Euclidean only)
        metric="euclidean",  # Ward linkage requires Euclidean; explicit for clarity
        compute_distances=True,  # retain distances for potential dendrogram in Phase 3
    )
    labels = agg.fit_predict(X)
    sil    = silhouette_score(X, labels, random_state=RANDOM_SEED)

    print(f"    ✓ Silhouette score : {sil:.6f}")
    counts = {c: int(np.sum(labels == c)) for c in range(k)}
    print(f"    Cluster sizes      : "
          + "  ".join(f"C{c}={n}" for c, n in counts.items()))
    return agg, labels


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — ARTIFACT EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_models(
    km_model:  KMeans,
    agg_model: AgglomerativeClustering,
    scaler:    StandardScaler,
    encoder:   OneHotEncoder,
) -> None:
    """
    Persist all four model artifacts to disk using joblib.

    Artifact manifest
    ─────────────────
    kmeans_model.joblib
        Production inference model.  Load and call .predict(X_new) on
        new encoded+scaled buyer records.

    agglomerative_model.joblib
        Comparative / dendrogram model.  Transductive — cannot predict
        on new records without re-fitting.

    scaler.joblib
        Fitted StandardScaler.  Apply to [age, satisfaction_score] columns
        of new records before inference.

    encoder.joblib
        Fitted OneHotEncoder (drop='if_binary').  Apply to all OHE_FEATURES
        columns of new records before inference.

    Inference recipe for new records
    ─────────────────────────────────
        scaler  = joblib.load('models/scaler.joblib')
        encoder = joblib.load('models/encoder.joblib')
        kmeans  = joblib.load('models/kmeans_model.joblib')
        X_num   = scaler.transform(new_df[NUMERICAL_FEATURES])
        X_cat   = encoder.transform(new_df[OHE_FEATURES])
        X_new   = np.hstack([X_num, X_cat])
        cluster = kmeans.predict(X_new)
    """
    print("\n── Stage 5a: Model Artifact Export ─────────────────────────────────────")
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifacts: dict[str, object] = {
        KMEANS_PATH:  km_model,
        AGGLO_PATH:   agg_model,
        SCALER_PATH:  scaler,
        ENCODER_PATH: encoder,
    }
    for path, obj in artifacts.items():
        joblib.dump(obj, path, compress=3)
        kb = os.path.getsize(path) / 1024
        print(f"    ✓ {path:<48s}  ({kb:.1f} KB)")


def export_segmented_csv(
    df_clean:       pd.DataFrame,
    cohort_labels:  pd.Series,
    km_labels:      np.ndarray,
    agg_labels:     np.ndarray,
) -> pd.DataFrame:
    """
    Assemble and export the fully-labelled buyer dataset.

    New columns appended to the clean DataFrame
    ─────────────────────────────────────────────
    cohort_label            Ground-truth cluster from Phase 1 (for validation)
    kmeans_cluster          KMeans assignment  (0-indexed integer)
    agglomerative_cluster   Agglomerative assignment  (0-indexed integer)

    Returns the assembled DataFrame for immediate use in print_summary().
    """
    print("\n── Stage 5b: Segmented CSV Export ──────────────────────────────────────")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_out = df_clean.copy()
    df_out["cohort_label"]          = cohort_labels.values
    df_out["kmeans_cluster"]        = km_labels
    df_out["agglomerative_cluster"] = agg_labels

    df_out.to_csv(OUTPUT_CSV, index=False)
    kb  = os.path.getsize(OUTPUT_CSV) / 1024
    print(f"    ✓ {OUTPUT_CSV}")
    print(f"      {len(df_out):,} rows × {len(df_out.columns)} columns  ({kb:.1f} KB)")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_profile_table(
    df: pd.DataFrame,
    cluster_col: str,
    k: int,
) -> pd.DataFrame:
    """
    Build a per-cluster business-metric profile table.

    Metrics
    ───────
    Size          Number of records in cluster
    Avg Age       Mean derived age
    Avg Sat.      Mean satisfaction_score
    Corp %        Fraction of Corporate client_type
    Invest %      Fraction of acquisition_purpose = Investment
    Loan Yes %    Fraction of loan_applied = Yes
    """
    d = df.copy()
    d["_corp"]   = (d["client_type"]         == "Corporate"  ).astype(int)
    d["_invest"] = (d["acquisition_purpose"] == "Investment" ).astype(int)
    d["_loan"]   = (d["loan_applied"]        == "Yes"        ).astype(int)

    grp = d.groupby(cluster_col)
    tbl = pd.DataFrame({
        "Size":       grp.size(),
        "Avg Age":    grp["age"].mean().round(1),
        "Avg Sat.":   grp["satisfaction_score"].mean().round(2),
        "Corp %":     (grp["_corp"].mean()   * 100).round(1),
        "Invest %":   (grp["_invest"].mean() * 100).round(1),
        "Loan Yes %": (grp["_loan"].mean()   * 100).round(1),
    })
    return tbl


def _top_country_per_cluster(df: pd.DataFrame, cluster_col: str) -> pd.Series:
    """Return the most common country in each cluster."""
    return df.groupby(cluster_col)["country"].agg(lambda x: x.value_counts().index[0])


def print_summary(
    df_out:        pd.DataFrame,
    km_labels:     np.ndarray,
    agg_labels:    np.ndarray,
    cohort_labels: pd.Series,
    k:             int,
    k_results:     dict[int, dict],
    feat_names:    list[str],
    scaler:        StandardScaler,
    encoder:       OneHotEncoder,
) -> None:
    """Print the full pipeline summary report to stdout."""
    sep  = "═" * 68
    dash = "─" * 68

    print(f"\n{sep}")
    print("  PIPELINE SUMMARY REPORT")
    print(sep)

    # ── [A] K Optimisation table ───────────────────────────────────────────────
    print("\n  [A] K Optimisation Results")
    print(f"  {'k':>3}  {'Silhouette':>11}  {'Inertia':>14}  {'Note'}")
    print(f"  {dash[:50]}")
    for kv in sorted(k_results):
        marker = "  ← SELECTED" if kv == k else ""
        s = k_results[kv]
        print(f"  k={kv}  {s['silhouette']:>11.6f}  {s['inertia']:>14,.1f}{marker}")

    # ── [B] KMeans cluster profiles ────────────────────────────────────────────
    print(f"\n  [B] KMeans Cluster Profiles  (k={k})")
    km_tbl = _cluster_profile_table(df_out, "kmeans_cluster", k)
    top_cc = _top_country_per_cluster(df_out, "kmeans_cluster")
    km_tbl["Top Country"] = top_cc
    print(km_tbl.to_string())

    # ── [C] Agglomerative cluster profiles ────────────────────────────────────
    print(f"\n  [C] Agglomerative Cluster Profiles  (k={k})")
    agg_tbl = _cluster_profile_table(df_out, "agglomerative_cluster", k)
    top_ac  = _top_country_per_cluster(df_out, "agglomerative_cluster")
    agg_tbl["Top Country"] = top_ac
    print(agg_tbl.to_string())

    # ── [D] Ground-truth recovery metrics ─────────────────────────────────────
    print(f"\n  [D] Cluster Quality vs Ground-Truth Labels (cohort_label)")
    ari_km   = adjusted_rand_score(cohort_labels, km_labels)
    nmi_km   = normalized_mutual_info_score(cohort_labels, km_labels)
    ari_agg  = adjusted_rand_score(cohort_labels, agg_labels)
    nmi_agg  = normalized_mutual_info_score(cohort_labels, agg_labels)

    print(f"  {'Model':<28}  {'ARI':>8}  {'NMI':>8}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}")
    print(f"  {'KMeans':<28}  {ari_km:>8.4f}  {nmi_km:>8.4f}")
    print(f"  {'AgglomerativeClustering':<28}  {ari_agg:>8.4f}  {nmi_agg:>8.4f}")
    print(f"\n  ARI: 0.0 = random assignment  |  1.0 = perfect match")
    print(f"  NMI: 0.0 = no mutual info     |  1.0 = perfect mutual information")

    # ── [E] KMeans → dominant ground-truth cohort mapping ─────────────────────
    print(f"\n  [E] KMeans Cluster → Dominant Cohort Mapping")
    print(f"  {'Cluster':>9}  {'Size':>6}  {'Dominant Cohort':<30}  {'Purity':>7}  {'2nd Cohort'}")
    print(f"  {'─'*9}  {'─'*6}  {'─'*30}  {'─'*7}  {'─'*25}")
    for c in sorted(np.unique(km_labels)):
        mask         = km_labels == c
        vc           = cohort_labels[mask].value_counts()
        dominant     = vc.index[0]
        purity       = vc.iloc[0] / mask.sum()
        second       = vc.index[1] if len(vc) > 1 else "—"
        second_pct   = f"({vc.iloc[1] / mask.sum():.0%})" if len(vc) > 1 else ""
        print(f"  Cluster {c}  {mask.sum():>6,}  {dominant:<30}  "
              f"{purity:>6.1%}  {second} {second_pct}")

    # ── [F] Feature engineering summary ───────────────────────────────────────
    n_ohe = len(feat_names) - len(NUMERICAL_FEATURES)
    print(f"\n  [F] Feature Matrix Summary")
    print(f"  Total features : {len(feat_names)}")
    print(f"    Scaled-numeric   : {len(NUMERICAL_FEATURES)}  → {NUMERICAL_FEATURES}")
    print(f"    OHE / binary     : {n_ohe}")
    for col, cats in zip(OHE_FEATURES, encoder.categories_):
        n_out = sum(1 for n in feat_names if n.startswith(col + "_"))
        print(f"      {col:<26}  {len(cats)} cats  →  {n_out} feature col(s)")

    # ── [G] Saved artifacts ────────────────────────────────────────────────────
    print(f"\n  [G] Saved Artifacts")
    for path in [KMEANS_PATH, AGGLO_PATH, SCALER_PATH, ENCODER_PATH, OUTPUT_CSV]:
        if os.path.exists(path):
            kb = os.path.getsize(path) / 1024
            print(f"  ✓  {path:<52}  {kb:>6.1f} KB")

    print(f"\n{sep}")
    print("  Phase 2 complete.")
    print("  Next step → visualisation.py  (Phase 3)")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("  Parcl Co. – ML Segmentation Pipeline")
    print("  Phase 2 of 4")
    print("=" * 68)

    # ── Stage 1: Load & Clean ──────────────────────────────────────────────────
    raw_df = load_data(INPUT_FILE)
    df_clean, cohort_labels = clean(raw_df)

    # ── Stage 2: Feature Engineering ──────────────────────────────────────────
    X, scaler, encoder, feat_names = engineer_features(df_clean)

    # ── Stage 3: Find Optimal k ───────────────────────────────────────────────
    optimal_k, k_results = find_optimal_k(X)

    # ── Stage 4: Train Models ─────────────────────────────────────────────────
    km_model,  km_labels  = train_kmeans(X, optimal_k)
    agg_model, agg_labels = train_agglomerative(X, optimal_k)

    # ── Stage 5: Export Artifacts ─────────────────────────────────────────────
    export_models(km_model, agg_model, scaler, encoder)
    df_out = export_segmented_csv(df_clean, cohort_labels, km_labels, agg_labels)

    # ── Summary Report ────────────────────────────────────────────────────────
    print_summary(
        df_out, km_labels, agg_labels, cohort_labels,
        optimal_k, k_results, feat_names, scaler, encoder,
    )


if __name__ == "__main__":
    main()
