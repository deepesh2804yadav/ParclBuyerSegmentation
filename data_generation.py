#!/usr/bin/env python3
"""
data_generation.py
══════════════════════════════════════════════════════════════════════════════
Synthetic Data Generation Pipeline
Project : Machine Learning-Based Buyer Segmentation and Investment Profiling
Client  : Parcl Co. Limited
══════════════════════════════════════════════════════════════════════════════

Generates 5,000 synthetic buyer records that faithfully mimic Parcl's real
estate market data.  Four cohort clusters are PROGRAMMATICALLY INJECTED so
that the downstream K-Means / DBSCAN / hierarchical clustering models have
meaningful structure to recover.

┌─────────────────────────────────────────────────────────────────────────┐
│  COHORT DESIGN                                                          │
├──────────────────────────┬────────┬────────────────────────────────────┤
│  Cohort                  │  Size  │  Key discriminating features       │
├──────────────────────────┼────────┼────────────────────────────────────┤
│  C1 – Global Investors   │  1 000 │  Intl. markets · low loan ·        │
│                          │        │  Investment purpose · sat ≈ 4-5    │
├──────────────────────────┼────────┼────────────────────────────────────┤
│  C2 – First-Time Buyers  │  1 500 │  Age 22-36 · Personal Use ·        │
│                          │        │  loan = Yes 88% · US-centric       │
├──────────────────────────┼────────┼────────────────────────────────────┤
│  C3 – Corporate Buyers   │  1 000 │  100% Corporate · Agent/Corp.      │
│                          │        │  Partnership referrals · business  │
│                          │        │  hub countries                     │
├──────────────────────────┼────────┼────────────────────────────────────┤
│  C4 – Luxury Investors   │    900 │  sat = 5 (80%) · Monaco/CH/UAE ·   │
│                          │        │  Investment 95% · loan = No 95%    │
├──────────────────────────┼────────┼────────────────────────────────────┤
│  Noise                   │    600 │  Fully random background pop.      │
└──────────────────────────┴────────┴────────────────────────────────────┘

DATA QUALITY FLAWS (for Phase 2 cleaning)
──────────────────────────────────────────
  • ~5 % missing values injected in five categorical columns:
      gender, region, referral_channel, acquisition_purpose, loan_applied
  • ~1 % duplicate rows appended (original client_id preserved so
    Phase 2 can detect them via composite key deduplication)

REQUIREMENTS
────────────
  Python ≥ 3.9
  pip install numpy pandas

USAGE
─────
  python data_generation.py
  → writes  data/raw/parcl_synthetic_data.csv
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_DIR: str  = os.path.join("data", "raw")
OUTPUT_FILE: str = os.path.join(OUTPUT_DIR, "parcl_synthetic_data.csv")

# Cohort sizes – must sum to exactly 5 000 base records
COHORT_SIZES: dict[str, int] = {
    "C1_Global_Investors":  1_000,
    "C2_First_Time_Buyers": 1_500,
    "C3_Corporate_Buyers":  1_000,
    "C4_Luxury_Investors":    900,
    "Noise":                  600,
}
assert sum(COHORT_SIZES.values()) == 5_000, (
    f"Cohort sizes must sum to 5 000.  Current sum: {sum(COHORT_SIZES.values())}"
)

# Data quality injection rates
MISSING_RATE: float   = 0.05   # 5 % missing per targeted column
DUPLICATE_RATE: float = 0.01   # 1 % duplicate rows

# ── Shared domain vocabularies ─────────────────────────────────────────────────

GENDERS: list[str] = ["Male", "Female", "Other"]

REFERRAL_CHANNELS: list[str] = [
    "Direct",
    "Agent",
    "Digital",
    "Corporate Partnership",
    "Friend Referral",
    "Social Media",
]

# Country → valid region list mapping (used for realistic location pairing)
COUNTRY_REGION_MAP: dict[str, list[str]] = {
    # ── North America ──────────────────────────────────────────────────────
    "United States": [
        "Northeast", "Southeast", "Midwest", "West Coast", "Southwest",
    ],
    "Canada": [
        "Ontario", "British Columbia", "Quebec", "Alberta",
    ],
    # ── Europe ────────────────────────────────────────────────────────────
    "United Kingdom": [
        "London", "South East", "North West", "Scotland", "Midlands",
    ],
    "Germany": [
        "Bavaria", "Berlin", "Hamburg", "North Rhine-Westphalia",
    ],
    "France": [
        "Île-de-France", "Provence", "Normandy", "French Riviera",
    ],
    "Switzerland": [
        "Zurich", "Geneva", "Basel", "Zug",
    ],
    "Monaco": [
        "La Condamine", "Monte Carlo", "Fontvieille", "Les Moneghetti",
    ],
    # ── Middle East ────────────────────────────────────────────────────────
    "UAE": [
        "Dubai", "Abu Dhabi", "Sharjah", "Ras Al Khaimah",
    ],
    "Saudi Arabia": [
        "Riyadh", "Jeddah", "Eastern Province", "Mecca Region",
    ],
    # ── Asia-Pacific ───────────────────────────────────────────────────────
    "Singapore": [
        "Central Region", "East Region", "West Region", "North Region",
    ],
    "Australia": [
        "New South Wales", "Victoria", "Queensland", "Western Australia",
    ],
    "Japan": [
        "Tokyo", "Osaka", "Kyoto", "Nagoya",
    ],
    "Hong Kong": [
        "Hong Kong Island", "Kowloon", "New Territories",
    ],
}

ALL_COUNTRIES: list[str] = list(COUNTRY_REGION_MAP.keys())


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_dob(min_age: int, max_age: int, n: int) -> pd.Series:
    """
    Return n ISO-8601-formatted date-of-birth strings.

    Ages are computed relative to 2024-01-01 so that the dataset remains
    deterministic regardless of when the script is executed.

    Parameters
    ----------
    min_age, max_age : int
        Inclusive age range (years).
    n : int
        Number of records to generate.

    Returns
    -------
    pd.Series of str  (format: 'YYYY-MM-DD')
    """
    reference = datetime(2024, 1, 1)
    earliest  = reference - timedelta(days=max_age * 365)
    latest    = reference - timedelta(days=min_age * 365)
    span_days = (latest - earliest).days

    offsets = np.random.randint(0, span_days + 1, size=n)
    dates   = [earliest + timedelta(days=int(d)) for d in offsets]
    return pd.Series([d.strftime("%Y-%m-%d") for d in dates])


def sample_location(
    countries: list[str],
    n: int,
    country_probs: list[float] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Sample n (country, region) pairs, respecting COUNTRY_REGION_MAP.

    Parameters
    ----------
    countries      : subset of COUNTRY_REGION_MAP keys to draw from.
    n              : number of samples.
    country_probs  : optional probability vector (must sum to 1.0).

    Returns
    -------
    (country_series, region_series) as pd.Series of str.
    """
    chosen_countries = np.random.choice(countries, size=n, p=country_probs)
    chosen_regions   = [
        random.choice(COUNTRY_REGION_MAP[c]) for c in chosen_countries
    ]
    return pd.Series(chosen_countries), pd.Series(chosen_regions)


def inject_missing(
    df: pd.DataFrame,
    columns: list[str],
    rate: float = MISSING_RATE,
) -> pd.DataFrame:
    """
    Randomly set approximately `rate` fraction of values in `columns` to NaN.

    Uses the globally seeded numpy state for full reproducibility.
    Missing values are independent per column (a cell is nulled with
    probability `rate` regardless of other columns).
    """
    df = df.copy()
    for col in columns:
        null_mask = np.random.random(len(df)) < rate
        df.loc[null_mask, col] = np.nan
    return df


def inject_duplicates(
    df: pd.DataFrame,
    rate: float = DUPLICATE_RATE,
) -> pd.DataFrame:
    """
    Append a random sample of existing rows to simulate real-world duplicates.

    Preserves the original client_id so Phase 2 can detect them via a
    (client_id, date_of_birth) composite key — a common production dedup
    pattern.

    Parameters
    ----------
    df   : DataFrame to duplicate from.
    rate : fraction of len(df) rows to duplicate.
    """
    n_dups   = max(1, int(len(df) * rate))
    dup_rows = df.sample(n=n_dups, random_state=RANDOM_SEED)
    return pd.concat([df, dup_rows], ignore_index=True)


def _build_cohort_frame(
    *,
    n: int,
    start_id: int,
    id_prefix: str,
    cohort_label: str,
    client_types: np.ndarray,
    genders: list[str],
    countries: pd.Series,
    regions: pd.Series,
    dob: pd.Series,
    acquisition_purpose: np.ndarray,
    loan_applied: np.ndarray,
    referral_channel: np.ndarray,
    satisfaction_score: np.ndarray,
) -> pd.DataFrame:
    """Shared DataFrame assembler so all cohort generators produce identical schemas."""
    return pd.DataFrame({
        "client_id":           [f"{id_prefix}_{start_id + i:05d}" for i in range(n)],
        "cohort_label":        cohort_label,          # ground-truth label (drop before unsupervised ML)
        "client_type":         client_types,
        "gender":              genders,
        "country":             countries.values,
        "region":              regions.values,
        "date_of_birth":       dob.values,
        "acquisition_purpose": acquisition_purpose,
        "loan_applied":        loan_applied,
        "referral_channel":    referral_channel,
        "satisfaction_score":  satisfaction_score,
    })


# ══════════════════════════════════════════════════════════════════════════════
# COHORT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

# ── C1: Global Investors ───────────────────────────────────────────────────────

def generate_c1_global_investors(n: int, start_id: int) -> pd.DataFrame:
    """
    C1 – Global Investors  (target size: 1 000)
    ════════════════════════════════════════════
    Profile
        High-net-worth individuals and corporates actively investing across
        major international real-estate markets.

    Programmatically injected signals
        • Countries     : UAE, Singapore, UK, Germany, Australia, HK, CA, JP
        • Purpose       : Investment 92 %
        • Loan applied  : No 85 %  (self-funded / liquidity-rich)
        • Satisfaction  : score ∈ {4, 5}  → 95 % of records
        • client_type   : Individual 55 % / Corporate 45 %
        • Age range     : 30–65
        • Referral      : Direct 35 %, Agent 30 %, Corporate Partnership 25 %
    """
    c1_countries = [
        "UAE", "Singapore", "United Kingdom", "Germany",
        "Australia", "Hong Kong", "Canada", "Japan",
    ]

    client_types = np.random.choice(
        ["Individual", "Corporate"], size=n, p=[0.55, 0.45]
    )
    genders = [
        random.choice(GENDERS) if t == "Individual" else "Not Applicable"
        for t in client_types
    ]
    countries, regions = sample_location(c1_countries, n)

    return _build_cohort_frame(
        n=n, start_id=start_id, id_prefix="C1",
        cohort_label="C1_Global_Investor",
        client_types=client_types,
        genders=genders,
        countries=countries,
        regions=regions,
        dob=generate_dob(min_age=30, max_age=65, n=n),
        acquisition_purpose=np.random.choice(
            ["Investment", "Personal Use"], size=n, p=[0.92, 0.08]
        ),
        loan_applied=np.random.choice(
            ["Yes", "No"], size=n, p=[0.15, 0.85]
        ),
        referral_channel=np.random.choice(
            ["Direct", "Agent", "Corporate Partnership", "Digital"],
            size=n, p=[0.35, 0.30, 0.25, 0.10],
        ),
        satisfaction_score=np.random.choice(
            [3, 4, 5], size=n, p=[0.05, 0.35, 0.60]
        ),
    )


# ── C2: First-Time Buyers ──────────────────────────────────────────────────────

def generate_c2_first_time_buyers(n: int, start_id: int) -> pd.DataFrame:
    """
    C2 – First-Time Buyers  (target size: 1 500)
    ════════════════════════════════════════════
    Profile
        Young domestic buyers acquiring their first primary residence.
        High financial dependency on mortgages; digitally engaged.

    Programmatically injected signals
        • Age range     : 22–36  (strongest separating feature from C1/C4)
        • Purpose       : Personal Use 95 %
        • Loan applied  : Yes 88 %
        • Countries     : US 60 %, UK 20 %, Canada 12 %, Australia 8 %
        • Referral      : Digital 35 %, Friend Referral 25 %, Agent 20 %
        • Satisfaction  : score ∈ {3, 4}  → 70 % of records
        • client_type   : Individual 97 %
    """
    c2_countries = ["United States", "United Kingdom", "Canada", "Australia"]
    c2_probs     = [0.60, 0.20, 0.12, 0.08]

    client_types = np.random.choice(
        ["Individual", "Corporate"], size=n, p=[0.97, 0.03]
    )
    genders = [random.choice(GENDERS) for _ in range(n)]
    countries, regions = sample_location(c2_countries, n, country_probs=c2_probs)

    return _build_cohort_frame(
        n=n, start_id=start_id, id_prefix="C2",
        cohort_label="C2_First_Time_Buyer",
        client_types=client_types,
        genders=genders,
        countries=countries,
        regions=regions,
        dob=generate_dob(min_age=22, max_age=36, n=n),
        acquisition_purpose=np.random.choice(
            ["Investment", "Personal Use"], size=n, p=[0.05, 0.95]
        ),
        loan_applied=np.random.choice(
            ["Yes", "No"], size=n, p=[0.88, 0.12]
        ),
        referral_channel=np.random.choice(
            ["Digital", "Friend Referral", "Agent", "Social Media", "Direct"],
            size=n, p=[0.35, 0.25, 0.20, 0.15, 0.05],
        ),
        satisfaction_score=np.random.choice(
            [1, 2, 3, 4, 5], size=n, p=[0.05, 0.10, 0.30, 0.40, 0.15]
        ),
    )


# ── C3: Corporate Buyers ───────────────────────────────────────────────────────

def generate_c3_corporate_buyers(n: int, start_id: int) -> pd.DataFrame:
    """
    C3 – Corporate Buyers  (target size: 1 000)
    ════════════════════════════════════════════
    Profile
        Institutional and corporate entities acquiring real estate at scale
        across major global business hubs.  Transact through formal
        intermediary channels.

    Programmatically injected signals
        • client_type   : Corporate 100 %  (strongest cluster signal)
        • gender        : Not Applicable 100 %
        • Countries     : UAE 30 %, US 25 %, UK 20 %, SG 15 %, DE 10 %
        • Referral      : Agent 40 %, Corporate Partnership 35 %
        • Loan applied  : Yes 40 %  (moderate leverage)
        • Satisfaction  : score ∈ {3, 4, 5}
        • Age range     : 30–60 (representative contact / director DOB)
    """
    c3_countries = ["UAE", "United States", "United Kingdom", "Singapore", "Germany"]
    c3_probs     = [0.30, 0.25, 0.20, 0.15, 0.10]

    client_types = np.full(n, "Corporate")
    genders      = ["Not Applicable"] * n
    countries, regions = sample_location(c3_countries, n, country_probs=c3_probs)

    return _build_cohort_frame(
        n=n, start_id=start_id, id_prefix="C3",
        cohort_label="C3_Corporate_Buyer",
        client_types=client_types,
        genders=genders,
        countries=countries,
        regions=regions,
        dob=generate_dob(min_age=30, max_age=60, n=n),
        acquisition_purpose=np.random.choice(
            ["Investment", "Personal Use"], size=n, p=[0.72, 0.28]
        ),
        loan_applied=np.random.choice(
            ["Yes", "No"], size=n, p=[0.40, 0.60]
        ),
        referral_channel=np.random.choice(
            ["Agent", "Corporate Partnership", "Direct", "Digital"],
            size=n, p=[0.40, 0.35, 0.15, 0.10],
        ),
        satisfaction_score=np.random.choice(
            [2, 3, 4, 5], size=n, p=[0.05, 0.25, 0.45, 0.25]
        ),
    )


# ── C4: Luxury Investors ───────────────────────────────────────────────────────

def generate_c4_luxury_investors(n: int, start_id: int) -> pd.DataFrame:
    """
    C4 – Luxury Investors  (target size: 900)
    ════════════════════════════════════════════
    Profile
        Ultra-high-net-worth individuals and family offices purchasing
        premium real-estate assets in exclusive global markets.

    Programmatically injected signals
        • Satisfaction  : score = 5  → 80 % of records
        • Countries     : Monaco 20 %, Switzerland 20 %, UAE 18 %,
                          UK 15 %, France 12 %, US 10 %, SG 5 %
        • Purpose       : Investment 95 %
        • Loan applied  : No 95 %  (entirely self-funded)
        • Age range     : 35–70
        • Referral      : Direct 40 %, Agent 30 %, Corp. Partnership 20 %
        • client_type   : Individual 60 % / Corporate 40 %
    """
    c4_countries = [
        "Monaco", "Switzerland", "UAE", "United Kingdom",
        "France", "United States", "Singapore",
    ]
    c4_probs = [0.20, 0.20, 0.18, 0.15, 0.12, 0.10, 0.05]

    client_types = np.random.choice(
        ["Individual", "Corporate"], size=n, p=[0.60, 0.40]
    )
    genders = [
        random.choice(GENDERS) if t == "Individual" else "Not Applicable"
        for t in client_types
    ]
    countries, regions = sample_location(c4_countries, n, country_probs=c4_probs)

    return _build_cohort_frame(
        n=n, start_id=start_id, id_prefix="C4",
        cohort_label="C4_Luxury_Investor",
        client_types=client_types,
        genders=genders,
        countries=countries,
        regions=regions,
        dob=generate_dob(min_age=35, max_age=70, n=n),
        acquisition_purpose=np.random.choice(
            ["Investment", "Personal Use"], size=n, p=[0.95, 0.05]
        ),
        loan_applied=np.random.choice(
            ["Yes", "No"], size=n, p=[0.05, 0.95]
        ),
        referral_channel=np.random.choice(
            ["Direct", "Agent", "Corporate Partnership", "Friend Referral"],
            size=n, p=[0.40, 0.30, 0.20, 0.10],
        ),
        satisfaction_score=np.random.choice(
            [3, 4, 5], size=n, p=[0.02, 0.18, 0.80]
        ),
    )


# ── Noise: Background Population ──────────────────────────────────────────────

def generate_noise(n: int, start_id: int) -> pd.DataFrame:
    """
    Noise – Background Population  (target size: 600)
    ══════════════════════════════════════════════════
    Fully random records drawn uniformly from all value spaces.

    Purpose
        • Makes the clustering problem non-trivial (avoids 100% clean separation)
        • Mimics real-world clients who do not neatly belong to any segment
        • Provides inter-cluster boundary records for silhouette score testing
    """
    client_types = np.random.choice(
        ["Individual", "Corporate"], size=n, p=[0.70, 0.30]
    )
    genders = [
        random.choice(GENDERS) if t == "Individual" else "Not Applicable"
        for t in client_types
    ]
    countries, regions = sample_location(ALL_COUNTRIES, n)

    return _build_cohort_frame(
        n=n, start_id=start_id, id_prefix="NX",
        cohort_label="Noise",
        client_types=client_types,
        genders=genders,
        countries=countries,
        regions=regions,
        dob=generate_dob(min_age=20, max_age=75, n=n),
        acquisition_purpose=np.random.choice(
            ["Investment", "Personal Use"], size=n
        ),
        loan_applied=np.random.choice(["Yes", "No"], size=n),
        referral_channel=np.random.choice(REFERRAL_CHANNELS, size=n),
        satisfaction_score=np.random.choice([1, 2, 3, 4, 5], size=n),
    )


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that all required schema columns are present and that no
    column has been inadvertently dropped or renamed.
    Raises AssertionError with an informative message on failure.
    """
    required_cols = {
        "client_id", "cohort_label", "client_type", "gender",
        "country", "region", "date_of_birth", "acquisition_purpose",
        "loan_applied", "referral_channel", "satisfaction_score",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    # satisfaction_score must be in [1, 5]
    valid_scores = {1, 2, 3, 4, 5}
    actual_scores = set(df["satisfaction_score"].dropna().unique())
    assert actual_scores.issubset(valid_scores), (
        f"Unexpected satisfaction_score values: {actual_scores - valid_scores}"
    )

    # client_type only allows known values
    valid_types = {"Individual", "Corporate"}
    actual_types = set(df["client_type"].dropna().unique())
    assert actual_types.issubset(valid_types), (
        f"Unexpected client_type values: {actual_types - valid_types}"
    )

    print("    ✓ Schema validation passed")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("  Parcl Co. – Synthetic Data Generation Pipeline")
    print("  Phase 1 of 4")
    print("=" * 68)

    # ── Step 1: Generate cohort DataFrames ────────────────────────────────────
    print("\n[1/6] Generating cohort records …")
    frames: list[pd.DataFrame] = [
        generate_c1_global_investors(
            n=COHORT_SIZES["C1_Global_Investors"],  start_id=10_000
        ),
        generate_c2_first_time_buyers(
            n=COHORT_SIZES["C2_First_Time_Buyers"], start_id=20_000
        ),
        generate_c3_corporate_buyers(
            n=COHORT_SIZES["C3_Corporate_Buyers"],  start_id=30_000
        ),
        generate_c4_luxury_investors(
            n=COHORT_SIZES["C4_Luxury_Investors"],  start_id=40_000
        ),
        generate_noise(
            n=COHORT_SIZES["Noise"],                start_id=50_000
        ),
    ]
    df = pd.concat(frames, ignore_index=True)
    print(f"    ✓ {len(df):,} base records generated across {len(frames)} cohorts")

    # ── Step 2: Validate schema ───────────────────────────────────────────────
    print("\n[2/6] Validating schema …")
    validate_schema(df)

    # ── Step 3: Shuffle rows ──────────────────────────────────────────────────
    print("\n[3/6] Shuffling rows (cohorts must not appear in sequential blocks) …")
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print("    ✓ Row order randomised")

    # ── Step 4: Inject ~5 % missing values ───────────────────────────────────
    print(f"\n[4/6] Injecting ≈{MISSING_RATE * 100:.0f} % missing values …")
    missing_target_cols = [
        "gender",
        "region",
        "referral_channel",
        "acquisition_purpose",
        "loan_applied",
    ]
    df = inject_missing(df, missing_target_cols)
    for col in missing_target_cols:
        cnt  = df[col].isna().sum()
        pct  = cnt / len(df) * 100
        bar  = "▓" * int(pct)
        print(f"    ✓ {col:<25s}  {cnt:>4} nulls  ({pct:.1f} %)  {bar}")

    # ── Step 5: Inject ~1 % duplicate rows ───────────────────────────────────
    print(f"\n[5/6] Injecting ≈{DUPLICATE_RATE * 100:.0f} % duplicate rows …")
    pre_dup_len = len(df)
    df = inject_duplicates(df)
    n_dups = len(df) - pre_dup_len
    print(
        f"    ✓ {n_dups} duplicate rows appended  →  "
        f"total rows: {len(df):,}"
    )

    # ── Step 6: Save to CSV ───────────────────────────────────────────────────
    print("\n[6/6] Saving dataset …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"    ✓ Saved  →  {OUTPUT_FILE}")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ══════════════════════════════════════════════════════════════════════════
    sep = "─" * 68
    print(f"\n{sep}")
    print("  DATASET SUMMARY REPORT")
    print(sep)

    print(f"\n  Total rows  (raw, including duplicates) : {len(df):,}")
    print(f"  Total columns                           : {len(df.columns)}")
    print(f"  Output file                             : {OUTPUT_FILE}")

    print(f"\n  {'Cohort':<30s}  {'Count':>6}  {'Share':>7}  Visual")
    print(f"  {'─' * 30}  {'─' * 6}  {'─' * 7}  {'─' * 20}")
    for label, cnt in df["cohort_label"].value_counts().sort_index().items():
        bar = "█" * (cnt // 75)
        pct = cnt / len(df) * 100
        print(f"  {label:<30s}  {cnt:>6,}  {pct:>6.1f}%  {bar}")

    print(f"\n  {'Column':<28s}  {'Nulls':>6}  {'Rate':>6}")
    print(f"  {'─' * 28}  {'─' * 6}  {'─' * 6}")
    null_series = df.isna().sum()
    for col, cnt in null_series.items():
        if cnt > 0:
            print(f"  {col:<28s}  {cnt:>6}  {cnt / len(df) * 100:>5.1f}%")

    print(f"\n  client_type breakdown:")
    for val, cnt in df["client_type"].value_counts().items():
        print(f"    {str(val):<20s}  {cnt:,}")

    print(f"\n  acquisition_purpose breakdown (non-null):")
    for val, cnt in df["acquisition_purpose"].value_counts().items():
        print(f"    {str(val):<20s}  {cnt:,}")

    print(f"\n  loan_applied breakdown (non-null):")
    for val, cnt in df["loan_applied"].value_counts().items():
        print(f"    {str(val):<10s}  {cnt:,}")

    print(f"\n  satisfaction_score distribution:")
    for score in sorted(df["satisfaction_score"].unique()):
        cnt = (df["satisfaction_score"] == score).sum()
        bar = "▪" * (cnt // 50)
        print(f"    Score {int(score)}:  {cnt:>5,}  {bar}")

    print(f"\n{sep}")
    print("  Phase 1 complete.")
    print("  Next step → run  data_cleaning.py  (Phase 2)")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
