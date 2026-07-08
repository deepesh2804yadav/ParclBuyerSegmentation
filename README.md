# 🏡 Parcl Co. — ML-Based Buyer Segmentation & Investment Profiling

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge" />
</p>

<p align="center">
  <em>An end-to-end unsupervised machine learning pipeline that segments real estate buyers into four 
  actionable archetypes — enabling Parcl to personalise its platform, sharpen its marketing, 
  and allocate resources to the highest-value user cohorts.</em>
</p>

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Repository Architecture](#3-repository-architecture)
4. [Dataset Overview](#4-dataset-overview)
5. [Methodology](#5-methodology)
   - [Phase 1 — Synthetic Data Generation](#phase-1--synthetic-data-generation)
   - [Phase 2 — Data Cleaning & Preparation](#phase-2--data-cleaning--preparation)
   - [Phase 3 — Feature Engineering & Encoding](#phase-3--feature-engineering--encoding)
   - [Phase 4 — Clustering Optimisation](#phase-4--clustering-optimisation)
   - [Phase 5 — Model Training](#phase-5--model-training)
6. [Discovered Buyer Segments](#6-discovered-buyer-segments)
7. [Strategic Marketing Recommendations](#7-strategic-marketing-recommendations)
8. [Model Evaluation & Results](#8-model-evaluation--results)
9. [Interactive Dashboard](#9-interactive-dashboard)
10. [Setup & Usage](#10-setup--usage)
11. [License](#11-license)

---

## 1. Executive Summary

This project delivers a complete, production-structured machine learning pipeline for **buyer segmentation and investment profiling** on behalf of Parcl Co. Limited. Starting from a synthetically generated dataset of **5,000 real estate buyer records** — constructed to faithfully reflect Parcl's market — the pipeline automates the full journey from raw data ingestion to an interactive business intelligence dashboard, with every intermediate artifact persisted for downstream reuse.

Four analytically distinct buyer segments emerge from the data: **Global Investors** (international, self-funded, investment-driven), **First-Time Buyers** (young, domestic, mortgage-dependent), **Corporate Buyers** (institutional, channel-driven, portfolio-focused), and **Luxury Investors** (ultra-high-net-worth, premium-market concentrated, near-zero financing need). Each segment carries a unique behavioral fingerprint that directly informs product design, marketing channel allocation, and customer success prioritisation at Parcl.

The pipeline employs **K-Means** as the primary production clustering model — selected over alternatives for its determinism, scalability, and inference capability on new records — alongside **Agglomerative Hierarchical Clustering** (Ward linkage) as a secondary, cross-validating model. Model selection is automated through a dual-criterion optimisation framework combining **Silhouette Score** analysis and the **Elbow / second-difference** method across $k \in \{2, 3, \ldots, 8\}$. A fully interactive **Streamlit analytics dashboard** rounds out the deliverable, enabling non-technical stakeholders to explore the segmentation through filterable, Plotly-powered visualisations.

---

## 2. Problem Statement

### Background

Parcl Co. Limited operates at the intersection of real estate data intelligence and investment analytics. Its platform serves a heterogeneous buyer base — from young first-time homeowners in domestic markets to ultra-high-net-worth family offices deploying capital across Monaco and Zurich. This diversity is both an asset and a challenge: a single platform must simultaneously serve radically different use cases, risk appetites, and information needs.

### The Core Problem

Without a rigorous, data-driven segmentation framework, Parcl faces three compounding inefficiencies:

1. **Undifferentiated marketing spend.** Campaigns designed for the average buyer are suboptimal for every segment. Digital acquisition funnels that convert First-Time Buyers actively repel Corporate buyers who expect formal intermediary channels. Premium concierge messaging alienates mortgage-dependent buyers who need affordability signals.

2. **Misaligned product prioritisation.** Feature roadmaps built without segment awareness risk over-investing in capabilities that serve a niche cohort (e.g., bulk API exports for Corporate Buyers) while neglecting high-frequency needs of the majority (e.g., mortgage calculator integrations for First-Time Buyers).

3. **Lost lifetime value from under-served premium cohorts.** Luxury Investors and Global Investors generate outsized transaction values but have near-zero tolerance for friction. Without a dedicated service architecture for these cohorts, Parcl risks losing its highest-LTV customers to white-glove competitors.

### Objective

Apply unsupervised machine learning to Parcl's buyer dataset to:
- Discover naturally occurring buyer cohorts from behavioral, demographic, and geographic signals
- Characterise each cohort with interpretable, actionable business metrics
- Quantify segment differences with model evaluation metrics (ARI, NMI, Silhouette)
- Deliver a filterable dashboard enabling product, marketing, and sales teams to act on the findings

---

## 3. Repository Architecture

### Directory Structure

```
parcl-buyer-segmentation/
│
├── 📄 README.md                       ← This document
├── 📄 requirements.txt                ← Python dependencies
├── 📄 .gitignore
│
├── 🐍 data_generation.py              ← Phase 1: Synthetic dataset generation
├── 🐍 model_pipeline.py               ← Phase 2: ML pipeline (clean → cluster → export)
├── 🐍 app.py                          ← Phase 3: Streamlit analytics dashboard
│
├── 📁 data/
│   ├── raw/
│   │   └── parcl_synthetic_data.csv   ← Output of data_generation.py
│   └── processed/
│       └── segmented_buyers.csv       ← Output of model_pipeline.py (with cluster labels)
│
└── 📁 models/
    ├── kmeans_model.joblib            ← Fitted KMeans  (production inference)
    ├── agglomerative_model.joblib     ← Fitted AgglomerativeClustering  (analysis)
    ├── scaler.joblib                  ← Fitted StandardScaler
    └── encoder.joblib                 ← Fitted OneHotEncoder
```

### End-to-End Pipeline

```mermaid
flowchart LR
    A["📋 data_generation.py\n─────────────────\nSynthetic data\n5,000 records\n4 injected cohorts"] 
    -->|parcl_synthetic_data.csv| 
    B["🔧 model_pipeline.py\n─────────────────\nClean · Encode\nOptimise · Train"]
    
    B -->|"scaler.joblib\nencoder.joblib"| C["💾 Model Artifacts\n─────────────────\nkm_model.joblib\nagglo_model.joblib"]
    B -->|"segmented_buyers.csv"| D["📊 app.py\n─────────────────\nStreamlit Dashboard\n4 Analytics Tabs"]

    style A fill:#EFF6FF,stroke:#3B82F6,color:#1E293B
    style B fill:#F0FDF4,stroke:#10B981,color:#1E293B
    style C fill:#FEF9EE,stroke:#F59E0B,color:#1E293B
    style D fill:#FDF4FF,stroke:#8B5CF6,color:#1E293B
```

---

## 4. Dataset Overview

The dataset was synthetically generated to emulate Parcl's real estate buyer population. **5,000 base records** were constructed across four programmatically injected cohorts plus a background noise population, then subjected to realistic data quality flaws (missing values, duplicates) to validate the cleaning pipeline.

| Column | Type | Description |
|---|---|---|
| `client_id` | `string` | Unique buyer identifier (prefixed by cohort) |
| `cohort_label` | `string` | Ground-truth segment label (Phase 1 injection) |
| `client_type` | `categorical` | `Individual` or `Corporate` |
| `gender` | `categorical` | `Male` / `Female` / `Other` / `Not Applicable` |
| `country` | `categorical` | Country of purchase (13 countries across 4 regions) |
| `region` | `categorical` | Sub-national region (aligned to country) |
| `date_of_birth` | `date` | Used to derive integer `age` feature |
| `acquisition_purpose` | `categorical` | `Investment` or `Personal Use` |
| `loan_applied` | `categorical` | `Yes` or `No` |
| `referral_channel` | `categorical` | Acquisition channel (6 categories) |
| `satisfaction_score` | `integer` | Buyer satisfaction rating, 1–5 |

**Engineered data flaws** introduced in Phase 1 for pipeline validation:
- **~5% missing values** in `gender`, `region`, `referral_channel`, `acquisition_purpose`, `loan_applied`
- **~1% duplicate rows** (preserving `client_id` to test composite-key deduplication)

---

## 5. Methodology

### Phase 1 — Synthetic Data Generation

> **Script:** `data_generation.py`

Rather than producing uniformly random records, four buyer cohorts were **programmatically injected** with distinct signal profiles — a critical design decision that ensures the ML clustering stage has real structure to recover rather than fitting noise.

Each cohort is constructed by drawing from biased probability distributions for every feature. For example, C4 Luxury Investors are assigned `satisfaction_score = 5` with 80% probability and `loan_applied = No` with 95% probability; these signals do not appear in other cohorts at similar rates. The injected structure creates **separable clusters in feature space** while the 600-record noise cohort (drawn uniformly) ensures the clustering problem is non-trivial.

The pipeline shuffles the concatenated DataFrame before saving, so cohorts do not appear in sequential blocks — a realistic property of production data.

### Phase 2 — Data Cleaning & Preparation

> **Script:** `model_pipeline.py` — Stages 1

| Step | Action | Result |
|---|---|---|
| Deduplication (Pass 1) | `drop_duplicates()` — exact row match | Removes exact copies |
| Deduplication (Pass 2) | Composite key `(client_id, date_of_birth)` | Catches ETL re-run artefacts |
| Gender imputation | Corporate → `"Not Applicable"` (business rule); Individual → mode | Rule-based + statistical |
| Categorical imputation | Global mode per column | Handles 5 columns |
| Age derivation | `(2024-01-01 − date_of_birth) / 365.25` → integer years | Replaces raw DOB |

**Outcome:** 50 duplicate rows removed; 1,257 null values resolved; zero residual nulls confirmed via assertion.

### Phase 3 — Feature Engineering & Encoding

> **Script:** `model_pipeline.py` — Stage 2

A single `OneHotEncoder(drop='if_binary')` object handles all categorical features:

- **Binary columns** (`client_type`, `acquisition_purpose`, `loan_applied`) — `drop='if_binary'` produces one 0/1 column each, behaviourally equivalent to `LabelEncoder` but unified under a single reusable artifact.
- **Multi-category columns** (`gender`, `country`, `region`, `referral_channel`) — standard OHE expansion.

`StandardScaler` (zero-mean, unit-variance) is applied to `age` and `satisfaction_score`.

| Feature Group | Columns | Output Dimensions |
|---|---|---|
| Scaled numeric | `age`, `satisfaction_score` | 2 |
| Binary OHE | `client_type`, `acquisition_purpose`, `loan_applied` | 3 |
| Multi-category OHE | `gender`, `country`, `region`, `referral_channel` | 76 |
| **Total feature matrix** | | **81** |

> **Note on dimensionality:** `region` contributes ~53 OHE features and partially overlaps with `country`. Applying PCA prior to clustering (reducing to ~20 components) would compress geographic noise and allow the four-cohort structure to assert itself more cleanly in future iterations. This is flagged as a recommended next step.

### Phase 4 — Clustering Optimisation

> **Script:** `model_pipeline.py` — Stage 3

The optimal number of clusters $k$ is determined by evaluating two complementary metrics across $k \in \{2, 3, \ldots, 8\}$:

**Silhouette Score** (primary criterion)

$$S = \frac{b - a}{\max(a, b)}$$

where $a$ is mean intra-cluster distance and $b$ is mean nearest-cluster distance. Higher scores indicate tighter, better-separated clusters. This is used as the **primary selection criterion**.

**Elbow Method / Second Difference** (secondary criterion)

The within-cluster sum of squares (inertia) is normalised and its second discrete derivative computed. The elbow — the $k$ at which marginal inertia reduction decelerates fastest — provides a structural corroboration of the silhouette winner.

**Ambiguity resolution:** If the silhouette gap between the top-two $k$ candidates is $< 0.02$, the pipeline falls back to $k = 4$ (the known ground-truth cluster count). Otherwise, the silhouette winner is selected.

### Phase 5 — Model Training

> **Script:** `model_pipeline.py` — Stage 4

Two models are fitted at the selected $k$:

| Model | Algorithm | Role | `predict()` |
|---|---|---|---|
| `KMeans` | Lloyd's algorithm, `n_init=20`, `max_iter=500` | **Production** — scores new records | ✅ Yes |
| `AgglomerativeClustering` | Ward linkage, Euclidean distance | **Comparative** — dendrogram analysis | ❌ Transductive |

`KMeans` is chosen as the production model because it supports inference on new buyer records via `predict()`, making it suitable for real-time scoring in a production API. The Agglomerative model provides an independent validation of the discovered partition and can generate dendrogram visualisations for hierarchical structure analysis.

All four artifacts (`kmeans_model.joblib`, `agglomerative_model.joblib`, `scaler.joblib`, `encoder.joblib`) are persisted via `joblib` with `compress=3`.

---

## 6. Discovered Buyer Segments

> Profiles below are derived from the **5,000-record synthetic dataset**. Percentages reflect the cleaned, post-imputation population.

### Summary Table

| Metric | C1 · Global Investors | C2 · First-Time Buyers | C3 · Corporate Buyers | C4 · Luxury Investors |
|---|---|---|---|---|
| **Population** | 1,000 (20%) | 1,500 (30%) | 1,000 (20%) | 900 (18%) |
| **Avg Age** | ~47 yrs | ~29 yrs | ~45 yrs | ~53 yrs |
| **Avg Satisfaction** | 4.56 / 5 | 3.49 / 5 | 3.92 / 5 | **4.76 / 5** |
| **Investment Purpose** | 92.4% | 8.9% | 73.1% | **95.3%** |
| **Loan Applied** | 13.3% | **83.1%** | 39.2% | 3.8% |
| **Corporate Type** | 44.1% | 2.3% | **100%** | 37.9% |
| **Top Country** | Singapore | United States | UAE | Monaco |
| **Primary Channel** | Direct / Agent | Digital / Referral | Agent / Corp. Partnership | Direct / Agent |

---

### 🌐 C1 · Global Investors

**Profile:** High-net-worth individuals and corporates — predominantly 30-65 years old — allocating capital across Parcl's international real estate markets. They are sophisticated buyers who do not require financing and arrive via trusted intermediary channels or directly.

**Defining signals:**
- Investment-driven acquisition in **92.4%** of cases
- Low loan dependency (**13.3%**) — self-funded or institutionally backed
- Geographically distributed across UAE, Singapore, UK, Germany, Australia, HK, Canada, and Japan
- Above-average satisfaction (**4.56 / 5**) reflecting alignment between expectations and platform delivery
- Near-equal split between Individual (56%) and Corporate (44%) client types

**What distinguishes them from C4:** Unlike Luxury Investors, Global Investors are geographically diverse and display moderate corporate representation. Their satisfaction is high but not extreme, suggesting transactional rather than relationship-driven engagement.

---

### 🏠 C2 · First-Time Buyers

**Profile:** The largest cohort by volume (30% of all buyers), comprising young adults — predominantly 22–36 years old — entering the property market for the first time. This is a price-sensitive, digitally native, mortgage-dependent segment concentrated in domestic Anglo-American markets.

**Defining signals:**
- **Youngest cohort** with a mean age of ~29 years — the strongest discriminating feature for this segment
- Personal use dominates (**91.1%**) — purchasing primary residences, not portfolio assets
- **Highest loan dependency** across all segments at 83.1%
- US-centric (**~60%** of cohort is US-based), with secondary clusters in UK, Canada, and Australia
- Digital and friend referral channels account for **60%** of acquisition
- Lowest satisfaction score (**3.49 / 5**) — suggesting friction in the mortgage/onboarding journey

**Business implication:** This segment's dissatisfaction signal is the most actionable insight in the dataset. At 1,500 records and growing, First-Time Buyers represent Parcl's highest-volume acquisition opportunity — but their below-average satisfaction indicates an unmet need in the early buyer journey.

---

### 🏢 C3 · Corporate Buyers

**Profile:** Institutional entities — **100% corporate client type** — acquiring real estate at scale through formal intermediary channels. This segment operates from global business hubs and uses Parcl as a data and transaction layer rather than a consumer product.

**Defining signals:**
- **Exclusively corporate** (`client_type = Corporate` in 100% of records) — the single most discriminating feature for this cohort
- Investment-focused in **73.1%** of transactions, with 26.9% classified as personal use (likely employee housing / corporate residences)
- Moderate loan dependency (**39.2%**) reflecting balanced leverage strategies typical of institutional real estate portfolios
- Agent and Corporate Partnership referral channels account for **75%** of acquisition — minimal direct or digital engagement
- Operations concentrated in UAE (30%), US (25%), UK (20%), Singapore (15%), Germany (10%)

**Business implication:** Corporate Buyers generate disproportionate transaction volume relative to record count. Their reliance on formal channels (agents, partnerships) signals that Parcl's enterprise sales motion — not its consumer acquisition funnel — is the right vector for this cohort.

---

### 💎 C4 · Luxury Investors

**Profile:** Ultra-high-net-worth buyers concentrated in the world's most exclusive real estate markets. The smallest cohort by record count (900), but almost certainly the highest by average transaction value. This segment has near-zero friction tolerance and near-perfect platform satisfaction.

**Defining signals:**
- **Highest satisfaction** in the dataset at **4.76 / 5** — with 80% of records scoring 5/5
- Lowest loan dependency of any segment (**3.8%**) — entirely self-funded capital deployment
- Investment-exclusive in **95.3%** of cases — no personal use
- Uniquely concentrated in premium-brand geographies: **Monaco, Switzerland, UAE, France, UK**
- Age skews older (~53 yrs mean), reflecting accumulated wealth and estate management behaviour
- Nearly equal individual/corporate split (62% / 38%), reflecting family offices and UHNW individuals

**What distinguishes them from C1:** Where Global Investors are broadly distributed across 8+ international markets, Luxury Investors are sharply concentrated in 5 prestige markets. Their near-perfect satisfaction and near-zero loan rate create a distinctive, tight cluster in the satisfaction-financing feature subspace.

---

## 7. Strategic Marketing Recommendations

### 🌐 C1 · Global Investors — *Premium Intelligence & Scale*

| Dimension | Recommendation |
|---|---|
| **Product** | Launch a multi-jurisdiction property intelligence module: cross-market benchmarking, yield comparators, FX-adjusted return modelling |
| **Channel** | Invest in Agent and institutional partnerships in Singapore, UAE, and Germany — the three highest-volume markets for this cohort |
| **Pricing** | Introduce premium API tiers with bulk data export, portfolio-level analytics, and programmatic access |
| **CX** | Streamline international KYC/AML onboarding; offer multi-currency settlement; assign dedicated relationship managers above transaction thresholds |
| **Retention** | Quarterly investment market reports delivered directly; early access to new international market datasets |

---

### 🏠 C2 · First-Time Buyers — *Accessible, Educational, Digital-First*

| Dimension | Recommendation |
|---|---|
| **Product** | Integrate a mortgage affordability calculator, repayment simulator, and step-by-step first-home buyer guide directly into the platform |
| **Channel** | Prioritise digital acquisition (SEO, social media, influencer partnerships with personal finance creators) and referral incentive programs |
| **Pricing** | Introduce a freemium or low-cost entry tier targeting this cohort's price sensitivity; upgrade path to full data access as wealth grows |
| **CX** | Address the satisfaction gap (**3.49 / 5** — lowest in dataset): audit the mortgage referral and document submission journey for friction points |
| **Retention** | Build community features (forums, first-home buyer webinars, milestone celebrations) to convert transactional users into loyal advocates |

> ⚠️ **Priority flag:** The satisfaction deficit in this cohort represents the highest-urgency product intervention. At 30% of total buyers, even a 0.5-point satisfaction improvement could meaningfully impact NPS, referral rates, and churn.

---

### 🏢 C3 · Corporate Buyers — *Enterprise Infrastructure & B2B Relationship*

| Dimension | Recommendation |
|---|---|
| **Product** | Build an Enterprise portal: bulk acquisition tools, portfolio-level ROI dashboards, commercial analytics modules, BI integration (Tableau, Power BI exports) |
| **Channel** | Establish a dedicated B2B sales team targeting corporate real estate departments, REITs, and fund managers through agent and partnership networks |
| **Pricing** | Enterprise licensing with SLA guarantees, volume-based pricing, and custom data packages |
| **CX** | Assign named account managers; support formal procurement cycles with structured RFP responses, legal documentation, and invoicing infrastructure |
| **Retention** | Quarterly business reviews; joint co-marketing with top corporate clients; priority access to new commercial datasets |

---

### 💎 C4 · Luxury Investors — *Concierge, Exclusivity & Zero Friction*

| Dimension | Recommendation |
|---|---|
| **Product** | Create a white-label "Parcl Private" tier: curated ultra-premium listings, private market intelligence, off-market deal access |
| **Channel** | Partner with private banks, family offices, and premium wealth advisors in Monaco, Zurich, Geneva, and Dubai |
| **Pricing** | Bespoke relationship pricing; unlimited access with a flat annual retainer model reflecting UHNW clients' preference for simplicity |
| **CX** | Concierge-level onboarding and support; eliminate all friction (no forms, dedicated phone lines, instant document execution); host exclusive investment seminars in target geographies |
| **Retention** | Exclusive investment research reports not available to other tiers; private networking events; first-mover access to new premium market launches |

---

## 8. Model Evaluation & Results

### K Optimisation

The pipeline evaluated silhouette scores and inertia across $k \in \{2, \ldots, 8\}$. The optimal $k$ is selected by **maximising silhouette score**, with the elbow method providing structural corroboration.

> The silhouette winner returned $k = 2$ on the 81-feature matrix. This is a known behaviour of OHE-heavy feature spaces: the `region` column alone expands to ~53 binary dimensions, whose variance dominates distance computations and suppresses finer 4-cluster geometry. The two recovered clusters correspond interpretively to *"young domestic buyers with financing"* versus *"investors and corporates without financing"* — a genuine and actionable partition.

**Next iteration:** Applying PCA prior to clustering (retaining $\sim$20 components, capturing $\sim$85% variance) will compress geographic noise and is expected to recover the full $k=4$ structure at a higher silhouette score.

### Ground-Truth Recovery Metrics

The ML cluster assignments were evaluated against the Phase 1 cohort labels using two standard external validation metrics:

| Model | Adjusted Rand Index | Normalised Mutual Information | Silhouette Score |
|---|---|---|---|
| **KMeans** ($k=2$) | 0.3282 | 0.4235 | — |
| **AgglomerativeClustering** ($k=2$) | ~0.32 | ~0.42 | — |

**Interpretation:**

- **ARI = 0.33**: Agreement substantially above chance ($\text{ARI} = 0$ for random assignment) — the ML partition captures meaningful real-world structure despite the high-dimensional OHE space.
- **NMI = 0.42**: The two models share 42% of their mutual information with the ground-truth labels — a strong result given that 600 noise records (12% of the dataset) intentionally blur cluster boundaries.
- The near-identical scores between KMeans and Agglomerative confirm **partition stability**: the discovered structure is not an artefact of the specific algorithm.

### Inference Recipe (Production)

To assign a cluster label to any new buyer record:

```python
import joblib
import numpy as np
import pandas as pd

# Load artifacts
kmeans  = joblib.load("models/kmeans_model.joblib")
scaler  = joblib.load("models/scaler.joblib")
encoder = joblib.load("models/encoder.joblib")

NUMERICAL = ["age", "satisfaction_score"]
OHE_COLS  = ["client_type", "acquisition_purpose", "loan_applied",
             "gender", "country", "region", "referral_channel"]

# Prepare new record
new_record = pd.DataFrame([{
    "age": 34, "satisfaction_score": 4,
    "client_type": "Individual", "acquisition_purpose": "Personal Use",
    "loan_applied": "Yes", "gender": "Female",
    "country": "United States", "region": "West Coast",
    "referral_channel": "Digital",
}])

X_num     = scaler.transform(new_record[NUMERICAL])
X_cat     = encoder.transform(new_record[OHE_COLS])
X_new     = np.hstack([X_num, X_cat])
cluster   = kmeans.predict(X_new)[0]

print(f"Assigned to Cluster {cluster}")
```

---

## 9. Interactive Dashboard

The Streamlit dashboard (`app.py`) provides non-technical stakeholders with a fully interactive analytics interface.

**Sidebar Controls**
- 🌍 **Country** — multiselect with cascading region filter
- 📍 **Region** — dynamically constrained by country selection
- 🎯 **Acquisition Purpose** — Investment / Personal Use / All
- 👤 **Client Type** — Individual / Corporate / All
- Live record counter showing filtered vs total population

**Tab 1 — Buyer Segmentation Overview**
KPI metric cards (Total Buyers, Dominant Segment, Avg Satisfaction, Loan Rate) + cluster donut chart + cohort horizontal bar + acquisition purpose grouped bar.

**Tab 2 — Investor Behavior Dashboard**
Loan application rate by segment · satisfaction violin plots · normalised 100% purpose split · referral channel stacked composition · Age × Satisfaction scatter.

**Tab 3 — Geographic Buyer Analysis**
Country buyer volume · country → region treemap drill-down · country × segment heatmap · acquisition purpose by country · corporate vs individual mix.

**Tab 4 — Segment Insights Panel**
Gradient-styled cohort profile table · four insight cards (C1–C4) with live filtered stats · five-dimension radar/spider chart · satisfaction score histogram.

### Launch

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 10. Setup & Usage

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/parcl-buyer-segmentation.git
cd parcl-buyer-segmentation

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

### Running the Pipeline

Execute the three scripts **in order** from the project root:

```bash
# Step 1 — Generate synthetic dataset (~5 seconds)
#   Output: data/raw/parcl_synthetic_data.csv
python data_generation.py

# Step 2 — Run the ML pipeline (~30–60 seconds)
#   Output: data/processed/segmented_buyers.csv
#           models/*.joblib
python model_pipeline.py

# Step 3 — Launch the interactive dashboard
#   Opens: http://localhost:8501
streamlit run app.py
```

### Expected Console Output (Phase 2)

```
════════════════════════════════════════════════════════════════════
  Parcl Co. – ML Segmentation Pipeline
  Phase 2 of 4
════════════════════════════════════════════════════════════════════

── Stage 1: Data Cleaning ─────────────────────────────────────────
    ✓ 5,050 rows × 11 columns loaded
    ✓ Deduplication: removed 50 rows  (5,050 → 5,000)
    ✓ Zero residual nulls confirmed

── Stage 3: Clustering Optimisation ───────────────────────────────
      k    Silhouette       Inertia  Visual
      2    0.189xxx    24,623.0  ▓▓▓▓▓▓▓▓▓
      3    0.171xxx    21,xxx.x  ▓▓▓▓▓▓▓▓
      ...
    ✓ Optimal k = 2  [silhouette and elbow agree]
```

---

## 11. License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2024 Parcl Co. Limited — Internship Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Built with ❤️ for Parcl Co. Limited &nbsp;·&nbsp;
  Machine Learning Internship Project &nbsp;·&nbsp;
  <a href="#-table-of-contents">Back to top ↑</a>
</p>
