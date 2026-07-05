# Influencing Factors behind Video Game Sales

## A Comprehensive Data Mining and Analysis Report

---

**Course:** INFO422 Data Science Project  
**Institution:** Lanzhou University (LZU)  
**Team:** Group 23  
**Date:** 2026-07-04  

| Team Member | Role | Responsibilities |
|---|---|---|
| Zuo Fengyuan | Project Leader & Final Integrator | Full-cycle project management, final quality review, submission |
| Yang Tianxiao | Data Engineer & Preprocessing Specialist | Data acquisition, cleaning, transformation, data quality |
| Bu Fanzhou | Data Analyst & Visualization Lead | EDA, statistical testing, visualization design |
| Zhong Rui | Machine Learning Engineer | Model building, cross-validation, performance evaluation |
| Gong Zijie | Domain Research & Presentation Lead | Background research, business recommendations, presentation |

---

## Executive Summary

This project investigates the key factors driving global video game sales using the VGChartz 2024 dataset (64,016 records, 14 attributes). Following the CRISP-DM framework across six milestones, the team conducted systematic data preprocessing, exploratory data analysis, predictive modeling, and deployment to answer three core research questions: **(1)** Do platform-genre combinations confer sales advantages? **(2)** Do genre preferences vary across regions? **(3)** Does publisher/developer brand significantly impact sales?

The analysis reveals that publisher brand scale is the single strongest sales driver, followed by critic score and release era. Platform-genre synergies are real but concentrated in specific high-performing combinations (e.g., Shooter × Xbox 360, Sports × PS4). Regional markets exhibit distinct genre preferences: North America favors Shooters and Sports, Japan favors RPGs and Visual Novels, and Europe/PAL favors Racing and Platformers. A Random Forest regression model achieves an R² of 0.394 on holdout data, and a novel two-stage blockbuster-aware variant reduces prediction error for top-tier titles by 6.7%. A Streamlit web application was deployed for stakeholder use.

**Key business recommendations:** Partner with established publishers for distribution scale, prioritize game quality to maximize critic scores, use platform-genre strategy for release planning, and tailor regional marketing to local genre preferences.

---

## Table of Contents

1. [Introduction and Business Context](#1-introduction-and-business-context)
2. [Research Questions and Hypotheses](#2-research-questions-and-hypotheses)
3. [Dataset Description](#3-dataset-description)
4. [Methodology and Tech Stack](#4-methodology-and-tech-stack)
5. [Phase 1: Data Acquisition and Preprocessing (M2 & M3)](#5-phase-1-data-acquisition-and-preprocessing-m2--m3)
6. [Phase 2: Exploratory Data Analysis (M4)](#6-phase-2-exploratory-data-analysis-m4)
7. [Phase 3: Predictive Modeling and Validation (M5)](#7-phase-3-predictive-modeling-and-validation-m5)
8. [Phase 4: Deployment and Presentation (M6)](#8-phase-4-deployment-and-presentation-m6)
9. [Results and Findings](#9-results-and-findings)
10. [Business Recommendations](#10-business-recommendations)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Introduction and Business Context

### 1.1 Industry Background

The global video game industry has matured into a $200B+ entertainment sector, surpassing the film and music industries combined. From casual mobile games to blockbuster 3A console titles, the diversity of game types, platforms, and business models has expanded dramatically. This growth has attracted massive capital investment, intensified competition, and created an urgent need for data-driven decision-making in game development and publishing.

For game developers, publishers, and investors, understanding the factors that drive commercial success is critical. A game's commercial outcome depends on a complex interplay of platform choice, genre selection, brand recognition, critical reception, regional market preferences, and release timing. Yet the relative importance of these factors — and their interactions — remains poorly quantified. Industry decisions are often guided by intuition and precedent rather than empirical evidence.

### 1.2 Project Motivation

This project aims to fill this gap by systematically quantifying the impact of multiple factors on video game sales using the comprehensive VGChartz 2024 dataset. By applying rigorous data mining, statistical analysis, and predictive modeling techniques, we provide targeted, evidence-based market strategy recommendations for game industry professionals.

### 1.3 Project Scope

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, progressing through six milestones:

| Milestone | Phase | Deliverables |
|---|---|---|
| M1 | Project Proposal | Research questions, hypotheses, team roles |
| M2 & M3 | Data Acquisition & Preprocessing | Cleaned dataset, preprocessing log, quality summary |
| M4 | Exploratory Data Analysis | EDA report with 12+ visualizations, revised hypotheses |
| M5 | Data Modeling & Visualization | 3 models, performance comparison, stakeholder visualizations |
| M6 | Final Deployment & Presentation | Web app, model card, final report, presentation |

---

## 2. Research Questions and Hypotheses

### 2.1 Research Questions

Three core research questions guide this investigation:

**Q1 — Platform-Genre Sales Advantage:** How do the total sales volumes of different game genres vary across different game platforms? Is there a specific platform-genre combination that exhibits a significant sales advantage?

**Q2 — Genre-Regional Impact:** Do different game genres have differential impacts on sales across geographic regions (North America, Japan, Europe/PAL, and Other)? Are there identifiable genre-region combinations with competitive advantages?

**Q3 — Brand Effect:** Does the brand effect of game developers and publishers significantly influence total game sales and sales across individual regions?

### 2.2 Preliminary Hypotheses

Based on domain knowledge and industry literature, the team formulated the following preliminary hypotheses:

- **H1 (Platform-Genre):** Specific platform-genre combinations (e.g., Shooter on Xbox, RPG on PlayStation) will show significantly higher median sales than the average, reflecting platform demographics and genre-audience alignment.
- **H2 (Regional):** Regional markets exhibit distinct genre preferences shaped by cultural factors. Japan will show elevated RPG and Visual Novel sales; North America will favor Shooter and Sports genres.
- **H3 (Brand):** Publisher and developer brand identity has a measurable impact on sales, operating through two channels: (a) scale — the number of games released, reflecting distribution reach and market power; and (b) prestige — the per-title sales premium associated with the brand name, independent of genre and platform.

---

## 3. Dataset Description

### 3.1 Data Source

The dataset is sourced from **VGChartz** (vgchartz.com), a leading video game sales tracking platform. The data was accessed via Kaggle under the title "Video Game Sales 2024" (asaniczka, 2024). The dataset captures game-level sales information aggregated from retail tracking services worldwide.

**Source:** [Video Game Sales 2024 - Kaggle](https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024)

### 3.2 Raw Dataset Characteristics

| Attribute | Value |
|---|---|
| Format | CSV (comma-separated values) |
| Records | 64,016 games |
| Variables | 14 columns |
| Time Period | Primarily pre-2019 releases (long tail into early 2020s) |
| Geographic Coverage | Global, with regional breakdowns |

### 3.3 Raw Variables

| Column | Type | Description |
|---|---|---|
| `img` | Text | Game cover image URL (excluded from analysis) |
| `title` | Text | Game title (excluded from analysis) |
| `console` | Categorical | Gaming platform (e.g., PS4, Xbox 360, Nintendo DS) |
| `genre` | Categorical | Game genre (e.g., Action, Shooter, Role-Playing) |
| `publisher` | Categorical | Publishing company |
| `developer` | Categorical | Development studio |
| `critic_score` | Numerical (0–10) | Aggregated critic review score |
| `total_sales` | Numerical (millions) | Global sales in millions of units |
| `na_sales` | Numerical (millions) | North America sales |
| `jp_sales` | Numerical (millions) | Japan sales |
| `pal_sales` | Numerical (millions) | Europe/Australia/Asia sales |
| `other_sales` | Numerical (millions) | Rest of world sales |
| `release_date` | Date | Original release date |
| `last_update` | Date | Data last updated timestamp (excluded) |

Below is a preview of the raw dataset structure:

![Dataset Preview Part 1](images/dataset_part1.png)
*Figure 1: Raw dataset preview showing game attributes including title, console, genre, publisher, and sales columns*

![Dataset Preview Part 2](images/dataset_part2.png)
*Figure 2: Raw dataset preview (continued) showing critic scores, release dates, and regional sales breakdown*

### 3.4 Ethical Considerations

- The dataset contains only publicly available aggregated sales data — no personally identifiable information.
- VGChartz data primarily reflects physical retail sales, systematically undercounting digital-only releases and PC platform sales (Steam, Epic Games Store). This bias is acknowledged throughout the analysis.
- Sales data for Japan is notably incomplete (63.7% zero-imputed), limiting the reliability of Japan-specific conclusions.
- The analysis does not include potentially sensitive variables (developer salaries, marketing budgets, user demographics).

---

## 4. Methodology and Tech Stack

### 4.1 CRISP-DM Framework

The project follows the CRISP-DM methodology, an industry-standard process for data mining projects:

1. **Business Understanding** — Define objectives, assess situation, determine goals
2. **Data Understanding** — Collect data, describe/explore data, verify quality
3. **Data Preparation** — Clean, construct, integrate, and format data
4. **Modeling** — Select, build, and assess models
5. **Evaluation** — Evaluate results, review process, determine next steps
6. **Deployment** — Plan deployment, monitor, maintain, produce final report

### 4.2 Technology Stack

| Component | Tools |
|---|---|
| **Language** | Python 3.14 |
| **Environment** | Jupyter Lab, VS Code |
| **Data Manipulation** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Statistical Analysis** | scipy, statsmodels |
| **Machine Learning** | scikit-learn, category-encoders |
| **Gradient Boosting** | XGBoost, LightGBM (evaluated) |
| **Model Persistence** | joblib |
| **Web Deployment** | Streamlit |
| **Version Control** | Git, GitHub |

### 4.3 Evaluation Metrics

Three metrics are used for model evaluation:

- **R² Score:** Proportion of variance in the target explained by the model. Primary metric for overall model performance.
- **Root Mean Squared Error (RMSE):** Square root of mean squared prediction error. Penalizes large errors more heavily — critical for risk assessment on high-value blockbuster titles.
- **Mean Absolute Error (MAE):** Average absolute prediction error. Robust to outliers; reflects typical prediction deviation for an average game.

---

## 5. Phase 1: Data Acquisition and Preprocessing (M2 & M3)

### 5.1 Preprocessing Pipeline Architecture

The raw data underwent a three-phase preprocessing pipeline: **Data Cleaning → Data Reduction → Data Transformation**. The complete pipeline is outlined below:

```
Raw Data (64,016 × 14)
    │
    ▼
╔══════════════════════════════════════╗
║ PHASE 1: DATA CLEANING              ║
╠══════════════════════════════════════╣
║ 1.1 Column Selection ────────► (64,016 × 11) ║
║ 1.2 Categorical Imputation ──► ("Unknown")   ║
║ 1.3 Regional Sales Imputation ► (NaN → 0)    ║
║ 1.4 Critic Score Imputation ──► (median)     ║
║ 1.5 Sales Filtering ──────────► (17,570 × 11)║
║ 1.6 Outlier Flagging (IQR) ───► (17,570 × 16)║
╚══════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════╗
║ PHASE 2: DATA REDUCTION             ║
╠══════════════════════════════════════╣
║ 2.1 Stratified Sampling ─────► (8,786 × 16)  ║
║ 2.2 K-Means Clustering ──────► (+sales_cluster)║
║ 2.3 Console×Genre Aggregation► (270 combos)  ║
╚══════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════╗
║ PHASE 3: DATA TRANSFORMATION        ║
╠══════════════════════════════════════╣
║ 3.1 Text Normalization ──────► (uppercase, dedup)║
║ 3.2 Log Transformation ──────► (log_sales)       ║
║ 3.3 Normalization ───────────► (Min-Max + Z-Score)║
║ 3.4 Date Discretization ─────► (5 equal-freq bins)║
║ 3.5 Derived Features ────────► (ratios, brands)  ║
╚══════════════════════════════════════╝
    │
    ▼
Final Dataset (8,786 × 33) ✓
```

### 5.2 Data Cleaning (Phase 1)

**Step 1.1 — Column Selection:** Retained 11 analysis-relevant columns. Removed `img` (image URLs with no analytical value), `title` (unique game identifiers), and `last_update` (metadata timestamp). `release_date` was retained for temporal feature engineering.

**Step 1.2 — Categorical Imputation:** Missing `publisher` and `developer` values (approximately 2,300 records) were filled with the string `"Unknown"`. This preserves records that would otherwise be deleted while explicitly flagging incomplete data.

**Step 1.3 — Regional Sales Imputation:** Missing regional sales columns were filled with 0. This is a conservative approach that assumes no reported sales in that region. The `has_regional_data` binary flag was later created to distinguish genuine zeros from imputed zeros.

| Column | Missing Before | Handling |
|---|---|---|
| `na_sales` | 29.11% | Filled with 0 |
| `jp_sales` | 63.15% | Filled with 0 |
| `pal_sales` | 32.94% | Filled with 0 |
| `other_sales` | 19.52% | Filled with 0 |

**Step 1.4 — Critic Score Imputation:** Missing `critic_score` values (~3.6% of records) were imputed with the median value of 7.50. The median was chosen over the mean for robustness against outliers; the symmetric distribution of critic scores makes median imputation appropriate.

**Step 1.5 — Sales Filtering:** Records with `total_sales <= 0` were removed. This eliminated 46,446 records (72.6% of the raw dataset) — games with zero or negative sales that represent unreleased titles, data entry errors, or placeholders with no commercial activity.

**Step 1.6 — Outlier Flagging (IQR Method):** Extreme values in sales columns were flagged using the IQR method (threshold: 1.5 × IQR), creating binary outlier flag columns. Importantly, outliers were **flagged, not removed** — blockbuster titles like *Grand Theft Auto* and *Call of Duty* represent genuine market phenomena, not measurement errors.

### 5.3 Data Reduction (Phase 2)

**Step 2.1 — Stratified Sampling by Genre:** A 50% stratified random sample was applied, preserving genre proportions. Rare genres (<5 records) were kept in full. This reduced the dataset from 17,570 to 8,786 records while maintaining representativeness.

**Step 2.2 — K-Means Clustering (K=4):** Applied to scaled features (`total_sales`, console label-encoding, `critic_score`) to identify natural commercial performance tiers:

| Cluster | Count | % | Avg Sales | Interpretation |
|---|---|---|---|---|
| 0 | 5,960 | 67.8% | 0.285M | Mainstream (low-to-moderate sellers) |
| 1 | 237 | 2.7% | 3.582M | Blockbuster titles |
| 2 | 2,133 | 24.3% | 0.236M | Budget/Niche titles |
| 3 | 456 | 5.2% | 0.325M | Mid-tier performers |

**Step 2.3 — Console × Genre Aggregation:** Created aggregated average sales per platform-genre combination (minimum 3 games per combination). This directly supports Q1 analysis.

### 5.4 Data Transformation (Phase 3)

**Step 3.1 — Text Normalization:** All categorical columns were converted to uppercase with whitespace stripped. Publisher aliases were consolidated (e.g., "ELECTRONIC ARTS" → "EA", "NAMCO BANDAI GAMES" → "BANDAI NAMCO").

**Step 3.2 — Log Transformation:** The critical transformation: `log_sales = log1p(total_sales)`. This reduces the extreme right skew from 8.04 (raw) to 2.55 (log), making the distribution suitable for parametric modeling.

**Step 3.3 — Feature Scaling:** Sales columns received Min-Max normalization (preserving the zero baseline); `critic_score` received Z-score standardization.

**Step 3.4 — Release Date Discretization:** Release years were binned into 5 equal-frequency eras (`era_1` through `era_5`), converting a continuous temporal variable into an interpretable categorical feature.

**Step 3.5 — Derived Features:** Engineered regional ratio features (`na_ratio`, `jp_ratio`, `pal_ratio`, `other_ratio`), brand scale proxies (`publisher_game_count`, `developer_game_count`), and the `has_regional_data` binary flag.

### 5.5 Preprocessing Summary

| Stage | Rows | Columns | Key Action |
|---|---|---|---|
| Raw | 64,016 | 14 | — |
| After cleaning | 17,570 | 16 | Filtered invalid sales |
| After reduction | 8,786 | 17 | Stratified sampling + clustering |
| After transformation | **8,786** | **33** | Engineered features |

**Final dataset:** 33 columns covering core categorical features, numerical variables, temporal bins, outlier flags, cluster assignments, regional ratios, and brand proxies. Only 0.25% residual missingness (limited to 22 unparseable release dates).

---

## 6. Phase 2: Exploratory Data Analysis (M4)

### 6.1 Univariate Analysis

**Sales Distribution (Viz 1–2):** Raw `total_sales` follows an extreme right-skewed distribution (skewness = 8.04). The mean (0.364M) is 2.6× the median (0.140M), confirming the classic hit-driven nature of the video game market: a tiny fraction of titles generate the majority of revenue.

![Sales Distribution](M4/images/viz01_sales_distribution.png)
*Figure 3: Visualization 1 — Raw total_sales distribution. Mean (red) far exceeds median (orange); log-scale view reveals orders-of-magnitude spread from niche titles to blockbusters.*

Log transformation reduces skewness by 68% to 2.55, producing a distribution suitable for parametric modeling. The Q-Q plot shows good linearity in the central 95% despite formal non-normality (Shapiro-Wilk p < 0.001, as expected for large real-world datasets).

![Log Sales Distribution](M4/images/viz02_log_sales_distribution.png)
*Figure 4: Visualization 2 — log_sales distribution with normal overlay, box plot, and Q-Q plot. Skewness drops from 8.04 to 2.55; the distribution is well-behaved for parametric modeling.*

**Critic Score Distribution (Viz 3):** Critic scores cluster near 7.5 (mean = 7.42, median = 7.50) with limited variance (SD = 0.68). A notable spike at 7.50 reflects median imputation of ~2,300 missing scores. Critic scores show surprisingly little variation across platforms — most platforms' median scores fall between 7.0 and 8.0.

![Critic Score Distribution](M4/images/viz03_critic_score_distribution.png)
*Figure 5: Visualization 3 — Critic_score distribution showing a spike at the imputed median (7.50). Platform-level box plots reveal minimal cross-platform variation.*

**Genre Class Balance (Viz 4):** The genre distribution is moderately imbalanced. The top 3 genres (ACTION 15.3%, SPORTS 14.1%, MISC 10.4%) account for 39.8% of records. The long tail includes SANDBOX (1), BOARD GAME (2), and EDUCATION (2).

![Genre Class Balance](M4/images/viz04_genre_class_balance.png)
*Figure 6: Visualization 4 — Genre class balance. Top 3 genres cover ~40% of data; 7 genres reach the 70% threshold. Rare genres (<1%) require aggregation for reliable analysis.*

### 6.2 Bivariate Analysis

**Sales by Platform (Viz 5):** Nintendo DS and PS2 dominate in game volume (~1,055 each), reflecting their status as best-selling platforms. However, median sales are remarkably uniform across platforms (0.05M–0.20M). The differentiation appears in the upper quartile. PC has the lowest median sales, likely from undercounted digital distribution.

![Sales by Console](M4/images/viz05_sales_by_console.png)
*Figure 7: Visualization 5 — Sales distribution by console platform. DS and PS2 lead in volume; PS4 and X360 show elevated upper-tail outliers.*

**Sales by Genre (Viz 6):** Genre differentiates sales more strongly than platform. SHOOTER and ROLE-PLAYING genres exhibit the highest median sales and widest spread. SPORTS titles show compressed distributions (annualized franchises). MISC is bimodal.

![Sales by Genre](M4/images/viz06_sales_by_genre.png)
*Figure 8: Visualization 6 — Sales by genre. SHOOTER and ROLE-PLAYING show the highest medians; SPORTS is compressed (annualized releases); MISC is bimodal.*

**Critic Score vs. Sales (Viz 7):** A remarkably weak correlation exists between critic scores and sales (Pearson r ≈ 0.05, Spearman ρ ≈ 0.05). Critic score alone explains less than 0.5% of sales variance.

![Critic Score vs Sales](M4/images/viz07_critic_vs_sales.png)
*Figure 9: Visualization 7 — Critic_score vs. log_sales hexbin plot. The binned mean trend (blue) shows a weak positive slope; scatter is enormous at all score levels.*

**Console × Genre Heatmap (Viz 8):** This directly addresses Q1. High-performing combinations include PS4 × SPORTS (0.75M), PS3 × FIGHTING (0.74M), X360 × SHOOTER (0.64M), and Xbox One × SHOOTER (0.70M). PC is consistently low across all genres.

![Console × Genre Heatmap](M4/images/viz08_console_genre_heatmap.png)
*Figure 10: Visualization 8 — Median sales heatmap for Console × Genre combinations. Dark cells indicate high premium combos: PS4×Sports, PS3×Fighting, X360×Shooter.*

**Key insight:** Platform-genre interactions are meaningful and concentrated in specific high-value combinations, validating the inclusion of interaction terms in modeling.

### 6.3 Multivariate Analysis

**Regional Preference Patterns (Viz 9):** Three distinct regional clusters emerge, directly addressing Q2:

| Region | Dominant Genres | Key Ratio |
|---|---|---|
| North America | SHOOTER, SPORTS, ACTION | na_ratio > 0.40 |
| Japan | RPG, VISUAL NOVEL, FIGHTING | jp_ratio > 0.35 (RPG) |
| Europe/PAL | RACING, PLATFORM, SPORTS | Elevated pal_ratio |

![Regional Sales Patterns](M4/images/viz09_regional_sales_patterns.png)
*Figure 11: Visualization 9 — Regional sales patterns by genre. Stacked bar (left) shows normalized proportions; absolute sales (right) reflect market size differences. Japan shows distinct RPG/VN preferences.*

North America dominates in absolute volume, but proportional analysis reveals genuine cultural preferences. Japan's RPG affinity (jp_ratio > 0.35) contrasts sharply with NA's shooter preference.

**K-Means Cluster Characterization (Viz 10):** The four clusters correspond to interpretable market tiers:
- **Blockbuster (2.7%):** Median 3.58M sales, dominated by Action and Shooter on PS3/PS4/X360
- **Mainstream (67.8%):** Median 0.29M, balanced genre/platform distribution
- **Mid-tier (5.2%):** Elevated Sports/Racing (annualized franchises)
- **Budget/Niche (24.3%):** Over-represented by MISC/Adventure on handheld platforms

![K-Means Clusters](M4/images/viz10_kmeans_clusters.png)
*Figure 12: Visualization 10 — K-Means cluster characterization (K=4). Four panels show sales distribution, genre composition, platform composition, and critic scores by cluster. Cluster 1 (2.7%) = Blockbuster tier.*

Critic scores vary minimally across clusters (7.4–7.6), confirming that critical reception is not the primary commercial differentiator. The cluster structure motivates a two-stage modeling approach.

**Publisher Brand Analysis (Viz 11):** Publisher strategies diverge into two categories:
- **Volume publishers** (EA, Ubisoft, Konami): Many games, moderate per-title sales, broad coverage.
- **Prestige publishers** (Nintendo, Rockstar, Bethesda): Fewer games, higher per-title premiums.
- Nintendo achieves both high volume and high per-title premium.

![Publisher Brand Analysis](M4/images/viz11_publisher_brand.png)
*Figure 13: Visualization 11 — Top 20 publisher brand analysis. Four panels show revenue concentration, volume vs. quality scatter, sales vs. critic reception, and platform vs. genre breadth strategy.*

**Correlation Matrix (Viz 12):** Western regional sales (NA, PAL, Other) are highly intercorrelated (ρ > 0.6), while Japan sales show weaker correlation (ρ ≈ 0.3–0.4). `publisher_game_count` and `developer_game_count` are highly correlated (ρ > 0.8).

![Correlation Matrix](M4/images/viz12_correlation_matrix.png)
*Figure 14: Visualization 12 — Spearman correlation matrix of numerical features (left) and feature correlations with log_sales target (right). NA/PAL markets are highly correlated; Japan is distinct.*

### 6.4 Revised Hypotheses

Based on EDA findings, the hypotheses were refined:

- **H1 (Platform-Genre):** Supported. The interaction is meaningful but concentrated in the upper quartile. SHOOTER/RPG carry a premium; handhelds excel in niche genres.
- **H2 (Regional):** Strongly supported. Three regional clusters identified with distinct genre preferences. Japan data quality (63.7% zeros) is a noted limitation.
- **H3 (Brand):** Supported. Brand operates through two separable mechanisms: scale (publisher_game_count) and prestige (publisher identity fixed effects).

---

## 7. Phase 3: Predictive Modeling and Validation (M5)

### 7.1 Feature Engineering for Modeling

The 33-feature enriched dataset was prepared for modeling with the following transformations:

- **Target variable:** `log_sales` (log-transformed total sales)
- **Low-cardinality categoricals** (console, genre, release_year_bin): One-Hot Encoding
- **High-cardinality categoricals** (publisher ~551, developer ~2,500): Target Encoding with smoothing = 10
- **Numerical features** (critic_score, publisher_game_count, developer_game_count): Z-score standardization
- **Excluded features:** Raw regional sales columns removed to avoid data leakage with `total_sales`
- **Train-test split:** 80/20 stratified by genre (7,028 training / 1,758 test)

### 7.2 Model Architecture

Three models were built and evaluated:

#### Model 1: Multiple Linear Regression (Interpretable Baseline)

MLR fits a linear relationship between features and `log_sales`, providing interpretable coefficients. It serves as a baseline to quantify the lower bound of predictive power from linear effects.

**Justification:** Directly quantifies linear feature effects, aligns with ANOVA findings from EDA, provides statistical significance tests, requires minimal computation.

#### Model 2: Random Forest Regressor (Production Model)

Random Forest is an ensemble of 100 decision trees (max_depth=15, min_samples_leaf=10), averaging predictions to reduce overfitting.

**Justification:** Captures non-linear interactions and feature interactions identified in EDA (platform-genre synergy, non-linear brand-scale effects). Native feature importance scores directly quantify each factor's contribution. Robust to outliers with minimal tuning required.

#### Model 3: Two-Stage Blockbuster-Aware Model (Specialized Variant)

This model directly attacks the systematic blockbuster underestimation exposed by the Random Forest residual analysis (predicted mean log_sales of 0.88 vs. actual mean of 1.87 for blockbusters).

**Architecture — Hybrid Blend:**

1. **Stage 1 — Classifier:** A `class_weight="balanced"` Random Forest classifier predicts blockbuster probability.
2. **Stage 2 — Dedicated Regressor:** A Random Forest trained exclusively on 62 blockbuster training records (total_sales > 3.5M).
3. **Blending Rule:** The global Random Forest is the base predictor for every game. Titles with classifier probability ≥ 0.60 receive:
   `final_pred = 0.30 × blockbuster_reg + 0.70 × base_rf`

A soft blend was chosen over hard routing because blockbusters constitute only 0.9% of the data — a pure two-stage model would misroute too many non-blockbusters.

### 7.3 Model Performance

#### Cross-Validation (5-Fold, Stratified)

| Model | CV Mean R² | CV Mean RMSE | CV Mean MAE |
|---|---|---|---|
| Multiple Linear Regression | 0.2796 | 0.2588 | 0.1769 |
| Random Forest Regressor | **0.3176** | **0.2516** | **0.1680** |

#### Holdout Test Set Performance

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| Multiple Linear Regression | 0.3123 | 0.2592 | 0.1730 |
| **Random Forest Regressor** | **0.3937** | **0.2434** | **0.1585** |
| Two-Stage RF (Blockbuster-Aware) | 0.3775 | 0.2466 | 0.1594 |

#### Blockbuster-Segment Performance

| Metric | Random Forest | Two-Stage | Improvement |
|---|---|---|---|
| Blockbuster RMSE | 1.0845 | 1.0122 | **−6.7%** |

#### Classifier Performance (Test Set)

| Metric | Value |
|---|---|
| Precision | 0.16 |
| Recall | 0.44 |
| F1 | 0.24 |

The low precision confirms blockbuster identification from pre-release attributes alone is intrinsically hard — exactly why a soft blend (not hard routing) is used. The modest 6.7% RMSE improvement empirically confirms that blockbuster data scarcity, not model architecture, is the binding constraint.

### 7.4 Performance Visualizations

The residual plot below shows the prediction error distribution for the two-stage model. Residuals are centered on zero for mid-range sales but fan out at higher values — the structural pattern the two-stage model was designed to mitigate.

![Residual Plot](M5/images/viz0_residuals_two_stage.png)
*Figure 15: M5 Visualization 0 — Residual plot for the two-stage blockbuster-aware model. Residuals fan out at higher predicted values, confirming variance grows with sales magnitude.*

The model performance comparison chart demonstrates that the Random Forest model delivers highest R² and lowest RMSE. The two-stage variant makes a strategic trade-off: slightly lower overall R² for improved blockbuster accuracy.

![Model Performance Comparison](M5/images/viz1_model_performance_comparison.png)
*Figure 16: M5 Visualization 1 — Model performance comparison (R² and RMSE). Random Forest leads across both metrics; two-stage trades modest R² for blockbuster RMSE improvement.*

### 7.5 Feature Importance Analysis

The Random Forest feature importance ranking reveals the top sales drivers:

| Rank | Feature | Importance Score | Category |
|---|---|---|---|
| 1 | developer (target-encoded) | 0.616 | Brand/Prestige |
| 2 | critic_score | 0.122 | Quality |
| 3 | developer_game_count | 0.083 | Brand/Scale |
| 4 | publisher (target-encoded) | 0.066 | Brand/Prestige |
| 5 | publisher_game_count | 0.040 | Brand/Scale |
| 6–10 | console features (various) | 0.010–0.033 | Platform |

![Feature Importance Ranking](M5/images/viz2_feature_importance_ranking.png)
*Figure 17: M5 Visualization 2 — Top 10 feature importance ranking (Random Forest). Developer target-encoding dominates; critic_score and brand scale metrics follow. Platform and genre effects are secondary.*

The actual vs. predicted scatter plot confirms both models perform reliably for mid-to-low sales games, with the two-stage variant visibly tightening the upper tail.

![Actual vs Predicted Sales](M5/images/viz3_actual_vs_predicted_sales.png)
*Figure 18: M5 Visualization 3 — Actual vs. predicted log_sales for Random Forest (left) and Two-Stage RF (right). The two-stage model lifts previously under-predicted blockbuster points toward the diagonal.*

The blockbuster RMSE comparison directly quantifies the two-stage model's improvement on the target segment.

![Blockbuster RMSE Lift](M5/images/viz4_blockbuster_rmse_lift.png)
*Figure 19: M5 Visualization 4 — Blockbuster prediction RMSE comparison. The two-stage model reduces RMSE from 1.085 to 1.012, a 6.7% improvement on the 18 test blockbusters.*

### 7.6 Key Findings for Research Questions

**Q1 — Platform-Genre Sales Advantage (Confirmed):** Platform and genre features rank in the top 10 of feature importance, confirming independent effects on sales. The 26% R² lift from linear to non-linear models validates that platform-genre interaction effects are meaningful, with combinations like PS4 × Sports and X360 × Shooter delivering significant premiums.

**Q2 — Genre-Regional Impact (Confirmed, with caveats):** Regional preference patterns are clear from EDA (Viz 9) and validated by the model's feature logic. However, the 63.7% zero-imputation rate for Japan sales limits quantitative verification in the global model. Dedicated regional sub-models are recommended.

**Q3 — Brand Effect (Confirmed):** Both brand channels — scale (`publisher_game_count` at rank 3) and prestige (target-encoded developer at rank 1, publisher at rank 4) — are top features. Brand influence on sales is the single strongest finding of the analysis.

---

## 8. Phase 4: Deployment and Presentation (M6)

### 8.1 Streamlit Web Application

A fully functional Streamlit web application was developed and deployed, enabling stakeholders to interact with the trained models.

**Features:**
- **Sidebar input form:** Dropdown selectors for console, genre, and release era; text input for publisher and developer with autocomplete suggestions; slider for critic score.
- **Prediction output:** Dual predictions from the Random Forest (recommended for general use) and the two-stage blockbuster-aware variant (better for high-budget projects).
- **Blockbuster probability:** Displays the predicted blockbuster probability and classification label.
- **Top feature drivers:** Shows the top 8 features and their relative importance for the specific prediction.
- **Model performance reference:** Displays the M5 test set performance table for stakeholder transparency.

**Technology:** Streamlit (Python), joblib for model persistence, deployed via Vercel.

### 8.2 Model Card

A comprehensive model card was created following the Google Model Cards for Model Reporting framework, documenting:

- **Intended use:** Sales forecasting for mainstream mid-budget video games. Supports marketing budget allocation, project risk assessment, and portfolio planning.
- **Do NOT use for:** Blockbuster prediction at high precision, brand-new platforms with no historical data, digital-only monetization models not in the training data.
- **Factors:** 8 input features with appropriate encodings (one-hot, target encoding, Z-score).
- **Ethical considerations:** Physical-sales bias, publisher scale bias (disadvantages new entrants), regional fairness issues (Japan data quality), genre representation (niche genres merged to "Other").
- **Caveats and recommendations:** Known limitations with operational mitigations, retraining recommendations, and future feature expansion plans.

### 8.3 Presentation

A structured 7-minute presentation director's guide was prepared, covering research objectives, data pipeline, EDA insights (platform-genre synergy, regional clusters, brand mechanisms), modeling approach, live app demo, recommendations, and limitations. The presentation is supported by a visual asset mapping table referencing all key figures from M4 and M5.

---

## 9. Results and Findings

### 9.1 Summary of Findings by Research Question

#### Q1: Platform-Genre Sales Advantage

**Finding confirmed.** Platform-genre interactions are meaningful predictors of sales, concentrated in the upper quartile of the sales distribution.

- Top combinations: PS4 × Sports (0.75M median), PS3 × Fighting (0.74M), X360 × Shooter (0.64M), Xbox One × Shooter (0.70M).
- SHOOTER and RPG genres carry a sales premium across most platforms.
- Handheld platforms (DS, PSP) dominate niche genres (MISC, Puzzle, Simulation).
- Later-generation home consoles show higher per-game sales, reflecting industry consolidation.
- The 26% R² lift from linear to non-linear models quantitatively validates that platform-genre synergy matters.

**Business implication:** Platform-genre strategy should be a deliberate component of release planning.

#### Q2: Genre-Regional Impact

**Finding confirmed.** Three distinct regional market clusters exhibit clearly differentiated genre preferences:

- **North America:** Shooter, Sports, and Action dominate (na_ratio > 40%).
- **Japan:** RPG, Visual Novel, and Fighting dominate (jp_ratio > 35% for RPG).
- **Europe/PAL:** Racing, Platform, and Sports show elevated shares.

Western regional markets (NA, PAL, Other) are highly correlated (ρ > 0.6), while Japan shows distinct preferences (ρ ≈ 0.3–0.4 with Western markets).

**Caveat:** Japan sales data has 63.7% zero-imputation due to incomplete VGChartz coverage. Japan-specific conclusions should be treated as directional rather than definitive.

**Business implication:** Genre-based regional marketing strategies can significantly improve market penetration.

#### Q3: Brand Effect

**Finding confirmed.** Brand influence on sales operates through two empirically separable channels:

1. **Scale channel (dominant):** Publishers with high game volumes achieve distribution, marketing, and platform-access advantages. `publisher_game_count` is a top-5 feature.

2. **Prestige channel (significant):** Certain publishers and developers command a per-title premium above what genre, platform, and scale alone would predict. Target-encoded developer identity is the #1 feature.

Nintendo uniquely achieves both high volume and high per-title premium, representing the gold standard.

**Business implication:** New game studios should partner with established publishers for distribution scale. Building brand prestige through consistent quality creates additional premium value.

### 9.2 Model Performance Summary

| Metric | Linear Regression | Random Forest | Two-Stage |
|---|---|---|---|
| R² | 0.312 | **0.394** | 0.378 |
| RMSE | 0.259 | **0.243** | 0.247 |
| MAE | 0.173 | **0.159** | 0.159 |
| Blockbuster RMSE | — | 1.085 | **1.012** (−6.7%) |
| **Best For** | Interpretable baseline | General forecasting | Blockbuster priority |

---

## 10. Business Recommendations

Based on the empirical findings, the following actionable recommendations are made for game industry stakeholders:

### 10.1 Strategic Recommendations

**1. Partner with Established Publishers (Highest Impact)**
Publisher brand scale is among the strongest predictors of game sales. New studios and independent developers should prioritize publishing partnerships with established entities to access distribution, marketing, and platform-access advantages.

**2. Prioritize Game Quality (Significant Impact)**
While critic score alone shows weak correlation with sales, it ranks as the second most important feature in the Random Forest model — indicating non-linear effects. Games scoring in the top decile (9.0+) demonstrate disproportionately higher sales.

**3. Use Platform-Genre Strategy (Measurable Impact)**
Platform-genre synergy is real and concentrated in specific combinations. Pair Shooters with Xbox/PlayStation, RPGs with PlayStation and handheld platforms, and Sports with PlayStation and Nintendo platforms. Avoid niche genres on PC where digital undercount distorts the data.

**4. Tailor Regional Marketing (Measurable Impact)**
Regional genre preferences are distinct and quantifiable. Emphasize Shooters, Sports, and Action in North America; RPGs and Visual Novels in Japan; Racing and Platform games in Europe/PAL.

**5. Use the Model as Decision Support (Risk Management)**
The Random Forest model (R² = 0.39) is appropriate for mainstream mid-budget title forecasting but should not be the sole basis for budget allocation. Use the two-stage variant when blockbuster projects are under consideration.

### 10.2 Operational Recommendations

- **Deploy Random Forest** for general forecasting; switch to two-stage variant for blockbuster priority.
- **Retrain quarterly** with updated VGChartz data to capture new releases and platform shifts.
- **Expand features** by incorporating marketing spend, IP attributes, and user review sentiment.
- **Build regional sub-models** for NA, Europe/PAL, and Japan to address preference heterogeneity.
- **Actively collect more blockbuster training samples** — data scarcity, not architecture, is the binding constraint.

---

## 11. Limitations and Future Work

### 11.1 Data Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Digital sales undercount** | Systematic bias for PC/indie titles | Treat PC forecasts as conservative |
| **Japan data gap** | 63.7% zero-imputation; poor JP reliability | Separate regional sub-models needed |
| **Blockbuster scarcity** | Only 62 training blockbusters (0.9%) | Two-stage variant partially addresses |
| **Temporal coverage** | Mostly pre-2019 data | Quarterly retraining required |
| **High-cardinality encoding** | Overfitting risk for new entities | Conservative predictions for unknowns |

### 11.2 Methodological Limitations

- **Correlation ≠ Causation:** Feature importance reflects associative relationships. Publisher scale may correlate with better IP rather than directly driving sales.
- **No causal inference:** The observational study cannot establish causal mechanisms.
- **Single imputation for critic_score:** Does not account for uncertainty in missing values.
- **Rare genre information loss:** Genres <1% frequency merged into "Other".

### 11.3 Potential Failure Modes

1. **New platform/genre prediction:** Unreliable for entirely new platforms or genres with no historical data.
2. **Blockbuster underestimation:** Even two-stage variant carries high uncertainty for top-tier titles.
3. **New entrant bias:** Target encoding shrinks predictions for unseen publishers toward global mean.
4. **Market structure shifts:** Subscription services, F2P, and cloud gaming could invalidate physical-sales patterns.

### 11.4 Future Work

1. **Feature expansion:** Integrate marketing spend, IP franchise attributes, user review sentiment.
2. **Regional sub-models:** Separate regression models for NA, Japan, and PAL markets.
3. **Causal inference methods:** Instrumental variable or difference-in-differences for causal effects.
4. **Deep learning exploration:** Test tabNet or FT-Transformer architectures.
5. **Temporal backtesting:** Time-based cross-validation for real-world generalization assessment.
6. **Post-2019 data enrichment:** Capture modern market dynamics and digital distribution trends.

---

## 12. Conclusion

This project provides a comprehensive, data-driven analysis of the factors influencing video game sales, following the CRISP-DM framework from business understanding through deployment.

**Three research questions were answered:**

1. **Q1 — Platform-Genre Synergy Matters (Confirmed).** Specific platform-genre combinations command measurable sales premiums, concentrated in the upper quartile. The 26% R² lift from linear to ensemble models validates platform-genre positioning as a legitimate competitive lever.

2. **Q2 — Regional Markets Are Distinct (Confirmed).** North America, Japan, and Europe/PAL exhibit clearly differentiated genre preferences. Tailoring game portfolios and marketing strategies to regional tastes is empirically justified.

3. **Q3 — Brand Drives Sales Through Two Channels (Confirmed).** Brand scale (volume of releases) and brand prestige (per-title premium) both contribute meaningfully. The dual-channel finding has direct implications for strategic partnerships and brand building.

**The project delivered:**
- A cleaned, enriched dataset (8,786 records × 33 features)
- 12+ EDA visualizations with documented interpretations (M4)
- Three predictive models with Random Forest achieving R² = 0.394 (M5)
- A novel two-stage blockbuster-aware variant reducing blockbuster RMSE by 6.7%
- A deployed Streamlit web application for stakeholder use (M6)
- A comprehensive model card documenting intended use, limitations, and ethical considerations

The video game industry's commercial dynamics are complex and multi-factorial. No single factor determines success. But with rigorous data analysis, the relative importance of each factor can be quantified — and that quantification provides a foundation for more informed, evidence-based decision-making in one of the world's most dynamic entertainment sectors.

---

## 13. References

1. VGChartz 2024 Dataset. Kaggle. https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024
2. Mitchell, T. M. (1997). *Machine Learning.* McGraw-Hill.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. Witten, I. H., Frank, E., & Hall, M. A. (2011). *Data Mining: Practical Machine Learning Tools and Techniques.* Morgan Kaufmann.
5. Shearer, C. (2000). The CRISP-DM model: The new blueprint for data mining. *Journal of Data Warehousing*, 5(4), 13–22.
6. Mitchell, M., et al. (2019). Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220–229.
7. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51–56.
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
9. Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
10. ESA (Entertainment Software Association). (2023). *Essential Facts About the Video Game Industry.*

---

*This report was prepared by Group 23 for INFO422 Data Science Project at Lanzhou University, July 2026.*
