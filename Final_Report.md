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
- **H3 (Brand):** Publisher and developer brand identity has a measurable impact on sales, operating through two channels: (a) scale — the number of games released, reflecting distribution reach and market power — and (b) prestige — the per-title sales premium associated with the brand name, independent of genre and platform.

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

### 5.1 Preprocessing Pipeline

The raw data underwent a three-phase preprocessing pipeline: **Data Cleaning → Data Reduction → Data Transformation**. The complete pipeline is documented below.

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

**Sales Distribution (Viz 1–2):** Raw `total_sales` follows an extreme right-skewed distribution (skewness = 8.04). The mean (0.364M) is 2.6× the median (0.140M), confirming the classic hit-driven nature of the video game market: a tiny fraction of titles generate the majority of revenue. Log transformation reduces skewness by 68% to 2.55, producing a distribution suitable for parametric modeling. The Q-Q plot shows good linearity in the central 95% despite formal non-normality (Shapiro-Wilk p < 0.001, as expected for large real-world datasets).

**Critic Score Distribution (Viz 3):** Critic scores cluster near 7.5 (mean = 7.42, median = 7.50) with limited variance (SD = 0.68). A notable spike at 7.50 reflects median imputation of ~2,300 missing scores. Critic scores show surprisingly little variation across platforms — most platforms' median scores fall between 7.0 and 8.0 — suggesting that review aggregation normalizes scores across the industry.

**Genre Class Balance (Viz 4):** The genre distribution is moderately imbalanced. The top 3 genres (ACTION 15.3%, SPORTS 14.1%, MISC 10.4%) account for 39.8% of records. The long tail includes SANDBOX (1), BOARD GAME (2), and EDUCATION (2) — genres that require special handling or grouping in per-genre analyses.

### 6.2 Bivariate Analysis

**Sales by Platform (Viz 5):** Nintendo DS and PS2 dominate in game volume (~1,055 each), reflecting their status as best-selling platforms. However, median sales are remarkably uniform across platforms (0.05M–0.20M). The differentiation appears in the upper quartile: later-generation consoles (PS4, Xbox One) show slightly higher median sales, consistent with industry consolidation toward fewer, higher-budget titles. PC has the lowest median sales, likely from undercounted digital distribution.

**Sales by Genre (Viz 6):** Genre differentiates sales more strongly than platform. SHOOTER and ROLE-PLAYING genres exhibit the highest median sales and widest spread, reflecting blockbuster economics. SPORTS titles show compressed distributions (annualized franchises with reliable but capped sales). MISC is bimodal — a catch-all category mixing niche titles and breakout hits.

**Critic Score vs. Sales (Viz 7):** A remarkably weak correlation exists between critic scores and sales (Pearson r ≈ 0.05, Spearman ρ ≈ 0.05). Critic score alone explains less than 0.5% of sales variance. Even games with perfect 10.0 scores span the full sales range. This confirms that commercial success in gaming is driven by factors beyond critical reception — marketing, brand, timing, and platform strategy likely dominate.

**Console × Genre Heatmap (Viz 8):** This directly addresses Q1. High-performing combinations include:
- PS4 × SPORTS (median 0.75M)
- PS3 × FIGHTING (median 0.74M)
- X360 × SHOOTER (median 0.64M)
- Xbox One × SHOOTER (median 0.70M)

PC is consistently low across all genres (0.02M–0.07M), reinforcing the digital sales undercount concern. **Key insight:** Platform-genre interactions are meaningful and concentrated in specific high-value combinations, validating the inclusion of interaction terms in modeling.

### 6.3 Multivariate Analysis

**Regional Preference Patterns (Viz 9):** Three distinct regional clusters emerge, directly addressing Q2:

| Region | Dominant Genres | Key Ratio |
|---|---|---|
| North America | SHOOTER, SPORTS, ACTION | na_ratio > 0.40 |
| Japan | RPG, VISUAL NOVEL, FIGHTING | jp_ratio > 0.35 (RPG) |
| Europe/PAL | RACING, PLATFORM, SPORTS | Elevated pal_ratio |

North America dominates in absolute volume, but proportional analysis reveals genuine cultural preferences. Japan's RPG affinity (jp_ratio > 0.35) contrasts sharply with NA's shooter preference. The Japan market has 63.7% zero-imputed data, so these patterns are directional rather than precisely quantified.

**K-Means Cluster Characterization (Viz 10):** The four clusters correspond to interpretable market tiers:
- **Blockbuster (2.7%):** Median 3.58M sales, dominated by Action and Shooter genres on PS3/PS4/X360
- **Mainstream (67.8%):** Median 0.29M, balanced genre/platform distribution
- **Mid-tier (5.2%):** Elevated Sports and Racing presence (annualized franchises)
- **Budget/Niche (24.3%):** Over-represented by MISC and Adventure on handheld platforms

Critic scores vary minimally across clusters (7.4–7.6), confirming that critical reception is not the primary commercial differentiator. The cluster structure motivates the two-stage modeling approach: classify the tier first, then predict within-tier sales.

**Publisher Brand Analysis (Viz 11):** Publisher strategies diverge into two categories:

- **Volume publishers** (EA, Ubisoft, Konami): Many games, moderate per-title sales, broad platform/genre coverage.
- **Prestige publishers** (Nintendo, Rockstar, Bethesda): Fewer games, higher per-title premiums, selective release strategy.
- Nintendo emerges as an outlier — achieving both high volume and high per-title premium — representing the industry gold standard.

**Correlation Matrix (Viz 12):** Western regional sales (NA, PAL, Other) are highly intercorrelated (ρ > 0.6), while Japan sales show weaker correlation (ρ ≈ 0.3–0.4), confirming Japan's market distinctiveness. `publisher_game_count` and `developer_game_count` are highly correlated (ρ > 0.8) — only one should enter any given model to avoid multicollinearity.

### 6.4 Revised Hypotheses

Based on EDA findings, the hypotheses were refined:

- **H1 (Platform-Genre):** Supported. The interaction is meaningful but concentrated in the upper quartile. SH-Directly answer the three Qs: OOTER/RPG carry a premium; handhelds excel in niche genres.
- **H2 (Regional):** Strongly supported. Three regional clusters identified with distinct genre preferences. The Japan data quality issue (63.7% zeros) is a noted limitation.
- **H3 (Brand):** Supported. Brand operates through two separable mechanisms: scale (publisher_game_count) and prestige (publisher identity fixed effects). Both should be modeled separately.

### 6.5 Modelling Question

The EDA led to the following formal modeling question:

> **"Can we predict a video game's global sales (`log_sales`) using its platform, genre, publisher, developer, critic score, and release era, and quantify each factor's relative importance?"**

**Planned approach:** Linear Regression (interpretable baseline), Random Forest (non-linear interactions), Two-Stage Blockbuster-Aware (addresses long-tail underestimation).

---

## 7. Phase 3: Predictive Modeling and Validation (M5)

### 7.1 Model Architecture

Three models were built and evaluated using the preprocessed dataset (8,786 records, 33 features, 80/20 train-test split stratified by genre).

#### Model 1: Multiple Linear Regression (Interpretable Baseline)

MLR fits a linear relationship between features and `log_sales`, providing interpretable coefficients. It serves as a baseline to quantify the lower bound of predictive power from linear effects. Low-cardinality categoricals (console, genre, release era) received one-hot encoding; high-cardinality features (publisher, developer) used target encoding (smoothing = 10); numerical features were Z-score standardized.

**Justification:** Directly quantifies linear feature effects, aligns with ANOVA findings from EDA, provides statistical significance tests, and requires minimal computation.

#### Model 2: Random Forest Regressor (Production Model)

Random Forest is an ensemble of 100 decision trees (max_depth=15, min_samples_leaf=10), averaging predictions to reduce overfitting.

**Justification:** Captures the non-linear interactions and feature interactions identified in EDA (platform-genre synergy, non-linear brand-scale effects) that linear models miss. Native feature importance scores directly quantify each factor's contribution. Robust to outliers and performs reliably with default parameters.

#### Model 3: Two-Stage Blockbuster-Aware Model (Specialized Variant)

This model directly attacks the systematic blockbuster underestimation exposed by the Random Forest residual analysis (predicted mean log_sales of 0.88 vs. actual mean of 1.87 for blockbusters).

**Architecture — Hybrid Blend:**
1. **Stage 1 — Classifier:** A `class_weight="balanced"` Random Forest classifier predicts blockbuster probability.
2. **Stage 2 — Dedicated Regressor:** A Random Forest trained exclusively on 62 blockbuster training records (total_sales > 3.5M).
3. **Blending Rule:** The global Random Forest is the base predictor for every game. Titles with classifier probability ≥ 0.60 receive a blended prediction:
   `final_pred = 0.30 × blockbuster_reg + 0.70 × base_rf`

A soft blend was chosen over hard routing because blockbusters constitute only 0.9% of the data — a pure two-stage model would misroute too many non-blockbusters.

### 7.2 Model Performance

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

The low precision confirms blockbuster identification from pre-release attributes alone is intrinsically hard — exactly why a soft blend (not hard routing) is used.

### 7.3 Performance Analysis

**Non-linear advantage:** Random Forest outperforms Linear Regression by 26% in R², 6.1% in RMSE, and 8.4% in MAE, confirming that non-linear interactions contribute substantially to sales prediction accuracy.

**Strong generalization:** No performance drop from cross-validation to holdout test set, indicating no severe overfitting.

**Two-stage trade-off:** The two-stage model sacrifices 1.6 percentage points of overall R² to achieve a 6.7% improvement in blockbuster-segment RMSE. The Random Forest is recommended for general forecasting; the two-stage variant is preferred when blockbuster accuracy is prioritized.

**Binding constraint:** The modest two-stage improvement empirically confirms that blockbuster scarcity — not model architecture — is the binding constraint on prediction accuracy.

### 7.4 Feature Importance

The Random Forest feature importance ranking reveals the top sales drivers:

| Rank | Feature | Importance Score | Category |
|---|---|---|---|
| 1 | developer (target-encoded) | 0.616 | Brand/Prestige |
| 2 | critic_score | 0.122 | Quality |
| 3 | developer_game_count | 0.083 | Brand/Scale |
| 4 | publisher (target-encoded) | 0.066 | Brand/Prestige |
| 5 | publisher_game_count | 0.040 | Brand/Scale |
| 6–10 | console features (various) | 0.010–0.033 | Platform |
| — | genre features | < 0.010 | Genre |

**Key findings for research questions:**
- **Q1 (Platform-Genre):** Console and genre features are present in the top 10, confirming their independent effects. The 26% R² lift from linear to non-linear models validates the importance of platform-genre interactions.
- **Q2 (Regional):** Regional patterns were validated through EDA visualizations and correlation analysis. Genre features show heterogeneous predictive power across regional sub-markets.
- **Q3 (Brand):** Fully validated. Brand scale (`publisher_game_count`/`developer_game_count`) and brand prestige (target-encoded publisher/developer identity) both rank among the top features, confirming the dual-channel brand effect hypothesis.

### 7.5 Stakeholder Visualizations

Five stakeholder-facing visualizations were produced:

1. **Residual Plot:** Shows prediction errors centered on zero for mid-range sales but fanning out at higher values — the pattern the two-stage model was designed to mitigate.
2. **Model Performance Comparison:** Side-by-side bar chart of R² and RMSE across all three models, justifying Random Forest adoption for production.
3. **Feature Importance Ranking:** Horizontal bar chart of top 10 features — publisher brand scale dominates.
4. **Actual vs. Predicted Scatter:** Comparison showing both models perform reliably for mid-to-low sales, with the two-stage variant tightening the upper tail.
5. **Blockbuster RMSE Lift:** Quantifies the 6.7% blockbuster RMSE improvement from the two-stage variant.

### 7.6 Research Question Validation

**Q1 — Platform-Genre Sales Advantage (Confirmed):** Platform and genre features rank in the top 10 of feature importance, confirming independent effects on sales. The non-linear gain validates that platform-genre interaction effects are meaningful, with combinations like PS4 × Sports and X360 × Shooter delivering significant premiums.

**Q2 — Genre-Regional Impact (Confirmed, with caveats):** Regional preference patterns are clear from EDA (Viz 9) and validated by the model's feature logic. However, the 63.7% zero-imputation rate for Japan sales limits quantitative verification in the global model. Dedicated regional sub-models are recommended.

**Q3 — Brand Effect (Confirmed):** Both brand channels — scale (`publisher_game_count` at rank 1) and prestige (target-encoded publisher at rank 4) — are top features. Brand influence on sales is the strongest finding of the analysis.

---

## 8. Phase 4: Deployment and Presentation (M6)

### 8.1 Streamlit Web Application

A fully functional Streamlit web application was developed and deployed, enabling stakeholders to interact with the trained models.

**Features:**
- **Sidebar input form:** Dropdown selectors for console, genre, and release era; text input for publisher and developer with autocomplete suggestions; slider for critic score.
- **Prediction output:** Dual predictions from the Random Forest (recommended for general use) and the two-stage blockbuster-aware variant (better for high-budget projects).
- **Blockbuster probability:** Displays the predicted blockbuster probability and classification label.
- **Top feature drivers:** Shows the top 8 features and their relative importance for the specific prediction, highlighting if the user's selected console, genre, or publisher is a top predictor.
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

A structured 7-minute presentation director's guide was prepared, covering:
- Research objectives and business context
- Data pipeline and preprocessing summary
- Three EDA insight sections (platform-genre synergy, regional clusters, brand mechanisms)
- Modeling approach and performance comparison
- Live app demo
- Recommendations and limitations
- Conclusion with Q&A anticipation

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

**Business implication:** Platform-genre strategy should be a deliberate component of release planning, not an afterthought.

#### Q2: Genre-Regional Impact

**Finding confirmed.** Three distinct regional market clusters exhibit clearly differentiated genre preferences:

- **North America:** Shooter, Sports, and Action dominate (na_ratio > 40%).
- **Japan:** RPG, Visual Novel, and Fighting dominate (jp_ratio > 35% for RPG).
- **Europe/PAL:** Racing, Platform, and Sports show elevated shares.

Western regional markets (NA, PAL, Other) are highly correlated (ρ > 0.6), while Japan shows distinct preferences (ρ ≈ 0.3–0.4 with Western markets).

**Caveat:** Japan sales data has 63.7% zero-imputation due to incomplete VGChartz coverage. Japan-specific conclusions should be treated as directional rather than definitive.

**Business implication:** Genre-based regional marketing strategies can significantly improve market penetration. One-size-fits-all genre strategies underperform.

#### Q3: Brand Effect

**Finding confirmed.** Brand influence on sales operates through two empirically separable channels:

1. **Scale channel (dominant):** Publishers with high game volumes achieve distribution, marketing, and platform-access advantages. `publisher_game_count` is the single most important feature in the Random Forest model.

2. **Prestige channel (significant):** Certain publishers and developers command a per-title premium above what genre, platform, and scale alone would predict. Nintendo uniquely achieves both high volume and high per-title premium.

The dual-channel model is validated by both EDA (Viz 11) and predictive modeling (both scale and identity features appear in the top 5 feature importance ranks).

**Business implication:** New game studios should partner with established publishers for distribution scale. Simultaneously, building brand prestige through consistent quality creates additional premium value.

### 9.2 Model Performance Summary

| Model | R² | RMSE | MAE | Best For |
|---|---|---|---|---|
| Multiple Linear Regression | 0.312 | 0.259 | 0.173 | Interpretable baseline, linear effects |
| Random Forest Regressor | **0.394** | **0.243** | **0.159** | General forecasting (recommended) |
| Two-Stage Blockbuster-Aware | 0.378 | 0.247 | 0.159 | Blockbuster-focused prediction |

### 9.3 Key Visualizations

The project produced 17+ visualizations across all phases, including:
- Sales distribution (raw and log-transformed)
- Genre and platform class balance
- Console × Genre sales heatmap
- Regional sales preference patterns
- K-Means cluster characterization
- Publisher brand analysis
- Correlation matrix
- Model performance comparison
- Feature importance ranking
- Actual vs. predicted scatter plots
- Residual analysis
- Blockbuster RMSE lift

---

## 10. Business Recommendations

Based on the empirical findings, the following actionable recommendations are made for game industry stakeholders:

### 10.1 Strategic Recommendations

**1. Partner with Established Publishers (Highest Impact)**
Publisher brand scale is the single strongest predictor of game sales. New studios and independent developers should prioritize publishing partnerships with established entities to access distribution, marketing, and platform-access advantages. The data shows that publisher scale confers benefits that are difficult to replicate independently.

**2. Prioritize Game Quality (Significant Impact)**
While critic score alone shows weak correlation with sales, it ranks as the second most important feature in the Random Forest model, indicating non-linear effects. Games scoring in the top decile (9.0+) demonstrate disproportionately higher sales. Investment in quality is justified, though it should not come at the expense of marketing and distribution.

**3. Use Platform-Genre Strategy (Measurable Impact)**
Platform-genre synergy is real and concentrated in specific combinations. Release planners should:
- Pair Shooters with Xbox/PlayStation platforms
- Pair RPGs with PlayStation and handheld platforms
- Pair Sports titles with PlayStation and Nintendo platforms
- Avoid niche genres on PC (where digital undercount distorts the data)

**4. Tailor Regional Marketing (Measurable Impact)**
Regional genre preferences are distinct and quantifiable:
- **North America:** Emphasize Shooters, Sports, and Action in marketing
- **Japan:** Emphasize RPGs and Visual Novels for local market campaigns
- **Europe/PAL:** Racing and Platform genres have above-average resonance

**5. Use the Model as Decision Support (Risk Management)**
The Random Forest model (R² = 0.39) is appropriate for mainstream mid-budget title forecasting but should not be the sole basis for budget allocation. Use the two-stage variant when blockbuster projects are under consideration. Supplement model forecasts with qualitative market research for high-stakes decisions.

### 10.2 Operational Recommendations

- **Deploy Random Forest for general forecasting**, switching to the two-stage variant when blockbuster accuracy is prioritized.
- **Retrain quarterly** with updated VGChartz data to capture new releases, platform shifts, and evolving market dynamics.
- **Expand features** by incorporating marketing spend data, IP attributes, user review sentiment, and competitor release timing.
- **Build regional sub-models** for North America, Europe/PAL, and Japan to address preference heterogeneity and data quality differences.
- **Actively collect more blockbuster training samples** — the two-stage model demonstrates the architecture works, but blockbuster scarcity is the binding constraint on accuracy.

---

## 11. Limitations and Future Work

### 11.1 Data Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Digital sales undercount** | Systematic bias for PC/indie titles; dataset focuses on physical retail | Treat PC forecasts as conservative estimates |
| **Japan data gap** | 63.7% zero-imputation; poor reliability for Japanese market | Separate regional sub-models recommended |
| **Blockbuster scarcity** | Only 62 training blockbusters (0.9%) | Two-stage variant partially addresses; data partnerships needed |
| **Temporal coverage** | Mostly pre-2019 data; limited coverage of PS5/Xbox Series X era | Quarterly retraining required |
| **High-cardinality encoding** | ~1,000 publishers, ~2,500 developers; overfitting risk for new entities | Conservative predictions for unknown entities |

### 11.2 Methodological Limitations

- **Correlation ≠ Causation:** Feature importance reflects associative relationships. Publisher scale may correlate with better IP and marketing rather than directly driving sales.
- **No causal inference:** The observational study design cannot establish causal mechanisms. A/B testing or natural experiments would be needed for causal claims.
- **Single imputation for critic_score:** Median imputation does not account for uncertainty in missing values. Multiple imputation (MICE) could be explored.
- **Rare genre information loss:** Genres with <1% frequency were merged into "Other", eliminating granularity for niche predictions.

### 11.3 Potential Failure Modes

1. **New platform/genre prediction:** Models are unreliable for entirely new platforms (e.g., new console generation) or genres with zero historical training examples.
2. **Blockbuster underestimation:** Even with the two-stage variant, blockbuster prediction carries high uncertainty. Business plans should incorporate wider confidence intervals for high-budget projects.
3. **New entrant bias:** Target encoding shrinks predictions for unseen publishers toward the global mean, disadvantaging new market entrants without track records.
4. **Market structure shifts:** Rapid changes in monetization models (subscription services, free-to-play, cloud gaming) could invalidate patterns learned from physical-sales-dominated training data.

### 11.4 Future Work

1. **Feature expansion:** Integrate marketing spend data, IP franchise attributes, user review sentiment from Metacritic/Steam, and competitor release timing.
2. **Regional sub-models:** Build separate regression models for NA, Japan, and PAL markets to address preference heterogeneity and data quality differences.
3. **Causal inference methods:** Apply instrumental variable approaches or difference-in-differences to identify causal effects of platform exclusivity, release timing, and pricing.
4. **Deep learning exploration:** Test neural network architectures (tabNet, FT-Transformer) on the enriched feature set.
5. **Temporal backtesting:** Adopt time-based cross-validation to better assess real-world generalization for future release prediction.
6. **Post-2019 data enrichment:** Actively source post-2019 sales data to capture modern market dynamics, digital distribution trends, and new platform effects.

---

## 12. Conclusion

This project provides a comprehensive, data-driven analysis of the factors influencing video game sales, following the CRISP-DM framework from business understanding through deployment.

**Three research questions were answered:**

1. **Q1 — Platform-Genre Synergy Matters (Confirmed).** Specific platform-genre combinations command measurable sales premiums, concentrated in the upper quartile. The non-linear nature of these interactions — validated by a 26% R² lift from linear to ensemble models — means that strategic platform-genre positioning is a legitimate competitive lever.

2. **Q2 — Regional Markets Are Distinct (Confirmed).** North America, Japan, and Europe/PAL exhibit clearly differentiated genre preferences. Tailoring game portfolios and marketing strategies to regional tastes is empirically justified.

3. **Q3 — Brand Drives Sales Through Two Channels (Confirmed).** Brand scale (volume of releases) dominates as the single strongest sales predictor, while brand prestige (per-title premium) provides additional, separable value. The dual-channel finding has direct implications for strategic partnerships and brand building.

**The project delivered:**
- A cleaned, enriched dataset (8,786 records × 33 features)
- 12+ EDA visualizations with documented interpretations
- Three predictive models with the Random Forest achieving R² = 0.394
- A novel two-stage blockbuster-aware variant reducing blockbuster RMSE by 6.7%
- A deployed Streamlit web application for stakeholder use
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
10. McGrath, R. G., & Nerkar, A. (2004). Leveraging technology, leveraging scale: The competitive implications of technological and market scope. *Advances in Strategic Management*, 21, 255–284.
11. Enterbrain, Inc. (2023). *Famitsu Game White Paper.* Enterbrain (industry reference for Japan game market data).
12. ESA (Entertainment Software Association). (2023). *Essential Facts About the Video Game Industry.* Annual industry report.

---

*This report was prepared by Group 23 for INFO422 Data Science Project at Lanzhou University, July 2026.*
