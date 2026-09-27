# Early Warning Machine Learning Model for High-Risk Opioid Prescribing

**A tool that predicts which prescribers are likely to start prescribing far more opioids than others in their field
within the next two years, built from 7.8 million real Medicare records.**

> **In short:** a health plan or state program can only review a small number of prescribers each year, so it needs to
> know where to look first. When this tool picks 1,000 prescribers to review, about **38%** of them really do become
> high prescribers. A simple rule of thumb gets **21%**, and picking at random gets **less than 1%**. It was tested on
> newer data it had never seen, and every prediction comes with a plain explanation of why the prescriber was flagged.
> A flag is a reason for a supportive review, not proof that anyone did anything wrong.

### ▶ [Try the live app: opioid-early-warning.streamlit.app](https://opioid-early-warning.streamlit.app)

[![Live app](Image/app_screenshot.png)](https://opioid-early-warning.streamlit.app)

---

*The sections below go into technical detail.*

## The problem

In [my previous project](https://github.com/Isaac-Agyapong/Medicare_Opioid_Prescribing) about 1% of Medicare
prescribers were extreme opioid outliers compared with their own specialty, and **78% of them stay outliers year
after year**. Flagging people who are already outliers is easy. The useful question is harder:

> **Among prescribers who are *not* outliers today, who will become one within two years?**

Only **0.45%** of them do (about 2,000 out of 460,000 a year), so this is a rare-event early-warning problem. A payer
or state monitoring program can only review a short list, so what matters is **how many of the flagged prescribers
really go on to become outliers**.

## Results (test year never used for training, tuning or calibration)

| Test year: 2022 snapshot → outcomes 2023-24 | Simple rule | Logistic regression | Random forest | **XGBoost** |
|---|---|---|---|---|
| **Precision of a 1,000-prescriber review list** | 21.1% | 33.5% | 36.4% | **37.7%** (95% CI 34.3-40.4%) |
| Lift over random selection | 47x | 75x | 82x | **84x** |
| Future outliers caught in the top 1% of scores | 43.5% | 39.4% | 50.0% | **52.1%** (95% CI 50.0-54.3%) |
| PR-AUC (right metric for rare events) | 0.149 | 0.159 | 0.219 | **0.233** (95% CI 0.215-0.253) |
| ROC-AUC | 0.921 | 0.941 | 0.950 | 0.949 |

- **The model beats the best simple rule by +16.6 points of precision (95% CI +12.8 to +19.7)**: 377 of its top
  1,000 flags became outliers, vs 211 for the rule. Random selection would find 4-5.
- **The 2021 backtest, also untouched, shows the same result**: 38.4% precision for the model vs 21.8% for the rule.
- **Probabilities are close to calibrated** (isotonic, fitted on the 2020 tune split): the model predicts 0.41% on
  average vs 0.45% observed on the test year (Brier 0.0038), slightly conservative.
- Confidence intervals come from 1,000 bootstrap resamples of the test year.

| | |
|---|---|
| ![](Image/01_model_comparison.png) | ![](Image/02_capture_curve.png) |
| ![](Image/04_feature_importance.png) | ![](Image/03_calibration.png) |

## What drives a flag (SHAP)

![SHAP summary](Image/05_shap_summary.png)

- **Where a prescriber already sits within their specialty** (percentile, and how close they are to the 99th
  percentile) is the strongest signal, followed by **prescribing volume**, **patient risk scores** and **specialty**.
- **A sudden one-year jump in percentile does not always raise risk**: the model learned that some spikes fall back
  the next year (regression to the mean), which a simple threshold rule cannot do.
- Nurse practitioners and PAs make up 57% of the top-1,000 review list, matching the shift found in the previous
  project.

![Review list by specialty](Image/06_review_list_by_specialty.png)

---

## How it's built

```
opioid_analytics (PostgreSQL, from the previous project)
   └── SQL/01_features.sql ──> ml.prescriber_snapshot   1.3M prescriber-years: features at year T, label = outlier in T+1 or T+2
          └── Python/02_train_models.py ──> rule / logistic regression / random forest / XGBoost, isotonic calibration,
                                            bootstrap confidence intervals
                 └── Python/03_model_results.ipynb ──> evaluation charts + SHAP
                        └── app/app.py (Streamlit) ──> interactive risk explorer with per-profile SHAP explanation
```

### Design decisions that make the evaluation honest

| Decision | Why |
|---|---|
| **No look-ahead**: fit on 80% of the 2020 snapshot, tune on the other 20%, backtest on 2021, test once on 2022 | Every decision (model choice, early stopping, calibration) uses outcomes up to 2022 only, exactly what would be known when scoring the 2022 snapshot. A first version tuned on 2021, whose outcome window overlapped the test window in 2023; fixing that lowered test precision from 39.9% to 37.7%, which is the number reported. |
| **Current outliers excluded** from the population | Otherwise the model would get credit for "predicting" people who are already outliers (78% stay flagged). |
| **Only information available at the end of year T** in the features | Year T and the change from T-1; nothing from the outcome years. |
| **Compared against a simple rule** | An ML model must beat the obvious heuristic to be worth deploying. |
| **Precision on a fixed review list and PR-AUC** as the main metrics | With a 0.45% base rate, accuracy and ROC-AUC look excellent for almost any model. |
| **Calibration fitted on the tune split** | Scores can be read as probabilities and used to set review thresholds. |
| **Bootstrap confidence intervals** | A single number hides uncertainty; the gain over the rule stays positive across all 1,000 resamples. |

### Features (19 numeric + specialty group + region)
Opioid share of prescriptions, percentile within specialty, ratio to the specialty median and 99th percentile,
volume (total and opioid claims, patients), long-acting opioid share, days supplied per claim, share of patients on
opioids, patient age and HCC risk score, rural practice, specialty size, and one-year changes (rate, percentile,
opioid claims growth, outlier last year).

---

## Interactive app

**Live:** https://opioid-early-warning.streamlit.app · locally: `streamlit run app/app.py` opens **Prescribing Insights**, a four-page app (Overview, Prescriber assessment,
Model evidence, Data & methods). On the assessment page, describe a prescriber profile (specialty, region, opioid
rate now and a year ago, volume, patient mix) and see the **calibrated probability**, a **review priority**
(low / elevated / high, relative to the 0.45% average) and a **SHAP chart explaining that specific score**.
The app runs entirely from the committed model files, so it deploys to Streamlit Community Cloud without a database.

![Model evidence page](Image/app_evidence.png)
A starting profile can be passed in the address, e.g. `?group=Primary%20Care&rate=18&prev=12`.

The app works on **profiles, not real prescribers**: no individual NPI is shown or stored in this repository.

---

## Model card

| | |
|---|---|
| **Intended use** | Prioritising prescribers for *supportive* review (education, prescription-monitoring checks) by a payer or public-health program |
| **Not intended for** | Penalties, network exclusion or any decision without human review. A flag is statistical, not evidence of inappropriate care |
| **Training data** | CMS Medicare Part D Prescribers by Provider, 2019-2022 features; outcomes 2021-2024 |
| **Population** | Individual prescribers with 100+ Part D claims and unsuppressed opioid counts, not currently an outlier |
| **Label** | Becomes a peer outlier (at or above the specialty's 99th percentile and 3x its median, 50+ opioid claims) in either of the next two years |
| **Performance (test)** | 37.7% precision in the top 1,000 (95% CI 34.3-40.4%); 52.1% recall in the top 1%; PR-AUC 0.233; Brier 0.0038 |
| **Known limitations** | Medicare Part D only (mostly 65+); prescribers must stay in Medicare for both follow-up years, so people who leave are not scored; the peer definition depends on CMS specialty labels; patterns change over time, so the model should be re-validated every year |
| **Fairness check to add** | Precision by specialty is shown above; a production version should also monitor rural/urban and regional error rates |

---

## Project structure

```
Opioid_Prescriber_Risk_Model/
├── SQL/01_features.sql            feature + label table in PostgreSQL (schema ml)
├── Python/
│   ├── 02_train_models.py         training, time-based evaluation, calibration
│   ├── 03_model_results.ipynb     evaluation charts and SHAP explanations
│   ├── 04_build_app_reference.py  group-level reference values for the app
│   └── viz_style.py
├── app/                           Streamlit app (app.py, assets/, requirements.txt for deployment)
├── models/                        XGBoost model, calibrator, metrics.json, app reference
├── Image/                         charts and app screenshot
└── run_all.py
```

## Reproduce

1. Build the `opioid_analytics` PostgreSQL database with the
   [Medicare_Opioid_Prescribing](https://github.com/Isaac-Agyapong/Medicare_Opioid_Prescribing) project.
2. `pip install -r requirements.txt`
3. `python run_all.py` (about 10 minutes)
4. `streamlit run app/app.py`

Prescriber-level feature files contain NPIs and are rebuilt locally; they are not committed.
