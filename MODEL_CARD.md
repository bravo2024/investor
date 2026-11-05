# Model card: Investor Propensity & Targeting Model

Companion to [README.md](README.md). Every figure below is read from the frozen files in
`artifacts/`; the README carries the full tables with confidence intervals.

## What the model does

| | |
|---|---|
| Task | Rank existing investment customers by the chance that they place at least one BUY order in the six months after a cut-off date |
| Output | A calibrated probability, a five-level propensity band, and a ranked contact list |
| Inputs | 37 behavioural and static features built only from records dated before the cut-off (33 for logistic regression after VIF pruning) |
| Population | 29,090 customers of one European financial institution, 2018-01 to 2022-11 |
| Headline model | LightGBM, selected on the validation block; ensembles tie it on ROC-AUC and capture |
| Training environment | Python 3.13.18, scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0 (`artifacts/run_summary.json`) |

## Intended use

* Choosing which existing customers a relationship manager or campaign tool should contact
  when only part of the base can be reached.
* Sizing a contact budget: the app reports the share of eventual buyers reached at any
  budget between 1% and 60%.
* Explaining a score to a customer-facing team (SHAP contributions per customer, per feature).

Not intended for: new-to-bank acquisition, advice on which asset to buy (the appendix in the
README is a separate experiment), credit decisions, or any use where a missed customer is a
regulatory event rather than a missed sale.

## Data and provenance

FAR-Trans (Sanz-Cruzado, Droukas, McCreadie; IJCAI-2024 FinRec workshop; arXiv 2407.08692),
published by the University of Glasgow under CC BY 4.0. The raw files are not committed; the
notebook downloads the zip and checks its SHA-256 (`artifacts/data_provenance.json`).

Two customer attributes are algorithm-predicted in the source data, not observed:
`riskLevel` for 7,141 customers and `investmentCapacity` for 7,318. Both are kept as flagged
features. An ablation that removes all 11 static customer attributes is reported in the README,
and the behaviour-only reading is the safer one.

## Evaluation

Strictly out of time, with a six-month purge gap between blocks: train 2019-10 to 2020-10,
validate 2021-04 to 2021-06, test 2021-12 to 2022-06 (seven monthly cut-offs, 197,513
customer-cut-off rows, 28,568 customers, 13.6% buy within six months).

| Model | Test ROC-AUC | Test PR-AUC | Buyers reached @20% |
|---|---|---|---|
| Logistic regression | 0.906 | 0.671 | 72.7% |
| LightGBM (headline) | 0.915 | 0.696 | 74.3% |
| XGBoost | 0.916 | 0.700 | 74.2% |
| Ensemble E3 (LR + RF + GB + XGB) | 0.915 | 0.704 | 74.3% |

Intervals come from a paired bootstrap over 1,000 customer resamples. The tree models beat
logistic regression by about +0.009 ROC-AUC with an interval that excludes zero; the ensembles
are statistically tied with the best single model on ROC-AUC and capture, and differ reliably
only on PR-AUC.

Threshold metrics, calibration, rolling-origin stability, segment level results and the
recommendation appendix are in README.md and in `artifacts/`.

## Limitations

* One institution's investment book; results are a benchmark, not a deployment claim. The test
  period has a lower uptake rate than training, and calibrators fitted on validation do not fully
  follow that drift.
* Customer attributes are a final snapshot (one value per customer, partly simulated);
behaviour-only models lose little but are not identical.
* The 2018-01-02 opening book counts as holdings, not trades, so tenure is left-censored.
* A customer appears at several cut-offs, which the bootstrap accounts for.
* Propensity is a ranking device: the top 20% list is 50.4% buyers, not 100%.

## Ethical and operational notes

* No protected attributes are used. Segment, risk profile and capacity describe the banking
  relationship, not the person.
* Contacting customers is a marketing act: scores should drive prioritisation, not eligibility,
  and a holdout group should be kept so the uplift of the list can be measured.
* The model is refit on a schedule that respects the six-month label window; scores go stale
  within a quarter because recent activity dominates the ranking.

## Licence

Code: MIT (see [LICENSE](LICENSE)). Dataset: CC BY 4.0, University of Glasgow.
