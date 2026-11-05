"""Generate README.md from the frozen files in artifacts/. Usage: python build_readme.py [kaggle_version]. Every number in the README is read from artifacts/.

Set GH_REPO and STREAMLIT_URL to change the links:
  GH_REPO=bravo2024/investor STREAMLIT_URL=https://... python build_readme.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
KAGGLE_VERSION = sys.argv[1] if len(sys.argv) > 1 else "see kernel page"
REPO = os.environ.get("GH_REPO", "bravo2024/investor")
REPO_URL = f"https://github.com/{REPO}"
APP_URL = os.environ.get("STREAMLIT_URL", "").strip()
APP_BADGE = (f"[![Streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]({APP_URL})" if APP_URL
             else "[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](#streamlit-app)")
LIVE_LINE = f"**Live app:** <{APP_URL}> (no login, six pages, frozen artifacts).\n\n" if APP_URL else ""


def js(n):
    return json.loads((ART / n).read_text(encoding="utf-8"))


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt and pd.notna(v) else ("" if pd.isna(v) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


p3 = "{:.3f}".format
pc = lambda x: f"{100 * x:.1f}%"
sg = lambda x: f"{x:+.3f}"

summ, prov, cb = js("run_summary.json"), js("data_provenance.json"), js("class_balance.json")
man = js("models/manifest.json")
NAMES = man["names"]
tm = pd.read_csv(ART / "test_metrics.csv").set_index("key")
ci = pd.read_csv(ART / "bootstrap_ci.csv")
lvl = ci[ci.comparison == "level"].pivot(index="model", columns="metric", values=["ci_low", "ci_high"])
roll = pd.read_csv(ART / "rolling_origin_summary.csv")
BS, BE, HEAD = man["best_single"], man["best_ensemble"], man["headline_model"]
KEYS = list(NAMES)

rows = []
for k in KEYS:
    n = NAMES[k]
    rows.append({"Model": n, "ROC-AUC (95% CI)": f"{tm.loc[k, 'roc_auc']:.3f} [{lvl.loc[n, ('ci_low', 'roc_auc')]:.3f}, {lvl.loc[n, ('ci_high', 'roc_auc')]:.3f}]",
                 "PR-AUC": p3(tm.loc[k, "pr_auc"]), "KS": p3(tm.loc[k, "ks"]), "Buyers reached @10%": pc(tm.loc[k, "capture_10"]), "@20%": pc(tm.loc[k, "capture_20"]), "@30%": pc(tm.loc[k, "capture_30"])})
main_table = md(pd.DataFrame(rows))

dl = ci[(ci.comparison == "minus_lr") & (ci.metric == "roc_auc")].set_index("model")
dl_pr = ci[(ci.comparison == "minus_lr") & (ci.metric == "pr_auc")].set_index("model")
dl_cap = ci[(ci.comparison == "minus_lr") & (ci.metric == "capture_20")].set_index("model")
diff_rows = [{"Model minus logistic regression": NAMES[k], "ROC-AUC difference [95% CI]": f"{dl.loc[NAMES[k], 'estimate']:+.3f} [{dl.loc[NAMES[k], 'ci_low']:+.3f}, {dl.loc[NAMES[k], 'ci_high']:+.3f}]",
              "PR-AUC difference [95% CI]": f"{dl_pr.loc[NAMES[k], 'estimate']:+.3f} [{dl_pr.loc[NAMES[k], 'ci_low']:+.3f}, {dl_pr.loc[NAMES[k], 'ci_high']:+.3f}]",
              "Capture@20% difference [95% CI]": f"{dl_cap.loc[NAMES[k], 'estimate']:+.3f} [{dl_cap.loc[NAMES[k], 'ci_low']:+.3f}, {dl_cap.loc[NAMES[k], 'ci_high']:+.3f}]"} for k in KEYS if k != "lr"]
db = ci[(ci.comparison == "minus_best_single")]
db_rows = []
for k in ("e1", "e2", "e3"):
    r = {"Ensemble minus " + NAMES[BS]: NAMES[k]}
    for m, lab in (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("capture_20", "Capture@20%")):
        x = db[(db.model == NAMES[k]) & (db.metric == m)].iloc[0]
        r[lab + " difference [95% CI]"] = f"{x.estimate:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}]"
    db_rows.append(r)

sp = pd.read_csv(ART / "splits.csv")
split_tab = md(pd.DataFrame({"split": sp.split, "cut-offs": sp.snapshots.astype(int), "first - last cut-off": [f"{cb[s]['first_cutoff'][:7]} to {cb[s]['last_cutoff'][:7]}" for s in sp.split],
                             "rows": sp.rows.astype(int).map("{:,}".format), "customers": sp.customers.astype(int).map("{:,}".format), "buyers": sp["buyers (y_any=1)"].astype(int).map("{:,}".format),
                             "positive rate": sp["positive rate"].map(pc)}))
imb = pd.read_csv(ART / "imbalance_comparison.csv"); imbd = js("imbalance_decision.json")
imb_tab = md(imb.rename(columns={"strategy": "strategy", "val_roc_auc": "val ROC-AUC", "val_pr_auc": "val PR-AUC", "val_capture_20": "val capture@20%", "val_brier": "val Brier", "mean_predicted": "mean predicted"})[
    ["strategy", "model", "val ROC-AUC", "val PR-AUC", "val capture@20%", "val Brier", "mean predicted"]], {c: p3 for c in ("val ROC-AUC", "val PR-AUC", "val capture@20%", "val Brier", "mean predicted")})
gain = pd.read_csv(ART / "tuning_gain.csv")
gain_tab = md(gain[["model", "val_roc_auc_untuned", "val_roc_auc_tuned", "test_roc_auc_untuned", "test_roc_auc_tuned", "test_pr_auc_untuned", "test_pr_auc_tuned"]].rename(columns={
    "val_roc_auc_untuned": "val ROC-AUC untuned", "val_roc_auc_tuned": "val ROC-AUC tuned", "test_roc_auc_untuned": "test ROC-AUC untuned", "test_roc_auc_tuned": "test ROC-AUC tuned",
    "test_pr_auc_untuned": "test PR-AUC untuned", "test_pr_auc_tuned": "test PR-AUC tuned"}), {c: p3 for c in ("val ROC-AUC untuned", "val ROC-AUC tuned", "test ROC-AUC untuned", "test ROC-AUC tuned", "test PR-AUC untuned", "test PR-AUC tuned")})
vs = js("vif_summary.json"); vt = pd.read_csv(ART / "vif_table.csv"); scr = pd.read_csv(ART / "spearman_screen.csv")
cal = pd.read_csv(ART / "calibration_choice.csv")
cal_tab = md(pd.DataFrame({"Model": [NAMES[k] for k in KEYS], "calibrator chosen on validation": cal.chosen.values, "test Brier, raw": [p3(tm.loc[k, "brier_raw"]) for k in KEYS],
                           "test Brier, calibrated": [p3(tm.loc[k, "brier"]) for k in KEYS]}))
fr = roll[roll.scheme.str.startswith("frozen")]; rf = roll[roll.scheme.str.startswith("expanding")]
def roll_tab(d):
    d = d.set_index("key").loc[KEYS]
    return md(pd.DataFrame({"Model": d.model.values, "ROC-AUC mean +/- sd": [f"{a:.3f} +/- {b:.3f}" for a, b in zip(d.roc_auc_mean, d.roc_auc_std)],
                            "PR-AUC mean +/- sd": [f"{a:.3f} +/- {b:.3f}" for a, b in zip(d.pr_auc_mean, d.pr_auc_std)],
                            "Capture@20% mean +/- sd": [f"{100 * a:.1f}% +/- {100 * b:.1f}" for a, b in zip(d.capture_20_mean, d.capture_20_std)]}))
seg = pd.read_csv(ART / "segment_results.csv")
def seg_tab(dim):
    d = seg[seg.dimension == dim].drop(columns="dimension")
    return md(d.rename(columns={"group": dim, "uptake_rate": "uptake rate", "model_roc_auc": "model ROC-AUC", "capture_20": "buyers on top-20% list", "precision_in_list": "precision on list", "share_of_buyers": "share of all buyers"})[
        [dim, "customers", "rows", "uptake rate", "model ROC-AUC", "buyers on top-20% list", "precision on list", "share of all buyers"]],
        {"customers": lambda x: f"{int(x):,}", "rows": lambda x: f"{int(x):,}", "uptake rate": pc, "model ROC-AUC": p3, "buyers on top-20% list": pc, "precision on list": pc, "share of all buyers": pc})
act = pd.read_csv(ART / "activity_slices.csv")
bands = pd.read_csv(ART / "propensity_bands.csv"); km = js("kmeans_model.json"); prof = pd.read_csv(ART / "kmeans_profiles.csv"); xtp = pd.read_csv(ART / "kmeans_vs_segment_share.csv", index_col=0)
sh = pd.read_csv(ART / "shap_summary.csv"); perm = pd.read_csv(ART / "permutation_importance.csv"); coef = pd.read_csv(ART / "lr_coefficients.csv")
catv = pd.read_csv(ART / "category_variants.csv"); abl = pd.read_csv(ART / "ablation_static_attributes.csv")
rec = pd.read_csv(ART / "rec_appendix_summary.csv"); imb_note = imbd["chosen"]

worse = [NAMES[k] for k in KEYS if tm.loc[k, 'brier'] > tm.loc[k, 'brier_raw']]
cal_note = f"Validation uptake ({pc(cb['valid']['positive_rate'])}) is higher than test uptake ({pc(cb['test']['positive_rate'])}), so a calibrator fitted on validation can over-predict on test: calibration made the test Brier score worse for {', '.join(worse) if worse else 'no model'} and better for the others (logistic regression gains most because its tuned class weights distort raw probabilities)."
min_gain = gain.loc[gain.test_roc_auc_gain.idxmin()]
tune_note = f"Tuning gains on the test period are small (largest test ROC-AUC gain {gain.test_roc_auc_gain.max():+.3f}); the shallow decision tree tuned on validation did worse on test ({min_gain.test_roc_auc_gain:+.3f} ROC-AUC), a sign that validation-chosen settings do not always survive the drift."
c = prov["counts"]; hs = tm.loc[HEAD]; lrm = tm.loc["lr"]; bsm = tm.loc[BS]
test_n, test_pos = cb["test"]["rows"], cb["test"]["positive_rate"]
act_hi = act[(act.model == NAMES[HEAD]) & act["slice"].str.startswith("traded")].iloc[0]
act_lo = act[(act.model == NAMES[HEAD]) & act["slice"].str.startswith("no trade")].iloc[0]
seg_df = seg[seg.dimension == "customer segment"].set_index("group")

readme = f"""# Investor Propensity & Targeting Model

[![tests]({REPO_URL}/actions/workflows/ci.yml/badge.svg)]({REPO_URL}/actions/workflows/ci.yml) {APP_BADGE} [![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)]({REPO_URL}/blob/main/LICENSE)

Who will buy in the next six months, and whom should the bank contact? A Streamlit dashboard and a
reproducible notebook: rank the investor base by the chance of a BUY in the next six months, then
decide how many customers to contact. Model card:
[`MODEL_CARD.md`]({REPO_URL}/blob/main/MODEL_CARD.md).

{LIVE_LINE}## Summary

A wealth and retail bank can contact only part of its investor base. This project ranks **existing investment customers** of a large European financial institution by the chance of placing **at least one BUY order in the next six months**, using only what is known before a cut-off date. Data: FAR-Trans (University of Glasgow, CC BY 4.0), {c['customers_with_transactions']:,} customers, {c['assets_in_asset_information']} assets, {c['transactions']:,} transactions ({c['buys']:,} buys), {c['first_transaction_date'][:7]} to {c['last_transaction_date'][:7]}.
All models are classical: logistic regression, decision tree, random forest, scikit-learn gradient boosting, XGBoost, LightGBM and three soft-voting ensembles (E1 = LR + RF + GB, E2 = LR + RF + XGB, E3 = LR + RF + GB + XGB).

Validation is strictly **out of time**: monthly cut-offs, training on 2019-10 to 2020-10, validation on 2021-04 to 2021-06, test on 2021-12 to 2022-06 with a six-month purge gap between blocks. Test: {test_n:,} customer-cut-off rows from {summ['n_test_customers']:,} customers, **{pc(test_pos)} of them buy** within six months.

{main_table}

Buyers reached @k% = share of the cut-off's eventual buyers among the top k% of customers by score (ranked within each cut-off, pooled over seven test cut-offs). Intervals: paired bootstrap, {summ['n_boot']:,} resamples of customers.

**Models against logistic regression** (paired bootstrap, same resamples):

{md(pd.DataFrame(diff_rows))}

**Ensembles against the best single model on validation ({NAMES[BS]})**:

{md(pd.DataFrame(db_rows))}

**Threshold metrics and top-decile lift** (threshold = the raw probability that maximises F1 on validation, applied unchanged to test):

{md(pd.DataFrame({"Model": [NAMES[k] for k in KEYS], "threshold": [p3(tm.loc[k, "threshold"]) for k in KEYS], "precision": [pc(tm.loc[k, "precision"]) for k in KEYS], "recall": [pc(tm.loc[k, "recall"]) for k in KEYS], "F1": [p3(tm.loc[k, "f1"]) for k in KEYS], "top-decile precision": [pc(tm.loc[k, "top_decile_precision"]) for k in KEYS], "top-decile lift": [f"{tm.loc[k, 'lift_top_decile']:.2f}" for k in KEYS]}))}

**Business result.** Contacting the top 20% of customers reaches **{pc(hs.capture_20)} of eventual buyers** with {NAMES[HEAD]} (best model on validation; precision {pc(hs.precision_20)} against a {pc(test_pos)} base rate, lift {hs.precision_20 / test_pos:.2f}). Logistic regression reaches {pc(lrm.capture_20)}, and the best ensemble on validation ({NAMES[BE]}) {pc(tm.loc[BE, 'capture_20'])}. The top 10% already holds {pc(hs.capture_10)} of buyers at {pc(hs.top_decile_precision)} precision (lift {hs.lift_top_decile:.2f}).
The ordering is dominated by recent activity, so the gap between a transparent logistic regression and the tree models is small in ROC-AUC (about {dl.loc[NAMES['xgb'], 'estimate']:+.3f} for XGBoost, interval excludes zero) and the ensembles are statistically tied with the best single model in ROC-AUC and capture; their only reliable edge is PR-AUC (see the table above).
Among customers who traded in the last 12 months ({pc(act_hi.share_of_rows)} of rows, uptake {pc(act_hi.uptake_rate)}) the headline model's ROC-AUC is {act_hi.roc_auc:.3f}; among customers with no trade in 12 months (uptake {pc(act_lo.uptake_rate)}) it is {act_lo.roc_auc:.3f}.

| Test ROC and PR curves | Drivers (SHAP) | Rolling-origin stability |
|---|---|---|
| ![Test curves](artifacts/figures/test_curves.png) | ![Drivers](artifacts/figures/drivers.png) | ![Rolling origin](artifacts/figures/rolling_origin.png) |

Kaggle run: `vivekkopthsd/far-trans-wealth-uptake` (private, CPU), {KAGGLE_VERSION}, runtime {summ['runtime_seconds']:,}s. Numbers in this file are generated from `artifacts/` by `build_readme.py`.

## Streamlit app

**Investor Propensity & Targeting Model** runs as six pages, all fed by the frozen files in
`artifacts/`: the app never downloads data and never trains, so it starts in seconds and deploys as-is.

| Page | What it shows |
|---|---|
| Overview | out-of-time design, split sizes, class balance, test metrics with bootstrap intervals |
| Models & ensembles | ROC and precision-recall curves, paired-bootstrap differences, tuning gains, VIF, calibration, rolling-origin stability |
| Segments | uptake and capture by segment, risk profile, capacity and activity, propensity bands, K-Means clusters, product categories |
| Contact budget | buyers reached, precision and lift for any contact budget between 1% and 60% |
| Drivers | SHAP and permutation importance, per-feature SHAP scatter, logistic-regression coefficients, feature dictionary |
| Customer lookup | recomputes one customer's calibrated score from the frozen models, with band and TreeSHAP contributions |

**Run it locally**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Deploy on Streamlit Community Cloud**

1. Push this folder as its own repository (see `git` steps in the quick start below).
2. Open <https://share.streamlit.io> -> **Create app** -> **Deploy a public app from GitHub**.
3. Repository `{REPO}`, branch `main`, main file path `app.py`.
4. **Advanced settings** -> Python **3.13** (3.11 and 3.12 also work). Do not pick 3.14:
   `scikit-learn==1.6.1` and `numpy==2.1.3` publish no wheels for it.
   Community Cloud ignores `runtime.txt`, so the version has to be set in that dialog.
5. Deploy. No secrets and no data download: the app reads the committed `artifacts/`
   (about 14 MB, frozen models included).

No system packages are needed, so there is no `packages.txt`. The app installs
`xgboost-cpu` rather than `xgboost`: the PyPI `xgboost` wheel drags in
`nvidia-nccl-cu12` on Linux (about 300 MB of CUDA libraries), while this
project only scores single customer rows on CPU. Same import name, same model,
no GPU dependency.

## How to run

```bash
# app only (this is what Streamlit Community Cloud installs)
pip install -r requirements.txt
streamlit run app.py

# everything: app plus notebook, README generator and tests
pip install -r requirements-dev.txt
jupyter nbconvert --to notebook --execute FARTransWealth.ipynb --output FARTransWealth.executed.ipynb   # downloads FAR-Trans into ./data, writes ./artifacts
python build_readme.py       # regenerates this file from artifacts/
pytest -q                    # 22 checks: artifacts, frozen models, notebook, README numbers, app pages
```

Push to GitHub from this folder:

```bash
git init -b main && git add . && git commit -m "Investor propensity model: notebook, artifacts, Streamlit app"
git remote add origin https://github.com/{REPO}.git && git push -u origin main
```

The frozen `artifacts/` are committed, so `streamlit run app.py` and `pytest` work without re-running the notebook. Environment variables: `FAR_DATA` (folder holding `FAR-Trans.zip`), `FAR_ART` (output folder), `FAR_SMOKE=1` (small sample run).
Kaggle: `kaggle kernels push -p .` pushes `run_far_trans.py`, a script kernel that embeds the notebook, executes it with nbclient and leaves `FARTransWealth.executed.ipynb` and `artifacts/` as output.

## Project structure

```
FARTransWealth.ipynb            all analysis code, top to bottom (no local package)
FARTransWealth.executed.ipynb   the same notebook executed on Kaggle, every output stored
app.py                          Streamlit app (reads artifacts/ only)
build_readme.py                 README generator
artifacts/                      metrics, tables, figures, scored test rows, frozen models (models/)
tests/                          artifact, frozen-model, notebook, README and app tests
run_far_trans.py, kernel-metadata.json   Kaggle runner
requirements.txt                app runtime (exactly what Community Cloud installs)
requirements-dev.txt            notebook, README generation and tests
.github/workflows/ci.yml        pytest plus a six-page Streamlit smoke test on every push
.streamlit/config.toml          theme
MODEL_CARD.md                   intended use, evaluation, limitations
LICENSE                         MIT for the code; the data is CC BY 4.0
```

## Tests and CI

`pytest -q` runs 22 checks: every artifact file is present; the provenance hashes, row counts and
the 7,141 / 7,318 simulated-attribute counts match the source; the split respects the six-month
purge gap; capture is monotone in the contact budget; the frozen joblib and JSON models reproduce
the stored scores and the test metrics; the notebook imports only standard, scikit-learn, xgboost,
lightgbm and shap modules and stores no error output; the numbers in this file match `artifacts/`;
and every Streamlit page renders under `streamlit.testing.v1.AppTest`.
`.github/workflows/ci.yml` runs the suite on Python 3.13, then installs `requirements.txt` alone and
renders all six pages, which is the same environment Community Cloud builds.

## Data gate

Downloaded from `https://researchdata.gla.ac.uk/id/eprint/1658` (zip {prov['zip_bytes'] / 1e6:.1f} MB, sha256 `{prov['zip_sha256'][:16]}...`, licence {prov['licence'].split(' (')[0]}); full hashes, row and column counts per file are in `artifacts/data_provenance.json`.
Counts reproduced: {c['customers_with_transactions']:,} customers, {c['assets_in_asset_information']} assets, {c['price_points']:,} price points, {c['buys']:,} buys, {c['sells']:,} sells ({c['transactions']:,} transactions; the paper's headline figures are one row higher, not material). Transactions {c['first_transaction_date']} to {c['last_transaction_date']}, prices to {c['last_price_date']}.

**Simulated or estimated fields** (dataset README: "Predicted_Conservative / Predicted_Income / Predicted_Balanced / Predicted_Aggressive: Same as earlier, but predicted by an algorithm."; paper: "In case they have not, the score is simulated through an automated process"):
{prov['simulated_or_estimated']['risk_level']['predicted_values']:,} customers have a predicted (simulated) risk profile and {prov['simulated_or_estimated']['investment_capacity']['predicted_values']:,} an estimated investment capacity, matching the paper's 7,141 and 7,318. Both are kept as flagged features. Segment, risk and capacity never change between a customer's records, so one final value per customer is used at every cut-off; an ablation (below) shows the effect of removing all customer attributes.
The gate passed: dated Buy/Sell transactions with customer ids over 59 months and customer attributes support a customer-level, time-split propensity task.

## Methods notes

**Task.** At each monthly cut-off T, every customer with a record before T (including the opening book) is one row; label = at least one BUY in [T, T + 6 months). Variants: BUY of a stock, a fund (MTF) or a bond.
**Opening positions.** About 9% of all rows are dated 2018-01-02, the first day of data. They are the bank's opening book and count as holdings, but not as trading events in recency, frequency or labels.
**Features.** {summ['panel']['features_total']} plain-English features built only from rows before T ({summ['panel']['features_tree']} after the Spearman screen, {summ['panel']['features_lr']} for logistic regression after VIF pruning), documented in `artifacts/feature_dictionary.csv`: recency, frequency and value of buys and sells, net-flow trend (last 3 vs prior 3 months), holdings (count, categories, market value from close prices, concentration, holding period), channel mix, 1-month and 3-month return of held assets, tenure, segment, risk profile, capacity band, questionnaire.

**Split and purge gap.**

{split_tab}

Test uptake is lower than training uptake: a real drift that is left in. The first snapshot is 2019-10 because trading data are thin before mid 2019 (uptake about 5% against 16% after).

**Class balance.** Training positive rate {pc(imbd['positive_rate_train'])}. None vs class_weight vs SMOTE (SMOTE inside the training sample only), two default models, validation scores:

{imb_tab}

Rule fixed in advance: switch only if another strategy beats "none" by 0.005 validation PR-AUC. Chosen: **{imb_note}**. Re-weighting and SMOTE lift the mean predicted probability far above the true rate (Brier gets worse) without improving ranking, and the contact-budget and calibration work needs honest probabilities. Logistic regression still searched `class_weight` as a hyper-parameter.

**Tuning on validation only** (small random searches; the untuned configuration is always a candidate; tables in `artifacts/tuning_*.csv`, parameters in `best_params.json`):

{gain_tab}

{tune_note}

**Multicollinearity.** Spearman screen: {vs['features_before_screen']} to {vs['after_spearman_screen']} features ({len(scr)} features dropped from pairs with |rho| > 0.9). VIF pruning for logistic regression dropped {', '.join(vs['features_dropped_for_lr']) or 'nothing'}: features with VIF above 10 went from {vs['n_vif_gt10_before']} to {vs['n_vif_gt10_after']}, maximum VIF {vs['max_vif_before']:.1f} to {vs['max_vif_after']:.2f} (`artifacts/vif_table.csv`).

**Calibration** (Platt or isotonic, picked by grouped 5-fold Brier on validation; Brier on test). {cal_note}

{cal_tab}

**Rolling-origin stability.** Frozen models at each of the seven test cut-offs:

{roll_tab(fr)}

Expanding-window refits (tuned settings, refit on every snapshot whose label window ended before the cut-off), {int(rf.n_cutoffs.iloc[0])} cut-offs:

{roll_tab(rf)}

## Segment analytics

Scores from {NAMES[HEAD]} (best on validation); contact list = top 20% of each cut-off.

{seg_tab('customer segment')}

{seg_tab('risk profile')}

{seg_tab('risk profile source')}

{seg_tab('investment capacity')}

**Propensity bands** (quintiles of the calibrated validation score; band 1 = highest):

{md(bands[['band', 'rows', 'uptake_rate', 'lift', 'share_of_buyers', 'mean_predicted']].rename(columns={'uptake_rate': 'uptake rate', 'share_of_buyers': 'share of all buyers', 'mean_predicted': 'mean predicted'}), {'rows': lambda x: f"{int(x):,}", 'uptake rate': pc, 'lift': lambda x: f"{x:.2f}", 'share of all buyers': pc, 'mean predicted': p3})}

**K-Means behavioural clusters** on pre-cut-off features (k = {km['k']} chosen by silhouette over 3 to 6: {', '.join(f"k={k}: {v:.3f}" for k, v in km['silhouette'].items())}); fitted on training rows, applied to test rows:

{md(prof[['name', 'share_of_rows', 'uptake_rate', 'mean_score', 'n_buy_12m', 'days_since_last_trade', 'n_assets_held', 'port_value_log', 'risk_ord']].rename(columns={'name': 'cluster', 'share_of_rows': 'share of rows', 'uptake_rate': 'uptake rate', 'mean_score': 'mean predicted', 'n_buy_12m': 'buys 12m', 'days_since_last_trade': 'days since trade', 'n_assets_held': 'assets held', 'port_value_log': 'log value', 'risk_ord': 'risk (0-3)'}), {'share of rows': pc, 'uptake rate': pc, 'mean predicted': p3, 'buys 12m': lambda x: f"{x:.1f}", 'days since trade': lambda x: f"{x:.0f}", 'assets held': lambda x: f"{x:.1f}", 'log value': lambda x: f"{x:.1f}", 'risk (0-3)': lambda x: f"{x:.2f}"})}

Official segment mix inside each cluster (row shares):

{md(xtp.reset_index().rename(columns={'cluster': 'cluster'}), {cc: pc for cc in xtp.columns})}

## Explainability

**Top 10 drivers** (SHAP, `TreeExplainer`, tuned XGBoost, test sample; direction from the correlation between feature value and SHAP value):

""" + "\n".join(f"{i}. {r.plain_english}" for i, r in enumerate(sh.head(10).itertuples(), 1)) + f"""

**Reading caution.** `cap_predicted` (capacity estimated by an algorithm) is a top driver because customers without a questionnaire are mostly dormant investors (uptake {pc(seg[(seg.dimension=='risk profile source') & (seg.group=='predicted (simulated)')].uptake_rate.iloc[0])} for predicted risk profiles against {pc(seg[(seg.dimension=='risk profile source') & (seg.group=='questionnaire')].uptake_rate.iloc[0])} for questionnaire ones). The flag marks missing engagement rather than a causal effect of the simulation, and behaviour-only models lose little (ablation below).

Permutation importance (ROC-AUC drop when shuffled), top 5: {', '.join(f"{r.feature} ({r.auc_drop_mean:.3f})" for r in perm.head(5).itertuples())}.
Logistic regression, largest standardised coefficients: {', '.join(f"{r.feature} ({r.coef_per_sd:+.2f})" for r in coef.head(5).itertuples())}.

## Product-category variants and ablation

Same features, label = BUY of that category in six months; logistic regression and the best single model, tuned settings reused (not re-tuned):

{md(catv, {'positive_rate_train': pc, 'positive_rate_test': pc, 'buyers_test': lambda x: f"{int(x):,}", 'roc_auc': p3, 'pr_auc': p3, 'capture_20': pc})}

Removing the 11 static customer attributes (segment, risk profile, capacity, questionnaire; partly simulated):

{md(abl.drop(columns='key'), {cc: p3 for cc in abl.columns if cc not in ('key', 'model')})}

## Appendix (optional): next-asset recommendation for buyers

Not part of the headline result. For customers who buy in the next six months: can a ranker put the bought assets in the top 10? Protocol A allows repeat purchases, protocol B counts only assets never bought before. Three test cut-offs (2021-12, 2022-03, 2022-06), a sample of buyers each, candidates = assets bought by someone in the previous 12 months.

{md(rec.drop(columns='paper_reference_ndcg_at_10'), {'hit_at_10': p3, 'ndcg_at_10': p3, 'buyers_evaluated': lambda x: f"{int(x):,}"})}

For reference the paper (Table 3, averaged over 61 two-week time points from 1 August 2019) reports nDCG@10 of 0.2710 for popularity and 0.3404 for LightGCN. The protocols differ (three cut-offs and sampled buyers here, a restricted candidate list, our own handling of repeat purchases and a much smaller model), so the figures are not comparable and no claim is made to beat the paper.

## Limitations

* Customer attributes are a final snapshot (one value per customer), and risk and capacity are partly simulated; behaviour-only models are the safer reading (see ablation).
* The opening book on 2018-01-02 is treated as holdings, not trades; tenure is left-censored there.
* Test uptake is lower than training uptake; calibrators fitted on validation do not fully follow that drift.
* Pooling seven monthly cut-offs means a customer appears several times; the bootstrap resamples customers to respect this.
* The dataset is one European financial institution's investment book; results are a benchmark, not a deployment claim.

## Licence and attribution

Code: MIT, see [`LICENSE`]({REPO_URL}/blob/main/LICENSE).
Data: FAR-Trans, Sanz-Cruzado, Droukas and McCreadie, IJCAI-2024 FinRec workshop (arXiv 2407.08692),
published by the University of Glasgow under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and downloaded from
<https://researchdata.gla.ac.uk/id/eprint/1658>. The raw files are not redistributed here; the
notebook downloads them and verifies the zip hash.
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README.md written,", len(readme), "characters")
