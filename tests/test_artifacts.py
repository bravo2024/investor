import ast
import json
import re

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from conftest import ART, ROOT

pytestmark = pytest.mark.skipif(not (ART / "run_summary.json").exists(), reason="artifacts not present")
KEYS = ["lr", "dt", "rf", "gb", "xgb", "lgbm", "e1", "e2", "e3"]
FILES = ["run_summary.json", "data_provenance.json", "class_balance.json", "imbalance_decision.json", "splits.csv", "snapshots.csv", "test_metrics.csv", "val_metrics.csv",
         "bootstrap_ci.csv", "tuning_gain.csv", "best_params.json", "vif_table.csv", "vif_summary.json", "calibration.json", "calibration_choice.csv", "reliability_table.csv",
         "contact_budget.csv", "curves_roc.csv", "curves_pr.csv", "rolling_origin_summary.csv", "rolling_origin_refit.csv", "frozen_by_cutoff.csv", "segment_results.csv",
         "kmeans_profiles.csv", "kmeans_model.json", "propensity_bands.csv", "shap_summary.csv", "permutation_importance.csv", "lr_coefficients.csv", "feature_dictionary.csv",
         "category_variants.csv", "ablation_static_attributes.csv", "test_scores.csv.gz", "latest_snapshot.csv.gz", "rec_appendix_summary.csv"]


def test_all_files_present():
    missing = [f for f in FILES if not (ART / f).exists()]
    assert not missing, missing
    assert all((ART / "models" / f).exists() for f in ("manifest.json", "lr.joblib", "dt.joblib", "rf.joblib", "gb.joblib", "lgbm.joblib", "xgb.json"))


def test_data_provenance_gate():
    p = json.loads((ART / "data_provenance.json").read_text())
    assert len(p["zip_sha256"]) == 64 and p["licence"].startswith("CC BY 4.0")
    c = p["counts"]
    assert c["customers_with_transactions"] == 29090 and c["buys"] == 228913 and c["price_points"] == 703303
    assert p["simulated_or_estimated"]["risk_level"]["predicted_values"] == 7141
    assert p["simulated_or_estimated"]["investment_capacity"]["predicted_values"] == 7318
    assert p["gate"]["supports_customer_level_time_split_task"] is True


def test_metrics_are_sane_and_complete():
    tm = pd.read_csv(ART / "test_metrics.csv").set_index("key")
    assert list(tm.index) == KEYS
    assert ((tm.roc_auc > 0.6) & (tm.roc_auc < 1)).all() and (tm.capture_20 > 0.2).all()
    assert (tm.capture_10 <= tm.capture_20).all() and (tm.capture_20 <= tm.capture_30).all()


def test_split_purge_gap_and_class_balance():
    sp = json.loads((ART / "class_balance.json").read_text())
    for a, b in (("train", "valid"), ("valid", "test")):
        gap = pd.Timestamp(sp[b]["first_cutoff"]) - (pd.Timestamp(sp[a]["last_cutoff"]) + pd.DateOffset(months=6))
        assert gap >= pd.Timedelta(0), (a, b)
    snaps = pd.read_csv(ART / "snapshots.csv", parse_dates=["snapshot"])
    last_ok = (pd.Timestamp("2022-11-30") + pd.Timedelta(days=1)) - pd.DateOffset(months=6)
    assert snaps[snaps.split == "test"].snapshot.max() <= last_ok
    assert all(0.02 < sp[s]["positive_rate"] < 0.5 for s in ("train", "valid", "test"))


def test_vif_pruning_result():
    vs = json.loads((ART / "vif_summary.json").read_text())
    assert vs["n_vif_gt10_after"] == 0 and vs["max_vif_after"] <= 10 and vs["after_spearman_screen"] <= vs["features_before_screen"]


def test_bootstrap_ci_contain_estimates():
    ci = pd.read_csv(ART / "bootstrap_ci.csv")
    assert set(ci.comparison) == {"level", "minus_lr", "minus_best_single"}
    lvl = ci[ci.comparison == "level"]
    assert ((lvl.ci_low <= lvl.estimate + 1e-9) & (lvl.estimate <= lvl.ci_high + 1e-9)).all()
    assert (ci.ci_low <= ci.ci_high).all()


def test_propensity_bands_rank_customers():
    b = pd.read_csv(ART / "propensity_bands.csv")
    b = b.sort_values("band")                       # "1 very high" ... "5 very low"
    assert len(b) == 5 and (np.diff(b.uptake_rate.values) < 0.01).all() and b.uptake_rate.iloc[0] > 2 * b.uptake_rate.iloc[-1]
    assert abs(b.share_of_buyers.sum() - 1) < 1e-9


def test_kmeans_k_in_range_and_matches_silhouette():
    km = json.loads((ART / "kmeans_model.json").read_text())
    best = int(max(km["silhouette"], key=lambda k: km["silhouette"][k]))
    assert 3 <= km["k"] <= 6 and km["k"] == best


def _frozen_scores(df):
    man = json.loads((ART / "models" / "manifest.json").read_text())
    lc = json.loads((ART / "lr_clip_bounds.json").read_text())
    X_lr = np.clip(df[lc["columns"]].to_numpy(float), lc["lo"], lc["hi"])
    X_tr = df[man["tree_columns"]]
    out = {}
    for k in man["base_models"]:
        if k == "xgb":
            m = xgb.XGBClassifier()
            m.load_model(str(ART / "models" / "xgb.json"))
        else:
            m = joblib.load(ART / "models" / f"{k}.joblib")
        out[k] = m.predict_proba(X_lr if k == "lr" else X_tr)[:, 1]
    for e, parts in man["ensembles"].items():
        out[e] = np.mean([out[p] for p in parts], axis=0)
    return out


def test_frozen_models_reproduce_scores():
    lat = pd.read_csv(ART / "latest_snapshot.csv.gz")
    s = _frozen_scores(lat)
    for k in KEYS:
        assert np.abs(s[k] - lat[f"p_{k}"].values).max() < 2e-5, k
    ts = pd.read_csv(ART / "test_scores.csv.gz")
    tm = pd.read_csv(ART / "test_metrics.csv").set_index("key")
    for k in KEYS:
        assert abs(roc_auc_score(ts.y_any, ts[f"p_{k}"]) - tm.loc[k, "roc_auc"]) < 1e-3, k
        assert abs(average_precision_score(ts.y_any, ts[f"p_{k}"]) - tm.loc[k, "pr_auc"]) < 1e-3, k


def test_notebook_is_self_contained():
    nb = json.loads((ROOT / "FARTransWealth.ipynb").read_text(encoding="utf-8"))
    ok = {"os", "json", "time", "hashlib", "warnings", "platform", "zipfile", "urllib", "math", "itertools", "pathlib", "numpy", "pandas", "matplotlib", "sklearn", "joblib",
          "xgboost", "lightgbm", "shap"}
    n_code = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(c["source"])
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] in ok for a in node.names), src[:80]
            if isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] in ok, node.module
    assert n_code > 20
    assert not (ROOT / "src").exists()


def test_executed_notebook_clean():
    p = ROOT / "FARTransWealth.executed.ipynb"
    if not p.exists():
        pytest.skip("executed notebook not present")
    nb = json.loads(p.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert not errs
    assert all(c["outputs"] for c in code), [i for i, c in enumerate(code) if not c["outputs"]]


def test_forbidden_words_absent():
    words = ["prof" + "it", "mar" + "gin"]
    acronyms = ["P" + "D", "LG" + "D", "E" + "AD"]
    pat = re.compile("(?i:" + "|".join(words) + ")|" + r"\b(?:" + "|".join(acronyms) + r")\b")
    for f in [ROOT / "README.md", ROOT / "app.py", ROOT / "FARTransWealth.ipynb"] + sorted(ART.glob("*.json")) + sorted(ART.glob("*.csv")):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        assert not pat.search(txt), f.name


def test_readme_numbers_match_artifacts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tm = pd.read_csv(ART / "test_metrics.csv").set_index("key")
    for k in KEYS:
        assert f"{tm.loc[k, 'roc_auc']:.3f}" in readme and f"{tm.loc[k, 'pr_auc']:.3f}" in readme, k
        assert f"{100 * tm.loc[k, 'capture_20']:.1f}%" in readme, k
    ci = pd.read_csv(ART / "bootstrap_ci.csv")
    r = ci[(ci.comparison == "minus_lr") & (ci.metric == "roc_auc")].iloc[0]
    assert f"{r.ci_low:+.3f}" in readme and f"{r.ci_high:+.3f}" in readme
    cb = json.loads((ART / "class_balance.json").read_text())
    assert f"{cb['test']['rows']:,}" in readme and f"{cb['train']['rows']:,}" in readme
