"""Investor Propensity & Targeting Model, a Streamlit app on the FAR-Trans data.

Reads the frozen files in artifacts/ and the frozen models in artifacts/models/.
Nothing is trained at request time and no notebook code is imported.

    streamlit run app.py
"""
import os

# Single-threaded BLAS/OpenMP: the app scores one row at a time, and thread
# pools sized to the host CPU are the usual cause of idle CPU spikes (and
# throttling) on small shared containers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parent / "artifacts"
PAGES = ["Overview", "Models & ensembles", "Segments", "Contact budget", "Drivers", "Customer lookup"]
SUBTITLES = {
    "Overview": "Which existing investment customers will place a BUY order in the next six months, and whom should the bank contact first?",
    "Models & ensembles": "Nine classical models, one out-of-time test period, paired bootstrap on customers.",
    "Segments": "Who ends up on the contact list, and does the ranking hold across the book?",
    "Contact budget": "How many eventual buyers does a contact list of a given size actually reach?",
    "Drivers": "What moves the score: globally through SHAP, and per customer through TreeSHAP.",
    "Customer lookup": "Recompute one customer's score from the frozen models, with its drivers.",
}

INK, MUTED, LINE = "#0F172A", "#5B6673", "#E8EBEF"
ACCENT, ACCENT2, NEUTRAL = "#2F5D8A", "#C2410C", "#9AA5B1"

st.set_page_config(page_title="Investor Propensity & Targeting Model", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
      html, body, [data-testid="stAppViewContainer"] {{
        font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      }}
      .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }}
      [data-testid="stHeading"] h1 {{ font-size: 0.78rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: {ACCENT}; }}
      [data-testid="stHeading"] h2 {{ font-size: 1.85rem; font-weight: 600; letter-spacing: -0.02em; color: {INK}; padding-top: 0; }}
      [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{ font-size: 1.02rem; font-weight: 600; color: {INK}; }}
      p, li {{ color: {INK}; }}
      [data-testid="stCaptionContainer"] p {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.5; }}
      [data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; color: {INK}; }}
      [data-testid="stMetricLabel"] p {{
        font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; color: {MUTED};
      }}
      [data-testid="stSidebar"] {{ background: #FAFBFC; border-right: 1px solid {LINE}; }}
      [data-testid="stSidebar"] .block-container {{ padding-top: 1.8rem; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 6px; }}
      hr {{ border-color: {LINE}; }}
      a {{ color: {ACCENT}; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def csv(name, **kw):
    return pd.read_csv(ART / name, **kw)


@st.cache_data
def js(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def dedupe(xs):
    return list(dict.fromkeys(xs))


def style(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(grid=False, domainColor=LINE, tickColor=LINE, labelColor=MUTED, titleColor=MUTED, labelFontSize=11, titleFontSize=11)
        .configure_legend(orient="bottom", direction="horizontal", labelColor=MUTED, titleColor=MUTED, labelFontSize=11)
        .configure_title(color=INK, fontSize=13, anchor="start")
    )


summary = js("run_summary.json")
manifest = js("models/manifest.json")
NAMES = manifest["names"]
BEST_SINGLE, BEST_ENS, HEAD = manifest["best_single"], manifest["best_ensemble"], manifest["headline_model"]
prov = js("data_provenance.json")
COUNTS = prov["counts"]
DATASET = (f"{COUNTS['customers_with_transactions']:,} customers · {COUNTS['transactions']:,} transactions · "
           f"{COUNTS['first_transaction_date'][:7]} to {COUNTS['last_transaction_date'][:7]} · "
           f"FAR-Trans benchmark dataset, University of Glasgow (CC BY 4.0)")

st.title("Investor Propensity & Targeting Model")

with st.sidebar:
    st.markdown(
        f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:{MUTED}'>Section</div>",
        unsafe_allow_html=True,
    )
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Six-month BUY propensity for the investment book of a European bank. FAR-Trans benchmark data, University of Glasgow.")
    st.caption("[Code, notebook and tests](https://github.com/bravo2024/investor) · every number on these pages is read from the frozen files in `artifacts/`")

st.header(page)
st.markdown(f"<div style='font-size:1rem;color:{MUTED};max-width:780px;padding-bottom:.4rem'>{SUBTITLES[page]}</div>", unsafe_allow_html=True)
st.caption(DATASET)

if summary.get("smoke"):
    st.warning("These artifacts come from a smoke run on a small sample, not the full run.")


# --------------------------------------------------------------------------------------------- Overview
if page == "Overview":
    with st.container(border=True):
        c = st.columns(4)
        c[0].metric("Customers", f"{COUNTS['customers_with_transactions']:,}")
        c[1].metric("Transactions", f"{COUNTS['transactions']:,}")
        c[2].metric("Buys", f"{COUNTS['buys']:,}")
        c[3].metric("Test buy rate", pct(summary["test_positive_rate"]))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Out-of-time design")
        snaps = csv("snapshots.csv", parse_dates=["snapshot"])
        color = alt.Color("split:N", title="Split", scale=alt.Scale(domain=["train", "gap", "valid", "test"], range=["#4C72B0", "#CFD6DE", "#DD8452", "#55A868"]))
        bar = alt.Chart(snaps).mark_bar().encode(
            x=alt.X("snapshot:T", title=None),
            y=alt.Y("positive_rate:Q", title="share who buy within 6 months", axis=alt.Axis(format="%")),
            color=color,
            tooltip=["snapshot", "split", "rows", "positive_rate"],
        ).properties(height=250)
        st.altair_chart(style(bar), width="stretch")
        st.caption("Monthly cut-offs. Blue trains, orange validates, green tests; grey cut-offs are dropped so a label window never "
                   "overlaps the next block. Test uptake is lower than training uptake: real drift, kept in.")
    with right:
        st.markdown("#### Split sizes")
        sp = csv("splits.csv")
        sp = pd.DataFrame({
            "Split": sp.split,
            "Cut-offs": sp.snapshots,
            "Customers": sp.customers,
            "Buy rate": 100 * sp["positive rate"],
        })
        st.dataframe(sp, hide_index=True, width="stretch", column_config={
            "Buy rate": st.column_config.NumberColumn(format="%.1f%%"),
        })

    st.markdown("#### Headline: out-of-time test (7 cut-offs, 2021-12 to 2022-06)")
    tm = csv("test_metrics.csv")
    ci = csv("bootstrap_ci.csv")
    lvl = ci[(ci.comparison == "level") & (ci.metric == "roc_auc")].set_index("model")
    t = pd.DataFrame({
        "Model": tm.model,
        "ROC-AUC (95% CI)": [f"{r.roc_auc:.3f} [{lvl.loc[r.model, 'ci_low']:.3f}, {lvl.loc[r.model, 'ci_high']:.3f}]" for r in tm.itertuples()],
        "PR-AUC": tm.pr_auc, "KS": tm.ks,
        "Top 10%": 100 * tm.capture_10, "Top 20%": 100 * tm.capture_20, "Top 30%": 100 * tm.capture_30,
    })
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        "KS": st.column_config.NumberColumn(format="%.3f"),
        "Top 10%": st.column_config.NumberColumn(format="%.1f%%"),
        "Top 20%": st.column_config.NumberColumn(format="%.1f%%"),
        "Top 30%": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.caption(f"Buyers reached @k% = share of the cut-off's eventual buyers in the top k% by score, pooled over the seven test cut-offs. "
               f"Intervals: paired bootstrap, {summary['n_boot']:,} customer resamples. Best single model on validation: {NAMES[BEST_SINGLE]}; "
               f"best ensemble: {NAMES[BEST_ENS]}.")

# --------------------------------------------------------------------------------------------- Models & ensembles
elif page == "Models & ensembles":
    roc, pr = csv("curves_roc.csv"), csv("curves_pr.csv")
    shown = st.multiselect("Models on the curves", list(NAMES.values()), default=dedupe([NAMES["lr"], NAMES[BEST_SINGLE], NAMES[HEAD]]))
    st.altair_chart(style(alt.Chart(roc[roc.model.isin(shown)]).mark_line().encode(
        x=alt.X("fpr", title="false positive rate"), y=alt.Y("tpr", title="true positive rate"), color=alt.Color("model:N", title="Model")).properties(title="ROC (test)", height=300)),
        width="stretch")
    st.altair_chart(style(alt.Chart(pr[pr.model.isin(shown)]).mark_line().encode(
        x=alt.X("recall", title="recall"), y=alt.Y("precision", title="precision"), color=alt.Color("model:N", title="Model")).properties(title="Precision-recall (test)", height=300)),
        width="stretch")

    tabs = st.tabs(["Bootstrap", "Tuning", "Collinearity", "Calibration", "Stability"])
    ci = csv("bootstrap_ci.csv")
    with tabs[0]:
        kind = st.radio("Show", ["Each model's own metric", "Minus logistic regression", "Ensembles minus best single model"], horizontal=True)
        sel = {"Each model's own metric": "level", "Minus logistic regression": "minus_lr", "Ensembles minus best single model": "minus_best_single"}[kind]
        mc = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "capture_20": "Buyers reached @20%"}
        d = ci[ci.comparison == sel].copy()
        fmt = (lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]") if sel == "level" \
            else (lambda r: f"{r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
        d["cell"] = d.apply(fmt, axis=1)
        d["metric"] = d.metric.map(mc).fillna(d.metric)
        d["Model"] = d.model.map(NAMES).fillna(d.model)
        out = d.pivot(index="Model", columns="metric", values="cell")
        out = out.reindex(columns=[c for c in mc.values() if c in out.columns])
        st.dataframe(out, width="stretch")
        st.caption(f"Paired bootstrap on the test period: {summary['n_boot']:,} resamples of customers, so a customer's repeated cut-offs move together."
                   + (" Cells are the estimate with its 95% interval." if sel == "level" else " Cells are the difference with its 95% interval; an interval that excludes zero is a real gap."))

    with tabs[1]:
        st.dataframe(csv("tuning_gain.csv").round(4), hide_index=True, width="stretch")
        model_key = st.selectbox("Random-search table", manifest["base_models"], format_func=lambda k: NAMES[k])
        st.dataframe(csv(f"tuning_{model_key}.csv"), hide_index=True, width="stretch")
        st.caption("Best parameters")
        st.json(js("best_params.json")["tuned"][model_key])

    with tabs[2]:
        vs = js("vif_summary.json")
        st.markdown(f"Spearman screen (|rho| > 0.9): {vs['features_before_screen']} to {vs['after_spearman_screen']} features. "
                    f"VIF pruning for logistic regression dropped {', '.join(vs['features_dropped_for_lr']) or 'nothing'}; features with VIF above 10 "
                    f"went from {vs['n_vif_gt10_before']} to {vs['n_vif_gt10_after']} (max VIF {vs['max_vif_before']:.1f} to {vs['max_vif_after']:.2f}).")
        st.dataframe(csv("vif_table.csv").round(2), hide_index=True, width="stretch")

    with tabs[3]:
        st.dataframe(csv("calibration_choice.csv").round(5), hide_index=True, width="stretch")
        rel = csv("reliability_table.csv")
        pick = st.multiselect("Reliability curves", list(NAMES.values()), default=dedupe([NAMES["lr"], NAMES[HEAD]]))
        base = alt.Chart(rel[rel.model.isin(pick)]).mark_line(point=True).encode(
            x=alt.X("mean_predicted", title="mean predicted"), y=alt.Y("observed_rate", title="observed buy rate"), color=alt.Color("model:N", title="Model"))
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=NEUTRAL, strokeDash=[4, 4]).encode(x="x", y="y")
        st.altair_chart(style(base + diag), width="stretch")
        st.caption("Platt or isotonic, picked by grouped 5-fold Brier on validation. Validation uptake is higher than test uptake, so a "
                   "calibrator fitted there can over-predict on test.")

    with tabs[4]:
        rs = csv("rolling_origin_summary.csv")
        st.dataframe(rs[["scheme", "model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "capture_20_mean", "capture_20_std", "n_cutoffs"]].round(4),
                     hide_index=True, width="stretch")
        st.caption("Frozen models and expanding-window refits at each test cut-off: mean and standard deviation across cut-offs.")

# --------------------------------------------------------------------------------------------- Segments
elif page == "Segments":
    seg = csv("segment_results.csv")
    dim = st.radio("Group by", list(seg.dimension.unique()), horizontal=True)
    d = seg[seg.dimension == dim].drop(columns="dimension")
    view = pd.DataFrame({
        dim: d.group, "Customers": d.customers, "Rows": d.rows,
        "Uptake": 100 * d.uptake_rate, "ROC-AUC": d.model_roc_auc,
        "Buyers on top-20% list": 100 * d.capture_20, "Precision on list": 100 * d.precision_in_list, "Share of all buyers": 100 * d.share_of_buyers,
    })
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Uptake": st.column_config.NumberColumn(format="%.1f%%"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Buyers on top-20% list": st.column_config.NumberColumn(format="%.1f%%"),
        "Precision on list": st.column_config.NumberColumn(format="%.1f%%"),
        "Share of all buyers": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.altair_chart(style(alt.Chart(d).mark_bar(color=ACCENT, size=16).encode(
        y=alt.Y("group:N", sort="-x", title=None),
        x=alt.X("uptake_rate:Q", axis=alt.Axis(format="%"), title="share of customers who buy within six months"),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("uptake_rate:Q", title="Uptake", format=".1%"),
                 alt.Tooltip("capture_20:Q", title="Buyers reached @20%", format=".1%")]).properties(height=160)),
        width="stretch")

    st.markdown("#### Propensity bands")
    bands = csv("propensity_bands.csv")
    bview = pd.DataFrame({"Band": bands.band, "Rows": bands.rows, "Uptake": 100 * bands.uptake_rate,
                          "Lift": bands.lift, "Share of buyers": 100 * bands.share_of_buyers, "Mean predicted": bands.mean_predicted})
    st.dataframe(bview, hide_index=True, width="stretch", column_config={
        "Uptake": st.column_config.NumberColumn(format="%.1f%%"),
        "Lift": st.column_config.NumberColumn(format="%.2fx"),
        "Share of buyers": st.column_config.NumberColumn(format="%.1f%%"),
        "Mean predicted": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption("Bands are cut on the calibrated probability; lift compares uptake with the base rate of the same test rows.")

    st.markdown("#### Dormant and active customers")
    act = csv("activity_slices.csv").drop_duplicates(subset=["slice", "model"])
    rows = []
    for name, g in act.groupby("slice", sort=False):
        r = {"Slice": name, "Uptake": 100 * g.uptake_rate.iloc[0]}
        for m in ["Logistic regression", "LightGBM"]:
            hit = g[g.model == m]
            if len(hit):
                r[f"ROC-AUC {m}"] = hit.roc_auc.iloc[0]
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Uptake": st.column_config.NumberColumn(format="%.1f%%"),
        "ROC-AUC Logistic regression": st.column_config.NumberColumn(format="%.3f"),
        "ROC-AUC LightGBM": st.column_config.NumberColumn(format="%.3f"),
    })
    dormant_share = act[act.slice.str.startswith("no trade")].share_of_rows.iloc[0]
    st.caption(f"Slices split by whether the customer traded in the previous 12 months. Dormant: {pct(dormant_share)} of test rows, "
               f"active: {pct(1 - dormant_share)}.")

    with st.expander("K-Means behavioural clusters"):
        km = js("kmeans_model.json")
        st.caption(f"k = {km['k']} chosen by silhouette over 3 to 6 ({', '.join(f'k={k}: {v:.3f}' for k, v in km['silhouette'].items())}), "
                   f"fitted on pre-cut-off behaviour of training rows.")
        st.dataframe(csv("kmeans_profiles.csv").drop(columns=["cluster_raw"]).round(3), hide_index=True, width="stretch")
        st.markdown("**Official segment mix inside each cluster** (row shares)")
        mix = csv("kmeans_vs_segment_share.csv", index_col=0)
        st.dataframe(mix.map(lambda v: f"{100 * v:.1f}%"), width="stretch")

    with st.expander("Buying by product category"):
        st.dataframe(csv("category_variants.csv").round(4), hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------------- Contact budget
elif page == "Contact budget":
    cb = csv("contact_budget.csv")
    budget = st.slider("Share of existing customers contacted at each cut-off", 1, 60, 20, format="%d%%")
    models = {"Logistic regression": NAMES["lr"], NAMES[BEST_SINGLE]: NAMES[BEST_SINGLE], NAMES[BEST_ENS]: NAMES[BEST_ENS]}
    rows = []
    for label, m in models.items():
        r = cb[(cb.model == m) & np.isclose(cb.contact_share, budget / 100)].iloc[0]
        rows.append({"model": label, "buyers reached": 100 * r.buyers_captured,
                     "precision (buyers among contacted)": 100 * r.precision, "lift vs random": r.lift})
    tab = pd.DataFrame(rows)

    st.markdown("#### Buyers reached")
    cols = st.columns(3)
    for col, (label, _) in zip(cols, models.items()):
        r = tab[tab.model == label].iloc[0]
        col.metric(label, f"{r['buyers reached']:.1f}%")
        col.caption(f"{r['lift vs random']:.2f}x lift · precision {r['precision (buyers among contacted)']:.1f}%")

    st.dataframe(tab, hide_index=True, width="stretch", column_config={
        "buyers reached": st.column_config.NumberColumn(format="%.1f%%"),
        "precision (buyers among contacted)": st.column_config.NumberColumn(format="%.1f%%"),
        "lift vs random": st.column_config.NumberColumn(format="%.2fx"),
    })
    base = summary["test_positive_rate"]
    st.caption(f"Contacting {budget}% at random would reach about {budget}% of buyers at {pct(base)} precision. Customers are ranked within each "
               f"cut-off and results are pooled over the seven test cut-offs.")
    sub = cb[cb.model.isin(models.values()) & (cb.contact_share <= 0.6)]
    line = alt.Chart(sub).mark_line().encode(
        x=alt.X("contact_share:Q", axis=alt.Axis(format="%"), title="share of customers contacted"),
        y=alt.Y("buyers_captured:Q", axis=alt.Axis(format="%"), title="buyers reached"), color=alt.Color("model:N", title="Model"))
    random_line = alt.Chart(pd.DataFrame({"x": [0, 0.6], "y": [0, 0.6]})).mark_line(color=NEUTRAL, strokeDash=[4, 4]).encode(x="x", y="y")
    st.altair_chart(style(line + random_line).properties(height=340), width="stretch")

# --------------------------------------------------------------------------------------------- Drivers
elif page == "Drivers":
    sh, perm = csv("shap_summary.csv"), csv("permutation_importance.csv")
    st.markdown(f"#### Top 10 drivers (SHAP on {NAMES['xgb']})")
    st.markdown("\n".join(f"{i}. {r.plain_english}" for i, r in enumerate(sh.head(10).itertuples(), 1)))
    st.altair_chart(style(alt.Chart(sh.head(15)).mark_bar(color=ACCENT).encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("mean_abs_shap:Q", title="mean |SHAP|")).properties(title="SHAP", height=330)),
        width="stretch")
    st.altair_chart(style(alt.Chart(perm.head(15)).mark_bar(color=ACCENT2).encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("auc_drop_mean:Q", title="ROC-AUC drop when shuffled")).properties(title="Permutation importance", height=330)),
        width="stretch")

    st.markdown("#### Feature value against its effect")
    ss = csv("shap_sample.csv.gz")
    feat = st.selectbox("Feature", list(sh.feature), index=0)
    st.altair_chart(style(alt.Chart(ss.sample(min(3000, len(ss)), random_state=0)).mark_circle(size=12, opacity=0.45, color=ACCENT).encode(
        x=alt.X(feat, title=feat), y=alt.Y(f"shap_{feat}", title="effect on log-odds of a buy")).properties(height=300)), width="stretch")

    with st.expander("Logistic regression coefficients (per one standard deviation)"):
        st.dataframe(csv("lr_coefficients.csv").round(3), hide_index=True, width="stretch")
    with st.expander("Feature dictionary"):
        st.dataframe(csv("feature_dictionary.csv"), hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------------- Customer lookup
else:
    import joblib
    import xgboost as xgb

    @st.cache_resource
    def load_models(keys):
        out = {}
        for k in keys:
            if k == "xgb":
                m = xgb.XGBClassifier(n_jobs=1)
                m.load_model(str(ART / "models" / "xgb.json"))
            else:
                m = joblib.load(ART / "models" / f"{k}.joblib")
                if hasattr(m, "n_jobs"):
                    m.set_params(n_jobs=1)
            out[k] = m
        return out

    def apply_cal(c, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1 / (1 + np.exp(-(c["a"] * z + c["b"]))) if c["kind"] == "platt" else np.interp(p, c["x"], c["y"])

    needed = sorted({"lr", BEST_SINGLE, HEAD, "xgb"})
    models = load_models(tuple(needed))
    lat = csv("latest_snapshot.csv.gz")
    cut = lat.snapshot.iloc[0]
    st.caption(f"Customers as of the latest cut-off, {cut}. Scores are recomputed here from the frozen models; the outcome column shows what "
               f"the customer actually did in the following six months.")

    f1, f2 = st.columns([1, 2])
    band_pick = f1.selectbox("Propensity band", ["all"] + sorted(lat.band.unique()))
    pool = lat if band_pick == "all" else lat[lat.band == band_pick]
    cid = f2.selectbox("Customer", pool.customerID.tolist()[:2000], index=0)
    row = lat[lat.customerID == cid].iloc[[0]]

    lc = js("lr_clip_bounds.json")
    X_lr = np.clip(row[lc["columns"]].to_numpy(float), lc["lo"], lc["hi"])
    X_tr = row[manifest["tree_columns"]]
    base = {k: float(models[k].predict_proba(X_lr if k == "lr" else X_tr)[:, 1][0]) for k in models}
    cal = js("calibration.json")[HEAD]
    p_cal = float(apply_cal(cal, np.array([base[HEAD]]))[0])
    edges = js("band_edges.json")["calibrated_probability_edges_from_validation"]
    band = manifest["band_names_low_to_high"][int(np.searchsorted(edges, p_cal))]

    with st.container(border=True):
        c = st.columns(4)
        c[0].metric("Chance of a buy in 6 months", pct(p_cal))
        c[1].metric("Propensity band", band)
        c[2].metric(NAMES["lr"], pct(base["lr"]))
        c[3].metric(NAMES[BEST_SINGLE], pct(base[BEST_SINGLE]))

    st.markdown(
        f"**Segment** {row.segment.iloc[0]} &nbsp;·&nbsp; **Risk profile** {row.risk_profile.iloc[0]} &nbsp;·&nbsp; "
        f"**Capacity** {row.capacity_band.iloc[0]} &nbsp;·&nbsp; **Cluster** {row.cluster.iloc[0]} &nbsp;·&nbsp; "
        f"**Actual outcome** {'bought' if int(row.y_any.iloc[0]) else 'no buy'}"
    )
    st.caption(f"Score shown is {NAMES[HEAD]}"
               + (", the average of its component probabilities" if HEAD in manifest.get("ensembles", {}) else "")
               + ", calibrated on validation.")

    contrib = models["xgb"].get_booster().predict(xgb.DMatrix(X_tr), pred_contribs=True)[0][:-1]
    desc = csv("feature_dictionary.csv").set_index("feature")["description"]
    d = pd.DataFrame({"feature": manifest["tree_columns"], "value": X_tr.iloc[0].values, "effect_on_log_odds": contrib})
    d = d.reindex(d.effect_on_log_odds.abs().sort_values(ascending=False).index).head(8)
    d["meaning"] = d.feature.map(desc)
    d["pushes"] = np.where(d.effect_on_log_odds > 0, "towards a buy", "away from a buy")

    st.markdown("#### What drives this customer's score")
    bar = alt.Chart(d).mark_bar().encode(
        x=alt.X("effect_on_log_odds:Q", title="effect on log-odds of a buy"),
        y=alt.Y("feature:N", sort=list(d.feature), title=None),
        color=alt.Color("pushes:N", scale=alt.Scale(domain=["towards a buy", "away from a buy"], range=[ACCENT, NEUTRAL]), title="Direction"),
        tooltip=["feature", "value", "effect_on_log_odds", "pushes"],
    ).properties(height=280)
    st.altair_chart(style(bar), width="stretch")
    show = pd.DataFrame({
        "Feature": d.feature, "Value": d.value, "Meaning": d.meaning,
        "Effect on log-odds": d.effect_on_log_odds, "Direction": d.pushes,
    })
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Value": st.column_config.NumberColumn(format="%.3g"),
        "Effect on log-odds": st.column_config.NumberColumn(format="%.3f"),
    })
