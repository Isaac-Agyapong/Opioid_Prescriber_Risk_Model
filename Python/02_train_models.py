"""
Train and evaluate the early-warning models without any look-ahead:

    fit      80% of the 2020 snapshot (features 2019-2020, outcome 2021-2022)   -> model fitting
    tune     20% of the 2020 snapshot (same years, held-out prescribers)       -> model choice, early stopping,
                                                                                  isotonic calibration
    backtest 2021 snapshot (outcome 2022-2023)                                 -> out-of-time check, no decisions
    test     2022 snapshot (outcome 2023-2024)                                 -> final result, reported once

Every decision uses only outcomes up to 2022, i.e. information available when scoring the 2022 snapshot.
Test metrics come with 95% bootstrap confidence intervals.

Models: a simple rule (closeness to the specialty's 99th percentile), logistic regression, random forest,
XGBoost. Outputs go to models/ (model, calibrator, metrics.json) and Data/ (test scores without NPIs).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "Data", ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
CONNINFO = "host=localhost port=5432 user=postgres dbname=opioid_analytics"
SEED = 42
N_BOOT = 1000

CATEGORICAL = ["specialty_group", "census_region"]
NUMERIC = ["rural", "peer_count", "total_claims", "opioid_claims", "opioid_rate", "peer_percentile",
           "rate_vs_peer_median", "rate_vs_peer_p99", "long_acting_share", "days_per_opioid_claim",
           "total_beneficiaries", "opioid_patient_share", "bene_avg_age", "bene_avg_risk_score",
           "present_prev_year", "rate_change_1y", "percentile_change_1y", "opioid_claims_growth_1y",
           "was_outlier_prev_year"]
REVIEW_LIST = 1000          # how many prescribers a payer's review team could realistically look at per year


def load():
    cache = DATA / "features.parquet"
    if not cache.exists():
        with psycopg.connect(CONNINFO) as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM ml.prescriber_snapshot")
            df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df.to_parquet(cache, index=False)
    return pd.read_parquet(cache)


def precision_at(y, score, k):
    return y[np.argsort(-score)[:k]].mean()


def metrics(y, score, name):
    order = np.argsort(-score)
    k = min(REVIEW_LIST, len(y))
    top = y[order[:k]]
    top1pct = y[order[: int(len(y) * 0.01)]]
    return {"model": name,
            "pr_auc": round(average_precision_score(y, score), 4),
            "roc_auc": round(roc_auc_score(y, score), 4),
            f"precision_top_{REVIEW_LIST}": round(top.mean(), 4),
            f"lift_top_{REVIEW_LIST}": round(top.mean() / y.mean(), 1),
            "recall_top_1pct": round(top1pct.sum() / y.sum(), 4),
            "base_rate": round(y.mean(), 5)}


def bootstrap(y, s_model, s_rule):
    """95% intervals for the model's test metrics and for its precision gain over the rule."""
    rng = np.random.default_rng(SEED)
    n, k1 = len(y), int(len(y) * 0.01)
    rows = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        yy, sm, sr = y[i], s_model[i], s_rule[i]
        om = np.argsort(-sm)
        pm, pr = yy[om[:REVIEW_LIST]].mean(), precision_at(yy, sr, REVIEW_LIST)
        rows.append({"precision_top": pm, "gain_vs_rule": pm - pr,
                     "recall_top_1pct": yy[om[:k1]].sum() / yy.sum(),
                     "pr_auc": average_precision_score(yy, sm)})
    b = pd.DataFrame(rows)
    return {c: [round(float(b[c].quantile(0.025)), 4), round(float(b[c].quantile(0.975)), 4)] for c in b}


def main():
    t0 = time.time()
    df = load()
    snap = {y: df[df.snapshot_year == y].reset_index(drop=True) for y in (2020, 2021, 2022)}
    fit, tune = train_test_split(snap[2020], test_size=0.2, stratify=snap[2020]["label"], random_state=SEED)
    fit, tune = fit.reset_index(drop=True), tune.reset_index(drop=True)
    sets = {"tune_2020": tune, "backtest_2021": snap[2021], "test": snap[2022]}
    X = lambda d: d[CATEGORICAL + NUMERIC]
    y_fit = fit["label"].to_numpy()
    ys = {k: d["label"].to_numpy() for k, d in sets.items()}
    print(f"fit {len(fit):,} ({y_fit.sum():,} pos) | " +
          " | ".join(f"{k} {len(d):,} ({ys[k].sum():,})" for k, d in sets.items()))

    pre_linear = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    pre_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])

    logit = Pipeline([("prep", pre_linear),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]).fit(X(fit), y_fit)
    forest = Pipeline([("prep", pre_tree),
                       ("model", RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=40,
                                                        class_weight="balanced_subsample", n_jobs=-1,
                                                        random_state=SEED))]).fit(X(fit), y_fit)
    prep_xgb = pre_tree.fit(X(fit))
    feat_names = [n.split("__", 1)[1] for n in prep_xgb.get_feature_names_out()]
    booster = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=(len(y_fit) - y_fit.sum()) / y_fit.sum(),
        eval_metric="aucpr", early_stopping_rounds=100, tree_method="hist", random_state=SEED, n_jobs=-1)
    booster.fit(prep_xgb.transform(X(fit)), y_fit, eval_set=[(prep_xgb.transform(X(tune)), ys["tune_2020"])],
                verbose=False)
    print(f"xgboost best iteration: {booster.best_iteration} (early stopping on the 2020 tune split)")

    predictors = {"Rule: closeness to peer 99th pct": lambda d: d["rate_vs_peer_p99"].fillna(0).to_numpy(),
                  "Logistic regression": lambda d: logit.predict_proba(X(d))[:, 1],
                  "Random forest": lambda d: forest.predict_proba(X(d))[:, 1],
                  "XGBoost": lambda d: booster.predict_proba(prep_xgb.transform(X(d)))[:, 1]}
    results, scores = {k: [] for k in sets}, {}
    for name, predict in predictors.items():
        for k, d in sets.items():
            s = predict(d)
            scores[(k, name)] = s
            results[k].append(metrics(ys[k], s, name))

    # model choice on the 2020 tune split only
    best = max((r for r in results["tune_2020"] if not r["model"].startswith("Rule")), key=lambda r: r["pr_auc"])["model"]
    print(f"selected on 2020 tune PR-AUC: {best}")

    calib = IsotonicRegression(out_of_bounds="clip").fit(scores[("tune_2020", best)], ys["tune_2020"])
    test_prob = calib.predict(scores[("test", best)])
    calibration = {"brier_test": round(brier_score_loss(ys["test"], test_prob), 5),
                   "mean_predicted": round(float(test_prob.mean()), 5), "observed_rate": round(float(ys["test"].mean()), 5)}
    ci = bootstrap(ys["test"], scores[("test", best)], scores[("test", "Rule: closeness to peer 99th pct")])

    joblib.dump({"prep": prep_xgb, "booster": booster, "calibrator": calib, "features": feat_names,
                 "categorical": CATEGORICAL, "numeric": NUMERIC}, MODELS / "early_warning_xgb.joblib")
    booster.save_model(MODELS / "early_warning_xgb.json")
    test = sets["test"]
    pd.DataFrame({"label": ys["test"], "specialty_group": test["specialty_group"], "census_region": test["census_region"],
                  "rule_score": scores[("test", "Rule: closeness to peer 99th pct")],
                  "model_score": scores[("test", best)], "model_probability": test_prob}
                 ).to_parquet(DATA / "test_scores.parquet", index=False)
    summary = {"protocol": "fit/tune = 80/20 of the 2020 snapshot; 2021 backtest and 2022 test untouched",
               "splits": {"fit_2020": [len(fit), int(y_fit.sum())],
                          **{k: [len(d), int(ys[k].sum())] for k, d in sets.items()}},
               "selected_model": best, "xgb_best_iteration": int(booster.best_iteration),
               "review_list_size": REVIEW_LIST, "results": results, "calibration": calibration,
               "test_bootstrap_95ci": ci, "bootstrap_resamples": N_BOOT}
    (MODELS / "metrics.json").write_text(json.dumps(summary, indent=2))

    for k in sets:
        print(f"\n{k.upper()}")
        print(pd.DataFrame(results[k]).to_string(index=False))
    print(f"\ncalibration (test): {calibration}")
    print(f"test 95% CI ({N_BOOT} bootstrap resamples): {ci}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
