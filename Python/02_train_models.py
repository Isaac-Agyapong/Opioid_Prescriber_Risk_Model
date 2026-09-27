"""
Train and evaluate the early-warning models with a time-based split:

    train      snapshot 2020 (features 2019-2020, outcome 2021-2022)
    validation snapshot 2021 (features 2020-2021, outcome 2022-2023)   -> model choice, early stopping, calibration
    test       snapshot 2022 (features 2021-2022, outcome 2023-2024)   -> reported once, never used for tuning

Models: a simple rule (closeness to the specialty's 99th percentile), logistic regression, random forest,
XGBoost. Outputs go to models/ (model, calibrator, metrics.json) and Data/ (test scores without NPIs).
"""
import importlib
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "Data", ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
CONNINFO = "host=localhost port=5432 user=postgres dbname=opioid_analytics"
SEED = 42

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


def metrics(y, score, name):
    order = np.argsort(-score)
    top = y[order[:REVIEW_LIST]]
    top1pct = y[order[: int(len(y) * 0.01)]]
    return {"model": name,
            "pr_auc": round(average_precision_score(y, score), 4),
            "roc_auc": round(roc_auc_score(y, score), 4),
            f"precision_top_{REVIEW_LIST}": round(top.mean(), 4),
            f"lift_top_{REVIEW_LIST}": round(top.mean() / y.mean(), 1),
            "recall_top_1pct": round(top1pct.sum() / y.sum(), 4),
            "base_rate": round(y.mean(), 5)}


def main():
    t0 = time.time()
    df = load()
    train, valid, test = (df[df.snapshot_year == y].reset_index(drop=True) for y in (2020, 2021, 2022))
    X = lambda d: d[CATEGORICAL + NUMERIC]
    y_tr, y_va, y_te = (d["label"].to_numpy() for d in (train, valid, test))
    print(f"train {len(train):,} ({y_tr.sum():,} pos) | valid {len(valid):,} ({y_va.sum():,}) | "
          f"test {len(test):,} ({y_te.sum():,})")

    results, scores = {"valid": [], "test": []}, {}

    # 1. baseline rule: already close to the peer 99th percentile -> most likely to cross it
    for split, d, y in (("valid", valid, y_va), ("test", test, y_te)):
        s = d["rate_vs_peer_p99"].fillna(0).to_numpy()
        results[split].append(metrics(y, s, "Rule: closeness to peer 99th pct"))
        scores[(split, "rule")] = s

    pre_linear = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    pre_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])

    # 2. logistic regression
    logit = Pipeline([("prep", pre_linear),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))])
    logit.fit(X(train), y_tr)
    # 3. random forest
    forest = Pipeline([("prep", pre_tree),
                       ("model", RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=40,
                                                        class_weight="balanced_subsample", n_jobs=-1,
                                                        random_state=SEED))])
    forest.fit(X(train), y_tr)
    # 4. XGBoost with early stopping on the validation year
    prep_xgb = pre_tree.fit(X(train))
    feat_names = [n.split("__", 1)[1] for n in prep_xgb.get_feature_names_out()]
    booster = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
        eval_metric="aucpr", early_stopping_rounds=100, tree_method="hist", random_state=SEED, n_jobs=-1)
    booster.fit(prep_xgb.transform(X(train)), y_tr,
                eval_set=[(prep_xgb.transform(X(valid)), y_va)], verbose=False)
    print(f"xgboost best iteration: {booster.best_iteration}")

    for name, predict in [("Logistic regression", lambda d: logit.predict_proba(X(d))[:, 1]),
                          ("Random forest", lambda d: forest.predict_proba(X(d))[:, 1]),
                          ("XGBoost", lambda d: booster.predict_proba(prep_xgb.transform(X(d)))[:, 1])]:
        for split, d, y in (("valid", valid, y_va), ("test", test, y_te)):
            s = predict(d)
            results[split].append(metrics(y, s, name))
            scores[(split, name)] = s

    # model choice on the validation year only
    best = max(results["valid"][1:], key=lambda r: r["pr_auc"])["model"]
    print(f"selected on validation PR-AUC: {best}")

    # calibration: map raw scores to real probabilities, fitted on the validation year
    calib = IsotonicRegression(out_of_bounds="clip").fit(scores[("valid", best)], y_va)
    test_prob = calib.predict(scores[("test", best)])
    calibration = {"brier_test": round(brier_score_loss(y_te, test_prob), 5),
                   "mean_predicted": round(float(test_prob.mean()), 5), "observed_rate": round(float(y_te.mean()), 5)}

    # artefacts
    joblib.dump({"prep": prep_xgb, "booster": booster, "calibrator": calib, "features": feat_names,
                 "categorical": CATEGORICAL, "numeric": NUMERIC}, MODELS / "early_warning_xgb.joblib")
    booster.save_model(MODELS / "early_warning_xgb.json")
    pd.DataFrame({"label": y_te, "specialty_group": test["specialty_group"], "census_region": test["census_region"],
                  "rule_score": scores[("test", "rule")], "model_score": scores[("test", best)],
                  "model_probability": test_prob}).to_parquet(DATA / "test_scores.parquet", index=False)
    summary = {"splits": {"train_2020": [len(train), int(y_tr.sum())], "valid_2021": [len(valid), int(y_va.sum())],
                          "test_2022": [len(test), int(y_te.sum())]},
               "selected_model": best, "xgb_best_iteration": int(booster.best_iteration),
               "review_list_size": REVIEW_LIST, "results": results, "calibration": calibration}
    (MODELS / "metrics.json").write_text(json.dumps(summary, indent=2))

    for split in ("valid", "test"):
        print(f"\n{split.upper()}")
        print(pd.DataFrame(results[split]).to_string(index=False))
    print(f"\ncalibration (test): {calibration}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
