"""Build models/app_reference.json: per-prescriber-group reference values the app needs to turn a
profile (opioid rate, volume, ...) into model features. Aggregates only, no individual prescribers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "Data" / "features.parquet")
ref = df[df.snapshot_year == 2022].copy()
ref["peer_median"] = ref["opioid_rate"] / ref["rate_vs_peer_median"]
ref["peer_p99"] = ref["opioid_rate"] / ref["rate_vs_peer_p99"]

out = {}
for group, g in ref.groupby("specialty_group"):
    if len(g) < 500:
        continue
    out[group] = {
        "prescribers": int(len(g)),
        "rate_quantiles": [round(float(v), 4) for v in np.quantile(g["opioid_rate"], np.linspace(0, 1, 101))],
        # 0 where the specialty median is 0: the ratio is then undefined, as in training (imputed by the model)
        "peer_median": round(float(np.nan_to_num(g["peer_median"].median())), 4),
        "peer_p99": round(float(g["peer_p99"].median()), 4),
        "peer_count": int(g["peer_count"].median()),
        "defaults": {c: round(float(g[c].median()), 3) for c in
                     ["opioid_rate", "total_claims", "long_acting_share", "days_per_opioid_claim",
                      "total_beneficiaries", "opioid_patient_share", "bene_avg_age", "bene_avg_risk_score"]},
    }
base_rate = float(ref["label"].mean())
(ROOT / "models" / "app_reference.json").write_text(json.dumps({"base_rate": base_rate, "groups": out}, indent=1))
print(f"{len(out)} prescriber groups, base rate {base_rate:.4%}")
