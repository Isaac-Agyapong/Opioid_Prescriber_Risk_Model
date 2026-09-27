"""
Rebuild the project end to end:

    python run_all.py

Prerequisite: the `opioid_analytics` PostgreSQL database built by the Medicare_Opioid_Prescribing project
(https://github.com/Isaac-Agyapong/Medicare_Opioid_Prescribing), with its password in pgpass.conf.

1. SQL/01_features.sql            -> ml.prescriber_snapshot (1.3M prescriber-years, features + label)
2. Python/02_train_models.py      -> models/ (XGBoost, calibrator, metrics.json), Data/test_scores.parquet
3. Python/03_model_results.ipynb  -> Image/*.png (evaluation, SHAP)
4. Python/04_build_app_reference.py -> models/app_reference.json (group-level reference for the app)

Then:  streamlit run app/app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PSQL = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
STEPS = [
    [PSQL, "-h", "localhost", "-U", "postgres", "-d", "opioid_analytics", "-w", "-v", "ON_ERROR_STOP=1", "-q",
     "-f", str(ROOT / "SQL" / "01_features.sql")],
    [sys.executable, str(ROOT / "Python" / "02_train_models.py")],
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
     "--ExecutePreprocessor.timeout=1200", str(ROOT / "Python" / "03_model_results.ipynb")],
    [sys.executable, str(ROOT / "Python" / "04_build_app_reference.py")],
]

# the feature cache must be rebuilt if the SQL changed
(ROOT / "Data" / "features.parquet").unlink(missing_ok=True)
for step in STEPS:
    print(f"\n>>> {Path(step[0]).name} {Path(step[-1]).name}", flush=True)
    subprocess.run(step, check=True, cwd=ROOT / "Python")
print("\nDone. Start the app with:  streamlit run app/app.py")
