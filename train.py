import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from app.datasource import FEATURES

rng = np.random.default_rng(42)
n = 8000

df = pd.DataFrame({
    "bk_score": rng.integers(380, 850, n),
    "overdue_days_max": rng.integers(0, 120, n),
    "inquiries_90d": rng.integers(0, 12, n),
    "revenue_ttm": rng.uniform(3e6, 1e8, n),
    "revenue_yoy": rng.uniform(-0.5, 0.6, n),
    "volatility": rng.uniform(0.05, 0.8, n),
    "avg_balance_ratio": rng.uniform(0.1, 3, n),
    "cash_gap_days_90": rng.integers(0, 45, n),
    "dsr": rng.uniform(0, 0.7, n),
    "tax_delay_days": rng.integers(0, 80, n),
    "account_blocked": (rng.random(n) < 0.07).astype(int),
    "lawsuit_ratio": rng.uniform(0, 0.3, n),
    "bailiff_ratio": rng.uniform(0, 0.25, n),
    "owner_bailiff": (rng.random(n) < 0.12).astype(int),
    "biz_age_months": rng.integers(6, 120, n),
    "top1_share": rng.uniform(0.05, 0.95, n),
    "counterparties": rng.integers(5, 120, n),
    "sector_risk": rng.uniform(0.1, 0.9, n),
    "requested_to_revenue": rng.uniform(0.02, 0.6, n),
    "term_months": rng.integers(3, 48, n),
    "collateral": (rng.random(n) < 0.3).astype(int),
})

logit = (
    -2.6
    - 0.006 * (df.bk_score - 650)
    + 0.02 * df.overdue_days_max
    + 1.6 * df.volatility
    - 1.2 * df.revenue_yoy
    + 0.03 * df.cash_gap_days_90
    + 2.0 * df.dsr
    + 1.5 * df.account_blocked
    + 3.0 * df.bailiff_ratio
    + 0.8 * df.owner_bailiff
    + 1.0 * df.top1_share
    - 0.003 * df.biz_age_months
    + 0.8 * df.requested_to_revenue
    - 0.6 * df.collateral
    + rng.normal(0, 0.6, n)
)
p = 1 / (1 + np.exp(-logit))
df["default"] = rng.binomial(1, p)

X, y = df[FEATURES], df["default"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = CatBoostClassifier(
    iterations=400, depth=5, learning_rate=0.05,
    loss_function="Logloss", eval_metric="AUC",
    allow_writing_files=False, verbose=100, random_seed=42,
)
model.fit(Xtr, ytr, eval_set=(Xte, yte), early_stopping_rounds=40)

out = Path("models")
out.mkdir(exist_ok=True)
model.save_model(out / "catboost.cfm")
(out / "features.json").write_text(json.dumps(FEATURES))

print("AUC:", roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
print("Default rate:", y.mean())