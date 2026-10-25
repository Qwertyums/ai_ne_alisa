"""Обучение CatBoost на синтетических данных малого и микробизнеса.

Генератор признаков — реалистичные для сегмента распределения (выручка от 1,5 до 800 млн ₽,
медиана ~30 млн; у большинства нет просрочек и судов). Коэффициенты в logit и состав
признаков (features.json) не меняются — сайт и сервис подходят к модели без правок.
Свободный член подобран так, чтобы доля дефолтов была ~7 %.

Выборки: обучение 70 %, валидация 15 % (ранняя остановка), отложенная 15 % (итоговый AUC).

Запуск:  python train.py
Подбор свободного члена:  INTERCEPT=-5.0 python train.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from app.datasource import FEATURES

SEED = 42
N = 40_000
INTERCEPT = float(os.environ.get("INTERCEPT", "-4.05"))

rng = np.random.default_rng(SEED)


def lognormal(median, sigma, low, high, size):
    return np.clip(np.exp(rng.normal(np.log(median), sigma, size)), low, high)


def mostly_zero(p_zero, values):
    """У доли p_zero значение 0, у остальных — из values."""
    return np.where(rng.random(len(values)) < p_zero, 0, values)


df = pd.DataFrame({
    # БКИ: центр ~690, хвост плохих историй до 300
    "bk_score": np.clip(rng.normal(690, 80, N), 300, 850).round(),
    # у ~75 % просрочек нет, у остальных — хвост до 120 дней
    "overdue_days_max": mostly_zero(0.75, np.clip(rng.exponential(20, N), 1, 120).round()),
    "inquiries_90d": np.clip(rng.poisson(2, N), 0, 12),
    # выручка: медиана ~30 млн ₽, микро до 120 млн, малое до 800 млн
    "revenue_ttm": lognormal(30e6, 1.0, 1.5e6, 800e6, N),
    "revenue_yoy": np.clip(rng.normal(0.08, 0.22, N), -0.5, 1.0),
    # у микробизнеса обороты неровные: центр ~0,25
    "volatility": 0.05 + 0.75 * rng.beta(2.2, 6, N),
    "avg_balance_ratio": lognormal(0.7, 0.7, 0.05, 3, N),
    "cash_gap_days_90": mostly_zero(0.7, rng.integers(1, 46, N)),
    # долговая нагрузка: медиана ~0,12, максимум 0,8
    "dsr": 0.8 * rng.beta(1.6, 8, N),
    "tax_delay_days": mostly_zero(0.85, rng.integers(1, 81, N)),
    "account_blocked": (rng.random(N) < 0.03).astype(int),
    "lawsuit_ratio": mostly_zero(0.85, rng.uniform(0, 0.3, N)),
    "bailiff_ratio": mostly_zero(0.9, rng.uniform(0, 0.25, N)),
    "owner_bailiff": (rng.random(N) < 0.08).astype(int),
    "biz_age_months": lognormal(40, 0.9, 3, 240, N).round(),
    # у микробизнеса крупный клиент даёт большую долю выручки: центр ~0,38
    "top1_share": 0.05 + 0.9 * rng.beta(2, 3.5, N),
    "counterparties": lognormal(25, 1.0, 1, 300, N).round(),
    "sector_risk": rng.uniform(0.1, 0.9, N),
    "requested_to_revenue": lognormal(0.12, 0.8, 0.01, 1.0, N),
    "term_months": rng.integers(1, 61, N),
    "collateral": (rng.random(N) < 0.25).astype(int),
})

# Коэффициенты — как в исходной модели; меняется только свободный член.
logit = (
    INTERCEPT
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
    + rng.normal(0, 0.6, N)
)
p = 1 / (1 + np.exp(-logit))
df["default"] = rng.binomial(1, p)
print(f"INTERCEPT: {INTERCEPT}  Default rate: {df['default'].mean():.4f}")

X, y = df[FEATURES], df["default"]
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_hold, y_val, y_hold = train_test_split(
    X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED
)

model = CatBoostClassifier(
    iterations=400, depth=5, learning_rate=0.05,
    loss_function="Logloss", eval_metric="AUC",
    allow_writing_files=False, verbose=100, random_seed=SEED,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40)

out = Path("models")
out.mkdir(exist_ok=True)
model.save_model(out / "catboost.cfm")
(out / "features.json").write_text(json.dumps(FEATURES))

auc = lambda X_, y_: roc_auc_score(y_, model.predict_proba(X_)[:, 1])  # noqa: E731
print(f"Деревьев: {model.tree_count_}")
print(f"AUC обучение: {auc(X_train, y_train):.4f}")
print(f"AUC валидация (ранняя остановка): {auc(X_val, y_val):.4f}")
print(f"AUC отложенная выборка: {auc(X_hold, y_hold):.4f}")
print(f"Default rate: {y.mean():.4f} (обучение {y_train.mean():.4f}, "
      f"валидация {y_val.mean():.4f}, отложенная {y_hold.mean():.4f})")
