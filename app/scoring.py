import json
import math
from pathlib import Path

import numpy as np

from .datasource import FEATURES

# Эвристика ONLY для демо, пока CatBoost не обучен.
# Не является кредитной моделью и не калибрована.
COEF = {
    "bk_score": -0.006, "overdue_days_max": 0.02, "inquiries_90d": 0.05,
    "revenue_yoy": -1.2, "volatility": 3.0, "avg_balance_ratio": -0.35,
    "cash_gap_days_90": 0.03, "dsr": 2.0, "tax_delay_days": 0.012,
    "account_blocked": 1.5, "lawsuit_ratio": 2.0, "bailiff_ratio": 4.0,
    "owner_bailiff": 0.8, "biz_age_months": -0.004, "top1_share": 1.2,
    "counterparties": -0.01, "sector_risk": 0.5,
    "requested_to_revenue": 1.0, "term_months": 0.008, "collateral": -0.7,
}
NEUTRAL = {
    "bk_score": 650, "overdue_days_max": 0, "inquiries_90d": 2,
    "revenue_yoy": 0.05, "volatility": 0.15, "avg_balance_ratio": 1.0,
    "cash_gap_days_90": 0, "dsr": 0.15, "tax_delay_days": 0,
    "account_blocked": 0, "lawsuit_ratio": 0, "bailiff_ratio": 0,
    "owner_bailiff": 0, "biz_age_months": 36, "top1_share": 0.2,
    "counterparties": 50, "sector_risk": 0.3,
    "requested_to_revenue": 0.1, "term_months": 12, "collateral": 0,
}


class Scorer:
    def __init__(self, model_dir: Path = Path("models")):
        self.model = None
        self.method = "heuristic"
        self.explainer = None
        path = model_dir / "catboost.cfm"
        if path.exists():
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(str(path))
            self.method = "catboost"
            try:
                import shap
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                pass

    def score(self, features: dict) -> dict:
        x = np.array([[features[k] for k in FEATURES]])

        if self.model is not None:
            pd = float(self.model.predict_proba(x)[0, 1])
            contributions = None
            if self.explainer is not None:
                sv = self.explainer.shap_values(x)
                values = sv[1] if isinstance(sv, list) else sv
                contributions = values[0]
            method = "shap" if contributions is not None else "catboost_no_attribution"
        else:
            z = sum(COEF.get(k, 0) * (features[k] - NEUTRAL.get(k, 0)) for k in FEATURES)
            pd = 1 / (1 + math.exp(-max(-20, min(20, -2.2 + z))))
            contributions = np.array([
                COEF.get(k, 0) * (features[k] - NEUTRAL.get(k, 0)) for k in FEATURES
            ])
            method = "heuristic"

        drivers = []
        if contributions is not None:
            for k, v in zip(FEATURES, contributions):
                if k == "revenue_ttm":  # масштабный признак, не интерпретируем напрямую
                    continue
                drivers.append({
                    "feature": k,
                    "value": features[k],
                    "contribution": round(float(v), 4),
                    "direction": "risk_up" if v > 0 else "risk_down",
                })
            drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)

        return {
            "probability_of_default": round(pd, 5),
            "internal_score": int(round(1000 * (1 - pd))),
            "model_type": self.method,
            "attribution_method": method,
            "top_drivers": drivers[:8],
        }