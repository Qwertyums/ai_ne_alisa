import random
from statistics import mean, pstdev

FEATURES = [
    "bk_score", "overdue_days_max", "inquiries_90d",
    "revenue_ttm", "revenue_yoy", "volatility",
    "avg_balance_ratio", "cash_gap_days_90", "dsr",
    "tax_delay_days", "account_blocked",
    "lawsuit_ratio", "bailiff_ratio", "owner_bailiff",
    "biz_age_months", "top1_share", "counterparties", "sector_risk",
    "requested_to_revenue", "term_months", "collateral",
]

SCENARIOS = {
    "7700000001": "healthy",   # примерно одобрение
    "7700000002": "stress",    # примерно отказ / review
    "7700000003": "blocked",   # hard rule: блокировка счёта
    "770000000001": "healthy_ip",
    "770000000002": "stress_ip",
}


def enrich(inn: str) -> dict:
    """Имитация ЕГРЮЛ + БКИ + ФНС + картотеки АС + банковской выписки.
    Детерминировано по ИНН: один ИНН = один и тот же профиль.
    """
    rng = random.Random(int(inn))
    scenario = SCENARIOS.get(inn, "random")
    bad = {"healthy": 0.05, "stress": 0.85, "blocked": 0.8}.get(scenario, rng.random())

    revenue_base = rng.uniform(600_000, 8_000_000)
    age_months = rng.randint(8, 110)

    # --- банковская выписка за 12 месяцев ---
    months, balance = [], revenue_base * 1.5
    trend = rng.uniform(-0.01, 0.025) - 0.02 * bad
    for i in range(12):
        income = revenue_base * (1 + trend) ** i * rng.uniform(0.78, 1.22)
        expense = income * rng.uniform(0.72, 1.02) + revenue_base * 0.08 * bad
        balance = max(0.0, balance * 0.65 + (income - expense) * 0.35)
        months.append({
            "month": f"2025-{i + 1:02d}",
            "income": round(income),
            "expense": round(expense),
            "closing_balance": round(balance),
            "cash_gap_days": rng.randint(4, 18) if expense > income else 0,
        })

    # --- БКИ ---
    bki = {
        "score": int(max(350, min(850, 730 - 260 * bad + rng.uniform(-45, 45)))),
        "active_loans": rng.randint(0, 4),
        "monthly_payment": round(revenue_base * rng.uniform(0.03, 0.12)),
        "overdue_days_max": int(bad * rng.uniform(20, 140)),
        "inquiries_90d": rng.randint(0, 12),
    }

    # --- ФНС ---
    fns = {
        "tax_regime": rng.choice(["УСН 6%", "УСН 15%", "ОСНО"]),
        "tax_delay_days": int(bad * rng.uniform(0, 70)),
        "account_blocked": scenario == "blocked",
        "checks_12m": rng.randint(0, 2),
    }

    # --- суды / приставы ---
    legal = {
        "as_defendant_cases": rng.randint(0, 8),
        "claims_amount": round(revenue_base * 12 * bad * rng.uniform(0.01, 0.12)),
        "bailiff_amount": round(revenue_base * 12 * bad * rng.uniform(0.0, 0.08)),
        "owner_has_bailiff": rng.random() < bad * 0.6,
    }

    return {
        "inn": inn,
        "ogrn": "1" + inn + "0",
        "name": f"ООО «Проект {inn[-4:]}»",
        "region": rng.choice(["77", "78", "50", "66", "16"]),
        "okved": rng.choice(["46.90", "41.20", "62.01", "56.10", "49.41"]),
        "biz_age_months": age_months,
        "bki": bki,
        "fns": fns,
        "legal": legal,
        "bank_statement": months,
        "counterparties": int(90 - 60 * bad + rng.uniform(-10, 20)),
        "top1_share": min(0.95, 0.15 + 0.6 * bad + rng.uniform(-0.05, 0.1)),
        "sector_risk": round(rng.uniform(0.1, 0.9), 3),
        "_source": "MOCK — синтетические данные, не реальные сведения",
    }


def build_features(data: dict, amount: float, term_months: int, collateral: bool) -> dict:
    st = data["bank_statement"]
    incomes = [m["income"] for m in st]
    balances = [m["closing_balance"] for m in st]
    revenue_ttm = sum(incomes)
    avg_monthly = revenue_ttm / 12

    first6, last6 = sum(incomes[:6]), sum(incomes[6:])
    f = {
        "bk_score": data["bki"]["score"],
        "overdue_days_max": data["bki"]["overdue_days_max"],
        "inquiries_90d": data["bki"]["inquiries_90d"],
        "revenue_ttm": revenue_ttm,
        "revenue_yoy": (last6 / first6 - 1) if first6 else 0.0,
        "volatility": pstdev(incomes) / mean(incomes) if mean(incomes) else 1.0,
        "avg_balance_ratio": mean(balances) / avg_monthly if avg_monthly else 0.0,
        "cash_gap_days_90": sum(m["cash_gap_days"] for m in st[-3:]),
        "dsr": data["bki"]["monthly_payment"] / avg_monthly if avg_monthly else 1.0,
        "tax_delay_days": data["fns"]["tax_delay_days"],
        "account_blocked": int(data["fns"]["account_blocked"]),
        "lawsuit_ratio": data["legal"]["claims_amount"] / revenue_ttm if revenue_ttm else 0,
        "bailiff_ratio": data["legal"]["bailiff_amount"] / revenue_ttm if revenue_ttm else 0,
        "owner_bailiff": int(data["legal"]["owner_has_bailiff"]),
        "biz_age_months": data["biz_age_months"],
        "top1_share": data["top1_share"],
        "counterparties": data["counterparties"],
        "sector_risk": data["sector_risk"],
        "requested_to_revenue": amount / revenue_ttm if revenue_ttm else 10.0,
        "term_months": term_months,
        "collateral": int(collateral),
    }
    return {k: round(float(f[k]), 6) for k in FEATURES}