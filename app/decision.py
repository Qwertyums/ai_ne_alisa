# Пороги PD подобраны под модель, обученную на данных малого и микробизнеса (дефолтов ~7 %):
# на отложенной выборке ~79 % одобрений, ~14 % ручной проверки, ~7 % отказов.
PD_REVIEW = 0.10
PD_REJECT = 0.25


def hard_rules(data: dict, f: dict) -> list[dict]:
    rules = [
        ("BLOCKED_ACCOUNT", data["fns"]["account_blocked"],
         "По данным ФНС заблокирован расчётный счёт."),
        ("SEVERE_OVERDUE", f["overdue_days_max"] > 90,
         "Есть просрочка более 90 дней по действующим обязательствам."),
        ("HIGH_ENFORCEMENT", f["bailiff_ratio"] > 0.2,
         "Сумма исполнительных производств превышает 20% годовой выручки."),
        ("TOO_YOUNG", data["biz_age_months"] < 6,
         "Бизнес существует менее 6 месяцев — нет достаточной истории оборотов."),
    ]
    return [{"code": code, "message": msg} for code, bad, msg in rules if bad]


def decide(data: dict, f: dict, score: dict, amount: float) -> dict:
    rules = hard_rules(data, f)
    pd = score["probability_of_default"]

    if rules:
        decision = "REJECT"
    elif pd > PD_REJECT:
        decision = "REJECT"
    elif pd > PD_REVIEW:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    monthly_income = f["revenue_ttm"] / 12
    limit = max(0, int(monthly_income * 2 * (1 - pd) / 50_000) * 50_000)

    conditions = []
    if f["top1_share"] > 0.5:
        conditions.append("Запросить договоры с ключевым контрагентом (доля > 50%).")
    if f["volatility"] > 0.35:
        conditions.append("Высокая волатильность оборотов: лимит на овердрафт не более 1 месяца.")
    if not f["collateral"] and pd > 0.08:
        conditions.append("Поручительство собственника.")
    if amount > limit > 0:
        conditions.append(f"Снизить запрашиваемую сумму до {limit:,} ₽.")

    reasons = [r["message"] for r in rules]
    if not rules and decision == "REJECT":
        reasons.append(f"Модель оценила вероятность дефолта в {pd:.1%} — выше допустимого порога {PD_REJECT:.0%}.")
    if not rules and decision == "REVIEW":
        reasons.append("Требуется ручная проверка: риск на границе допустимого.")

    return {
        "decision": decision,
        "decision_source": "rule_engine + ml",  # LLM здесь не участвует
        "rules_triggered": rules,
        "reasons": reasons,
        "conditions": conditions,
        "max_amount": limit,
        "alternative_products": (
            ["Факторинг", "Кредит под поручительство"] if decision != "APPROVE" else []
        ),
    }