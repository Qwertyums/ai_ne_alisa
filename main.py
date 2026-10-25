import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.datasource import FEATURES, enrich, build_features
from app.scoring import Scorer
from app.decision import decide
from app.llm import explain, load_local_model

SCORER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SCORER
    SCORER = Scorer(Path("models"))

    load_local_model()

    yield


app = FastAPI(title="Smart Credit Cabinet — Mock Scoring", version="0.1", lifespan=lifespan)


class Application(BaseModel):
    inn: str = Field(pattern=r"^\d{10,12}$")
    amount: float = Field(gt=0)
    term_months: int = Field(default=12, ge=1, le=60)
    purpose: Literal["working_capital", "investment", "refinance"] = "working_capital"
    collateral: bool = False


class WhatIf(BaseModel):
    application: Application
    overrides: dict[str, float | bool] = Field(default_factory=dict)


def run(app_req: Application, overrides: dict | None = None):
    data = enrich(app_req.inn)
    f = build_features(data, app_req.amount, app_req.term_months, app_req.collateral)
    for k, v in (overrides or {}).items():
        if k not in FEATURES:
            raise HTTPException(400, f"Неизвестный признак: {k}")
        f[k] = float(v)
    score = SCORER.score(f)
    decision = decide(data, f, score, app_req.amount)
    return {
        "application": app_req.model_dump(),
        "data_source": data["_source"],
        "features": f,
        "score": score,
        "decision": decision,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": SCORER.method if SCORER else "not_loaded"}


@app.get("/v1/enrichment/{inn}")
def enrichment(inn: str):
    if not inn.isdigit() or len(inn) not in (10, 12):
        raise HTTPException(400, "ИНН должен содержать 10 цифр (для юрлиц) или 12 цифр (для ИП/физлиц)")
    return enrich(inn)


@app.post("/v1/decision")
def decision(req: Application):
    return run(req)


@app.post("/v1/what-if")
def what_if(req: WhatIf):
    """Например: {"collateral": true} или {"volatility": 0.15}."""
    return run(req.application, req.overrides)


@app.post("/v1/explain")
def explain_endpoint(payload: dict):
    if "decision" not in payload or "score" not in payload:
        raise HTTPException(400, "Передайте ответ /v1/decision как есть")
    return explain(payload)


@app.get("/v1/features")
def features():
    return FEATURES


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)