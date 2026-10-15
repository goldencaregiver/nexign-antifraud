"""HTTP-сервис скоринга.
uvicorn antifraud.api:app --port 8000
"""
from __future__ import annotations
from datetime import datetime
from functools import lru_cache
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .features import Call, FeatureStore
from .model import Scorer
from .presets import PRESETS, build_store

app = FastAPI(title="Антифрод по метаданным вызова", version="1.0")
live_store = FeatureStore()

class CallIn(BaseModel):
    a_number: str = Field(examples=["+74951234512"])
    b_number: str = Field(examples=["+79169876583"])
    ts: Optional[datetime] = None
    ingress_country: str = "RU"
    trunk_id: str = ""
    connection_type: Literal["mobile", "landline", "voip", "intl"] = "mobile"
    imei: Optional[str] = None
    imsi: Optional[str] = None
    outcome: Optional[Literal["answered", "no_answer", "rejected", "ring_drop"]] = None
    duration_sec: Optional[float] = None
    stage: Optional[Literal["pre", "post"]] = Field(
        None, description="pre — до ответа, когда можно блокировать; post — после вызова. По умолчанию по наличию outcome."
    )
    history: Literal["live", "fraud", "normal", "ambulance"] = Field(
        "live", description="live — накопленная история сервиса; остальные — готовые истории для демо."
    )
    commit: bool = Field(False, description="Записать вызов в live-историю после скоринга.")

@lru_cache(maxsize=1)
def scorer() -> Scorer:
    return Scorer()

@app.get("/health")
def health() -> dict:
    s = scorer()
    return {"status": "ok", "thresholds": {k: vars(v) for k, v in s.thresholds.items()}}

@app.get("/presets")
def presets() -> dict:
    return PRESETS

@app.post("/score")
def score(body: CallIn) -> dict:
    call = Call(
        ts=body.ts or datetime.now(), a_number=body.a_number, b_number=body.b_number,
        ingress_country=body.ingress_country, trunk_id=body.trunk_id,
        connection_type=body.connection_type, imei=body.imei, imsi=body.imsi,
        outcome=body.outcome, duration_sec=body.duration_sec,
    )
    if body.stage == "post" and call.outcome is None:
        call.outcome = "answered"
    if body.stage == "pre":
        call.outcome, call.duration_sec = None, None
        
    try:
        store = live_store if body.history == "live" else build_store(body.history, call)
    except KeyError:
        raise HTTPException(400, f"Неизвестная история {body.history}")
        
    result = scorer().score(call, store, body.stage)
    
    if body.commit and body.history == "live":
        if call.outcome is None:
            call.outcome = "answered"
        live_store.update(call)
        
    return result