"""Загрузка моделей и скоринг одного вызова: балл, факторы, решение."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier, Pool

from . import explain, numbering as nb
from .features import CAT_FEATURES, POST_FEATURES, PRE_FEATURES, Call, FeatureStore
from .policy import Thresholds, decide

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
STAGE_FEATURES = {"pre": PRE_FEATURES, "post": POST_FEATURES}


def to_frame(rows: list[dict[str, Any]] | pd.DataFrame, stage: str) -> pd.DataFrame:
    cols = STAGE_FEATURES[stage]
    df = pd.DataFrame(rows)[cols].copy()
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    return df


class Scorer:
    def __init__(self, model_dir: Path = MODEL_DIR):
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.models: dict[str, CatBoostClassifier] = {}
        self.thresholds: dict[str, Thresholds] = {}
        for stage in ("pre", "post"):
            m = CatBoostClassifier()
            m.load_model(str(model_dir / f"{stage}.cbm"))
            self.models[stage] = m
            self.thresholds[stage] = Thresholds.from_dict(meta["thresholds"][stage])

    def predict(self, stage: str, X: pd.DataFrame):
        return self.models[stage].predict_proba(X)[:, 1] * 100

    def score(self, call: Call, store: FeatureStore, stage: str | None = None) -> dict[str, Any]:
        stage = stage or ("post" if call.completed else "pre")
        feats = store.features(call, stage)
        ctx = store.context(call)
        X = to_frame([feats], stage)
        score = float(self.predict(stage, X)[0])
        shap = self.models[stage].get_feature_importance(
            Pool(X, cat_features=CAT_FEATURES), type="ShapValues"
        )[0]
        base_score = 100 / (1 + math.exp(-float(shap[-1])))
        points = explain.to_points(shap[:-1], score, base_score)
        decision = decide(call, score, feats, ctx, self.thresholds[stage], stage)
        return {
            "stage": stage,
            "call": {
                "a": nb.mask(call.a_number),
                "b": nb.mask(call.b_number),
                "ts": call.ts.isoformat(timespec="seconds"),
                "duration_sec": call.duration_sec if stage == "post" else None,
                "ingress_country": call.ingress_country,
                "trunk_id": call.trunk_id,
            },
            "score": round(score),
            "base_score": round(base_score, 1),
            "thresholds": vars(self.thresholds[stage]),
            "factors": explain.factors(STAGE_FEATURES[stage], feats, points),
            "decision": decision.as_dict(),
            "features": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in feats.items()},
        }
