"""Признаки -> CatBoost (до ответа и после) -> пороги по ложным срабатываниям -> отчёт.

    python -m antifraud.train
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from .features import CAT_FEATURES, Call, FeatureStore
from .generate_data import DATA_DIR, generate
from .model import MODEL_DIR, STAGE_FEATURES, to_frame
from .policy import Thresholds, decide

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
# Целиком в тесте: модель не видела этот сценарий — проверка обобщения.
HELD_OUT_SCENARIOS = {"simbank_slow"}
# Пороги подбираем по квантилям легальных баллов на валидации (доля ложных).
FPR_TARGETS = {"label": 0.01, "review": 0.001, "block": 0.0002}


def build_features(calls: pd.DataFrame) -> pd.DataFrame:
    store = FeatureStore()
    rows = []
    t0 = time.time()
    for rec in calls.to_dict("records"):
        call = Call.from_mapping(rec)
        feats = store.features(call, "post")
        ctx = store.context(call)
        feats["b_in_emergency_callback_window"] = ctx["b_in_emergency_callback_window"]
        feats["b_called_emergency_hours_ago"] = ctx["b_called_emergency_hours_ago"]
        rows.append(feats)
        store.update(call)
    print(f"признаки: {len(rows)} вызовов за {time.time() - t0:.0f} c")
    feats = pd.DataFrame(rows)
    raw = calls.reset_index(drop=True).drop(columns=[c for c in feats.columns if c in calls.columns])
    return pd.concat([raw, feats], axis=1)


def split(df: pd.DataFrame) -> pd.Series:
    """По сущностям (абонент, шлюз, кампания), стратифицированно по сценарию и подтипу.

    Нельзя делить по строкам: звонки одного мошенника попадут и в train, и в test,
    и оценка будет завышена. Страты (simbank|shared / simbank|per_sim) нужны, чтобы
    оба типа шлюзов попали в обучение — иначе модель «срезает угол» на одном признаке.
    """
    ent = df[["entity", "scenario"]].drop_duplicates("entity").copy()
    # entity вида "gw:shared:IMEI" → stratum "simbank|gw:shared"
    ent["stratum"] = ent.scenario + "|" + ent.entity.str.rpartition(":")[0]
    ent["h"] = ent.entity.map(lambda e: hashlib.md5(str(e).encode()).hexdigest())
    ent["q"] = ent.groupby("stratum")["h"].rank(method="first", pct=True)
    ent["part"] = np.where(ent.q <= 0.56, "fit", np.where(ent.q <= 0.7, "val", "test"))
    part = df.entity.map(ent.set_index("entity")["part"])
    part[df.scenario.isin(HELD_OUT_SCENARIOS)] = "test"
    part[df.warmup] = "warmup"
    return part


def train_stage(stage: str, fit: pd.DataFrame, val: pd.DataFrame) -> CatBoostClassifier:
    # rsm=0.4: в каждом дереве случайная доля признаков — модель не залипает на одном.
    # Ранняя остановка по Logloss, не по AUC: AUC на синтетике слишком рано становится 1.0.
    model = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05, l2_leaf_reg=5, rsm=0.4,
        loss_function="Logloss", od_type="Iter", od_wait=200, random_seed=7, verbose=0,
    )
    model.fit(
        Pool(to_frame(fit, stage), fit.label, cat_features=CAT_FEATURES),
        eval_set=Pool(to_frame(val, stage), val.label, cat_features=CAT_FEATURES),
        use_best_model=True,
    )
    return model


def calibrate(scores: np.ndarray, legit: np.ndarray) -> dict[str, float]:
    """Порог = квантиль баллов легальных вызовов: ровно (1 − FPR) из них ниже порога.

    block не ниже 90: поведенческий балл сам по себе не должен включать автоблокировку
    (см. policy: до уровня 3 доводит только подмена маршрута).
    """
    legit_scores = scores[legit]
    th = {name: float(np.quantile(legit_scores, 1 - fpr)) for name, fpr in FPR_TARGETS.items()}
    th["review"] = max(th["review"], th["label"])
    th["block"] = max(th["block"], th["review"], 90.0)
    return {k: round(v, 2) for k, v in th.items()}


def apply_policy(df: pd.DataFrame, scores: np.ndarray, thresholds: Thresholds, stage: str) -> np.ndarray:
    levels = np.zeros(len(df), dtype=int)
    for i, (rec, score) in enumerate(zip(df.to_dict("records"), scores)):
        call = Call.from_mapping(rec)
        ctx = {k: rec[k] for k in ("b_in_emergency_callback_window", "b_called_emergency_hours_ago")}
        levels[i] = decide(call, float(score), rec, ctx, thresholds, stage).level
    return levels


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Пороги из исходного Падабук.xlsx, как они записаны."""
    return pd.DataFrame({
        "foreign_gateway": df.cli_trunk_mismatch == 1,
        "short_calls": (df.short_ratio > 0.8) & (df.calls_24h >= 10),
        "burst": (df.calls_1h > 60) | (df.calls_1m > 5),
        "unique_contacts": df.unique_b_ratio_24h > 0.95,
        "regions": df.regions_count_1h > 15,
        "imsi_change": df.imsi_per_imei_24h > 5,
        "night": (df.hour_local_b >= 1) & (df.hour_local_b < 5),
        "in_out": (df.in_out_ratio_24h < 0.02) & (df.calls_24h >= 100),
        "emergency_spoof": df.short_cli_from_foreign == 1,
    })


def rates(flag: np.ndarray, y: np.ndarray) -> dict[str, float]:
    flag = np.asarray(flag, dtype=bool)
    return {
        "recall": round(float(flag[y == 1].mean()), 4),
        "fpr": round(float(flag[y == 0].mean()), 5),
    }


def load_dataset() -> pd.DataFrame:
    calls_path = DATA_DIR / "calls.csv"
    cache = DATA_DIR / "features.pkl"
    if not calls_path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        generate().to_csv(calls_path, index=False)
    if cache.exists() and cache.stat().st_mtime > calls_path.stat().st_mtime:
        return pd.read_pickle(cache)
    calls = pd.read_csv(calls_path, parse_dates=["ts"], dtype={
        "a_number": str, "b_number": str, "imei": str, "imsi": str, "trunk_id": str, "entity": str,
    })
    df = build_features(calls)
    df.to_pickle(cache)
    return df


def main() -> None:
    df = load_dataset()
    df["part"] = split(df)
    fit, val, test = (df[df.part == p] for p in ("fit", "val", "test"))
    print({p: int((df.part == p).sum()) for p in ("warmup", "fit", "val", "test")})

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    meta: dict = {"thresholds": {}, "features": STAGE_FEATURES, "fpr_targets": FPR_TARGETS}
    report: dict = {"rows": {p: int((df.part == p).sum()) for p in ("fit", "val", "test")}, "stages": {}}
    # Обучаемся на label (с шумом жалоб), оцениваемся на is_fraud (истинный сценарий).
    y = test.is_fraud.to_numpy()

    for stage in ("pre", "post"):
        print(f"\n=== модель {stage} ===")
        model = train_stage(stage, fit, val)
        model.save_model(str(MODEL_DIR / f"{stage}.cbm"))
        val_scores = model.predict_proba(to_frame(val, stage))[:, 1] * 100
        th = calibrate(val_scores, (val.is_fraud == 0).to_numpy())
        meta["thresholds"][stage] = th

        scores = model.predict_proba(to_frame(test, stage))[:, 1] * 100
        levels = apply_policy(test, scores, Thresholds.from_dict(th), stage)
        by_scenario = {}
        for scen, idx in test.groupby("scenario").groups.items():
            lv = levels[test.index.get_indexer(idx)]
            by_scenario[scen] = {
                "calls": int(len(lv)),
                "level>=1": round(float((lv >= 1).mean()), 4),
                "level>=2": round(float((lv >= 2).mean()), 4),
                "level3": int((lv == 3).sum()),
            }
        importance = model.get_feature_importance(prettified=True).head(12)
        report["stages"][stage] = {
            "auc": round(float(roc_auc_score(y, scores)), 4),
            "thresholds": th,
            "level>=1": rates(levels >= 1, y),
            "level>=2": rates(levels >= 2, y),
            "level3": rates(levels == 3, y),
            "by_scenario": by_scenario,
            "top_features": dict(zip(importance["Feature Id"], importance["Importances"].round(2))),
        }

    flags = rule_flags(test)
    n_flags = flags.sum(axis=1).to_numpy()
    report["rules_baseline"] = {
        "any_flag": rates(n_flags >= 1, y),
        "two_flags": rates(n_flags >= 2, y),
        "per_rule": {c: rates(flags[c].to_numpy(), y) for c in flags.columns},
        "any_flag_by_scenario": {
            scen: round(float((n_flags[test.index.get_indexer(idx)] >= 1).mean()), 4)
            for scen, idx in test.groupby("scenario").groups.items()
        },
    }

    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "metrics.md").write_text(render_report(report), encoding="utf-8")
    print("\n" + render_report(report))


def render_report(r: dict) -> str:
    lines = ["# Качество на тесте", "",
             "Тест — абоненты, шлюзы и кампании, которых не было в обучении. "
             "Сценарий simbank_slow целиком отложен: модель его не видела.", ""]
    for stage, s in r["stages"].items():
        title = "До ответа (можно блокировать)" if stage == "pre" else "После вызова (есть длительность)"
        lines += [f"## {title}", "",
                  f"AUC {s['auc']}. Пороги: метка {s['thresholds']['label']}, "
                  f"проверка {s['thresholds']['review']}, блок {s['thresholds']['block']}.", "",
                  "| Действие | Охват мошенников | Ложные срабатывания |", "|---|---|---|"]
        for key, name in (("level>=1", "Метка и выше"), ("level>=2", "Проверка и выше"), ("level3", "Автоблокировка")):
            lines.append(f"| {name} | {s[key]['recall']:.1%} | {s[key]['fpr']:.3%} |")
        lines += ["", "| Сценарий | Вызовов | Метка и выше | Проверка и выше | Автоблокировок |", "|---|---|---|---|---|"]
        for scen, v in sorted(s["by_scenario"].items()):
            lines.append(f"| {scen} | {v['calls']} | {v['level>=1']:.1%} | {v['level>=2']:.1%} | {v['level3']} |")
        lines.append("")
    b = r["rules_baseline"]
    lines += ["## Бейзлайн: пороги из исходной таблицы", "",
              f"Хотя бы одно правило: охват {b['any_flag']['recall']:.1%}, ложные {b['any_flag']['fpr']:.2%}. "
              f"Два и больше: охват {b['two_flags']['recall']:.1%}, ложные {b['two_flags']['fpr']:.2%}.", "",
              "| Правило | Охват | Ложные |", "|---|---|---|"]
    for name, v in b["per_rule"].items():
        lines.append(f"| {name} | {v['recall']:.1%} | {v['fpr']:.2%} |")
    lines += ["", "| Сценарий | Сработало хоть одно правило |", "|---|---|"]
    for scen, v in sorted(b["any_flag_by_scenario"].items()):
        lines.append(f"| {scen} | {v:.1%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
