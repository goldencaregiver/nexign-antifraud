"""Перевод SHAP-вкладов в баллы и фразы для оператора."""
from __future__ import annotations
import math
from typing import Any

CONNECTION_RU = {
    "mobile": "мобильная сеть",
    "landline": "городская сеть",
    "voip": "VoIP",
    "intl": "международный шлюз"
}

def _plural(n: float, one: str, few: str, many: str) -> str:
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        word = one
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = few
    else:
        word = many
    return f"{n} {word}"

def _calls(n: float) -> str:
    return _plural(n, "вызов", "вызова", "вызовов")

def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"

def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))

def describe(name: str, v: Any, feats: dict[str, Any]) -> str:
    if _missing(v):
        return {
            "short_ratio": "Нет истории вызовов номера",
            "unanswered_ratio": "Нет истории вызовов номера",
            "ring_drop_ratio": "Нет истории вызовов номера",
            "unique_b_ratio_24h": "Нет вызовов за сутки",
            "reciprocity": "Никому ещё не звонил",
            "in_out_ratio_24h": "Нет исходящих за сутки",
            "fan_b_overlap": "Одно устройство — один номер",
        }.get(name, f"{name}: нет данных")

    # Заменяем match-case на словарь с функциями для совместимости с Python 3.9
    rules = {
        "cli_trunk_mismatch": lambda: "Номер не совпадает со страной входящего шлюза" if v else "Номер и шлюз из одной страны",
        "short_cli_from_foreign": lambda: "Служебный номер пришёл из-за рубежа или через VoIP" if v else "Не служебный номер",
        "a_is_landline": lambda: "Звонит городской номер" if v else "Звонит не городской номер",
        "connection_type": lambda: f"Тип соединения: {CONNECTION_RU.get(v, v)}",
        "source_operator": lambda: f"Источник: {str(v).replace('intl', 'зарубежный шлюз')}",
        "hour_local_b": lambda: f"У получателя {int(v):02d}:{int((v - int(v)) * 60):02d} по местному времени",
        "pair_calls_before": lambda: f"Ранее общались {_plural(v, 'раз', 'раза', 'раз')}" if v else "Ранее не общались",
        "b_called_a_before": lambda: "Получатель сам звонил этому номеру" if v else "Получатель никогда не звонил этому номеру",
        "number_age_days": lambda: (
            "Номер раньше не встречался" if v <= 0 
            else f"Номер появился {v * 24:.0f} ч назад" if v < 1 
            else f"Номер активен {_plural(v, 'день', 'дня', 'дней')}"
        ),
        "calls_1m": lambda: f"{_calls(v)} за последнюю минуту",
        "calls_1h": lambda: f"{_calls(v)} за час",
        "calls_24h": lambda: f"{_calls(v)} за сутки",
        "short_ratio": lambda: f"{_pct(v)} разговоров короче 3 сек",
        "unanswered_ratio": lambda: f"{_pct(v)} вызовов без ответа",
        "ring_drop_ratio": lambda: f"{_pct(v)} сбросов до ответа",
        "unique_b_ratio_24h": lambda: f"{_pct(v)} собеседников за сутки — разные номера",
        "regions_count_1h": lambda: f"Регионов получателей за час: {int(v)}",
        "regions_entropy_1h": lambda: f"Равномерность охвата регионов {v:.2f}",
        "reciprocity": lambda: f"{_pct(v)} собеседников перезванивали",
        "in_out_ratio_24h": lambda: f"Входящих на один исходящий: {v:.2f}",
        "imsi_per_imei_24h": lambda: f"SIM-карт в одном устройстве за сутки: {int(v)}" if v else "Устройство не передано",
        "gateway_tac": lambda: "Устройство — модель GSM-шлюза" if v else "Обычное устройство",
        "fan_group_size_24h": lambda: f"Номеров с одного устройства: {int(v)}",
        "fan_b_overlap": lambda: f"Пересечение списков обзвона {_pct(v)}",
        "trunk_distinct_a_1h": lambda: f"Разных номеров через этот шлюз за час: {int(v)}",
        "answered": lambda: "Вызов отвечен" if v else "Вызов без ответа",
        "duration_sec": lambda: f"Длительность {v:.0f} сек",
        "short_call": lambda: "Разговор короче 3 сек" if v else "Разговор дольше 3 сек",
        "ring_drop": lambda: "Сброс до ответа" if v else "Без сброса до ответа",
    }

    if name in rules:
        return rules[name]()
    
    return f"{name} = {v}"

def to_points(shap_row, score: float, base_score: float) -> list[float]:
    """SHAP у CatBoost — вклады в логит, а оператору нужны баллы 0–100."""
    total = float(sum(shap_row))
    if abs(total) < 1e-9:
        return [0.0 for _ in shap_row]
    scale = (score - base_score) / total
    return [float(s) * scale for s in shap_row]

def factors(names: list[str], feats: dict[str, Any], points: list[float], top: int = 6) -> list[dict[str, Any]]:
    order = sorted(range(len(names)), key=lambda i: -abs(points[i]))
    out = []
    for i in order[:top]:
        if abs(points[i]) < 0.5:
            continue
        name = names[i]
        out.append({
            "feature": name, 
            "text": describe(name, feats.get(name), feats),
            "points": round(points[i], 1), 
            "value": feats.get(name)
        })
    return out