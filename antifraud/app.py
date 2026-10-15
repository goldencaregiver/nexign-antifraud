"""Интерфейс оператора: форма + имитация входящего звонка.
streamlit run antifraud/app.py
"""
from __future__ import annotations
import os
import re
import sys
from datetime import date, datetime, time
from pathlib import Path
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from antifraud.presets import PRESETS
from antifraud.api import CallIn, score as api_score

API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Справочники ---
COUNTRIES = ["RU", "KZ", "LV", "EE", "GB", "DE", "PL", "NL", "TR", "AE", "GE", "AM"]
COUNTRY_NAMES = {
    "RU": "Россия", "KZ": "Казахстан", "LV": "Латвия", "EE": "Эстония",
    "GB": "Великобритания", "DE": "Германия", "PL": "Польша", "NL": "Нидерланды",
    "TR": "Турция", "AE": "ОАЭ", "GE": "Грузия", "AM": "Армения",
}
COUNTRY_CODES = {
    "RU": "+7", "KZ": "+7", "LV": "+371", "EE": "+372",
    "GB": "+44", "DE": "+49", "PL": "+48", "NL": "+31",
    "TR": "+90", "AE": "+971", "GE": "+995", "AM": "+374",
}
FLAG_EMOJIS = {
    "RU": "🇷🇺", "KZ": "🇰🇿", "LV": "🇱🇻", "EE": "🇪🇪",
    "GB": "🇬🇧", "DE": "🇩🇪", "PL": "🇵🇱", "NL": "🇳🇱",
    "TR": "🇹🇷", "AE": "🇦🇪", "GE": "🇬🇪", "AM": "🇦🇲",
}
# Длина локальной части номера (без кода страны)
COUNTRY_LENGTHS = {
    "RU": 10, "KZ": 10, "LV": 8, "EE": 8, "GB": 10, "DE": 10, 
    "PL": 9, "NL": 9, "TR": 10, "AE": 9, "GE": 9, "AM": 8,
}

CONNECTIONS = {"mobile": "Обычный (мобильный)", "landline": "Городской", "voip": "VoIP", "intl": "Международный шлюз"}
OUTCOMES = {"answered": "Отвечен", "no_answer": "Без ответа", "rejected": "Отклонён", "ring_drop": "Сброс до ответа"}
HISTORIES = {"fraud": "Пример: Мошенник", "normal": "Пример: Обычный", "ambulance": "Пример: Экстренная служба", "live": "Без истории (живой поток)"}

PRESET_UI = {
    "fraud": {"label": "Мошенник", "icon": "🕵️", "bg": "#c62828"},
    "normal": {"label": "Обычный", "icon": "👤", "bg": "#2e7d32"},
    "ambulance": {"label": "Экстренная служба", "icon": "🚑", "bg": "#1565c0"},
}
LEVEL_COLORS = {0: "#2e7d32", 1: "#f9a825", 2: "#ef6c00", 3: "#c62828"}
LEVEL_TITLES = {0: "Обычный вызов", 1: "Подозрительный вызов", 2: "Высокий риск", 3: "Вероятная подмена номера"}

st.set_page_config(page_title="Антифрод Nexign", layout="centered", initial_sidebar_state="collapsed")

# --- Логика валидации и нормализации ---
def validate_phone(phone: str, country: str, is_source: bool = True) -> tuple[bool, str]:
    """
    is_source=True — проверяем номер источника (A) по коду страны
    is_source=False — проверяем номер получателя (B), он ВСЕГДА российский
    """
    if not phone or not phone.strip():
        return False, "Номер не может быть пустым"
    
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    
    if re.search(r'[a-zA-Zа-яА-ЯёЁ]', cleaned):
        return False, "Номер не может содержать буквы"
    if not re.match(r'^[\d+]+$', cleaned):
        return False, "Номер должен содержать только цифры и символ +"
    
    if is_source:
        # Номер источника — проверяем по выбранной стране
        expected_code = COUNTRY_CODES.get(country, "+7")
        if not cleaned.startswith(expected_code):
            return False, f"Номер источника должен начинаться с кода страны {expected_code}"
        expected_length = COUNTRY_LENGTHS.get(country, 10)
        local_part = cleaned[len(expected_code):]
        if len(local_part) != expected_length:
            return False, f"Для {COUNTRY_NAMES[country]} номер должен содержать {expected_length} цифр после кода {expected_code} (сейчас {len(local_part)})"
    else:
        # Номер получателя — ВСЕГДА российский (+7)
        if not cleaned.startswith("+7"):
            return False, "Номер получателя должен быть российским (начинаться с +7)"
        local_part = cleaned[2:]  # Убираем "+7"
        if len(local_part) != 10:
            return False, f"Российский номер должен содержать 10 цифр после +7 (сейчас {len(local_part)})"
    
    return True, ""

def normalize_phone(phone: str, country: str) -> str:
    cleaned = re.sub(r'[^\d+]', '', phone)
    expected_code = COUNTRY_CODES.get(country, "+7")
    
    if cleaned.startswith(expected_code):
        return cleaned
    if country in ("RU", "KZ") and cleaned.startswith("8"):
        return "+7" + cleaned[1:]
    return expected_code + cleaned

def on_country_change():
    """Callback при изменении страны - обновляет только номер источника (A).
    Номер получателя (B) всегда российский (+7...), так как мы делаем систему для РФ оператора."""
    country = st.session_state.get("country", "RU")
    code = COUNTRY_CODES.get(country, "+7")
    
    # Обновляем ТОЛЬКО номер источника (A)
    a_num = st.session_state.get("a_number", "")
    if a_num:
        # Убираем старый код страны если есть
        for old_code in COUNTRY_CODES.values():
            if a_num.startswith(old_code):
                a_num = a_num[len(old_code):]
                break
        st.session_state.a_number = code + a_num
    
    # Номер получателя (B) НЕ трогаем - он всегда российский!
    # Если вдруг он не российский - приводим к +7
    b_num = st.session_state.get("b_number", "")
    if b_num and not b_num.startswith("+7"):
        for old_code in COUNTRY_CODES.values():
            if b_num.startswith(old_code):
                b_num = b_num[len(old_code):]
                break
        st.session_state.b_number = "+7" + b_num

# --- Вспомогательные функции ---
def load_preset(name: str) -> None:
    c = PRESETS[name]["call"]
    ts = datetime.fromisoformat(c["ts"])
    st.session_state.update(
        a_number=c["a_number"], b_number=c["b_number"], duration=int(c["duration_sec"]),
        country=c["ingress_country"], call_date=ts.date(), call_time=ts.time(),
        connection=c["connection_type"], outcome=c["outcome"], history=name, last_action=None,
    )

def request_score(payload: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}/score", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return api_score(CallIn(**payload))

def build_payload(stage: str) -> dict:
    ts = datetime.combine(st.session_state.call_date or date.today(), st.session_state.call_time or time(12))
    payload = {
        "a_number": normalize_phone(st.session_state.a_number, st.session_state.country),
        "b_number": normalize_phone(st.session_state.b_number, st.session_state.country),
        "ts": ts.isoformat(), "ingress_country": st.session_state.country,
        "connection_type": st.session_state.connection, "stage": stage, "history": st.session_state.history,
    }
    if stage == "post":
        payload.update(duration_sec=st.session_state.duration, outcome=st.session_state.outcome)
    return payload

@st.dialog("Входящий вызов", width="large")
def call_dialog(result: dict) -> None:
    d = result["decision"]
    call = result["call"]
    color = LEVEL_COLORS[d["level"]]
    
    st.markdown(f"### {LEVEL_TITLES[d['level']]}")
    st.markdown(f"**От:** `{call['a']}`  →  **Кому:** `{call['b']}`")
    st.markdown(f"**Время:** {call['ts'].replace('T', ' ')} | **Длительность:** {call.get('duration_sec', 'ожидание')} сек")
    
    st.markdown("---")
    st.markdown(f"#### 🎯 Риск: <span style='color:{color}; font-weight:bold; font-size:1.5em'>{result['score']}</span>", unsafe_allow_html=True)
    
    if result.get("factors"):
        st.markdown("**Факторы риска:**")
        for f in result["factors"][:5]:
            pts_color = "#ff4b4b" if f['points'] > 0 else "#00cc66"
            st.markdown(f"- {f['text']} <span style='color:{pts_color}'>({f['points']:+.0f})</span>", unsafe_allow_html=True)
            
    if d.get("reasons"):
        st.info("💡 " + "\n".join(d["reasons"]))
        
    st.markdown("---")
    st.markdown(f"**Рекомендация системы:** {d['action']}")
    
    col1, col2, col3 = st.columns(3)
    action = None
    if col1.button("📞 Ответить", use_container_width=True): action = "answer"
    if col2.button(" Заблокировать", use_container_width=True): action = "block"
    if col3.button(" Отклонить", use_container_width=True): action = "decline"
    
    if action:
        st.session_state.last_action = {"action": action, "level": d["level"], "score": result["score"]}
        st.session_state.show_call = False
        st.rerun()

# --- Основная страница ---
st.title("🛡️ Антифрод-система Nexign")
st.caption("Скоринг вызовов по метаданным без прослушивания")

if "a_number" not in st.session_state: load_preset("fraud")
if "show_call" not in st.session_state: st.session_state.show_call = False

# Кнопки пресетов
cols = st.columns(3)
for col, name in zip(cols, ("fraud", "normal", "ambulance")):
    ui = PRESET_UI[name]
    col.button(f"{ui['icon']} {ui['label']}", use_container_width=True, type="secondary", on_click=load_preset, args=(name,))

st.markdown("---")

# Форма ввода
c1, c2 = st.columns(2)
a_raw = c1.text_input("Номер источника (A)", value=st.session_state.get("a_number", ""), key="a_number")
b_raw = c2.text_input("Номер получателя (B)", value=st.session_state.get("b_number", ""), key="b_number")

c1, c2 = st.columns(2)
c1.date_input("Дата", key="call_date")
c2.time_input("Время", key="call_time")

c1, c2 = st.columns(2)
with c1:
    # Флаги теперь прямо в выпадающем списке!
    st.selectbox(
        "Страна", 
        COUNTRIES, 
        format_func=lambda c: f"{FLAG_EMOJIS.get(c, '')} {COUNTRY_NAMES[c]} ({COUNTRY_CODES[c]})", 
        key="country",
        on_change=on_country_change
    )
c2.selectbox("Тип соединения", list(CONNECTIONS), format_func=CONNECTIONS.get, key="connection")

stage = st.radio("Момент оценки", ["pre", "post"], horizontal=True, format_func={"pre": "До ответа (входящий)", "post": "После вызова (разбор)"}.get, index=0)

if stage == "post":
    c1, c2 = st.columns(2)
    c1.slider("Длительность (сек)", 0, 600, key="duration")
    c2.selectbox("Исход", list(OUTCOMES), format_func=OUTCOMES.get, key="outcome")

st.selectbox("История номера", list(HISTORIES), format_func=HISTORIES.get, key="history")

# Валидация перед отправкой
validation_errors = []
country = st.session_state.get("country", "RU")

# Валидация номера источника (A) — по выбранной стране
a_valid, a_error = validate_phone(a_raw, country, is_source=True)
if not a_valid:
    validation_errors.append(f"Номер источника: {a_error}")

# Валидация номера получателя (B) — ВСЕГДА российский
b_valid, b_error = validate_phone(b_raw, country, is_source=False)
if not b_valid:
    validation_errors.append(f"Номер получателя: {b_error}")

# Проверка на одинаковые номера
a_norm = normalize_phone(a_raw, country) if a_raw else ""
b_norm = normalize_phone(b_raw, country) if b_raw else ""
if a_norm and b_norm and a_norm == b_norm:
    validation_errors.append("Номер источника и получателя совпадают!")

# Показываем ошибки
if validation_errors:
    for error in validation_errors:
        st.error(f"❌ {error}")

# Кнопка с блокировкой если есть ошибки
if st.button("🔍 СИМУЛИРОВАТЬ ВЫЗОВ", type="primary", use_container_width=True, disabled=len(validation_errors) > 0):
    if not validation_errors:
        st.session_state.call_result = request_score(build_payload(stage))
        st.session_state.show_call = True

if st.session_state.show_call and st.session_state.get("call_result"):
    call_dialog(st.session_state.call_result)

if st.session_state.get("last_action"):
    act = st.session_state.last_action
    color = LEVEL_COLORS[act["level"]]
    st.success(f"Оператор выбрал: **{act['action']}** (Риск: {act['score']}, Уровень: {act['level']})")