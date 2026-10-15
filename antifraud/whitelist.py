"""Реестры доверенных номеров (White List) на основе данных Даши Д."""
from __future__ import annotations
from typing import Any

# Экстренные службы (жесткий бан на блокировку)
EMERGENCY_SERVICE_NUMBERS = {"101", "102", "103", "104", "112"}

# Сервисные номера (банки, операторы, госуслуги). 
# ВАЖНО: Это должен быть СЛОВАРЬ {номер: название}, иначе generate_data.py упадет!
SERVICE_REGISTRY = {
    "900": "Сбер",
    "1000": "ВТБ",
    "400": "Газпромбанк",
    "700": "Совкомбанк",
    "800": "Открытие",
    "7787": "Россельхозбанк",
    "0707": "ОТП Банк",
    "3040": "АТБ",
    "999": "Т-Банк",
    "0890": "МТС",
    "0611": "Билайн",
    "0500": "МегаФон",
    "611": "T2",
    "111": "Мотив",
    "107": "Ростелеком",
    "110": "Администрация",
    "115": "Госуслуги",
    "122": "Единая служба",
}

def is_emergency(number: str) -> bool:
    """Проверяет, является ли номер экстренной службой."""
    clean = number.replace("+7", "").replace("8", "", 1).replace(" ", "").replace("-", "")
    return clean in EMERGENCY_SERVICE_NUMBERS or number in EMERGENCY_SERVICE_NUMBERS

def is_service(number: str) -> bool:
    """Проверяет, является ли номер сервисным (банк, оператор, госуслуги)."""
    clean = number.replace("+7", "").replace("8", "", 1).replace(" ", "").replace("-", "")
    return clean in SERVICE_REGISTRY or number in SERVICE_REGISTRY

def lookup(number: str) -> tuple[str, str] | None:
    """Проверяет номер по белому списку. Возвращает (категория, описание) или None."""
    if is_emergency(number):
        return "emergency", "Экстренная служба (101-112)"
    if is_service(number):
        clean = number.replace("+7", "").replace("8", "", 1).replace(" ", "").replace("-", "")
        name = SERVICE_REGISTRY.get(clean) or SERVICE_REGISTRY.get(number, "Сервисный номер")
        return "service", name
    
    # Проверка на 8-800
    if number.startswith("8800") or number.startswith("+7800"):
        return "service", "Федеральный номер 8-800"
    
    return None

EMERGENCY_NUMBERS = EMERGENCY_SERVICE_NUMBERS
BANK_NUMBERS = {"900", "1000", "400", "700", "800", "7787", "0707", "3040", "999"}
OPERATOR_NUMBERS = {"0890", "0611", "0500", "611", "111", "107"}
GOVERNMENT_NUMBERS = {"110", "115", "122"}