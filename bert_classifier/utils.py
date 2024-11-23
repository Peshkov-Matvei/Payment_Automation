import re
import dateparser
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()

ABBREVIATIONS = {
    "сч.": "счет",
    "з/п": "зарплата",
    "No": "номер",
    "ООБез": "ООО Без",
    'НДС': 'налог на добавленную стоимость',
    'ОАО': 'открытое акционерное общество',
    'ГСМ': 'горюче-смазочные материалы',
    'VIN': 'номер транспортного средства',
    'ДОГ': 'договор',
    'ЖК': 'жилой комплекс',
    "АКБ": "акционерное общество банка",
    "АО": "акционерное общество",
    'ЭП': 'электронная подпись',
    "ГОСТ": "государственный стандарт",
    "WC": "туалет",
    'ГА': 'генеральная ассистентская компания'
}


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    for short, full in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(short)}\b', full, text)
    text = re.sub(r'(\d)([А-Яа-я])', r'\1 \2', text)
    text = re.sub(r'([А-Яа-я])(\d)', r'\1 \2', text)
    text = re.sub(
        r'\b(\d{1,2}[./\-\s]\d{1,2}[./\-\s]\d{2,4})\b',
        lambda m: dateparser.parse(m.group(0)).strftime('%d.%m.%Y') if dateparser.parse(m.group(0)) else m.group(0),
        text)
    text = re.sub(r'(\d+)[.,-](\d+)', r'\1.\2', text)
    text = re.sub(r'[^\w\sа-яА-Я0-9]', '', text)
    text = ' '.join([morph.parse(word)[0].normal_form for word in text.split()])
    return text
