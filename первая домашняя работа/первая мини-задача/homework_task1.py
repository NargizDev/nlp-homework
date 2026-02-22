# =============================================================
# Домашняя мини-задача
# =============================================================
# Возьмите русский роман (любой .txt) и повторите блок Zipf + длинный хвост.
# Сравните предобработки: lowercasing, удаление пунктуации, замена "ё"→"е", лемматизация (если есть инструменты).
# Добавьте 2 новых интента и правила (Matcher/regex).

#   pip install requests numpy pandas matplotlib pymorphy3 pymorphy3-dicts-ru spacy
#   python -m spacy download ru_core_news_sm

import re
import subprocess
import sys
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

print('Библиотеки загружены!')


# =============================================================
# ЗАДАЧА 1 — Закон Zipf + длинный хвост
# =============================================================

# Скачиваем «Белые ночи» Достоевского (5 страниц)
print('\nСкачиваем текст...')
pages = []
for i in range(1, 6):
    url = f'https://ilibrary.ru/text/29/p.{i}/index.html'
    resp = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
    pages.append(resp.text)

# Убираем HTML-теги и лишние пробелы
html = ' '.join(pages)
text = re.sub(r'<[^>]+>', ' ', html)
text = re.sub(r'&[a-z]+;', ' ', text)
text = re.sub(r'\s+', ' ', text)

# Токенизация — только русские слова
tokens = re.findall(r'[А-Яа-яЁё]+', text)

print(f'Токенов всего: {len(tokens):,}')
print(f'Уникальных слов: {len(set(tokens)):,}')

# --- График Zipf ---
counter = Counter(tokens)
freqs = sorted(counter.values(), reverse=True)  # частоты по убыванию

plt.figure(figsize=(8, 5))
plt.loglog(range(1, len(freqs) + 1), freqs, color='steelblue')
plt.title('Закон Zipf — Белые ночи (Достоевский)')
plt.xlabel('Ранг слова (log)')
plt.ylabel('Частота (log)')
plt.grid(alpha=0.3)
plt.show()

# Топ-20 слов
print('\nТоп-20 слов:')
print(pd.DataFrame(counter.most_common(20), columns=['слово', 'частота']))

# --- Длинный хвост ---
hapax = sum(1 for f in counter.values() if f == 1)  # слова, что встретились 1 раз
total = len(counter)
print(f'\nДлинный хвост:')
print(f'  Встречается 1 раз (hapax): {hapax} слов = {100*hapax/total:.1f}% словаря')
print(f'  Встречается ≤3 раз: {sum(1 for f in counter.values() if f<=3)} слов')


# =============================================================
# ЗАДАЧА 2 — Сравнение предобработок
# =============================================================

# Шаг 1: без изменений
t1 = re.findall(r'[А-Яа-яЁё]+', text)

# Шаг 2: нижний регистр
t2 = [w.lower() for w in t1]

# Шаг 3: ё → е (чтобы "её" и "ее" считались одним словом)
t3 = [w.replace('ё', 'е') for w in t2]

# Шаг 4: убрать однобуквенные слова
t4 = [w for w in t3 if len(w) > 1]

# Шаг 5: лемматизация (привести слова к начальной форме)
try:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    # лемматизируем только уникальные слова — так намного быстрее
    lemmas = {w: morph.parse(w)[0].normal_form for w in set(t3)}
    t5 = [lemmas[w] for w in t3]
    print('\npymorphy3 работает!')
except ImportError:
    t5 = t3
    lemmas = {}
    print('\npymorphy3 не установлен — лемматизация пропущена')

# Таблица сравнения
comparison = pd.DataFrame({
    'Шаг': ['1. Raw', '2. Lowercase', '3. ё→е', '4. Без коротких', '5. Леммы'],
    'Уникальных слов': [len(set(t)) for t in [t1, t2, t3, t4, t5]]
})
print('\nКак меняется словарь:')
print(comparison.to_string(index=False))

# График
plt.figure(figsize=(8, 4))
plt.bar(comparison['Шаг'], comparison['Уникальных слов'], color='steelblue', alpha=0.8)
plt.xticks(rotation=15, ha='right')
plt.title('Размер словаря при разных предобработках')
plt.ylabel('Уникальных слов')
plt.tight_layout()
plt.show()

# Пример трансформации конкретных слов
print('\nКак меняются слова на каждом шаге:')
print(f'{"Исходное":15} {"lower":15} {"ё→е":15} {"лемма"}')
print('-' * 60)
for w in ['Настенька', 'Мечтатель', 'Её', 'быть']:
    lo  = w.lower()
    yo  = lo.replace('ё', 'е')
    lem = lemmas.get(yo, yo)
    print(f'{w:15} {lo:15} {yo:15} {lem}')


# =============================================================
# ЗАДАЧА 3 — Интенты: базовые + 2 новых
# =============================================================

import spacy
from spacy.matcher import Matcher

def load_model(name):
    try:
        return spacy.load(name)
    except OSError:
        print(f'Скачиваем модель {name}...')
        subprocess.run([sys.executable, '-m', 'spacy', 'download', name], check=True)
        return spacy.load(name)

nlp = load_model('ru_core_news_sm')
matcher = Matcher(nlp.vocab)

# --- Базовые интенты ---
matcher.add('GREETING',  [[{'LOWER': {'IN': ['привет', 'здравствуй', 'здравствуйте', 'хай']}}]])
matcher.add('GOODBYE',   [[{'LOWER': {'IN': ['пока', 'прощай', 'до свидания']}}]])
matcher.add('ASK_WEATHER',[[{'LOWER': {'IN': ['погода', 'погоду', 'погоды']}}]])
matcher.add('ASK_TIME',  [[{'LOWER': 'сколько'}, {'LOWER': {'IN': ['времени', 'часов']}, 'OP': '?'}]])

# --- Новые интенты ---
# просьба что-то посоветовать
matcher.add('ASK_RECOMMENDATION', [
    [{'LOWER': {'IN': ['посоветуй', 'посоветуйте', 'порекомендуй']}}],
    [{'LOWER': 'что'}, {'LOWER': {'IN': ['посмотреть', 'почитать', 'послушать']}}],
])

# благодарность
matcher.add('EXPRESS_GRATITUDE', [
    [{'LOWER': {'IN': ['спасибо', 'благодарю', 'благодарен', 'благодарна']}}],
    [{'LOWER': {'IN': ['большое', 'огромное']}}, {'LOWER': 'спасибо'}],
])

# Функция определения интента
def get_intent(phrase):
    doc = nlp(phrase.lower())
    matches = matcher(doc)
    if matches:
        return nlp.vocab.strings[matches[0][0]]  # первый найденный
    return 'UNKNOWN'

# Тестируем
test_phrases = [
    'Привет, как дела?',
    'До свидания!',
    'Какая сегодня погода?',
    'Сколько сейчас времени?',
    'Посоветуй что-нибудь интересное',
    'Что посмотреть вечером?',
    'Спасибо за помощь!',
    'Большое спасибо!',
    'Расскажи про динозавров',   # неизвестный интент
]

print('\nРезультаты определения интентов:')
print('-' * 45)
for phrase in test_phrases:
    print(f'{get_intent(phrase):25} | {phrase}')