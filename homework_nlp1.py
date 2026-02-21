
# # Домашняя мини-задача: Zipf + предобработка + интенты

# Установка (запустить один раз)
# !pip -q install spacy pymorphy3 pymorphy3-dicts-ru
# !python -m spacy download ru_core_news_sm -q

# %pip install requests -q

import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

print('Библиотеки загружены!')


# ## Задача 1 — Zipf + длинный хвост (русский текст)
# Загружаем «Анну Каренину» Толстого с Project Gutenberg.

# Загружаем Анну Каренину (ID 17876 на Gutenberg)
URL = 'https://www.gutenberg.org/cache/epub/17876/pg17876.txt'
response = requests.get(URL, timeout=30)
raw = response.text

# Убираем шапку и подвал Gutenberg
start = re.search(r'\*\*\* START OF', raw)
end   = re.search(r'\*\*\* END OF', raw)
if start and end:
    text = raw[raw.find('\n', start.end())+1 : end.start()]
else:
    text = raw

print(f'Загружено символов: {len(text):,}')
print('Начало текста:', text[:200])

# Простая токенизация: только русские буквы
tokens_raw = re.findall(r'[А-Яа-яЁё]+', text)
print(f'Всего токенов (tokens): {len(tokens_raw):,}')
print(f'Уникальных слов (types): {len(set(tokens_raw)):,}')
print(f'Type/Token ratio: {len(set(tokens_raw))/len(tokens_raw):.4f}')

# Zipf: строим rank-frequency график
counter = Counter(tokens_raw)
freqs_sorted = np.array(sorted(counter.values(), reverse=True))
ranks = np.arange(1, len(freqs_sorted) + 1)

plt.figure(figsize=(9, 5))
plt.loglog(ranks, freqs_sorted, linewidth=1.2, color='steelblue')
plt.title('Закон Zipf — Анна Каренина (Толстой)')
plt.xlabel('Ранг (log)')
plt.ylabel('Частота (log)')
plt.grid(True, alpha=0.3)
plt.show()

# Топ-20 слов
print('\nТоп-20 самых частых слов:')
print(pd.DataFrame(counter.most_common(20), columns=['слово', 'частота']))

# Длинный хвост: слова которые встречаются 1, 2, 3 раза
hapax    = sum(1 for f in counter.values() if f == 1)  # встречается 1 раз
dis      = sum(1 for f in counter.values() if f == 2)
tris     = sum(1 for f in counter.values() if f == 3)
total_v  = len(counter)

print(f'Словарь всего: {total_v:,} уникальных слов')
print(f'Hapax legomena (f=1): {hapax:,} ({100*hapax/total_v:.1f}% словаря)')
print(f'f=2: {dis:,} ({100*dis/total_v:.1f}%)')
print(f'f=3: {tris:,} ({100*tris/total_v:.1f}%)')
print(f'Слова с f<=3 — это «длинный хвост»: {hapax+dis+tris:,} ({100*(hapax+dis+tris)/total_v:.1f}% словаря)')

# График хвоста
freq_bins = [1, 2, 3, 4, 5, 10, 50, 100, 500]
counts = [sum(1 for f in counter.values() if f <= b) for b in freq_bins]
plt.figure(figsize=(8, 4))
plt.bar([str(b) for b in freq_bins], counts, color='tomato', alpha=0.8)
plt.title('Сколько слов встречается не чаще N раз (длинный хвост)')
plt.xlabel('Максимальная частота N')
plt.ylabel('Кол-во уникальных слов')
plt.show()


# ---
# ## Задача 2 — Сравнение предобработок
# Сравниваем: raw → lowercase → без пунктуации → замена ё→е → лемматизация

def vocab_size(tokens):
    return len(set(tokens))

# 1. Raw (без изменений)
t1 = re.findall(r'[А-Яа-яЁё]+', text)

# 2. Lowercase
t2 = [w.lower() for w in t1]

# 3. Lowercase + замена ё→е
t3 = [w.replace('ё', 'е') for w in t2]

# 4. Lowercase + ё→е + удаление коротких слов (len <= 1)
t4 = [w for w in t3 if len(w) > 1]

# 5. Лемматизация через pymorphy3
try:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    # Лемматизируем только уникальные слова — быстрее
    unique = list(set(t3))
    lemma_dict = {w: morph.parse(w)[0].normal_form for w in unique}
    t5 = [lemma_dict[w] for w in t3]
    HAS_MORPH = True
    print('pymorphy3 работает!')
except:
    t5 = t3  # если нет pymorphy3 — используем t3
    HAS_MORPH = False
    print('pymorphy3 не найден, лемматизация пропущена')

# Сравниваем размер словаря
results = pd.DataFrame({
    'Предобработка': [
        '1. Raw (как есть)',
        '2. Lowercase',
        '3. Lowercase + ё→е',
        '4. + удаление слов длиной 1',
        '5. Лемматизация' if HAS_MORPH else '5. Лемматизация (нет pymorphy3)'
    ],
    'Токенов': [len(t1), len(t2), len(t3), len(t4), len(t5)],
    'Уникальных слов': [vocab_size(t1), vocab_size(t2), vocab_size(t3), vocab_size(t4), vocab_size(t5)]
})
print(results)

# Визуализируем уменьшение словаря на каждом шаге
plt.figure(figsize=(9, 4))
plt.bar(results['Предобработка'], results['Уникальных слов'], color='steelblue', alpha=0.8)
plt.xticks(rotation=20, ha='right')
plt.title('Размер словаря при разных предобработках')
plt.ylabel('Уникальных слов')
plt.tight_layout()
plt.show()

# Пример: как меняются слова на каждом шаге
sample_words = ['Анна', 'Каренина', 'Её', 'Лёвин', 'быть']
print('\nПример трансформации слов:')
for w in sample_words:
    lo = w.lower()
    yo = lo.replace('ё', 'е')
    lem = lemma_dict.get(yo, yo) if HAS_MORPH else yo
    print(f'  {w:15} → {lo:15} → {yo:15} → {lem}')


# ---
# ## Задача 3 — 2 новых интента (Matcher + regex)
# Добавляем к базовым интентам (greeting, weather, time, помощь) два новых:
# - **ask_recommendation** — просьба что-то посоветовать
# - **express_gratitude** — благодарность

import spacy
from spacy.matcher import Matcher

# Загружаем русскую модель
try:
    nlp_ru = spacy.load('ru_core_news_sm')
    print('ru_core_news_sm загружена!')
except:
    # Если нет русской — используем английскую
    nlp_ru = spacy.load('en_core_web_sm')
    print('Используем en_core_web_sm (установите ru_core_news_sm для лучшего результата)')

matcher = Matcher(nlp_ru.vocab)

# ── Базовые интенты из семинара ──────────────────────────────────────────────

# Приветствие
matcher.add('GREETING', [
    [{'LOWER': {'IN': ['привет', 'здравствуй', 'здравствуйте', 'добрый', 'добрый день', 'хай']}}],
])

# Прощание
matcher.add('GOODBYE', [
    [{'LOWER': {'IN': ['пока', 'до свидания', 'досвидания', 'прощай', 'удачи']}}],
])

# Погода
matcher.add('ASK_WEATHER', [
    [{'LOWER': {'IN': ['погода', 'погоду', 'погоды']}},
     {'OP': '?'},
     {'OP': '?'}],
])

# Время
matcher.add('ASK_TIME', [
    [{'LOWER': {'IN': ['время', 'времени', 'сколько']}},
     {'LOWER': {'IN': ['времени', 'часов', 'час']}, 'OP': '?'}],
])

# ── НОВЫЕ ИНТЕНТЫ ────────────────────────────────────────────────────────────

# Интент 1: ask_recommendation — просьба что-то посоветовать
matcher.add('ASK_RECOMMENDATION', [
    [{'LOWER': {'IN': ['посоветуй', 'посоветуйте', 'порекомендуй', 'порекомендуйте']}}],
    [{'LOWER': 'что'}, {'LOWER': {'IN': ['посмотреть', 'почитать', 'послушать', 'поиграть']}}],
    [{'LOWER': {'IN': ['можешь', 'можете', 'мог', 'могли']}}, 
     {'LOWER': {'IN': ['порекомендовать', 'посоветовать', 'предложить']}}],
])

# Интент 2: express_gratitude — благодарность
matcher.add('EXPRESS_GRATITUDE', [
    [{'LOWER': {'IN': ['спасибо', 'благодарю', 'благодарен', 'благодарна']}}],
    [{'LOWER': 'большое'}, {'LOWER': 'спасибо'}],
    [{'LOWER': 'огромное'}, {'LOWER': 'спасибо'}],
    [{'LOWER': {'IN': ['спасибо', 'благодарю']}}, {'LOWER': 'тебе', 'OP': '?'}],
])

print('Все интенты добавлены!')
print('Список интентов:', matcher.get_rules().keys() if hasattr(matcher, 'get_rules') else 'greeting, goodbye, ask_weather, ask_time, ask_recommendation, express_gratitude')

# Функция определения интента
def detect_intent(text):
    doc = nlp_ru(text.lower())
    matches = matcher(doc)
    if matches:
        # Берём первый найденный интент
        match_id = matches[0][0]
        return nlp_ru.vocab.strings[match_id]
    return 'UNKNOWN'

# Тестируем на примерах
test_phrases = [
    # Базовые интенты
    'Привет, как дела?',
    'До свидания!',
    'Какая сегодня погода?',
    'Сколько сейчас времени?',
    # Новые интенты
    'Посоветуй что-нибудь интересное',
    'Что посмотреть вечером?',
    'Можешь порекомендовать хороший фильм?',
    'Спасибо за помощь!',
    'Большое спасибо!',
    'Благодарю тебя',
    # Неизвестный интент
    'Расскажи про динозавров',
]

print('Тестирование интентов:')
print('-' * 45)
for phrase in test_phrases:
    intent = detect_intent(phrase)
    print(f'{intent:25} | {phrase}')

# Бонус: regex-версия для тех же интентов (более простая, но работает без spaCy)
INTENT_PATTERNS = {
    'GREETING':           r'привет|здравствуй|добрый день|хай',
    'GOODBYE':            r'пока|до свидания|прощай',
    'ASK_WEATHER':        r'погод[ауы]?',
    'ASK_TIME':           r'сколько (времени|часов)|который час',
    'ASK_RECOMMENDATION': r'посоветуй|порекомендуй|что (посмотреть|почитать|послушать)|можешь (посоветовать|порекомендовать)',
    'EXPRESS_GRATITUDE':  r'спасибо|благодар(ю|ен|на)|большое спасибо',
}

def detect_intent_regex(text):
    text_lower = text.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, text_lower):
            return intent
    return 'UNKNOWN'

print('Тестирование regex-версии:')
print('-' * 45)
for phrase in test_phrases:
    intent = detect_intent_regex(phrase)
    print(f'{intent:25} | {phrase}')
