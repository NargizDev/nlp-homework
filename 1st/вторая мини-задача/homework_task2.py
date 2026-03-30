
# # Мини-задача 10: токены vs стеммы vs леммы + POS/DEP/NER + персонажи + приветствия

# Установка (один раз)
# !pip -q install spacy pymorphy3 pymorphy3-dicts-ru nltk
# !python -m spacy download ru_core_news_sm -q

import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

print('Библиотеки загружены!')


# ## 1. Загружаем фрагмент русского романа (~50k символов)
# Используем «Преступление и наказание» Достоевского.

# «Белые ночи» на ilibrary.ru — кодировка windows-1251, 5 страниц
print('Скачиваем текст...')
parts = []
for i in range(1, 6):
    url = f'https://ilibrary.ru/text/29/p.{i}/index.html'
    resp = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
    # Сайт отдаёт windows-1251 — декодируем правильно
    page = resp.content.decode('windows-1251', errors='ignore')
    # Вырезаем только тело страницы, убираем HTML-теги
    page = re.sub(r'<script[^>]*>.*?</script>', '', page, flags=re.DOTALL)  # убираем JS
    page = re.sub(r'<style[^>]*>.*?</style>',  '', page, flags=re.DOTALL)  # убираем CSS
    page = re.sub(r'<[^>]+>', ' ', page)                                    # убираем теги
    page = re.sub(r'\s+', ' ', page).strip()
    parts.append(page)
    print(f'  Страница {i} загружена')

text = ' '.join(parts)

# Проверяем что загрузился русский текст
ru_words = re.findall(r'[А-Яа-яЁё]+', text)
if not ru_words:
    raise RuntimeError('Текст не загрузился — проверьте интернет')

print(f'\nЗагружено символов: {len(text):,}')
print('Начало:', text[:300])


# ---
# ## 2. Сравнение частот: токены vs стеммы vs леммы

# Шаг 1: токены (lowercase + ё→е)
def base_tokenize(text):
    t = text.lower().replace('ё', 'е')
    return re.findall(r'[а-яa-z]+', t)

tokens = base_tokenize(text)
print(f'Токенов: {len(tokens):,}, уникальных: {len(set(tokens)):,}')

# Шаг 2: стеммы (Snowball)
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt', quiet=True)

stemmer = SnowballStemmer('russian')
stems = [stemmer.stem(t) for t in tokens]
print(f'Стеммов: {len(stems):,}, уникальных: {len(set(stems)):,}')

# Шаг 3: леммы (pymorphy3)
try:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    # Лемматизируем только уникальные слова (быстрее)
    unique_tokens = list(set(tokens))
    lemma_map = {w: morph.parse(w)[0].normal_form for w in unique_tokens}
    lemmas = [lemma_map[t] for t in tokens]
    HAS_MORPH = True
    print(f'Лемм: {len(lemmas):,}, уникальных: {len(set(lemmas)):,}')
except:
    lemmas = tokens
    HAS_MORPH = False
    print('pymorphy3 не найден, леммы = токены')

# Топ-20 для каждого вида
top_tokens = Counter(tokens).most_common(20)
top_stems  = Counter(stems).most_common(20)
top_lemmas = Counter(lemmas).most_common(20)

df_top = pd.DataFrame({
    'Токен':       [w for w, _ in top_tokens],
    'f(токен)':    [f for _, f in top_tokens],
    'Стемма':      [w for w, _ in top_stems],
    'f(стемма)':   [f for _, f in top_stems],
    'Лемма':       [w for w, _ in top_lemmas],
    'f(лемма)':    [f for _, f in top_lemmas],
})
print('Топ-20 по токенам / стеммам / леммам:')
print(df_top)

# Размер словаря и доля «длинного хвоста» (f<=3)
def tail_stats(counter_items):
    c = dict(counter_items) if not isinstance(counter_items, dict) else counter_items
    total = len(c)
    tail  = sum(1 for f in c.values() if f <= 3)
    return total, tail, tail / total * 100

c_tok  = Counter(tokens)
c_stem = Counter(stems)
c_lem  = Counter(lemmas)

stats = pd.DataFrame([
    {'Вид':    'Токены',  **dict(zip(['Словарь', 'f≤3', '% хвост'], tail_stats(c_tok)))},
    {'Вид':    'Стеммы',  **dict(zip(['Словарь', 'f≤3', '% хвост'], tail_stats(c_stem)))},
    {'Вид':    'Леммы',   **dict(zip(['Словарь', 'f≤3', '% хвост'], tail_stats(c_lem)))},
])
print('Сравнение словаря и длинного хвоста:')
print(stats)

# График
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(stats['Вид'], stats['Словарь'], color=['steelblue', 'tomato', 'seagreen'])
axes[0].set_title('Размер словаря')
axes[0].set_ylabel('Уникальных форм')

axes[1].bar(stats['Вид'], stats['% хвост'], color=['steelblue', 'tomato', 'seagreen'])
axes[1].set_title('Доля длинного хвоста (f≤3)')
axes[1].set_ylabel('%')
plt.tight_layout()
plt.show()


# ---
# ## 3. POS / DEP / NER от spaCy + леммы от pymorphy3 (5 предложений)

import spacy

try:
    nlp = spacy.load('ru_core_news_sm')
    print('ru_core_news_sm загружена!')
except:
    raise RuntimeError('Установите: python -m spacy download ru_core_news_sm')

# Разбиваем текст на предложения и берём 5 интересных
doc_full = nlp(text[:5000])  # небольшой фрагмент для скорости
sentences = [s.text.strip() for s in doc_full.sents if len(s.text.strip()) > 30]

# Выбираем 5 предложений
chosen = sentences[2:7]
print('Выбранные предложения:')
for i, s in enumerate(chosen, 1):
    print(f'{i}. {s}')

# Анализируем каждое предложение
for i, sent in enumerate(chosen, 1):
    print(f'\n{'='*60}')
    print(f'Предложение {i}: {sent}')
    print('='*60)
    
    doc = nlp(sent)
    rows = []
    for token in doc:
        if token.is_space:
            continue
        pym_lemma = lemma_map.get(token.text.lower().replace('ё','е'), token.text.lower()) if HAS_MORPH else '—'
        rows.append({
            'Токен':         token.text,
            'POS (spaCy)':   token.pos_,
            'DEP (spaCy)':   token.dep_,
            'Лемма spaCy':   token.lemma_,
            'Лемма pymorphy': pym_lemma,
        })
    print(pd.DataFrame(rows))
    
    # NER
    if doc.ents:
        print('NER:', [(e.text, e.label_) for e in doc.ents])
    else:
        print('NER: сущностей не найдено')


# ---
# ## 4. Список имён персонажей (через NER + частоты)

# Обрабатываем весь фрагмент и собираем NER PER (персонажей)
# Обрабатываем кусками — spaCy не любит очень длинные тексты
CHUNK = 10000
all_ents = Counter()

for start_i in range(0, len(text), CHUNK):
    chunk_doc = nlp(text[start_i:start_i+CHUNK])
    for ent in chunk_doc.ents:
        if ent.label_ == 'PER':  # только персонажи
            name = ent.text.strip()
            if len(name) > 2:    # убираем слишком короткие
                all_ents[name] += 1

print(f'Найдено упоминаний персонажей: {sum(all_ents.values())}')
print(f'Уникальных имён: {len(all_ents)}')
print('\nТоп-20 персонажей:')
print(pd.DataFrame(all_ents.most_common(20), columns=['Персонаж', 'Упоминаний']))


# ---
# ## 5. Предложения с приветствиями

# Ищем предложения с приветствиями через regex
# Разбиваем текст на предложения простым способом
raw_sentences = re.split(r'(?<=[.!?])\s+', text)

GREETING_PATTERN = re.compile(
    r'здравствуй|здравствуйте|добрый день|добрый вечер|добрый утро|'
    r'добро пожаловать|привет|приветствую|рад(а)? вас|рад(а)? видеть|'
    r'мое почтение|мое уважение|позвольте представиться',
    re.IGNORECASE
)

greeting_sents = [
    s.strip() for s in raw_sentences
    if GREETING_PATTERN.search(s) and len(s.strip()) > 10
]

print(f'Найдено предложений с приветствиями: {len(greeting_sents)}')
print('='*60)
for i, s in enumerate(greeting_sents, 1):
    # Подсвечиваем найденное слово
    highlighted = GREETING_PATTERN.sub(lambda m: f'>>>{m.group()}<<<', s)
    print(f'{i}. {highlighted[:200]}')
    print()

# Дополнительно: ищем через spaCy Matcher (по POS-паттернам)
from spacy.matcher import Matcher

matcher_greet = Matcher(nlp.vocab)

# Приветствия как отдельные слова
matcher_greet.add('GREETING_WORD', [
    [{'LOWER': {'IN': ['здравствуйте', 'здравствуй', 'привет', 'приветствую']}}],
    [{'LOWER': 'добрый'}, {'LOWER': {'IN': ['день', 'вечер', 'утро', 'ночь']}}],
    [{'LOWER': {'IN': ['рад', 'рада']}}, {'LOWER': {'IN': ['вас', 'вам', 'видеть']}}],
])

spacy_greeting_sents = []
for start_i in range(0, min(len(text), 30000), CHUNK):
    chunk_doc = nlp(text[start_i:start_i+CHUNK])
    for sent in chunk_doc.sents:
        matches = matcher_greet(nlp(sent.text))
        if matches:
            spacy_greeting_sents.append(sent.text.strip())

print(f'\nspaCy Matcher нашёл предложений с приветствиями: {len(spacy_greeting_sents)}')
for i, s in enumerate(spacy_greeting_sents, 1):
    print(f'{i}. {s[:200]}')

# Итоговое сравнение двух подходов
print('Сравнение подходов к поиску приветствий:')
print(f'  Regex:         {len(greeting_sents)} предложений')
print(f'  spaCy Matcher: {len(spacy_greeting_sents)} предложений')
print()