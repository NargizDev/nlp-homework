
# # Домашка: Domain Shift + Наивный Байес
# **Часть A** — русский датасет, log-odds, графики
# **Часть B** — классификатор авторов, метрики, domain shift

# Установка нужных библиотек (запустить один раз)
# !pip install datasets pymorphy2 -q

import re
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Попробуем загрузить pymorphy2 для лемматизации
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    HAS_MORPH = True
    print('pymorphy2 загружен успешно')
except:
    HAS_MORPH = False
    print('pymorphy2 не найден — лемматизация будет пропущена')

print('Библиотеки загружены!')


# ## Вспомогательные функции

# Стоп-слова русского языка
STOPWORDS_RU = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'как', 'а', 'то', 'все',
    'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
    'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из',
    'ему', 'когда', 'ну', 'ли', 'если', 'или', 'ни', 'быть', 'был', 'до', 'вас',
    'уже', 'вам', 'там', 'потом', 'себя', 'ей', 'они', 'тут', 'где', 'есть', 'для',
    'мы', 'их', 'чем', 'была', 'сам', 'будет', 'кто', 'этот', 'этого', 'здесь', 'этом',
}

LEMMA_CACHE = {}

def tokenize(text):
    """Токенизация: только буквы (русские и английские), нижний регистр."""
    text = str(text).lower().replace('ё', 'е')
    return re.findall(r'[а-яa-z]+', text)

def lemmatize(token):
    """Лемматизация одного слова через pymorphy2."""
    if not HAS_MORPH or token.isdigit():
        return token
    if token not in LEMMA_CACHE:
        LEMMA_CACHE[token] = morph.parse(token)[0].normal_form
    return LEMMA_CACHE[token]

def preprocess(text, use_lemma=False, remove_stop=False):
    """Предобработка текста: токенизация + опционально лемматизация и стоп-слова."""
    tokens = tokenize(text)
    if use_lemma:
        tokens = [lemmatize(t) for t in tokens]
    if remove_stop:
        tokens = [t for t in tokens if t not in STOPWORDS_RU and len(t) > 1]
    return ' '.join(tokens) if tokens else '_empty_'

def log_odds(texts_a, texts_b, alpha=0.1, top_n=30, use_lemma=False, remove_stop=False):
    """Считаем log-odds для всех токенов между двумя доменами."""
    ca = Counter()
    cb = Counter()
    for t in texts_a:
        ca.update(preprocess(t, use_lemma, remove_stop).split())
    for t in texts_b:
        cb.update(preprocess(t, use_lemma, remove_stop).split())
    
    vocab = set(ca) | set(cb)
    V = len(vocab)
    Na, Nb = sum(ca.values()), sum(cb.values())
    
    scores = {}
    for w in vocab:
        pa = (ca[w] + alpha) / (Na + alpha * V)
        pb = (cb[w] + alpha) / (Nb + alpha * V)
        scores[w] = math.log(pa / pb)
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    pos = sorted_scores[:top_n]   # характерно для домена A
    neg = sorted_scores[-top_n:]  # характерно для домена B
    return pos, neg[::-1]

def chunk_tokens(tokens, chunk_size=150, min_frac=0.5):
    """Режем список токенов на чанки фиксированного размера."""
    chunks = []
    min_len = int(chunk_size * min_frac)
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        if len(chunk) >= min_len:
            chunks.append(' '.join(chunk))
    return chunks

print('Функции готовы!')


# ---
# ## Часть A — Domain shift на русском датасете
# Используем [rus_news_classifier](https://huggingface.co/datasets/data-silence/rus_news_classifier) — новости по темам.

# A1. Загружаем датасет
ds = load_dataset('data-silence/rus_news_classifier', split='train')
df = ds.to_pandas()

print('Колонки:', df.columns.tolist())
print('Размер:', df.shape)
print('\nКлассы и количество:')
print(df['topic'].value_counts())

# A2. Выбираем 2 домена и берём по 2000 текстов из каждого
# Выберем два самых разных по тематике — посмотри на вывод выше и при желании поменяй
DOMAIN_A = 'Политика'   # поменяй если нужно
DOMAIN_B = 'Спорт'      # поменяй если нужно
N = 2000  # сколько текстов брать из каждого домена

texts_a = df[df['topic'] == DOMAIN_A]['text'].astype(str).tolist()
texts_b = df[df['topic'] == DOMAIN_B]['text'].astype(str).tolist()

# Ограничиваем размер выборки
random.shuffle(texts_a)
random.shuffle(texts_b)
texts_a = texts_a[:N]
texts_b = texts_b[:N]

print(f'Домен A ({DOMAIN_A}): {len(texts_a)} текстов')
print(f'Домен B ({DOMAIN_B}): {len(texts_b)} текстов')
print(f'\nПример из {DOMAIN_A}:', texts_a[0][:200])
print(f'\nПример из {DOMAIN_B}:', texts_b[0][:200])

# A3. Log-odds: baseline (без предобработки)
pos_base, neg_base = log_odds(texts_a, texts_b, top_n=30)

print(f'Топ-30 слов для {DOMAIN_A} (против {DOMAIN_B}):')
print(pd.DataFrame(pos_base, columns=['слово', 'log-odds']))

print(f'\nТоп-30 слов для {DOMAIN_B} (против {DOMAIN_A}):')
print(pd.DataFrame(neg_base, columns=['слово', 'log-odds']))

# A4. Log-odds с лемматизацией и стоп-словами
pos_proc, neg_proc = log_odds(texts_a, texts_b, top_n=30, use_lemma=HAS_MORPH, remove_stop=True)

print(f'Топ-30 слов для {DOMAIN_A} (с предобработкой):')
print(pd.DataFrame(pos_proc, columns=['слово', 'log-odds']))

print(f'\nТоп-30 слов для {DOMAIN_B} (с предобработкой):')
print(pd.DataFrame(neg_proc, columns=['слово', 'log-odds']))

# A5. График 1 — распределение длин документов
lens_a = [len(tokenize(t)) for t in texts_a]
lens_b = [len(tokenize(t)) for t in texts_b]

plt.figure(figsize=(10, 4))
plt.hist(lens_a, bins=50, alpha=0.6, label=DOMAIN_A, color='steelblue')
plt.hist(lens_b, bins=50, alpha=0.6, label=DOMAIN_B, color='tomato')
plt.title('Распределение длины документов')
plt.xlabel('Токенов в документе')
plt.ylabel('Кол-во')
plt.legend()
plt.show()

print(f'{DOMAIN_A}: среднее={np.mean(lens_a):.0f}, медиана={np.median(lens_a):.0f}')
print(f'{DOMAIN_B}: среднее={np.mean(lens_b):.0f}, медиана={np.median(lens_b):.0f}')

# A6. График 2 — Zipf (топ-100 токенов по частоте)
def get_top_freqs(texts, top_n=100):
    freq = Counter()
    for t in texts:
        freq.update(tokenize(t))
    return [f for _, f in freq.most_common(top_n)]

freqs_a = get_top_freqs(texts_a)
freqs_b = get_top_freqs(texts_b)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 101), freqs_a, label=DOMAIN_A, color='steelblue')
plt.plot(range(1, 101), freqs_b, label=DOMAIN_B, color='tomato')
plt.xscale('log')
plt.yscale('log')
plt.title('Zipf: топ-100 токенов')
plt.xlabel('Ранг (log)')
plt.ylabel('Частота (log)')
plt.legend()
plt.show()

# A7. Наблюдения
# Сравниваем топ-10 слов до и после предобработки
top10_base_a = [w for w, _ in pos_base[:10]]
top10_proc_a = [w for w, _ in pos_proc[:10]]
overlap = len(set(top10_base_a) & set(top10_proc_a))

print('=== Наблюдения ===')
print(f'1. Средняя длина текстов: {DOMAIN_A}={np.mean(lens_a):.0f} токенов, {DOMAIN_B}={np.mean(lens_b):.0f} токенов')
print(f'2. Пересечение топ-10 слов до/после предобработки: {overlap}/10')
print(f'3. Baseline топ-10 для {DOMAIN_A}: {top10_base_a}')
print(f'4. После предобработки топ-10 для {DOMAIN_A}: {top10_proc_a}')
print('5. Вывод: лемматизация и удаление стоп-слов убирают служебные слова,\n   оставляя более содержательные тематические маркеры.')


# ---
# ## Часть B — Наивный Байес: классификатор авторов
# Используем [IlyaGusev/versa](https://huggingface.co/datasets/IlyaGusev/versa) — стихи русских поэтов.
# Domain shift: обучаемся на обычном тексте, тестируем на тексте с ударениями (`stressed_text`).

# B1. Загружаем датасет
df_versa = load_dataset('IlyaGusev/versa', split='train').to_pandas()

print('Колонки:', df_versa.columns.tolist())
print('Размер:', df_versa.shape)
print('\nТоп авторов:')
print(df_versa['author'].value_counts().head(10))

# B2. Выбираем 2 авторов с наибольшим числом текстов
top_authors = df_versa['author'].value_counts().head(2).index.tolist()
AUTHOR_A, AUTHOR_B = top_authors[0], top_authors[1]
print(f'Авторы: {AUTHOR_A} и {AUTHOR_B}')

# Оставляем только нужных авторов
df_ab = df_versa[df_versa['author'].isin([AUTHOR_A, AUTHOR_B])].copy()
df_ab = df_ab[['author', 'text', 'stressed_text']].dropna(subset=['author', 'text'])
df_ab['stressed_text'] = df_ab['stressed_text'].fillna(df_ab['text'])  # если нет — берём обычный

print(f'\nТекстов по авторам:')
print(df_ab['author'].value_counts())
print('\nПример текста:', df_ab[df_ab['author']==AUTHOR_A]['text'].iloc[0][:200])

# B3. Нарезаем тексты на чанки
CHUNK_SIZE = 150  # токенов в чанке

def texts_to_chunks(texts, chunk_size=150):
    """Объединяем все токены и режем на чанки."""
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize(str(t)))
    return chunk_tokens(all_tokens, chunk_size=chunk_size)

rows_train, rows_test_plain, rows_test_shift = [], [], []

for author in [AUTHOR_A, AUTHOR_B]:
    sub = df_ab[df_ab['author'] == author].sample(frac=1, random_state=RANDOM_STATE)
    split = int(len(sub) * 0.8)
    
    train_texts  = sub['text'].iloc[:split].tolist()
    test_plain   = sub['text'].iloc[split:].tolist()
    test_shift   = sub['stressed_text'].iloc[split:].tolist()
    
    for ch in texts_to_chunks(train_texts, CHUNK_SIZE):
        rows_train.append({'chunk': ch, 'label': author})
    for ch in texts_to_chunks(test_plain, CHUNK_SIZE):
        rows_test_plain.append({'chunk': ch, 'label': author})
    for ch in texts_to_chunks(test_shift, CHUNK_SIZE):
        rows_test_shift.append({'chunk': ch, 'label': author})

train_df = pd.DataFrame(rows_train)
test_df  = pd.DataFrame(rows_test_plain)
shift_df = pd.DataFrame(rows_test_shift)

print('Чанков в train:', len(train_df))
print('Чанков в test (обычный):', len(test_df))
print('Чанков в test (с ударениями):', len(shift_df))
print('\nРаспределение по авторам в train:')
print(train_df['label'].value_counts())

# B4. Обучаем MultinomialNB (baseline)
vec = CountVectorizer(min_df=3)
X_train = vec.fit_transform(train_df['chunk'])
y_train = train_df['label'].values

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)

# Оцениваем in-domain
X_test = vec.transform(test_df['chunk'])
y_test = test_df['label'].values
y_pred = nb.predict(X_test)

print('=== In-domain (обычный текст → обычный текст) ===')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(classification_report(y_test, y_pred, target_names=[AUTHOR_A, AUTHOR_B]))

cm = confusion_matrix(y_test, y_pred, labels=[AUTHOR_A, AUTHOR_B])
print('Confusion matrix:\n', cm)

# B5. Топ-20 важных токенов по Δ(w)
feature_names = np.array(vec.get_feature_names_out())
classes = list(nb.classes_)
idx_a = classes.index(AUTHOR_A)
idx_b = classes.index(AUTHOR_B)

delta = nb.feature_log_prob_[idx_a] - nb.feature_log_prob_[idx_b]

top_a = np.argsort(-delta)[:20]
top_b = np.argsort(delta)[:20]

print(f'Топ-20 токенов для {AUTHOR_A}:')
print(pd.DataFrame({'токен': feature_names[top_a], 'Δ': delta[top_a]}))

print(f'\nТоп-20 токенов для {AUTHOR_B}:')
print(pd.DataFrame({'токен': feature_names[top_b], 'Δ': delta[top_b]}))

# B6. Улучшенная предобработка: стоп-слова + биграммы
vec2 = CountVectorizer(min_df=3, stop_words=list(STOPWORDS_RU), ngram_range=(1, 2))
X_train2 = vec2.fit_transform(
    [preprocess(t, use_lemma=HAS_MORPH, remove_stop=True) for t in train_df['chunk']]
)

nb2 = MultinomialNB(alpha=0.1)
nb2.fit(X_train2, y_train)

X_test2 = vec2.transform(
    [preprocess(t, use_lemma=HAS_MORPH, remove_stop=True) for t in test_df['chunk']]
)
y_pred2 = nb2.predict(X_test2)

print('=== In-domain (с предобработкой) ===')
print(f'Accuracy: {accuracy_score(y_test, y_pred2):.3f}')
print(classification_report(y_test, y_pred2, target_names=[AUTHOR_A, AUTHOR_B]))

# B7. Domain shift: обучились на обычном тексте, тестируем на тексте с ударениями
X_shift = vec.transform(shift_df['chunk'])
y_shift = shift_df['label'].values
y_pred_shift = nb.predict(X_shift)

print('=== Domain shift (обычный → с ударениями) — baseline ===')
print(f'Accuracy: {accuracy_score(y_shift, y_pred_shift):.3f}')
print(classification_report(y_shift, y_pred_shift, target_names=[AUTHOR_A, AUTHOR_B]))

cm_shift = confusion_matrix(y_shift, y_pred_shift, labels=[AUTHOR_A, AUTHOR_B])
print('Confusion matrix:\n', cm_shift)

# Сводная таблица
acc_in = accuracy_score(y_test, y_pred)
acc_in2 = accuracy_score(y_test, y_pred2)
acc_shift = accuracy_score(y_shift, y_pred_shift)

print('\n=== Итоговое сравнение ===')
results = pd.DataFrame({
    'Модель': ['Baseline (unigram)', 'С предобработкой (bigram+stop)', 'Baseline → Domain shift'],
    'Accuracy': [acc_in, acc_in2, acc_shift]
})
print(results)

# B8. Итоговые наблюдения
print('=== Наблюдения по Части B ===')
print(f'1. Baseline (in-domain) accuracy: {acc_in:.3f}')
print(f'2. С предобработкой (in-domain) accuracy: {acc_in2:.3f}')
print(f'3. Domain shift accuracy: {acc_shift:.3f} (упало на {acc_in - acc_shift:.3f})')
print('4. Добавление стоп-слов и биграмм', 'улучшило' if acc_in2 > acc_in else 'не улучшило', 'качество in-domain')
print('5. Domain shift (ударения) меняет написание слов — многие токены становятся')
print('   незнакомыми для модели, обученной на обычном тексте.')
