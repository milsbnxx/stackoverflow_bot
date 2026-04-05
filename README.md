# Stack Overflow Retrieval Bot

Проект для тестового задания по ML/retrieval.

Бот не генерирует новые ответы, а ищет самый релевантный ответ в базе Stack Overflow:
1. вопрос пользователя переводится в эмбеддинг;
2. сравнивается с эмбеддингами вопросов из индекса;
3. показываются top-k похожих ответов.

## Что есть в проекте

- `src/prepare_data.py` — подготовка датасета из `Questions.csv` и `Answers.csv`
- `build_index.py` — построение индекса эмбеддингов
- `src/retriever.py` — поиск релевантных вопросов/ответов
- `app.py` — Flask backend + API `/api/search`
- `templates/index.html` — HTML интерфейс
- `static/styles.css` — стили
- `static/app.js` — логика фронтенда
- `data/index/question_embeddings.npy` — векторы вопросов
- `data/index/metadata.json` — тексты и метаданные для выдачи

## Требования

- Python 3.10+ (лучше 3.11+)
- интернет на первом запуске (чтобы скачать модель с Hugging Face)

## Запуск (если индекс уже есть)

```bash
cd /Users/sabinaamilova/Documents/stackoverflow_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Открой в браузере: `http://127.0.0.1:7860`

## Полный запуск с нуля

1. Установить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Подготовить данные (если нет `data/processed_qa.csv`):

```bash
python3 src/prepare_data.py
```

3. Построить индекс:

```bash
python3 build_index.py --max-rows 50000
```

4. Запустить веб-интерфейс:

```bash
python3 app.py
```

## Как обновить базу новыми данными

После замены CSV-файлов в `data/` нужно пересобрать:

```bash
python3 src/prepare_data.py
python3 build_index.py --max-rows 50000
```

## Полезные параметры индексации

`build_index.py`:
- `--max-rows` — сколько строк брать в индекс (`0` = все)
- `--batch-size` — размер батча для расчета эмбеддингов
- `--model-name` — название модели эмбеддингов

## Частые проблемы

1. `Индекс не найден`
   Запусти `python3 build_index.py --max-rows 50000`.

2. Ошибки Hugging Face при первом запуске
   Проверь интернет и запусти команду снова.

3. Медленно работает на большом индексе
   Уменьши `--max-rows`, например до `20000`.
