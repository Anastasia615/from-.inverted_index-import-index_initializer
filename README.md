# Инвертированный индекс с сжатием

**Версия:** 0.1.0

Краткая реализация поиска по текстам из SQLite с опциональным сжатием постинг-листов (Elias Gamma/Delta).

## Установка

```bash
git clone <repo_url>
cd inverted_index-main
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install click
```

## Быстрый запуск

```bash
python cli.py \
  --database-path path/to/db.sqlite \
  --methods lowcase normalize_spaces special_chars remove_stopwords lemmatize_text \
  [--encoding delta|gamma] \
  search "Ректор СПбГУ"
```