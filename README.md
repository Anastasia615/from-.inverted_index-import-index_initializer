# Инвертированный индекс с сжатием

Демо-проект для курса по информационному поиску.  
Реализация системы полнотекстового поиска на основе инвертированного индекса с возможностью сжатия постинг-листов.

## Ключевые возможности

- Быстрый поиск по ~40 000 документов  
- Сжатие постинг-листов (Gamma / Delta)  
- CLI и HTTP API  
- SQLite для хранения исходных текстов  

## Prerequisites

- Python 3.8+
- Poetry (для управления зависимостями)

## Быстрый старт

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Anastasia615/inverted-index.git
cd inverted-index
```

2. Установите зависимости с помощью Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 - --user
poetry install
```

3. Создайте файл `.env` на основе `.env.example` и заполните его своими данными:
```bash
cp .env.example .env
```

4. Запустите сбор данных и создание индекса:
```bash
poetry run collect-and-index
```

## Использование

### Консольный интерфейс

```bash
poetry run search --database-path data/telegram_data.sqlite --methods lowcase normalize_spaces special_chars remove_stopwords lemmatize_text "ваш запрос"
```

### HTTP API

Запустите сервер:
```bash
poetry run serve --database-path data/telegram_data.sqlite --methods lowcase normalize_spaces special_chars remove_stopwords lemmatize_text
```

Выполните поиск через HTTP:
```bash
curl "http://localhost:5000/documents?q=ваш+запрос"
```

## Описание скриптов

### collect-and-index
Собирает данные из Telegram-каналов университетов и создает инвертированный индекс.

### search
Консольный интерфейс для поиска по индексу.

Аргументы:
- `--database-path`, `-d`: Путь к файлу SQLite (обязательный)
- `--methods`, `-m`: Методы предобработки текста (по умолчанию: lowcase, normalize_spaces, special_chars, remove_stopwords, lemmatize_text)
- `--encoding`, `-e`: Алгоритм сжатия (gamma или delta)

### serve
Запускает HTTP API сервер для поиска.

Аргументы:
- `--database-path`, `-d`: Путь к файлу SQLite (обязательный)
- `--methods`, `-m`: Методы предобработки текста (по умолчанию: lowcase, normalize_spaces, special_chars, remove_stopwords, lemmatize_text)
- `--encoding`, `-e`: Алгоритм сжатия (gamma или delta)

## Тестирование

Для запуска тестов выполните:

```bash
poetry install --no-root
poetry run python -m unittest test_inverted_index.py -v
```

Тесты проверяют:
- Корректность построения инвертированного индекса
- Поиск по индексу (в том числе с учетом регистра, спецсимволов, стоп-слов, лемматизации, пустого запроса)
- Скорость индексирования большого количества документов

Ожидается, что все тесты проходят успешно и индекс работает корректно даже на больших данных.
