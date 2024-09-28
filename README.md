# 🛠️ Rutube AI Supporter

![Rutube AI Helper](./media/git_logo.webp)

**Rutube AI Supporter** — это проект, предназначенный для автоматизации взаимодействия с платформой Rutube и поддержки пользователей через Telegram. Проект включает в себя API сервер, Telegram бота и модели машинного обучения для анализа и обработки данных.

Проект состоит из трёх основных частей:

1. **API сервер** — обеспечивает взаимодействие с моделями и управление данными.
2. **Telegram Bot** — предоставляет удобный интерфейс для общения и запросов к AI через Telegram.
3. **vLLM Inference Server** — предоставляет сервис для инференса моделей машинного обучения.

## 📋 Основные компоненты

### 1. API Сервер 🌐

API сервер предоставляет интерфейс для взаимодействия с моделями машинного обучения, позволяя отправлять запросы и получать результаты.

### 2. Telegram Bot 🤖

Telegram Bot обеспечивает удобный доступ к моделям, позволяя пользователям отправлять команды и получать ответы напрямую в мессенджере.

### 3. vLLM Inference Server 🚀

vLLM Inference Server предлагает высокопроизводительный сервис для инференса моделей, управляемый через Docker и интегрированный с командами Makefile для автоматизации.

## 🚀 Как запустить проект

### Предварительные требования

1. Установлен [Docker](https://docs.docker.com/get-docker/).
2. Установлен [Poetry](https://python-poetry.org/docs/#installation) для управления зависимостями Python.
3. Установлен [Make](https://www.gnu.org/software/make/) для автоматизации команд.

### Скачивание и подготовка проекта

```bash
# Клонируем репозиторий
git clone https://github.com/yourusername/rutube-ai-supporter.git

# Переходим в директорию проекта
cd rutube-ai-supporter
```

### Установка зависимостей для API сервера

API сервер имеет свои зависимости и `pyproject.toml` файл внутри директории `api`. Перейдите в директорию `api` и установите зависимости:

```bash
# Переходим в директорию API
cd api

# Устанавливаем зависимости с помощью Poetry
poetry install

# Возвращаемся в корневую директорию
cd ..
```

### Установка зависимостей для Telegram бота

Аналогично, Telegram бот имеет свои зависимости и `pyproject.toml` файл внутри директории `tg_bot`. Перейдите в директорию `tg_bot` и установите зависимости:

```bash
# Переходим в директорию Telegram бота
cd tg_bot

# Устанавливаем зависимости с помощью Poetry
poetry install

# Возвращаемся в корневую директорию
cd ..
```

### Скачивание необходимых данных и моделей для API

```bash
# Скачиваем необходимые данные и модели
make download-data
```

### Запуск API сервера

Сборка и запуск Docker-контейнера для API:

```bash
# Сборка Docker-образа для API
make build-api

# Запуск Docker-контейнера для API
make run-api

# Просмотр логов API
make logs-api
```

API сервер запустится на порту `8001`. Вы можете изменить порт, указав его в `Makefile` или в переменных окружения.

### Запуск Telegram Bot

#### Запуск бота локально

Для отладки или разработки можно запустить бота локально:

```bash
# Локальный запуск бота
make run-bot-local
```

#### Запуск Telegram Bot через Docker

Для развертывания бота в Docker:

```bash
# Сборка Docker-образа для бота
make build-bot

# Запуск Docker-контейнера для бота
make run-bot

# Просмотр логов бота
make logs-bot
```

### Запуск модели на vLLM

Для запуска модели на сервере vLLM используйте команды из `Makefile`:

```bash
# Запуск Docker-контейнера для vLLM
make run-vllm HUGGING_FACE_HUB_TOKEN=<your_token>

# Просмотр логов vLLM
make logs-vllm

# Остановка Docker-контейнера для vLLM
make stop-vllm
```

## 🛠️ Команды Makefile

Ниже представлен список доступных команд `Makefile`:

### Для API

- **build-api**: Сборка Docker-образа для API.
- **run-api**: Запуск Docker-контейнера для API.
- **restart-api**: Перезапуск Docker-контейнера для API.
- **stop-api**: Остановка и удаление Docker-контейнера для API.
- **clean-api**: Удаление Docker-образа для API.
- **logs-api**: Просмотр логов контейнера для API.
- **shell-api**: Вход в контейнер для API.
- **download-data**: Скачивание файлов с Google Диска для API.

### Для Telegram Bot

- **build-bot**: Сборка Docker-образа для Telegram бота.
- **run-bot**: Запуск Docker-контейнера для Telegram бота.
- **run-bot-local**: Запуск Telegram бота локально.
- **restart-bot**: Перезапуск Docker-контейнера для Telegram бота.
- **stop-bot**: Остановка и удаление Docker-контейнера для Telegram бота.
- **clean-bot**: Удаление Docker-образа для Telegram бота.
- **logs-bot**: Просмотр логов контейнера для Telegram бота.
- **shell-bot**: Вход в контейнер для Telegram бота.

### Для модели vLLM

- **run-vllm**: Запуск Docker-контейнера для vLLM.
- **stop-vllm**: Остановка Docker-контейнера для vLLM.
- **logs-vllm**: Просмотр логов контейнера для vLLM.

## 🗂️ Структура репозитория

```
rutube-ai-supporter/
├── api/                            # Директория с кодом API сервера
│   ├── data/                       # Данные для работы API (необходимо скачать с помощью make download-data)
│   │   ├── knowledge_base.xlsx     # База знаний
│   ├── models/                     # Модели машинного обучения (необходимо скачать с помощью make download-data)
│   ├── scripts/                    # Скрипты для API
│   │   ├── model_config.py         # Конфигурация модели
│   │   ├── inference.py            # Скрипт для инференса
│   │   └── api_models.py           # Скрипт для управления моделями
│   ├── pyproject.toml              # Зависимости и настройки для API
│   ├── poetry.lock                 # Блокировка версий зависимостей для API
│   ├── Dockerfile                  # Dockerfile для API сервера
│   └── main.py                     # Основной файл запуска API
├── tg_bot/                         # Директория с кодом Telegram бота
│   ├── handlers/                   # Директория для скриптов команд и сообщений бота
│   │   └── handlers.py             # Обработчики команд и сообщений бота
│   ├── .env                        # Файл окружения для Telegram бота (не добавляется в репозиторий)
│   ├── pyproject.toml              # Зависимости и настройки для Telegram бота
│   ├── poetry.lock                 # Блокировка версий зависимостей для Telegram бота
│   ├── Dockerfile                  # Dockerfile для Telegram бота
│   └── main.py                     # Основной файл запуска бота
├── Makefile                        # Makefile для автоматизации команд
├──pyproject.toml                   # Зависимости и настройки для основного проекта
├── poetry.lock                     # Блокировка версий зависимостей для основного проекта
├── .gitignore                      # Файл .gitignore для исключения файлов и директорий из репозитория
└── README.md                       # Описание проекта и инструкции
```

## 📞 Контакты

Если у вас есть вопросы или предложения, вы можете связаться с командой разработчиков:

- **Email**: r.v.lazovskiy@gmail.com
- **Telegram**: [@rvlazovskiy](https://t.me/rvlazovskiy)
