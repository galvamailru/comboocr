# Combo OCR (Docling + VLM)

Сервис объединяет **Docling** (сегментация PDF: текст, таблицы, изображения) и **VLM** (классификация вырезанных изображений: печать, подпись, логотип и т.п.). Сервер vLLM с vision-моделью запускается **отдельно**.

## Запуск через Docker (docker-compose)

1. Скопируйте пример переменных окружения и при необходимости отредактируйте (URL vLLM, модель):

   ```bash
   cp .env.example .env
   ```

   В `.env` обязательно укажите адрес, по которому контейнер достучится до vLLM:
   - **Windows/Mac:** `VLLM_BASE_URL=http://host.docker.internal:8000/v1`
   - **Linux:** `VLLM_BASE_URL=http://172.17.0.1:8000/v1` (или IP хоста)

2. Соберите и запустите сервис:

   ```bash
   docker-compose up -d --build
   ```

3. API доступен по адресу **http://localhost:8020** (порт 8020 снаружи, 8000 внутри контейнера).

   Пример запроса:

   ```bash
   curl -X POST "http://localhost:8020/parse" -F "file=@document.pdf"
   ```

## Запуск vLLM отдельно

vLLM с vision-моделью (Qwen2.5-VL, Ministral-3 и т.д.) нужно запускать на хосте или в отдельном контейнере. Пример на хосте:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000
```

Убедитесь, что в `.env` указан правильный `VLLM_BASE_URL` и что имя модели совпадает с тем, что возвращает `curl http://localhost:8000/v1/models`.

## Переменные окружения (.env)

| Переменная | Описание |
|------------|----------|
| `VLLM_BASE_URL` | URL OpenAI-совместимого API vLLM (обязательно указать адрес хоста при работе в Docker) |
| `VLLM_MODEL` | Имя модели на vLLM |
| `VLLM_API_KEY` | Опционально |
| `VLLM_TIMEOUT_SECONDS` | Таймаут запроса к VLM (сек) |
| `VLLM_MAX_TOKENS` | Макс. токенов ответа |
