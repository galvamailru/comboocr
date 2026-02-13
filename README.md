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

vLLM с vision-моделью (Qwen2.5-VL, Qwen3-VL-235B, Ministral-3 и т.д.) нужно запускать на хосте или в отдельном контейнере.

### Qwen2.5-VL-7B (одна GPU)

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000
```

### Qwen2.5-VL-32B-Instruct (2–4× GPU)

Модель ~32B параметров, веса в FP16/BF16 ~64 GB. Рекомендуется 2× A100 80GB или 4× A100 40GB.

- **2× GPU (например, 2× A100 80GB):**

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data
```

- **4× GPU (A100 40GB или 80GB), только изображения (без видео):**

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

Опция `--mm-encoder-tp-mode data` разносит vision-encoder по data-parallel и снижает нагрузку на TP. Для экономии памяти можно задать `--max-model-len 65536`.

В `.env` comboocr укажите: `VLLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct`.

### Qwen3-VL-235B-A22B (8× GPU, ~80 GB каждая)

Модель MoE (~235B параметров, ~22B активных). Требуется несколько GPU (рекомендуется 8× H100 80GB или аналог).

**Установка (vLLM ≥ 0.11, утилиты Qwen-VL):**

```bash
pip install -U vllm "qwen-vl-utils>=0.0.14"
```

**Запуск (примеры по железу):**

- **H100, только изображения (без видео), FP8, 4 GPU:**

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128
```

- **H100, FP8, 8 GPU (image + video):**

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling
```

- **A100 / H100, BF16, 8 GPU:**

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
```

После старта vLLM в `.env` comboocr укажите имя модели так, как его возвращает API (обычно совпадает с путём на Hugging Face):

```env
VLLM_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
# или для FP8:
# VLLM_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```

Проверка: `curl http://localhost:8000/v1/models` — в ответе должно быть нужное `id` модели.

## Переменные окружения (.env)

| Переменная | Описание |
|------------|----------|
| `VLLM_BASE_URL` | URL OpenAI-совместимого API vLLM (обязательно указать адрес хоста при работе в Docker) |
| `VLLM_MODEL` | Имя модели на vLLM |
| `VLLM_API_KEY` | Опционально |
| `VLLM_TIMEOUT_SECONDS` | Таймаут запроса к VLM (сек) |
| `VLLM_MAX_TOKENS` | Макс. токенов ответа |
