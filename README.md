# Combo OCR (Docling + VLM)

**Comboocr** — сервис с уже встроенной поддержкой VLM: загружаете PDF, comboocr сегментирует его через **Docling** (текст, таблицы, изображения с bbox) и отправляет каждую страницу в **VLM** для распознавания и классификации (текст, таблица, изображение, печать, подпись и т.д.). Результат — структурированные объекты, markdown и визуализация сегментации.

## Быстрый старт

### 1. Запуск comboocr

**Через Docker (рекомендуется):**

```bash
cp .env.example .env
# В .env укажите VLLM_BASE_URL и VLLM_MODEL (см. ниже)
docker-compose up -d --build
```

API: **http://localhost:8020**  
Пример: `curl -X POST "http://localhost:8020/parse" -F "file=@document.pdf"`

**Локально (без Docker):**

```bash
cp .env.example .env
# Отредактируйте .env: VLLM_BASE_URL=http://localhost:8000/v1 и VLLM_MODEL=...
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8020
```

### 2. Подключение VLM-бэкенда

Comboocr обращается к VLM по **OpenAI-совместимому API** (vLLM и др.). Нужно:

1. Запустить сервер с vision-моделью (на этом же хосте или отдельной машине).
2. В `.env` comboocr указать его адрес и имя модели.

| Переменная | Описание |
|------------|----------|
| `VLLM_BASE_URL` | URL API (например `http://localhost:8000/v1`). В Docker: Windows/Mac — `http://host.docker.internal:8000/v1`, Linux — `http://172.17.0.1:8000/v1` |
| `VLLM_MODEL` | Имя модели, как в ответе `curl http://<vllm>:8000/v1/models` (обычно путь на Hugging Face) |
| `VLLM_API_KEY` | По желанию (vLLM часто без ключа) |
| `VLLM_TIMEOUT_SECONDS` | Таймаут запроса к VLM (сек) |
| `VLLM_MAX_TOKENS` / `VLLM_MAX_TOKENS_TABLE` | Лимит токенов ответа |
| `PDF_DPI` | DPI при рендере страниц PDF (влияет на размер картинки для VLM) |

Проверка бэкенда: `curl http://localhost:8000/v1/models` — в списке должен быть нужный `id` модели.

**Любая vision-модель в vLLM:** запустите сервер своей командой (как в примере ниже), затем укажите в `.env` тот же `VLLM_MODEL`, что возвращает API:

```bash
vllm serve <org>/<model-name> --host 0.0.0.0 --port 8000 ...
```

В `.env`: `VLLM_BASE_URL=http://localhost:8000/v1`, `VLLM_MODEL=<org>/<model-name>` (например `mistralai/Ministral-3-14B-Instruct-2512`). Модель должна поддерживать ввод изображений в chat API.

---

## Варианты VLM-бэкенда (vLLM)

Ниже — как запустить vLLM с разными моделями. После старта укажите в `.env` соответствующий `VLLM_MODEL`.

### Qwen2.5-VL-7B (одна GPU)

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000
```

В `.env`: `VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`

### Qwen2.5-VL-32B-Instruct (2–4× GPU)

Модель ~32B параметров. Рекомендуется 2× A100 80GB или 4× A100 40GB.

**2× GPU:**
```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data
```

**4× GPU (только изображения):**
```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

В `.env`: `VLLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct`

### Qwen3-VL-235B-A22B (8× GPU, ~80 GB каждая)

Модель MoE (~235B параметров, ~22B активных). Установка: `pip install -U vllm "qwen-vl-utils>=0.0.14"`

**H100, только изображения, FP8, 4 GPU:**
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128
```

**A100 / H100, BF16, 8 GPU:**
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
```

В `.env`: `VLLM_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct` или `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`

### DeepSeek-OCR-2 (отдельный сценарий)

Модель **DeepSeek-OCR-2** (~3B) — отдельный OCR с другим API и промптами. В официальном репозитории нет стандартного `vllm serve` с OpenAI-совместимым API; идёт batch-обработка папок (изображения/PDF). Для использования с comboocr потребовался бы отдельный HTTP-обёртка (например, FastAPI вокруг `model.infer()` из Transformers) и адаптер запросов. Официальная установка и скрипты — в [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) (vLLM 0.8.5, CUDA 11.8, `run_dpsk_ocr2_image.py` / `run_dpsk_ocr2_pdf.py`).

---

## Итоговая схема

1. **Запустить VLM** (vLLM с одной из моделей выше) на порту 8000.
2. **В `.env` comboocr** задать `VLLM_BASE_URL` и `VLLM_MODEL`.
3. **Запустить comboocr** (Docker или `uvicorn`).
4. Отправлять PDF на `POST /parse` — comboocr сам вызовет Docling и VLM и вернёт объекты, markdown и страницы для UI.
