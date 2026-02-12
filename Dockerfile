# Combo OCR — Docling сегментация PDF + VLM классификация изображений.
# Сервер vLLM с vision-моделью запускается отдельно; URL задаётся через .env.
FROM python:3.11-slim

# Системные зависимости: Docling, OCR (Tesseract), pdf2image (poppler), рендеринг
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-rus \
        libmagic1 \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
