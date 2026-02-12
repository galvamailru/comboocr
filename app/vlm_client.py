"""
Simple VLM client for classifying cropped document image segments (печать, подпись, логотип, фото и т.п.).

Uses OpenAI-compatible API served by vLLM (Qwen-VL / Ministral / etc.).
"""
import base64
import logging
from typing import Any, Dict, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


SEGMENT_SYSTEM_PROMPT = """You are a precise classifier of small document image fragments.

You receive ONE small image fragment cut from a scanned document.
Your task is to determine WHAT this fragment is.

Return ONLY a SHORT JSON object with the following shape:
{
  "category": "<one of: stamp, signature, logo, text_block, table_fragment, photo, other>",
  "description": "<1 short Russian phrase describing what is visible>"
}

Rules:
- "stamp": round or rectangular stamps with company/organization text.
- "signature": handwritten or stylized personal signature (without the descriptive text like "Подпись").
- "logo": company / brand logo (icon + wordmark etc.), not a general picture.
- "text_block": if this is mostly text without clear borders as a separate object.
- "table_fragment": grid-like structure of rows/columns.
- "photo": real-world photo (people, objects, scenes).
- "other": everything that does not fit above.

Do NOT wrap the JSON in markdown. Do NOT add explanations. ONLY the JSON object.
"""


TABLE_OCR_SYSTEM_PROMPT = """You are an OCR system for document tables.

You receive ONE image: a cropped region of a document page containing a TABLE (rows and columns).
The table may contain:
- Printed text (typewriter, printer)
- Handwritten text (рукописный текст)

Your task: extract ALL text from the table and return it as a single Markdown table.

Rules:
- Use pipe | to separate columns.
- Use a row of dashes with pipes (e.g. |---|---|) after the header row.
- Preserve row and column structure. If a cell is empty, leave it empty between pipes.
- Recognize both printed and handwritten text; transcribe handwritten text as accurately as possible.
- Use Russian and English as in the source.
- Do NOT add any text before or after the table (no "Here is the table", no explanations).
- Output ONLY the Markdown table.
"""

TABLE_OCR_USER_PROMPT = """Extract the table from this image as Markdown. Include all text (printed and handwritten). Output only the Markdown table, nothing else."""


def run_table_ocr(
    image_png_bytes: bytes,
    segment_id: str,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Отправить вырезанное изображение таблицы в VLM; вернуть Markdown-таблицу.
    Returns: {"markdown": str, "raw": str}
    """
    settings = get_settings()
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai package: pip install openai") from e

    client = OpenAI(
        base_url=settings.vllm_base_url.rstrip("/"),
        api_key=settings.vllm_api_key or "dummy",
    )

    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    payload_size_kb = (len(b64) * 3 // 4) // 1024
    tokens_limit = max_tokens if max_tokens is not None else settings.vllm_max_tokens_table
    logger.info(
        "VLM: отправка таблицы %s (размер PNG ~%s КБ, max_tokens=%s)",
        segment_id,
        payload_size_kb,
        tokens_limit,
    )

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": TABLE_OCR_USER_PROMPT},
    ]

    try:
        resp = client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {"role": "system", "content": TABLE_OCR_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=tokens_limit,
            timeout=settings.vllm_timeout_seconds,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("VLM: ошибка при распознавании таблицы %s: %s", segment_id, e)
        return {"markdown": "", "raw": f"VLM error: {e}"}

    choice = resp.choices[0] if resp.choices else None
    raw = (choice.message.content if choice and choice.message else "").strip()
    logger.info("VLM: таблица %s — ответ получен, длина %s символов", segment_id, len(raw))
    return {"markdown": raw, "raw": raw}


def classify_image_segment(image_png_bytes: bytes, segment_id: str) -> Dict[str, Any]:
    """
    Send one cropped image segment to VLM and return a dict:
    {"category": str, "description": str, "raw": str}
    """
    settings = get_settings()
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - runtime dependency
        raise RuntimeError("Install openai package: pip install openai") from e

    client = OpenAI(
        base_url=settings.vllm_base_url.rstrip("/"),
        api_key=settings.vllm_api_key or "dummy",
    )

    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    payload_size_kb = (len(b64) * 3 // 4) // 1024
    logger.info(
        "VLM: отправка сегмента %s (размер PNG ~%s КБ) в модель %s",
        segment_id,
        payload_size_kb,
        settings.vllm_model,
    )

    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=settings.vllm_max_tokens,
            timeout=settings.vllm_timeout_seconds,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("VLM: ошибка при классификации сегмента %s: %s", segment_id, e)
        return {"category": "error", "description": f"VLM error: {e}", "raw": ""}

    choice = resp.choices[0] if resp.choices else None
    raw = (choice.message.content if choice and choice.message else "") or ""
    raw = raw.strip()
    logger.info("VLM: сегмент %s — ответ получен, длина %s символов", segment_id, len(raw))

    # Минимальный JSON-парсинг без сложных восстановлений
    import json

    parsed: Optional[Dict[str, Any]] = None
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw.replace(",}", "}"))
        except json.JSONDecodeError:
            parsed = None

    result: Dict[str, Any] = {
        "category": (parsed or {}).get("category") or "other",
        "description": (parsed or {}).get("description") or raw[:200],
        "raw": raw,
    }
    return result

