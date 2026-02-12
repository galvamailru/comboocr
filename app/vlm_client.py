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

