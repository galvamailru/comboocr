"""
VLM: отправка целой страницы PDF с промптом, дополненным разметкой Docling (тип, страница, bbox в PDF-координатах).
Один запрос на страницу → JSON с элементами (type, bbox, text).
"""
import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


PAGE_OCR_SYSTEM_PROMPT = """У вас есть разметка документа в виде списка объектов. Каждый объект задан как:
тип объекта, страница, координаты,
где координаты — четыре числа: x1, y_top, x2, y_bottom,
и система координат стандартная для PDF:
начало (0, 0) — левый нижний угол страницы,
ось X → вправо, ось Y ↑ вверх,
значит: чем больше y_top, тем выше расположен блок.

Также приложены изображения всех страниц.

Ваша задача:
Для каждого объекта определите его тип по содержимому и визуальному контексту, строго по следующим правилам:

text: любой связный текстовый блок — заголовки, абзацы, списки, колонтитулы, номера страниц, подписи к рисункам. Группируйте только логически связанные фрагменты (например, один пункт списка, один абзац). Не объединяйте всю страницу в один элемент.

image: графические объекты без структурированного текста — логотипы, фотографии, диаграммы, иконки. Если внутри есть краткий текст (напр., «Рис. 1»), укажите его в "text" как описание. Внимательно проверь, что изображено на изображении, и расскажи об этом в его text-описании.

table: явные таблицы с ячейками, строками и столбцами (даже без сетки). В "text" верните Markdown-таблицу (|...|), восстанавливая структуру.

stamp: официальные оттиски (круглые, прямоугольные, овальные) с текстом внутри. Извлеките весь видимый текст максимально полно — не заменяйте на «штамп». Если подпись рукой внутри штампа — создайте два элемента: stamp + signature.

signature: исключительно рукописные автографы/росчерки. Не путать с печатным текстом.

❌ Не включайте водяные знаки, фоновые узоры, артефакты без смысла.

Сгруппируйте и объедините только те текстовые фрагменты, которые:
- имеют близкие y_top (разница < 15 pt),
- выровнены по x1 или образуют последовательность (например, нумерованный список),
- визуально принадлежат одному абзацу/пункту.

Отсортируйте элементы внутри каждой страницы:
- по y_top убыванию (сверху вниз),
- при равных y_top — по x1 возрастанию (слева направо).

Верните только JSON, без пояснений, в строгом формате:
{
  "page_rotation_degrees": 0,
  "elements": [
    {
      "type": "text",
      "bbox": [x1, y_top, x2, y_bottom],
      "text": "Содержимое блока"
    },
    ...
  ]
}

page_rotation_degrees всегда 0 (если не указано иное).
"text" — строка, без экранирования \\n внутри (используйте \\n для переносов строк в одном блоке).
Не добавляйте поля, кроме указанных.
Действуйте только на основе предоставленных координат и изображений. Не выдумывайте текст — выводите только то, что видно или логически следует из макета."""


def _build_objects_list_for_prompt(objects: List[Dict[str, Any]]) -> str:
    """Форматирует список объектов для вставки в промпт. bbox уже в формате [x1, y_top, x2, y_bottom]."""
    lines = []
    for i, obj in enumerate(objects):
        t = obj.get("type") or "unknown"
        p = obj.get("page") or "?"
        bbox = obj.get("bbox")
        if bbox and len(bbox) >= 4:
            bstr = ", ".join(f"{float(x):.1f}" for x in bbox[:4])
            lines.append(f"  {i+1}. type={t}, page={p}, bbox=[{bstr}]")
        else:
            lines.append(f"  {i+1}. type={t}, page={p}, bbox=нет координат")
    return "\n".join(lines) if lines else "  (нет объектов)"


def _extract_json_from_response(raw: str) -> Optional[str]:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        return m.group(1).strip()
    if raw.startswith("{"):
        return raw
    start = raw.find("{")
    if start >= 0:
        return raw[start:]
    return None


def run_page_ocr(
    page_image_png_bytes: bytes,
    docling_objects_for_page: List[Dict[str, Any]],
    page_num: int,
) -> Dict[str, Any]:
    """
    Отправить изображение одной страницы и разметку Docling (объекты с bbox в PDF-формате)
    в VLM; вернуть распознанные элементы.
    Returns: {"elements": [...], "page_rotation_degrees": float, "raw": str}
    Каждый element: {"type": str, "bbox": [x1, y_top, x2, y_bottom], "text": str}
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

    b64 = base64.b64encode(page_image_png_bytes).decode("ascii")
    payload_kb = (len(b64) * 3 // 4) // 1024
    max_tokens = getattr(settings, "vllm_max_tokens_table", None) or getattr(settings, "vllm_max_tokens", 2048)
    logger.info(
        "VLM: страница %s — отправка (PNG ~%s КБ, объектов Docling: %s)",
        page_num,
        payload_kb,
        len(docling_objects_for_page),
    )

    objects_block = _build_objects_list_for_prompt(docling_objects_for_page)
    user_text = (
        f"Текущая страница: {page_num}. Изображение страницы приложено.\n\n"
        f"Объекты Docling для этой страницы (координаты в PDF: x1, y_top, x2, y_bottom):\n{objects_block}\n\n"
        "Верните JSON с полем \"elements\" для элементов этой страницы (и \"page_rotation_degrees\": 0)."
    )

    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": user_text},
    ]

    try:
        resp = client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {"role": "system", "content": PAGE_OCR_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            timeout=settings.vllm_timeout_seconds,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("VLM: ошибка для страницы %s: %s", page_num, e)
        return {
            "elements": [],
            "page_rotation_degrees": 0.0,
            "raw": f"VLM error: {e}",
            "user_prompt": user_text,
            "system_prompt": PAGE_OCR_SYSTEM_PROMPT,
        }

    choice = resp.choices[0] if resp.choices else None
    raw = (choice.message.content if choice and choice.message else "").strip()
    logger.info("VLM: страница %s — ответ получен, %s символов", page_num, len(raw))

    extracted = _extract_json_from_response(raw)
    if not extracted:
        return {
            "elements": [],
            "page_rotation_degrees": 0.0,
            "raw": raw,
            "user_prompt": user_text,
            "system_prompt": PAGE_OCR_SYSTEM_PROMPT,
        }
    normalized = extracted.replace(",]", "]").replace(",}", "}")
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        return {
            "elements": [],
            "page_rotation_degrees": 0.0,
            "raw": raw,
            "user_prompt": user_text,
            "system_prompt": PAGE_OCR_SYSTEM_PROMPT,
        }
    elements = data.get("elements") if isinstance(data.get("elements"), list) else []
    rotation = float(data.get("page_rotation_degrees", 0) or 0)
    return {
        "elements": elements,
        "page_rotation_degrees": rotation,
        "raw": raw,
        "user_prompt": user_text,
        "system_prompt": PAGE_OCR_SYSTEM_PROMPT,
    }
