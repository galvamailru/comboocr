import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pdf2image import convert_from_path

from app.config import get_settings
from app.vlm_client import run_page_ocr

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="Combo OCR (Docling + VLM)",
    description=(
        "1) Docling сегментирует PDF (объекты с bbox); "
        "2) координаты приводятся к PDF-формату [x1, y_top, x2, y_bottom]; "
        "3) каждая страница целиком отправляется в VLM с промптом, дополненным разметкой Docling; "
        "4) VLM возвращает элементы (type, bbox, text) для улучшенного распознавания."
    ),
    version="0.2.0",
)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


def _bbox_to_list(bbox: Any) -> Optional[List[float]]:
    """Safely normalize bbox to list [left, top, right, bottom] or [x0, y0, x1, y1]."""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    if isinstance(bbox, dict):
        for keys in (("left", "top", "right", "bottom"), ("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
            if all(k in bbox for k in keys):
                try:
                    return [float(bbox[k]) for k in keys]
                except (TypeError, ValueError):
                    pass

    # Try attribute names used by Docling / docling_core BoundingBox
    for attrs in (
        ("left", "top", "right", "bottom"),
        ("l", "t", "r", "b"),
        ("x0", "y0", "x1", "y1"),
    ):
        coords: List[float] = []
        for attr in attrs:
            if hasattr(bbox, attr):
                try:
                    coords.append(float(getattr(bbox, attr)))
                except (TypeError, ValueError):
                    break
        if len(coords) == 4:
            return coords

    # Fallback: any object with 4 numeric fields (e.g. dataclass)
    if hasattr(bbox, "__iter__") and not isinstance(bbox, (str, bytes)):
        try:
            vals = list(bbox)[:4]
            if len(vals) == 4:
                return [float(v) for v in vals]
        except (TypeError, ValueError):
            pass
    return None


def bbox_to_pdf_format(bbox: Optional[List[float]]) -> Optional[List[float]]:
    """
    Привести bbox к формату PDF: [x1, y_top, x2, y_bottom].
    PDF: (0,0) — левый нижний угол, Y растёт вверх → y_top (верх блока) > y_bottom (низ блока).
    Из любых четырёх чисел: x1 <= x2, y_top >= y_bottom.
    """
    if not bbox or len(bbox) < 4:
        return None
    x1, x2 = min(float(bbox[0]), float(bbox[2])), max(float(bbox[0]), float(bbox[2]))
    y_bottom, y_top = min(float(bbox[1]), float(bbox[3])), max(float(bbox[1]), float(bbox[3]))
    return [round(x1, 1), round(y_top, 1), round(x2, 1), round(y_bottom, 1)]


def _extract_page_and_bbox(element: Any) -> tuple[Optional[int], Optional[List[float]]]:
    """
    Извлечь номер страницы и bbox из Docling-элемента.
    """
    prov = getattr(element, "prov", None)
    if isinstance(prov, list) and prov:
        for p in prov:
            page_no = getattr(p, "page_no", None) or getattr(p, "page", None)
            bbox = getattr(p, "bbox", None) or getattr(p, "bounding_box", None) or getattr(p, "box", None)
            bbox_list = _bbox_to_list(bbox)
            if bbox_list is not None or page_no is not None:
                return (int(page_no) if page_no is not None else None, bbox_list)

    # fallback: bbox на самом элементе
    for attr in ("bbox", "bounding_box", "box"):
        bbox_list = _bbox_to_list(getattr(element, attr, None))
        if bbox_list is not None:
            return (None, bbox_list)
    return (None, None)


def _iterate_items(doc: Any) -> Iterable[Tuple[Any, int]]:
    """Wrapper around Docling's iterate_items()."""
    if hasattr(doc, "iterate_items"):
        yield from doc.iterate_items()
    else:
        for el in getattr(doc, "elements", []):
            yield el, 0


def _picture_to_base64(element: PictureItem, doc: Any) -> Optional[str]:
    """Получить изображение PictureItem как PNG в base64 (вырезанный по bbox сегмент)."""
    get_image = getattr(element, "get_image", None)
    if not callable(get_image):
        return None
    try:
        img = get_image(doc)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _extract_objects(doc: Any) -> List[Dict[str, Any]]:
    """
    Извлечь объекты из Docling-документа (логика как в doclingocr):
    - таблицы (TableItem)
    - изображения (PictureItem)
    - текстовые блоки (остальное с текстом)
    - при пустом результате — fallback из export_to_dict (blocks/elements).
    """
    objects: List[Dict[str, Any]] = []

    for element, _level in _iterate_items(doc):
        page_no, bbox = _extract_page_and_bbox(element)

        if isinstance(element, TableItem):
            kind = "table"
            text = ""
            export_md = getattr(element, "export_to_markdown", None)
            if callable(export_md):
                try:
                    text = export_md(doc=doc)
                except Exception:  # noqa: BLE001
                    text = ""
        elif isinstance(element, PictureItem):
            kind = "image"
            text = getattr(element, "caption", "") or ""
        else:
            kind = getattr(element, "category", None) or "text"
            text = getattr(element, "text", None) or getattr(element, "content", None) or ""

        bbox_rounded: Optional[List[float]] = None
        if bbox and len(bbox) >= 4:
            bbox_rounded = [round(float(x), 1) for x in bbox[:4]]

        obj: Dict[str, Any] = {"type": str(kind), "page": page_no, "bbox": bbox_rounded, "text": text}

        if isinstance(element, PictureItem):
            image_base64 = _picture_to_base64(element, doc)
            if image_base64:
                obj["image_base64"] = image_base64

        objects.append(obj)

    # Fallback to exported dict if list is still empty (как в doclingocr)
    if not objects and hasattr(doc, "export_to_dict"):
        data = doc.export_to_dict()
        blocks = data.get("blocks") or data.get("elements") or []
        for block in blocks:
            raw_bbox = block.get("bbox")
            bbox_rounded = None
            if raw_bbox is not None:
                bl = _bbox_to_list(raw_bbox)
                if bl and len(bl) >= 4:
                    bbox_rounded = [round(float(x), 1) for x in bl[:4]]
            objects.append({
                "type": block.get("category") or block.get("type") or "unknown",
                "page": block.get("page_no") or block.get("page"),
                "bbox": bbox_rounded,
                "text": block.get("text", ""),
            })

    return objects


def convert_pdf_with_docling(pdf_path: Path) -> Dict[str, Any]:
    """
    Конвертация PDF через Docling — та же логика и опции, что в doclingocr,
    для одинаковой точности сегментации.
    """
    ocr_options = TesseractCliOcrOptions(lang=["rus", "eng"])
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
        generate_picture_images=True,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document
    text = doc.export_to_text()
    objects = _extract_objects(doc)
    return {"text": text, "objects": objects}


def document_to_markdown(objects: List[Dict[str, Any]], page_separator: str = "\n\n---\n\n") -> str:
    """Собрать markdown из списка объектов (по страницам). Элементы от VLM: type, text."""
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for el in objects:
        p = el.get("page") if el.get("page") is not None else 1
        by_page.setdefault(p, []).append(el)

    parts: List[str] = []
    for page_num in sorted(by_page.keys()):
        page_blocks: List[str] = []
        for el in by_page[page_num]:
            el_type = (el.get("type") or "text").lower()
            text = (el.get("text") or el.get("content") or "").strip()
            if el_type == "table":
                page_blocks.append(text if text else "*(таблица)*")
            elif el_type == "stamp":
                page_blocks.append(f"*[Печать: {text or '—'}]*")
            elif el_type == "signature":
                page_blocks.append(f"*[Подпись: {text or '—'}]*")
            elif el_type == "image":
                page_blocks.append(text if text else "*(изображение)*")
            else:
                if text:
                    page_blocks.append(text)
        parts.append("\n\n".join(page_blocks))
    return page_separator.join(parts)


def _docling_table_bboxes(objects: List[Dict[str, Any]]) -> List[List[float]]:
    """Список bbox объектов Docling с type=table (формат [x1, y_top, x2, y_bottom])."""
    return [obj["bbox"] for obj in objects if obj.get("type") == "table" and obj.get("bbox") and len(obj.get("bbox", [])) >= 4]


def _bbox_matches_docling_table(bbox: List[float], docling_table_bboxes: List[List[float]]) -> bool:
    """True, если bbox заметно пересекается с одной из таблиц Docling (сохранение type=table)."""
    if not bbox or len(bbox) < 4 or not docling_table_bboxes:
        return False
    x1, y_top, x2, y_bottom = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    for tb in docling_table_bboxes:
        if len(tb) < 4:
            continue
        tx1, ty_top, tx2, ty_bottom = float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3])
        # пересечение по осям (PDF: y_top > y_bottom)
        if x1 >= tx2 or x2 <= tx1 or y_top <= ty_bottom or y_bottom >= ty_top:
            continue
        # есть пересечение — считаем совпадением региона таблицы
        return True
    return False


def build_pages_for_ui(
    pdf_path: Path,
    objects: List[Dict[str, Any]],
    dpi: int = 150,
    page_images: Optional[List[Image.Image]] = None,
) -> List[Dict[str, Any]]:
    """Рендер страниц PDF в PNG и для каждой страницы — элементы с bbox_norm (0–1) для отрисовки в UI. Если передан page_images, повторный рендер не выполняется."""
    pages_for_ui: List[Dict[str, Any]] = []
    try:
        if page_images is None:
            page_images = convert_from_path(str(pdf_path), dpi=dpi)
        for i, pil_img in enumerate(page_images):
            page_num = i + 1
            w_px, h_px = pil_img.size
            w_pt = w_px * 72 / dpi
            h_pt = h_px * 72 / dpi
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
            elements_on_page: List[Dict[str, Any]] = []
            for obj in objects:
                if obj.get("page") != page_num:
                    continue
                bbox = obj.get("bbox")
                if not bbox or len(bbox) < 4:
                    elements_on_page.append({**obj, "bbox_norm": None})
                    continue
                # bbox в PDF-формате: [x1, y_top, x2, y_bottom]; y_top > y_bottom
                x1_pt, y_top_pt, x2_pt, y_bottom_pt = [float(bbox[j]) for j in range(4)]
                # PDF origin bottom-left → нормализованные 0–1 (top-left origin для canvas)
                x1 = x1_pt / w_pt
                x2 = x2_pt / w_pt
                y1 = 1.0 - (y_bottom_pt / h_pt)
                y2 = 1.0 - (y_top_pt / h_pt)
                elements_on_page.append({
                    **obj,
                    "bbox_norm": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                })
            pages_for_ui.append({
                "page": page_num,
                "image_base64": image_base64,
                "image_width_px": w_px,
                "image_height_px": h_px,
                "elements": elements_on_page,
            })
    except Exception:  # noqa: BLE001
        pass
    return pages_for_ui


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": "comboocr"}


@app.get("/")
def index():
    return JSONResponse(
        status_code=307,
        headers={"Location": "/static/index.html"},
        content={},
    )


@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    """
    Пайплайн:
    1) Docling сегментирует PDF → объекты (type, page, bbox).
    2) Bbox приводятся к PDF-формату [x1, y_top, x2, y_bottom].
    3) Для каждой страницы: изображение страницы + список объектов Docling для этой страницы
       отправляются в VLM с промптом извлечения; VLM возвращает элементы (type, bbox, text).
    4) Результат — объединённые элементы по всем страницам.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        return JSONResponse(
            status_code=400,
            content={"error": "Файл должен быть PDF"},
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        logger.info("comboocr: начало обработки %s, размер %s байт", file.filename, len(content))
        base_result = convert_pdf_with_docling(tmp_path)
        docling_objects = base_result["objects"]

        # Приводим bbox к PDF-формату [x1, y_top, x2, y_bottom] для каждой объекта
        for obj in docling_objects:
            bbox = obj.get("bbox")
            pdf_bbox = bbox_to_pdf_format(bbox)
            if pdf_bbox is not None:
                obj["bbox"] = pdf_bbox
            # Номер страницы обязателен для группировки
            if obj.get("page") is None and docling_objects:
                obj["page"] = 1

        settings = get_settings()
        dpi = settings.pdf_dpi
        try:
            page_images_list = convert_from_path(str(tmp_path), dpi=dpi)
        except Exception:  # noqa: BLE001
            page_images_list = []

        # Группируем объекты Docling по страницам
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for obj in docling_objects:
            p = obj.get("page") if obj.get("page") is not None else 1
            by_page.setdefault(p, []).append(obj)
        # Страницы без объектов Docling тоже обрабатываем VLM (пустой список в промпте)
        for i in range(1, len(page_images_list) + 1):
            if i not in by_page:
                by_page[i] = []

        enhanced_objects: List[Dict[str, Any]] = []
        page_prompts: List[Dict[str, Any]] = []
        vlm_system_prompt: Optional[str] = None
        for page_num in sorted(by_page.keys()):
            objects_on_page = by_page[page_num]
            if page_num < 1 or page_num > len(page_images_list):
                for obj in objects_on_page:
                    enhanced_objects.append({**obj})
                continue
            pil_img = page_images_list[page_num - 1]
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            page_png_bytes = buf.getvalue()
            try:
                result = run_page_ocr(page_png_bytes, objects_on_page, page_num)
                elements = result.get("elements") or []
                # Сохраняем тип table от Docling, если VLM вернул text для того же региона (переклассификация таблиц)
                docling_tables_by_bbox = _docling_table_bboxes(objects_on_page)
                for el in elements:
                    el["page"] = page_num
                    if el.get("type") == "text" and el.get("bbox") and len(el["bbox"]) >= 4:
                        if _bbox_matches_docling_table(el["bbox"], docling_tables_by_bbox):
                            el["type"] = "table"
                    enhanced_objects.append(el)
                if vlm_system_prompt is None and result.get("system_prompt"):
                    vlm_system_prompt = result.get("system_prompt")
                page_prompts.append({
                    "page": page_num,
                    "user_prompt": result.get("user_prompt") or "",
                    "system_prompt": result.get("system_prompt") or "",
                })
            except Exception as e:  # noqa: BLE001
                logger.exception("comboocr: ошибка VLM для страницы %s: %s", page_num, e)
                for obj in objects_on_page:
                    enhanced_objects.append({**obj, "vlm_error": str(e)})
                page_prompts.append({
                    "page": page_num,
                    "user_prompt": "",
                    "system_prompt": vlm_system_prompt or "",
                })

        # Сегментацию для UI строим из объектов Docling (как в doclingocr), а не из VLM —
        # тогда рамки на странице совпадают с doclingocr по качеству.
        pages = build_pages_for_ui(
            tmp_path, docling_objects, dpi=dpi, page_images=page_images_list
        )
        markdown = document_to_markdown(enhanced_objects)
        full_text = "\n\n".join(
            (el.get("text") or "").strip() for el in enhanced_objects if (el.get("text") or "").strip()
        )

        return {
            "filename": file.filename,
            "text": full_text,
            "objects": enhanced_objects,
            "structure": enhanced_objects,
            "docling_objects": docling_objects,
            "markdown": markdown,
            "pages": pages,
            "num_pages": len(pages),
            "page_prompts": page_prompts,
            "vlm_system_prompt": vlm_system_prompt or "",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("comboocr: ошибка обработки PDF %s: %s", file.filename, exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Не удалось обработать PDF: {exc}"},
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

