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

from app.vlm_client import classify_image_segment, run_table_ocr

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="Combo OCR (Docling + VLM)",
    description=(
        "1) Docling сегментирует PDF в текст, таблицы, изображения; "
        "2) текст и таблицы берутся из Docling; "
        "3) для изображений вызывается VLM для классификации (печать, подпись, логотип и т.п.)."
    ),
    version="0.1.0",
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
    Извлечь объекты из Docling-документа:
    - таблицы (TableItem)
    - изображения (PictureItem)
    - текстовые блоки (остальное с текстом)
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

    return objects


def _make_pipeline_options() -> PdfPipelineOptions:
    """Создать опции пайплайна; при поддержке — не пропускать колонтитулы (header/footer)."""
    ocr_options = TesseractCliOcrOptions(lang=["rus", "eng"])
    base_kw: Dict[str, Any] = {
        "do_ocr": True,
        "ocr_options": ocr_options,
        "generate_picture_images": True,
    }
    # Включить захват колонтитулов, если Docling поддерживает соответствующую опцию
    extra_opts = [
        ("skip_header_footer", False),
        ("skip_header_footers", False),
        ("include_headers_footers", True),
    ]
    for opt_name, opt_value in extra_opts:
        try:
            pipeline_options = PdfPipelineOptions(**{**base_kw, opt_name: opt_value})
            logger.info("comboocr: используется опция %s=%s для захвата колонтитулов", opt_name, opt_value)
            return pipeline_options
        except TypeError:
            continue
    return PdfPipelineOptions(**base_kw)


def convert_pdf_with_docling(pdf_path: Path) -> Dict[str, Any]:
    """
    1) Docling конвертирует PDF в документ;
    2) Извлекаются объекты (text/table/image), включая колонтитулы при поддержке;
    3) Возвращается общий текст + список объектов.
    """
    pipeline_options = _make_pipeline_options()
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
    """Собрать markdown из списка объектов (по страницам). Для image используем vlm_category и vlm_description."""
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
            elif el_type == "image":
                cat = el.get("vlm_category") or ""
                desc = (el.get("vlm_description") or "").strip()
                parts_img = []
                if cat or desc:
                    parts_img.append(f"*[Изображение: {cat}" + (f" — {desc}" if desc else "") + "]*")
                if el.get("vlm_handwritten_text"):
                    parts_img.append(f"Рукописный текст: {el.get('vlm_handwritten_text')}")
                if el.get("vlm_has_signature"):
                    parts_img.append("*[Подпись]*")
                if el.get("vlm_date"):
                    parts_img.append(f"Дата: {el.get('vlm_date')}")
                if parts_img:
                    page_blocks.append(" ".join(parts_img))
                else:
                    page_blocks.append(text if text else "*(изображение)*")
            else:
                if text:
                    page_blocks.append(text)
        parts.append("\n\n".join(page_blocks))
    return page_separator.join(parts)


def _crop_table_from_page_image(pil_img: Image.Image, bbox: List[float], dpi: int = 150) -> bytes:
    """
    Вырезать область таблицы из изображения страницы.
    bbox: [left, top, right, bottom] в PDF-точках (origin bottom-left, y вверх).
    Возвращает PNG bytes.
    """
    if not bbox or len(bbox) < 4:
        return b""
    w_px, h_px = pil_img.size
    w_pt = w_px * 72 / dpi
    h_pt = h_px * 72 / dpi
    left_pt, top_pt, right_pt, bottom_pt = [float(bbox[i]) for i in range(4)]
    # PDF: y вверх → в пикселях (y вниз): crop_y_upper = (h_pt - bottom_pt) / h_pt * h_px
    x1 = max(0, min(w_px, left_pt * w_px / w_pt))
    x2 = max(0, min(w_px, right_pt * w_px / w_pt))
    y1 = max(0, min(h_px, (h_pt - bottom_pt) * h_px / h_pt))
    y2 = max(0, min(h_px, (h_pt - top_pt) * h_px / h_pt))
    if x2 <= x1 or y2 <= y1:
        return b""
    cropped = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


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
                left, top, right, bottom = [float(bbox[j]) for j in range(4)]
                # PDF: origin bottom-left → нормализованные 0–1 (top-left origin для canvas)
                x1 = left / w_pt
                x2 = right / w_pt
                y1 = 1.0 - (bottom / h_pt)
                y2 = 1.0 - (top / h_pt)
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
    Основной маршрут:
    - сегментация PDF Docling'ом (text, table, image);
    - для text/table берём текст Docling;
    - для image отправляем crop в VLM для классификации (печать/подпись/логотип/...).
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
        objects = base_result["objects"]

        # Один раз рендерим страницы: для кропа таблиц и для UI
        dpi = 150
        try:
            page_images_list = convert_from_path(str(tmp_path), dpi=dpi)
        except Exception:  # noqa: BLE001
            page_images_list = []

        enhanced_objects: List[Dict[str, Any]] = []
        for idx, obj in enumerate(objects):
            obj_out = dict(obj)
            obj_type = (obj.get("type") or "").lower()

            if obj_type == "table":
                page_no = obj.get("page")
                bbox = obj.get("bbox")
                if page_no is not None and bbox and len(bbox) >= 4 and 1 <= page_no <= len(page_images_list):
                    try:
                        pil_page = page_images_list[page_no - 1]
                        crop_png = _crop_table_from_page_image(pil_page, bbox, dpi=dpi)
                        if crop_png:
                            seg_id = f"page{page_no}_table{idx}"
                            table_result = run_table_ocr(crop_png, seg_id)
                            vlm_md = (table_result.get("markdown") or "").strip()
                            if vlm_md:
                                obj_out["text"] = vlm_md
                                obj_out["vlm_table_markdown"] = vlm_md
                            obj_out["docling_text"] = obj.get("text") or ""
                    except Exception as e:  # noqa: BLE001
                        logger.exception("comboocr: ошибка VLM для таблицы %s: %s", idx, e)
                        obj_out["vlm_table_error"] = str(e)
                        obj_out["docling_text"] = obj.get("text") or ""
                else:
                    obj_out["docling_text"] = obj.get("text") or ""

            elif obj_type == "image" and obj.get("image_base64"):
                try:
                    img_bytes = base64.b64decode(obj["image_base64"])
                    seg_id = f"page{obj.get('page') or 0}_img{idx}"
                    cls = classify_image_segment(img_bytes, seg_id)
                    obj_out["vlm_category"] = cls.get("category")
                    obj_out["vlm_description"] = cls.get("description")
                    obj_out["vlm_handwritten_text"] = cls.get("handwritten_text")
                    obj_out["vlm_has_signature"] = cls.get("has_signature")
                    obj_out["vlm_date"] = cls.get("date")
                except Exception as e:  # noqa: BLE001
                    logger.exception("comboocr: ошибка VLM для сегмента %s: %s", idx, e)
                    obj_out["vlm_category"] = "error"
                    obj_out["vlm_description"] = str(e)

            enhanced_objects.append(obj_out)

        pages = build_pages_for_ui(tmp_path, enhanced_objects, dpi=dpi, page_images=page_images_list)
        markdown = document_to_markdown(enhanced_objects)

        return {
            "filename": file.filename,
            "text": base_result["text"],
            "objects": enhanced_objects,
            "structure": enhanced_objects,
            "markdown": markdown,
            "pages": pages,
            "num_pages": len(pages),
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

