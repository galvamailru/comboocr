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
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.vlm_client import classify_image_segment

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


def convert_pdf_with_docling(pdf_path: Path) -> Dict[str, Any]:
    """
    1) Docling конвертирует PDF в документ;
    2) Извлекаются объекты (text/table/image);
    3) Возвращается общий текст + список объектов.
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


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": "comboocr"}


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
        objects: List[Dict[str, Any]] = base_result["objects"]

        enhanced_objects: List[Dict[str, Any]] = []
        for idx, obj in enumerate(objects):
            obj_out = dict(obj)
            if (obj.get("type") or "").lower() == "image" and obj.get("image_base64"):
                try:
                    img_bytes = base64.b64decode(obj["image_base64"])
                    seg_id = f"page{obj.get('page') or 0}_img{idx}"
                    cls = classify_image_segment(img_bytes, seg_id)
                    obj_out["vlm_category"] = cls.get("category")
                    obj_out["vlm_description"] = cls.get("description")
                except Exception as e:  # noqa: BLE001
                    logger.exception("comboocr: ошибка VLM для сегмента %s: %s", idx, e)
                    obj_out["vlm_category"] = "error"
                    obj_out["vlm_description"] = str(e)
            enhanced_objects.append(obj_out)

        return {
            "filename": file.filename,
            "text": base_result["text"],
            "objects": enhanced_objects,
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

