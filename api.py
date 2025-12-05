# api.py
from typing import Any, Dict
import pathlib
import time
import datetime
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from modules.anpr import ANPR

app = FastAPI(
    title="Hikvision ANPR Wrapper",
    description="HTTP API для распознавания гос-номеров по кадру камеры",
    version="1.0.0",
)

# создаём движок один раз, чтобы модели не грузились на каждый запрос
engine = ANPR()

@app.get("/health", summary="Health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/anpr", summary="Recognize Plate")
async def recognize_plate_anpr(
    file: UploadFile = File(..., description="Изображение (JPEG/PNG)"),
) -> JSONResponse:
    """
    Принимает изображение (multipart/form-data, поле: file),
    возвращает JSON с номером и метаданными:

    {
      "plate": "850ZEX15",
      "det_conf": 0.87,
      "ocr_conf": 0.91,
      "bbox": [x1, y1, x2, y2]
    }
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    result: Dict[str, Any] = engine.infer(img)
    return JSONResponse(content=result)


BASE_DIR = pathlib.Path("hik_raws")
BASE_DIR.mkdir(parents=True, exist_ok=True)

def get_paths():
    """
    Возвращает:
      time_str  - строка времени для имени файла
      RAWS_DIR  - папка для raw-запросов
      PARTS_DIR - папка для multipart-частей
      IMAGES_DIR- папка для картинок
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")  # 23-59-12_123456

    date_root = BASE_DIR / date_str
    raws_dir = date_root / "raws"
    parts_dir = date_root / "parts"
    images_dir = date_root / "images"

    for d in (raws_dir, parts_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)

    return time_str, raws_dir, parts_dir, images_dir

def parse_anpr_xml(xml_bytes: bytes) -> Dict[str, Any]:
    """
    Парсим anpr.xml от Hikvision.
    Извлекаем:
      - номер, который дала камера
      - confidenceLevel (0..1)
      - время события
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return {}

    ns = {"isapi": "http://www.isapi.org/ver20/XMLSchema"}

    def txt(path: str) -> str | None:
        return root.findtext(path, default=None, namespaces=ns)

    event_type = txt("isapi:eventType")
    date_time = txt("isapi:dateTime")

    anpr = root.find("isapi:ANPR", ns)
    if anpr is None:
        return {
            "event_type": event_type,
            "date_time": date_time,
        }

    def txt_anpr(path: str) -> str | None:
        return anpr.findtext(path, default=None, namespaces=ns)

    # для твоего XML: <licensePlate> и <originalLicensePlate> одинаковые
    plate = txt_anpr("isapi:licensePlate") or txt_anpr("isapi:originalLicensePlate")
    original_plate = txt_anpr("isapi:originalLicensePlate")

    conf_str = txt_anpr("isapi:confidenceLevel")
    camera_conf = None
    if conf_str:
        try:
            camera_conf = float(conf_str) / 100.0
        except ValueError:
            camera_conf = None

    return {
        "event_type": event_type,
        "date_time": date_time,
        "plate": plate,
        "original_plate": original_plate,
        "confidence": camera_conf,
    }


@app.post("/api/v1/anpr/hikvision")
async def hikvision_isapi(request: Request):
    # Директории вида hik_raws/YYYY-MM-DD/{raws,parts,images}
    time_str, RAWS_DIR, PARTS_DIR, IMAGES_DIR = get_paths()
    headers = dict(request.headers)

    print("=== HIKVISION REQUEST HEADERS ===")
    for k, v in headers.items():
        print(f"{k}: {v}")

    body = await request.body()

    # 1) Сохраняем сырой HTTP-запрос
    raw_path = RAWS_DIR / f"hik_raw_{time_str}.bin"
    raw_path.write_bytes(body)
    print(f"[HIK] raw saved to {raw_path}, size={len(body)} bytes")

    content_type = headers.get("content-type", "")

    # Информация камеры (из anpr.xml)
    camera_info: Dict[str, Any] = {}
    camera_xml_path: str | None = None

    # Результат нашего ANPR по detectionPicture.jpg
    model_plate: str | None = None
    model_det_conf: float | None = None
    model_ocr_conf: float | None = None
    model_bbox: Any = None
    model_image_path: str | None = None

    # === ВАРИАНТ 1: multipart/form-data (основной для Hikvision) ===
    if "multipart/form-data" in content_type:
        form = await request.form()
        found_files = 0

        for key, value in form.items():
            if hasattr(value, "filename"):
                found_files += 1
                file_bytes = await value.read()
                fname = value.filename or f"hik_file_{time_str}_{found_files}.bin"
                ftype = value.content_type or "application/octet-stream"

                print(
                    f"[HIK] part field={key}, name={fname}, "
                    f"type={ftype}, size={len(file_bytes)}"
                )

                # 2) сохраняем part как есть (xml/jpg/…)
                part_path = PARTS_DIR / f"{time_str}_{fname}"
                part_path.write_bytes(file_bytes)

                # anpr.xml → парсим номер камеры
                if fname.lower().endswith("anpr.xml"):
                    camera_info = parse_anpr_xml(file_bytes)
                    camera_xml_path = str(part_path)
                    continue

                # 3) ВСЕ изображения сохраняем в images/
                if ftype.startswith("image/"):
                    np_arr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if img is not None:
                        img_path = IMAGES_DIR / f"{time_str}_{fname}"
                        cv2.imwrite(str(img_path), img)
                        print(f"[HIK] image saved to {img_path}")

                        # Только detectionPicture.jpg → в наш ANPR
                        if fname.lower() == "detectionpicture.jpg":
                            anpr_res = engine.infer(img)

                            model_plate = anpr_res.get("plate")
                            model_det_conf = anpr_res.get("det_conf")
                            model_ocr_conf = anpr_res.get("ocr_conf")
                            model_bbox = anpr_res.get("bbox")
                            model_image_path = str(img_path)

            else:
                # текстовые части (если будут) — просто логируем
                print(f"[HIK] form field: {key} = {value}")

        # 4) Логируем детекцию в detections.log
        log_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_path": str(raw_path),
            "xml_path": camera_xml_path,
            "detection_image_path": model_image_path,
            "camera_plate": camera_info.get("plate"),
            "camera_original_plate": camera_info.get("original_plate"),
            "camera_confidence": camera_info.get("confidence"),
            "anpr_plate": model_plate,
            "anpr_det_conf": model_det_conf,
            "anpr_ocr_conf": model_ocr_conf,
            "anpr_bbox": model_bbox,
        }
        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        # 5) Ответ с двумя номерами
        return JSONResponse(
            {
                "status": "ok",
                "kind": "hikvision_anpr_multipart"
                if found_files
                else "multipart_no_files",
                "camera_plate": camera_info.get("plate"),
                "camera_original_plate": camera_info.get("original_plate"),
                "camera_confidence": camera_info.get("confidence"),
                "anpr_plate": model_plate,
                "anpr_det_conf": model_det_conf,
                "anpr_ocr_conf": model_ocr_conf,
                "anpr_bbox": model_bbox,
                "paths": {
                    "raw": str(raw_path),
                    "xml": camera_xml_path,
                    "detection_image": model_image_path,
                },
            }
        )

    # === ВАРИАНТ 2: fallback — ищем JPEG прямо в body (на всякий случай) ===
    start = body.find(b"\xff\xd8\xff")
    end = body.find(b"\xff\xd9", start + 2)

    if start == -1 or end == -1:
        # вообще нет jpeg
        return JSONResponse(
            {
                "status": "ok",
                "kind": "no_jpeg_in_body",
                "size": len(body),
            }
        )

    jpg_bytes = body[start : end + 2]
    img_path = IMAGES_DIR / f"hik_frame_{time_str}.jpg"
    img_path.write_bytes(jpg_bytes)
    print(f"[HIK] jpeg saved to {img_path}")

    np_arr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            {"status": "error", "message": "cannot decode jpeg"}, status_code=400
        )

    anpr_res = engine.infer(img)
    model_plate = anpr_res.get("plate")
    model_det_conf = anpr_res.get("det_conf")
    model_ocr_conf = anpr_res.get("ocr_conf")
    model_bbox = anpr_res.get("bbox")

    # Лог для fallback-сценария
    log_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "raw_path": str(raw_path),
        "xml_path": None,
        "detection_image_path": str(img_path),
        "camera_plate": None,
        "camera_original_plate": None,
        "camera_confidence": None,
        "anpr_plate": model_plate,
        "anpr_det_conf": model_det_conf,
        "anpr_ocr_conf": model_ocr_conf,
        "anpr_bbox": model_bbox,
    }
    log_path = BASE_DIR / "detections.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return JSONResponse(
        {
            "status": "ok",
            "kind": "hikvision_anpr_body_jpeg",
            "camera_plate": None,
            "camera_original_plate": None,
            "camera_confidence": None,
            "anpr_plate": model_plate,
            "anpr_det_conf": model_det_conf,
            "anpr_ocr_conf": model_ocr_conf,
            "anpr_bbox": model_bbox,
            "paths": {
                "raw": str(raw_path),
                "xml": None,
                "detection_image": str(img_path),
            },
        }
    )
