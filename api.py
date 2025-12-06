from typing import Any, Dict
import pathlib
import datetime
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import httpx
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

# URL внешнего сервиса, куда шлём JSON + фото
UPSTREAM_URL = "https://snowops-anpr-service.onrender.com/api/v1/anpr/events"


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


# === Работа с файловой структурой для логов ===

BASE_DIR = pathlib.Path("hik_raws")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def get_paths():
    """
    Возвращает:
      time_str  - строка времени для имени файла
      PARTS_DIR - папка для multipart-частей (xml)
      IMAGES_DIR - пока не используем, но создаём на будущее
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")  # 23-59-12

    date_root = BASE_DIR / date_str
    parts_dir = date_root / "parts"
    images_dir = date_root / "images"

    for d in (parts_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)

    return time_str, parts_dir, images_dir


# === Парсер anpr.xml от Hikvision ===

def parse_anpr_xml(xml_bytes: bytes) -> Dict[str, Any]:
    """
    Парсим anpr.xml от Hikvision.
    Извлекаем:
      - plate / original_plate
      - confidenceLevel (0..1)
      - eventType и dateTime
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


# === Отправка события и фотографий на внешний сервис ===

async def send_to_upstream(
    event_data: Dict[str, Any],
    detection_bytes: bytes | None,
    feature_bytes: bytes | None,
    license_bytes: bytes | None,
) -> Dict[str, Any]:
    """
    Отправка события и фотографий на внешний ANPR-сервис.

    Формат, ожидаемый бэкендом:
      Content-Type: multipart/form-data

      Поля формы:
        - event  (обязательное)  — JSON-строка с данными события
        - photos (опционально)   — файлы фотографий, одно или несколько полей photos

    Возвращает:
      {
        "sent": bool,
        "status": int | None,
        "error": str | None,
      }
    """
    if not UPSTREAM_URL:
        msg = "UPSTREAM_URL is not set"
        print(f"[UPSTREAM] {msg}")
        return {
            "sent": False,
            "status": None,
            "error": msg,
        }

    try:
        # event — как строка JSON в поле формы
        event_str = json.dumps(event_data, ensure_ascii=False)
        data = {
            "event": event_str,
        }

        print("[UPSTREAM] EVENT JSON:")
        print(event_str)

        # photos — список файлов под одним и тем же ключом "photos"
        files = []

        if detection_bytes:
            print(
                f"[UPSTREAM] add photo: field='photos', name='detectionPicture.jpg', "
                f"size={len(detection_bytes)}"
            )
            files.append(
                (
                    "photos",
                    ("detectionPicture.jpg", detection_bytes, "image/jpeg"),
                )
            )

        if feature_bytes:
            print(
                f"[UPSTREAM] add photo: field='photos', name='featurePicture.jpg', "
                f"size={len(feature_bytes)}"
            )
            files.append(
                (
                    "photos",
                    ("featurePicture.jpg", feature_bytes, "image/jpeg"),
                )
            )

        if license_bytes:
            print(
                f"[UPSTREAM] add photo: field='photos', name='licensePlatePicture.jpg', "
                f"size={len(license_bytes)}"
            )
            files.append(
                (
                    "photos",
                    ("licensePlatePicture.jpg", license_bytes, "image/jpeg"),
                )
            )

        async with httpx.AsyncClient(timeout=10.0) as client:
            # data + files => multipart/form-data
            resp = await client.post(UPSTREAM_URL, data=data, files=files or None)
            print(f"[UPSTREAM] status={resp.status_code}, body={resp.text[:400]}")

            return {
                "sent": resp.is_success,
                "status": resp.status_code,
                "error": None if resp.is_success else resp.text[:400],
            }

    except Exception as e:
        # Не ломаем обработку камеры, просто логируем
        print(f"[UPSTREAM] error while sending event: {e}")
        return {
            "sent": False,
            "status": None,
            "error": str(e),
        }


# === Основной хендлер для Hikvision ANPR ===

@app.post("/api/v1/anpr/hikvision")
async def hikvision_isapi(request: Request):
    # Директории вида hik_raws/YYYY-MM-DD/{parts,images}
    time_str, PARTS_DIR, IMAGES_DIR = get_paths()
    headers = dict(request.headers)

    print("=== HIKVISION REQUEST HEADERS ===")
    for k, v in headers.items():
        print(f"{k}: {v}")

    body = await request.body()
    content_type = headers.get("content-type", "")

    # Информация камеры (из anpr.xml)
    camera_info: Dict[str, Any] = {}
    camera_xml_path: str | None = None

    # Результат нашего ANPR по detectionPicture.jpg
    model_plate: str | None = None
    model_det_conf: float | None = None
    model_ocr_conf: float | None = None
    model_bbox: Any = None

    # Байты трёх картинок (их отправим дальше, но не сохраняем на диск)
    detection_bytes: bytes | None = None
    feature_bytes: bytes | None = None
    license_bytes: bytes | None = None

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

                # anpr.xml → номер камеры, сохраняем XML на диск
                if fname.lower().endswith("anpr.xml"):
                    part_path = PARTS_DIR / f"{time_str}_{fname}"
                    part_path.write_bytes(file_bytes)
                    camera_xml_path = str(part_path)
                    camera_info = parse_anpr_xml(file_bytes)
                    continue

                # Картинки держим в памяти, на диск не кладём
                if ftype.startswith("image/"):
                    lower_name = fname.lower()

                    if lower_name == "detectionpicture.jpg":
                        detection_bytes = file_bytes

                        # Гоняем через наш ANPR
                        np_arr = np.frombuffer(file_bytes, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            anpr_res = engine.infer(img)
                            model_plate = anpr_res.get("plate")
                            model_det_conf = anpr_res.get("det_conf")
                            model_ocr_conf = anpr_res.get("ocr_conf")
                            model_bbox = anpr_res.get("bbox")

                    elif lower_name == "featurepicture.jpg":
                        feature_bytes = file_bytes

                    elif lower_name == "licenseplatepicture.jpg":
                        license_bytes = file_bytes

            else:
                # текстовые части (если будут) — просто логируем
                print(f"[HIK] form field: {key} = {value}")

        # === Формируем JSON-событие в простом виде ===

        now_iso = datetime.datetime.now().isoformat()

        camera_plate = camera_info.get("plate")
        camera_conf = camera_info.get("confidence")
        event_time = camera_info.get("date_time") or now_iso

        # основной номер события для поля plate (обязателен для бэкенда)
        main_plate = model_plate or camera_plate

        event_data: Dict[str, Any] = {
            # контракт бэкенда
            "camera_id": "camera-001",  # TODO: подставь реальный ID камеры
            "event_time": event_time,
            "plate": main_plate,

            # понятные поля
            "camera_plate": camera_plate,
            "camera_confidence": camera_conf,
            "model_plate": model_plate,
            "model_det_conf": model_det_conf,
            "model_ocr_conf": model_ocr_conf,

            # доп. служебное время
            "timestamp": now_iso,
        }

        # 5) Отправляем JSON + фото на внешний сервис и получаем результат
        upstream_result = await send_to_upstream(
            event_data=event_data,
            detection_bytes=detection_bytes,
            feature_bytes=feature_bytes,
            license_bytes=license_bytes,
        )

        # 6) Логируем в detections.log (включая статус отправки)
        log_event = {
            **event_data,
            "upstream_sent": upstream_result["sent"],
            "upstream_status": upstream_result["status"],
            "upstream_error": upstream_result["error"],
        }

        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

        # 7) Ответ камере — просто "ok"
        return JSONResponse({"status": "ok"})

    # === ВАРИАНТ 2: fallback — ищем JPEG прямо в body (редкий случай) ===
    start = body.find(b"\xff\xd8\xff")
    end = body.find(b"\xff\xd9", start + 2)

    if start == -1 or end == -1:
        # вообще нет jpeg — логируем, камере просто "ok"
        log_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "kind": "no_jpeg_in_body",
            "body_size": len(body),
            "upstream_sent": False,
            "upstream_status": None,
            "upstream_error": "no_jpeg_in_body",
        }
        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

        return JSONResponse({"status": "ok"})

    jpg_bytes = body[start: end + 2]
    np_arr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        # Декод не удался — логируем, камере отдаём ошибку
        log_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "kind": "jpeg_decode_error",
            "upstream_sent": False,
            "upstream_status": None,
            "upstream_error": "cannot_decode_jpeg",
        }
        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

        return JSONResponse(
            {"status": "error", "message": "cannot decode jpeg"}, status_code=400
        )

    anpr_res = engine.infer(img)
    model_plate = anpr_res.get("plate")
    model_det_conf = anpr_res.get("det_conf")
    model_ocr_conf = anpr_res.get("ocr_conf")
    model_bbox = anpr_res.get("bbox")

    now_iso = datetime.datetime.now().isoformat()

    # тут камеры нет, только модель
    main_plate = model_plate

    event_data: Dict[str, Any] = {
        "camera_id": "camera-001",
        "event_time": now_iso,
        "plate": main_plate,
        "camera_plate": None,
        "camera_confidence": None,
        "model_plate": model_plate,
        "model_det_conf": model_det_conf,
        "model_ocr_conf": model_ocr_conf,
        "timestamp": now_iso,
    }

    # Отправляем только одну картинку как detection_picture
    upstream_result = await send_to_upstream(
        event_data=event_data,
        detection_bytes=jpg_bytes,
        feature_bytes=None,
        license_bytes=None,
    )

    log_event = {
        **event_data,
        "upstream_sent": upstream_result["sent"],
        "upstream_status": upstream_result["status"],
        "upstream_error": upstream_result["error"],
        "anpr_bbox": model_bbox,
    }

    log_path = BASE_DIR / "detections.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

    # Ответ камере — снова просто "ok"
    return JSONResponse({"status": "ok"})
