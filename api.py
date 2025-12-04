# api.py
from typing import Any, Dict
import pathlib
import time
import datetime
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
    time_str = now.strftime("%H-%M-%S_%f")  # 23-59-12_123456

    date_root = BASE_DIR / date_str
    raws_dir = date_root / "raws"
    parts_dir = date_root / "parts"
    images_dir = date_root / "images"

    for d in (raws_dir, parts_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)

    return time_str, raws_dir, parts_dir, images_dir

@app.post("/api/v1/anpr/hikvision")
async def hikvision_isapi(request: Request):
    time_str, RAWS_DIR, PARTS_DIR, IMAGES_DIR = get_paths()
    headers = dict(request.headers)

    print("=== HIKVISION REQUEST HEADERS ===")
    for k, v in headers.items():
        print(f"{k}: {v}")

    body = await request.body()

    # 1) сохраняем сырой запрос
    raw_path = RAWS_DIR / f"hik_raw_{time_str}.bin"
    raw_path.write_bytes(body)
    print(f"[HIK] raw saved to {raw_path}, size={len(body)} bytes")

    content_type = headers.get("content-type", "")

    # === Вариант 1: multipart/form-data ===
    if "multipart/form-data" in content_type:
        form = await request.form()
        results = []
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

                # 2) сохраняем сам part (как пришёл)
                part_path = PARTS_DIR / f"{time_str}_{fname}"

                part_path.write_bytes(file_bytes)

                # 3) если это картинка — декодируем и гоняем через ANPR
                if ftype.startswith("image/"):
                    np_arr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if img is not None:
                        # сохраняем исходный кадр
                        img_path = IMAGES_DIR / f"{time_str}_{fname}"

                        cv2.imwrite(str(img_path), img)
                        print(f"[HIK] image saved to {img_path}")

                        anpr_res = engine.infer(img)

                        results.append(
                            {
                                "field": key,
                                "filename": str(img_path),
                                "plate": anpr_res.get("plate"),
                                "det_conf": anpr_res.get("det_conf"),
                                "ocr_conf": anpr_res.get("ocr_conf"),
                                "bbox": anpr_res.get("bbox"),
                            }
                        )

            else:
                # текстовые поля из multipart — просто логируем
                print(f"[HIK] form field: {key} = {value}")

        if not found_files:
            return JSONResponse(
                {
                    "status": "ok",
                    "kind": "multipart_no_files",
                    "raw_size": len(body),
                }
            )

        return JSONResponse(
            {
                "status": "ok",
                "kind": "multipart_with_files",
                "files": results,
            }
        )

    # === Вариант 2: fallback — ищем JPEG прямо в теле ===
    start = body.find(b"\xff\xd8\xff")
    end = body.find(b"\xff\xd9", start + 2)

    if start == -1 or end == -1:
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

    result = engine.infer(img)
    return JSONResponse(
        {
            "status": "ok",
            "kind": "jpeg_in_body",
            "filename": str(img_path),
            "plate": result.get("plate"),
            "det_conf": result.get("det_conf"),
            "ocr_conf": result.get("ocr_conf"),
            "bbox": result.get("bbox"),
        }
    )