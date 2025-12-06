import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2
import numpy as np
from google import genai
from PIL import Image
from ultralytics import YOLO

from combined_merger import init_merger

# === Конфиг через переменные окружения ===

SNOW_VIDEO_SOURCE_URL = os.getenv("SNOW_VIDEO_SOURCE_URL", "")
SNOW_CAMERA_ID = os.getenv("SNOW_CAMERA_ID", "camera-snow")
SNOW_YOLO_MODEL_PATH = os.getenv("SNOW_YOLO_MODEL_PATH", "yolov8n.pt")

TRUCK_CLASS_ID = int(os.getenv("SNOW_TRUCK_CLASS_ID", "7"))
CONFIDENCE_THRESHOLD = float(os.getenv("SNOW_CONFIDENCE_THRESHOLD", "0.55"))

CENTER_ZONE_START_X = float(os.getenv("SNOW_CENTER_ZONE_START_X", "0.35"))
CENTER_ZONE_END_X = float(os.getenv("SNOW_CENTER_ZONE_END_X", "0.65"))
CENTER_LINE_X = float(os.getenv("SNOW_CENTER_LINE_X", "0.5"))
MIN_DIRECTION_DELTA = int(os.getenv("SNOW_MIN_DIRECTION_DELTA", "5"))

SNAPSHOT_BASE_DIR = os.getenv("SNAPSHOT_BASE_DIR", "snapshots")
SHOW_WINDOW = os.getenv("SNOW_SHOW_WINDOW", "false").lower() == "true"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_snow_thread: threading.Thread | None = None
_stop_event = threading.Event()
_gemini_client: genai.Client | None = None


# === Вспомогательные функции из старого снежного сервиса ===

def _detect_truck_bbox(frame: np.ndarray, model: YOLO) -> Optional[Tuple[int, int, int, int]]:
    """
    Находит bbox грузовика (class=TRUCK_CLASS_ID) с максимальной площадью.
    """
    best_box = None
    best_area = 0.0

    results = model(frame, verbose=False)
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            if cls_id != TRUCK_CLASS_ID or conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
    return best_box


def _check_center_zone(bbox, frame_width: int):
    """
    Проверка: центр bbox внутри центральной зоны.
    """
    x1, _, x2, _ = bbox
    center_x = x1 + (x2 - x1) // 2
    zone_start_px = int(frame_width * CENTER_ZONE_START_X)
    zone_end_px = int(frame_width * CENTER_ZONE_END_X)
    in_zone = zone_start_px < center_x < zone_end_px
    return in_zone, center_x, zone_start_px, zone_end_px


def _is_moving_left_to_right(current_center_x: int, last_center_x: Optional[int]) -> bool:
    if last_center_x is None:
        return False
    return (current_center_x - last_center_x) > MIN_DIRECTION_DELTA


def _save_frame(frame: np.ndarray):
    """
    Сохранить кадр в snapshots/YYYY-MM-DD/HH-MM-SS.jpg.
    """
    now = datetime.now(tz=timezone.utc)
    date_dir = os.path.join(SNAPSHOT_BASE_DIR, now.strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    filename = now.strftime("%H-%M-%S") + ".jpg"
    path = os.path.join(date_dir, filename)
    cv2.imwrite(path, frame)
    return path, now


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _analyze_snow_gemini(image_path: str) -> dict:
    """
    Отправка кадра в Gemini, ожидаем JSON с percentage/confidence/direction.
    """
    try:
        client = _get_gemini_client()
        image = Image.open(image_path)
        prompt = (
            "Оцени изображение кузова грузовика.\n"
            "Верни JSON с полями percentage (0-100), confidence (0..1), direction "
            "(left_to_right|right_to_left|unknown) без дополнительных комментариев.\n"
            "Пример:\n"
            "{\n"
            '  \"percentage\": 42,\n'
            '  \"confidence\": 0.9,\n'
            '  \"direction\": \"left_to_right\"\n'
            "}\n"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image, prompt],
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            data = json.loads(text)
        except Exception:
            data = {"raw": text}
        return data
    except Exception as e:
        print(f"[GEMINI] error: {e}")
        return {"error": str(e)}


def _save_analysis_json(image_path: str, timestamp: datetime, gemini_result: dict) -> str:
    json_path = image_path.rsplit(".", 1)[0] + ".json"
    payload = {
        "timestamp": timestamp.isoformat(),
        "image_path": image_path,
        "gemini": gemini_result,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return json_path


def _extract_gemini_fields(gemini_result: dict):
    percentage = None
    confidence = None
    direction = None

    if isinstance(gemini_result, dict):
        percentage = gemini_result.get("percentage")
        confidence = gemini_result.get("confidence")
        direction = gemini_result.get("direction")
        raw = gemini_result.get("raw")
    else:
        raw = None

    if (percentage is None or confidence is None or direction is None) and raw:
        raw_s = str(raw).strip()
        try:
            if raw_s.startswith("```"):
                raw_s = raw_s.strip("`")
                if raw_s.lower().startswith("json"):
                    raw_s = raw_s[4:].strip()
            parsed = json.loads(raw_s)
            percentage = parsed.get("percentage") if percentage is None else percentage
            confidence = parsed.get("confidence") if confidence is None else confidence
            direction = parsed.get("direction") if direction is None else direction
        except Exception:
            pass

    try:
        if percentage is not None:
            percentage = int(round(float(percentage)))
    except Exception:
        percentage = None

    try:
        if confidence is not None:
            confidence = float(confidence)
    except Exception:
        confidence = None

    if direction is not None:
        direction = str(direction).strip().lower()

    return percentage, confidence, direction


# === Основной цикл снежной камеры ===

def _snow_loop(upstream_url: str):
    model = YOLO(SNOW_YOLO_MODEL_PATH)
    merger = init_merger(upstream_url)

    cap = cv2.VideoCapture(SNOW_VIDEO_SOURCE_URL)
    if not cap.isOpened():
        print(f"[SNOW] cannot open video source: {SNOW_VIDEO_SOURCE_URL}")
        return

    window_name = "Snow Camera" if SHOW_WINDOW else None
    if SHOW_WINDOW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

    last_center_x = None
    event_sent_for_current_truck = False

    frame_width = None
    frame_height = None
    center_start_px = None
    center_end_px = None
    center_x_geom = None

    fail_count = 0
    MAX_FAILS = 50

    print("[SNOW] worker started")
    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            fail_count += 1
            print(f"[SNOW] read fail {fail_count}")
            if fail_count >= MAX_FAILS:
                print("[SNOW] reopening stream...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(SNOW_VIDEO_SOURCE_URL)
                fail_count = 0
            time.sleep(0.05)
            continue

        fail_count = 0
        raw_frame = frame.copy()

        if frame_width is None:
            frame_height, frame_width = frame.shape[:2]
            center_start_px = int(frame_width * CENTER_ZONE_START_X)
            center_end_px = int(frame_width * CENTER_ZONE_END_X)
            center_x_geom = int(frame_width * CENTER_LINE_X)
            print(f"[SNOW] center zone: {center_start_px}px .. {center_end_px}px")

        # Отрисовка вспомогательных линий (только для визуального контроля)
        if SHOW_WINDOW:
            cv2.line(frame, (center_x_geom, 0), (center_x_geom, frame_height), (0, 255, 255), 1)
            cv2.line(frame, (center_start_px, 0), (center_start_px, frame_height), (0, 255, 0), 2)
            cv2.line(frame, (center_end_px, 0), (center_end_px, frame_height), (0, 255, 0), 2)

        bbox = _detect_truck_bbox(raw_frame, model)
        if bbox:
            in_zone, center_x_obj, _, _ = _check_center_zone(bbox, frame_width)
            moving_right = _is_moving_left_to_right(center_x_obj, last_center_x)
            last_center_x = center_x_obj

            if SHOW_WINDOW:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if in_zone and moving_right and not event_sent_for_current_truck:
                ts = datetime.now(tz=timezone.utc)
                image_path, ts_saved = _save_frame(raw_frame)

                gemini_result = _analyze_snow_gemini(image_path)
                _save_analysis_json(image_path, ts_saved, gemini_result)

                percentage, confidence, direction = _extract_gemini_fields(gemini_result)
                if not direction:
                    direction = "left_to_right" if moving_right else "unknown"

                payload = {
                    "camera_id": SNOW_CAMERA_ID,
                    "event_time": ts_saved.replace(microsecond=0).isoformat() + "Z",
                    "snow_volume_percentage": percentage,
                    "snow_volume_confidence": confidence,
                    "snow_direction_ai": direction,
                    "snow_gemini_raw": gemini_result,
                }

                try:
                    with open(image_path, "rb") as f:
                        photo_bytes = f.read()
                except Exception as e:
                    print(f"[SNOW] cannot read snapshot {image_path}: {e}")
                    photo_bytes = None

                merger.add_snow_event(payload, photo_bytes)
                event_sent_for_current_truck = True
        else:
            event_sent_for_current_truck = False
            last_center_x = None

        if SHOW_WINDOW:
            cv2.imshow(window_name, cv2.resize(frame, (960, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                _stop_event.set()
                break

        # небольшая пауза, чтобы не грузить CPU
        time.sleep(0.005)

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("[SNOW] worker stopped")


def start_snow_worker(upstream_url: str):
    """
    Запуск снегового воркера в отдельном потоке.
    """
    global _snow_thread
    if _snow_thread is not None:
        return
    if not SNOW_VIDEO_SOURCE_URL:
        print("[SNOW] SNOW_VIDEO_SOURCE_URL is empty, snow worker disabled")
        return

    _stop_event.clear()
    _snow_thread = threading.Thread(
        target=_snow_loop,
        args=(upstream_url,),
        daemon=True,
        name="snow-worker",
    )
    _snow_thread.start()


def stop_snow_worker():
    """
    Остановка снегового воркера (используется только если нужно мягко завершить).
    """
    if _snow_thread is None:
        return
    _stop_event.set()
