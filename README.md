# Unified ANPR + Snow Service

One service that:
- Accepts Hikvision ANPR webhooks (FastAPI) and extracts plate data (YOLOv8 + PaddleOCR).
- Continuously watches a snow RTSP camera, detects trucks (YOLOv8), estimates snow volume via Gemini, and buffers those events.
- Merges the latest snow event with the next plate event (snow -> plate order) and sends a single multipart request to the upstream endpoint with full JSON + all photos.

## Components
- `api.py`: FastAPI app with endpoints `GET /health`, `POST /anpr`, `POST /api/v1/anpr/hikvision`. On startup can launch the snow worker.
- `snow_worker.py`: Background thread for the snow camera (RTSP), truck detection, Gemini analysis, snapshot saving, and publishing snow events to the merger.
- `combined_merger.py`: In-memory buffer (TTL + window) to match snow events with plate events and send one multipart to `UPSTREAM_URL`.
- `modules/anpr.py`: Plate detection/OCR pipeline.

## Environment variables (.env example)
```
UPSTREAM_URL=https://snowops-anpr-service.onrender.com/api/v1/anpr/events

# merge timing
MERGE_WINDOW_SECONDS=30      # max allowed delta (snow earlier, plate later) to merge
MERGE_TTL_SECONDS=60         # how long to keep unmatched snow events

# snow worker
ENABLE_SNOW_WORKER=true
SNOW_VIDEO_SOURCE_URL=rtsp://user:pass@host:port/Streaming/Channels/101
SNOW_CAMERA_ID=camera-snow
SNOW_YOLO_MODEL_PATH=yolov8n.pt
SNAPSHOT_BASE_DIR=snapshots
SNOW_CENTER_ZONE_START_X=0.35
SNOW_CENTER_ZONE_END_X=0.65
SNOW_CENTER_LINE_X=0.5
SNOW_MIN_DIRECTION_DELTA=5
SNOW_SHOW_WINDOW=false       # set true to see preview window

# Gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate   # or source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --env-file .env
```
- Snow worker starts automatically if `ENABLE_SNOW_WORKER=true`.
- Stop with Ctrl+C (background worker stops with the app).

## API endpoints
- `GET /health` -> `{"status": "ok"}`
- `POST /anpr` -> accepts `multipart/form-data` field `file` (JPEG/PNG), returns plate JSON.
- `POST /api/v1/anpr/hikvision` -> Hikvision webhook (multipart with `anpr.xml` + images or raw JPEG fallback). This triggers plate inference and merging.

## Upstream request (single merged event)
Multipart `POST {UPSTREAM_URL}` with:
- Field `event`: JSON string. Keys include:
  - `camera_id` (plate camera id)
  - `event_time` (from camera XML or now, ISO8601)
  - `plate`, `camera_plate`, `camera_confidence`, `model_plate`, `model_det_conf`, `model_ocr_conf`, `timestamp`
  - If matched snow: `snow_volume_percentage`, `snow_volume_confidence`, `snow_gemini_raw`, `matched_snow=true`
  - Note: `snow_volume_m3` вычисляется на стороне Go сервиса: `(snow_volume_percentage / 100) * body_volume_m3`
  - Note: для времени снежного события используется `event_time`, для камеры - `camera_id` (не дублируются)
  - If no snow match yet: `matched_snow=false`
- Field `photos` (one or several):
  - `detectionPicture.jpg` (ANPR frame)
  - `featurePicture.jpg` (optional)
  - `licensePlatePicture.jpg` (optional)
  - `snowSnapshot.jpg` (from snow worker, if matched)

## Data and logs
- Plate webhook logs: `hik_raws/detections.log`
- Snow snapshots/analysis: `snapshots/YYYY-MM-DD/HH-MM-SS.{jpg,json}`

## Notes
- Snow -> plate ordering is assumed; window/TTL control pairing. Adjust `MERGE_WINDOW_SECONDS`/`MERGE_TTL_SECONDS` to match camera spacing/speed.
- Gemini must be configured (`GEMINI_API_KEY`). If Gemini fails, direction falls back to motion (left_to_right), percentage/confidence may be null.
