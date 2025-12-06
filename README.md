# Hikvision ANPR Wrapper

FastAPI‑сервис, который принимает события от камер Hikvision, прогоняет изображение через свою модель распознавания номерных знаков (YOLOv8 + PaddleOCR) и при необходимости пересылает событие в другой сервис в виде `multipart/form-data`.

## Что внутри
- `api.py` — HTTP API: `GET /health`, `POST /anpr`, `POST /api/v1/anpr/hikvision`.
- `modules/anpr.ANPR` — пайплайн: детекция номера YOLOv8 (`runs/detect/train4/weights/best.pt`), препроцессинг кадра, OCR (PaddleOCR), нормализация под номера РК (`limitations/plate_rules.py`).
- `modules/detector.PlateDetector` — обертка над YOLO для детекции номера без OCR.
- Входные артефакты камер складываются в `hik_raws/<YYYY-MM-DD>/parts` (XML) и `hik_raws/<YYYY-MM-DD>/images` (зарезервировано), логи отправок — `hik_raws/detections.log`.

## Установка
Требуется Python 3.11.8.
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
> PaddleOCR подтянет весовые файлы при первом запуске (скачивание ~200 МБ).

## Запуск API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Значение апстрима настраивается в `api.py` через константу `UPSTREAM_URL`. Если оставить пустой строкой, события наружу не отправляются, но продолжают логироваться.

## Эндпоинты
### `GET /health`
Проверка доступности. Ответ: `{"status": "ok"}`.

### `POST /anpr`
Одно изображение на вход, распознанный номер на выходе.
- Тело: `multipart/form-data`, поле `file` (JPEG/PNG).
- Успешный ответ: `{"plate":"850ZEX15","det_conf":0.87,"ocr_conf":0.91,"bbox":[x1,y1,x2,y2]}`.
- Ошибки: `400 Empty file`, `400 Cannot decode image`.
Пример:
```bash
curl -X POST -F "file=@img/sample.jpg" http://localhost:8000/anpr
```

### `POST /api/v1/anpr/hikvision`
Точка приёма webhook от Hikvision. Поддерживаются два формата запроса:
1) **Стандартный multipart от камеры.**
   - `anpr.xml` — сохраняется в `hik_raws/<date>/parts/<time>_anpr.xml`, парсятся поля `licensePlate`/`originalLicensePlate`, `confidenceLevel`, `eventType`, `dateTime`.
   - `detectionPicture.jpg` — используется как основное изображение для своей ANPR-модели.
   - `featurePicture.jpg`, `licensePlatePicture.jpg` — дополнительные кадры, просто пробрасываются дальше.
   - Любые текстовые поля формы печатаются в лог (stdout).
2) **Fallback: JPEG прямо в теле** (для случаев без multipart). Из тела вырезается первый JPEG, остальной контент игнорируется.

Общий поток обработки:
1. Если пришёл `detectionPicture.jpg`, запускается `ANPR.infer()`; без него модель не вызывается.
2. Формируется `event_data`:
   - `camera_id` — всегда `"camera-001"` (зашито, при необходимости поменяйте в `api.py`).
   - `event_time` — `dateTime` из XML, иначе текущее время.
   - `plate` — приоритетно модель (`model_plate`), иначе номер из камеры.
   - `camera_plate`, `camera_confidence` — из XML; могут быть `null`.
   - `model_plate`, `model_det_conf`, `model_ocr_conf` — из собственной модели; `null`, если модель не запускалась.
   - `timestamp` — текущее время формирования события.
3. Событие отправляется в апстрим (`send_to_upstream`), результат фиксируется в `hik_raws/detections.log` вместе со статусом отправки (`upstream_sent`, `upstream_status`, `upstream_error`). В fallback‑режиме в лог дополнительно пишется `anpr_bbox`.
4. Ответ сервиса — `{"status": "ok"}`. Если JPEG не извлечён из тела — тоже `ok` (ничего не отправляется). При невозможности декодировать JPEG — `{"status":"error","message":"cannot decode jpeg"}` с кодом 400.

Пример запроса отлаженного multipart:
```bash
curl -X POST http://localhost:8000/api/v1/anpr/hikvision ^
  -F "anpr.xml=@hik_raws/2025-12-06/parts/sample_anpr.xml;type=text/xml" ^
  -F "detectionPicture.jpg=@img/sample.jpg;type=image/jpeg" ^
  -F "featurePicture.jpg=@img/sample.jpg;type=image/jpeg"
```

## Формат отправки в внешний сервис
`send_to_upstream` делает `POST {UPSTREAM_URL}` с `multipart/form-data`:
- Поле `event` — строка JSON с вышеописанной структурой `event_data`.
- Поле (повторяющееся) `photos` — JPEG-файлы:
  - `detectionPicture.jpg` — основное изображение (если было в запросе, либо JPEG из тела при fallback).
  - `featurePicture.jpg` — если камера прислала.
  - `licensePlatePicture.jpg` — если камера прислала.

Пример полезной нагрузки `event` при multipart:
```json
{
  "camera_id": "camera-001",
  "event_time": "2025-12-06T15:20:18",
  "plate": "850ZEX15",
  "camera_plate": "850ZEX15",
  "camera_confidence": 0.83,
  "model_plate": "850ZEX15",
  "model_det_conf": 0.91,
  "model_ocr_conf": 0.89,
  "timestamp": "2025-12-06T15:20:19.123456"
}
```
В fallback‑режиме `camera_plate` и `camera_confidence` будут `null`, `event_time` = текущее время, фотография — единственный `detectionPicture.jpg` из тела.

## Отладочные файлы
- `debug_raw_crop.jpg`, `debug_proc_crop.jpg` — последний raw/обработанный кроп номера из OCR.
- `debug_no_det/` — исходные кадры, где YOLO не нашла номер.
- `hik_raws/detections.log` — построчно JSON событий и статусов отправки.

## Использование как библиотеки
```python
from modules.anpr import ANPR

engine = ANPR()  # можно передать yolo_weights="path/to/weights.pt"
result = engine.infer("img/sample.jpg")  # либо np.ndarray (BGR)
print(result)  # {'plate': '850ZEX15', 'det_conf': ..., 'ocr_conf': ..., 'bbox': [...]}
```
