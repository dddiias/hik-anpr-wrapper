# Hikvision ANPR Wrapper

Сервис для распознавания госномеров (детекция -> кроп -> OCR) для камер Hikvision и обычных HTTP-запросов. Детекция работает на YOLOv8 (`runs/detect/train4/weights/best.pt`), чтение текста — через PaddleOCR с нормализацией казахстанских форматов.

## Кратко о сервисе
- FastAPI API в `api.py`: `GET /health`, `POST /anpr`, `POST /api/v1/anpr/hikvision`.
- Пайплайн: YOLOv8 ищет номер, кроп нормализуется, PaddleOCR читает текст, `limitations/plate_rules.py` приводит результат к шаблонам KZ (01-20 регион).
- Веса YOLO уже лежат в `runs/detect/train4/weights/best.pt`, примеры изображений — в `img/`.
- Можно использовать как библиотеку (`modules/anpr.ANPR`) или только детектор (`modules/detector.PlateDetector`).

## Как работает пайплайн
1. Детектор YOLOv8 (порог `det_conf_thr=0.15`) выбирает bbox с максимальной уверенностью. При отсутствии детекции исходник сохраняется в `debug_no_det/no_det_<timestamp>.jpg`, ответ — с `plate=None`.
2. Кроп приводится к ~240 px по большей стороне, применяется CLAHE и билатеральные/морфологические фильтры. Сохраняются `debug_raw_crop.jpg` (сырое) и `debug_proc_crop.jpg` (бинаризованное).
3. OCR (PaddleOCR, CPU): делаются две попытки — по CLAHE-кропу и по бинарному; обе логируются, выбирается лучший вариант.
4. Нормализация номера: чистка символов, карта путаниц (O/0, S/5, 2/Z, 8/B), проверка форматов KZ и регионов 01..20. Если вариант не прошёл валидацию, возвращается `None` или лучший найденный.
5. Выход `dict`: `plate`, `det_conf`, `ocr_conf`, `bbox=(x1,y1,x2,y2)`.

## Требования и установка
1) Python 3.11.8.  
2) Виртуальное окружение и зависимости:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
> При первом запуске PaddleOCR скачает модели (~200 МБ), потребуется интернет.

## Запуск API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Эндпойнт `GET /health`
Простой healthcheck: `{"status": "ok"}`.

### Эндпойнт `POST /anpr`
- Ожидает `multipart/form-data` с полем `file` (JPEG/PNG).
- Возвращает результат пайплайна: `{"plate":"850ZEX15","det_conf":0.87,"ocr_conf":0.91,"bbox":[x1,y1,x2,y2]}`.
- Ошибки: пустой файл (`400 Empty file`), не удалось декодировать изображение (`400 Cannot decode image`).
Пример:
```bash
curl -X POST -F "file=@img/sample.jpg" http://localhost:8000/anpr
```

### Эндпойнт `POST /api/v1/anpr/hikvision`
Принимает нативные запросы камер Hikvision. Поддерживает два режима и всегда сохраняет сырые данные в `hik_raws/<YYYY-MM-DD>`:
- `raws/hik_raw_<time>.bin` - тело запроса как есть;
- `parts/<time>_<filename>` - каждую multipart-часть;
- `images/<...>.jpg` - все найденные картинки.

**Multipart/form-data**  
- Каждый файл сохраняется в `parts/`. Если имя заканчивается на `anpr.xml`, парсится `<licensePlate>`, `<originalLicensePlate>`, `<confidenceLevel>`.  
- Все картинки сохраняются в `images/`. Если файл называется `detectionpicture.jpg`, по нему сразу запускается ANPR.  
- В `hik_raws/detections.log` добавляется JSON-строка с путями, данными камеры и результатом модели.
- Ответ:
```json
{
  "status": "ok",
  "kind": "hikvision_anpr_multipart",
  "camera_plate": "...",
  "camera_original_plate": "...",
  "camera_confidence": 0.73,
  "anpr_plate": "850ZEX15",
  "anpr_det_conf": 0.87,
  "anpr_ocr_conf": 0.91,
  "anpr_bbox": [x1, y1, x2, y2],
  "paths": {
    "raw": "hik_raws/2025-12-04/raws/hik_raw_12-34-56.bin",
    "xml": "hik_raws/2025-12-04/parts/12-34-56_anpr.xml",
    "detection_image": "hik_raws/2025-12-04/images/12-34-56_detectionpicture.jpg"
  }
}
```
`kind` будет `hikvision_anpr_multipart` если в форме есть файлы, либо `multipart_no_files`.

**JPEG в теле запроса** (fallback)  
- Из тела извлекается JPEG по сигнатурам `FFD8...FFD9`, сохраняется в `images/hik_frame_<time>.jpg`, после чего запускается ANPR.  
- Если в теле нет JPEG, ответ: `{"status":"ok","kind":"no_jpeg_in_body","size":<len>}`.
- Ответ для успешно извлечённого JPEG:
```json
{
  "status": "ok",
  "kind": "hikvision_anpr_body_jpeg",
  "camera_plate": null,
  "camera_original_plate": null,
  "camera_confidence": null,
  "anpr_plate": "850ZEX15",
  "anpr_det_conf": 0.87,
  "anpr_ocr_conf": 0.91,
  "anpr_bbox": [x1, y1, x2, y2],
  "paths": {
    "raw": "hik_raws/2025-12-04/raws/hik_raw_12-34-56.bin",
    "xml": null,
    "detection_image": "hik_raws/2025-12-04/images/hik_frame_12-34-56.jpg"
  }
}
```

## Использование из Python
```python
from modules.anpr import ANPR

engine = ANPR()  # по умолчанию берет runs/detect/train4/weights/best.pt
res = engine.infer("img/sample.jpg")  # можно передать и numpy.ndarray BGR
print(res)  # {'plate': '850ZEX15', 'det_conf': ..., 'ocr_conf': ..., 'bbox': [...]}
```

### Детектор отдельно
```python
from modules.detector import PlateDetector
import cv2

detector = PlateDetector("runs/detect/train4/weights/best.pt")
img = cv2.imread("img/sample.jpg")
for det in detector.detect(img, conf=0.25):
    print(det["bbox"], det["conf"])
```

## Директории и артефакты
- `runs/detect/train4/weights/best.pt` - веса YOLOv8 для детекции.
- `img/` - примеры (`sample*.jpg`) и тестовый набор `img/test/`.
- `hik_raws/<YYYY-MM-DD>/{raws,parts,images}` - сохраненные запросы/части/картинки от камер Hikvision; можно периодически чистить.  
- `hik_raws/detections.log` - лог JSON-строк с путями файлов, данными камеры и результатом модели.
- `debug_raw_crop.jpg`, `debug_proc_crop.jpg` - последний кроп и его бинаризация.
- `debug_no_det/` - исходники без детекции.

## Тесты и отладка
- Проверка пайплайна: `python -m modules.anpr img/sample.jpg` или `python tests/test_anpr.py` (при необходимости поправьте путь к картинке).
- Детектор: `python tests/test_detect.py`.
- OCR отдельно: `python tests/test_ocr_standalone.py`.
- Сравнение кастомных OCR-моделей: `python tests/compare_models.py` (использует `img/test` и модели в `models/infer_*`, поправьте пути под свои файлы).

## Настройки и кастомизация
- Задать другие веса YOLO: `ANPR(yolo_weights="path/to/weights.pt")`.
- Порог детекции (`det_conf_thr`) и пороги/предобработку можно менять в `modules/anpr.py`.
- OCR по умолчанию работает на CPU; переключение на GPU настраивается в `modules/ocr.py` (`device="gpu:0"`).
