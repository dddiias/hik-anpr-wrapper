# Hikvision ANPR Wrapper

Обертка над пайплайном распознавания госномеров (детекция -> кроп -> OCR) для камер Hikvision и обычных HTTP-клиентов. Детекция работает на YOLOv8 (`runs/detect/train4/weights/best.pt`), чтение текста — через PaddleOCR с нормализацией казахстанских форматов.

## Что умеет
- FastAPI API (`api.py`): `GET /health`, `POST /anpr`, `POST /api/v1/anpr/hikvision`.
- Пайплайн: YOLOv8 ищет номер, кроп нормализуется, PaddleOCR читает текст, `limitations/plate_rules.py` приводит результат к валидным шаблонам KZ (регионы 01-20).
- Примеры изображений лежат в `img/`, веса YOLO - в `runs/detect/train4/weights/best.pt`.
- Можно использовать как библиотеку (`modules/anpr.ANPR`) или только детектор (`modules/detector.PlateDetector`).
- Hikvision-эвенты форвардятся наружу на `UPSTREAM_URL` как multipart: поле `event` (JSON) + массив `photos` (detection/feature/licensePicture).

## Как работает пайплайн
1) YOLOv8 с порогом `det_conf_thr=0.15` выбирает bbox с максимальной уверенностью. При отсутствии детекции исходник сохраняется в `debug_no_det/no_det_<timestamp>.jpg`, ответ — с `plate=None`.  
2) Кроп ресайзится до ~240 px по большей стороне, проходит CLAHE, блюр, морфологию; сохраняются `debug_raw_crop.jpg` (сырой кроп) и `debug_proc_crop.jpg` (бинаризация).  
3) OCR (PaddleOCR, CPU): две попытки — по CLAHE-кропу и по бинарному; логируются в консоль, берется лучший вариант.  
4) Нормализация номера: чистка символов, исправление путаниц (O/0, S/5, 2/Z, 8/B), проверка форматов KZ и регионов 01..20.  
5) Результат `dict`: `plate`, `det_conf`, `ocr_conf`, `bbox=(x1,y1,x2,y2)`.

## Установка
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

### `GET /health`
Простой healthcheck: `{"status":"ok"}`.

### `POST /anpr`
- Ожидает `multipart/form-data` с полем `file` (JPEG/PNG).  
- Возвращает результат пайплайна: `{"plate":"850ZEX15","det_conf":0.87,"ocr_conf":0.91,"bbox":[x1,y1,x2,y2]}`.  
- Ошибки: пустой файл (`400 Empty file`), не удалось декодировать изображение (`400 Cannot decode image`).
Пример:
```bash
curl -X POST -F "file=@img/sample.jpg" http://localhost:8000/anpr
```

### `POST /api/v1/anpr/hikvision`
Принимает нативные запросы камер Hikvision. Поддерживает два режима и создает каталоги `hik_raws/<YYYY-MM-DD>/parts` (сохраняется только `anpr.xml`; `images/` создается, но сейчас не используется).

**Ожидаемые части multipart/form-data**
- `anpr.xml` — сохраняется в `hik_raws/<date>/parts/<time>_anpr.xml`, парсятся `licensePlate`, `originalLicensePlate`, `confidenceLevel`, `eventType`, `dateTime`.  
- `detectionPicture.jpg` — запускает ANPR, чтобы заполнить `anpr_*`; байты уходят наружу.  
- `featurePicture.jpg`, `licensePlatePicture.jpg` — просто форвардятся наружу.  
- Остальные поля игнорируются (только логируются в stdout).

**Форвардинг события наружу**  
- `UPSTREAM_URL` в `api.py` указывает сервис получателя (`https://snowops-anpr-service.onrender.com/api/v1/anpr/events`).  
- Отправляется `multipart/form-data` с полями:  
  - `event` — JSON (`timestamp`, `camera_plate`, `camera_confidence`, `anpr_plate`, `anpr_det_conf`, `anpr_ocr_conf`);  
  - `photos` — массив файлов: detectionPicture/featurePicture/licensePlatePicture (что пришло, то и уходит); для fallback JPEG кладется в `photos` как detectionPicture.  
- Таймаут отправки 10 секунд. Чтобы отключить форвардинг, поставьте `UPSTREAM_URL = ""`.

**Локальное логирование**  
- В `hik_raws/detections.log` одна JSON-строка на запрос: все поля `event`, плюс `upstream_sent`, `upstream_status`, `upstream_error`.  
- Для кейсов без JPEG в теле и ошибок декодирования тоже пишутся записи с `kind=no_jpeg_in_body`/`jpeg_decode_error`.

**Ответы API**
- Multipart: всегда `{"status": "ok"}` (детали только в логе и отправке наружу).  
- Body-JPEG fallback: если JPEG не найден — `{"status": "ok"}` (и запись в лог о `no_jpeg_in_body`); если JPEG не декодируется — `{"status": "error", "message": "cannot decode jpeg"}`; если декодируется — `{"status": "ok"}` (детали в логе и форвардинге).

## Использование из Python
```python
from modules.anpr import ANPR

engine = ANPR()  # по умолчанию берет runs/detect/train4/weights/best.pt
res = engine.infer("img/sample.jpg")  # можно передать и numpy.ndarray (BGR)
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
- `runs/detect/train4/weights/best.pt` — веса YOLOv8.  
- `img/` — примеры (`sample*.jpg`) и тестовый набор `img/test/`.  
- `hik_raws/<YYYY-MM-DD>/parts/` — сохраненные anpr.xml от камер (по времени запроса). Каталог `images/` создается, но сейчас не используется.  
- `hik_raws/detections.log` — построчный JSON с данными камеры и результатом модели.  
- `debug_raw_crop.jpg`, `debug_proc_crop.jpg` — последний кроп и его бинаризация.  
- `debug_no_det/` — исходники, где номер не найден.

## Тесты и отладка
- Быстрая проверка пайплайна: `python -m modules.anpr img/sample.jpg` или `python tests/test_anpr.py` (при необходимости поправьте путь к картинке).  
- Детектор: `python tests/test_detect.py`.  
- OCR отдельно: `python tests/test_ocr_standalone.py`.  
- Сравнение кастомных OCR-моделей: `python tests/compare_models.py` (использует `img/test` и модели в `models/infer_*`; поправьте пути под свои данные).

## Настройки
- Другие веса YOLO: `ANPR(yolo_weights="path/to/weights.pt")`.  
- Порог детекции и препроцессинг правятся в `modules/anpr.py`.  
- OCR по умолчанию CPU; переключение на GPU — в `modules/ocr.py` (`device="gpu:0"`).  
- Адрес внешнего сервиса для Hikvision-эвентов — `UPSTREAM_URL` в `api.py`.
