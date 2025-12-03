# Hikvision ANPR Wrapper

Обертка над пайплайном распознавания госномеров (детекция → кроп → OCR) для камер Hikvision. Используются YOLOv8 для поиска номера (`runs/detect/train4/weights/best.pt`) и PaddleOCR для чтения текста с нормализацией казахстанских шаблонов.

## Возможности
- HTTP API на FastAPI (`api.py`) с эндпойнтами `GET /health` и `POST /anpr`.
- Класс `ANPR` для вызова из Python: принимает путь к файлу или `numpy.ndarray`, возвращает `dict` с полями `plate`, `det_conf`, `ocr_conf`, `bbox`.
- Можно отдельно использовать детектор (`modules/detector.py`) или OCR (`modules/ocr.py`).
- Примеры изображений: `img/sample.jpg`, `img/test/*.jpg`; веса YOLO уже лежат в `runs/detect/train4/weights/best.pt`.

## Установка
1) Установите Python 3.11.8.
2) Создайте окружение и установите зависимости:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
> PaddleOCR при первом запуске докачает модели (~200 МБ) и может потребовать интернет.

## Запуск API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Проверка запроса:
```bash
curl -X POST -F "file=@img/sample.jpg" http://localhost:8000/anpr
```
Ответ: `{"plate":"850ZEX15","det_conf":0.87,"ocr_conf":0.91,"bbox":[x1,y1,x2,y2]}`.

## Локальный инференс из кода
```python
from modules.anpr import ANPR

engine = ANPR()  # по умолчанию берет runs/detect/train4/weights/best.pt
result = engine.infer("img/sample.jpg")
print(result)
```
`infer` возвращает словарь, совместимый с ответом API.

## Использование детектора отдельно
```python
from modules.detector import PlateDetector
import cv2

detector = PlateDetector("runs/detect/train4/weights/best.pt")
img = cv2.imread("img/sample.jpg")
for det in detector.detect(img, conf=0.25):
    print(det["bbox"], det["conf"])
```

## Структура проекта
- `api.py` — FastAPI обертка.
- `modules/anpr.py` — основной пайплайн: детекция, препроцессинг, OCR.
- `modules/ocr.py` — обертка над PaddleOCR.
- `modules/detector.py` — отдельный YOLOv8-детектор.
- `limitations/plate_rules.py` — нормализация и валидаторы форматов KZ.
- `runs/detect/train4/weights/best.pt` — обученные веса YOLO.
- `img/` — примеры и тестовые изображения.
- `tests/` — скрипты-проверки; при необходимости поправьте пути к картинкам.

## Отладочные файлы
- Последний crop и бинаризация сохраняются в `debug_raw_crop.jpg` и `debug_proc_crop.jpg`.
- При отсутствии детекции исходное изображение складывается в `debug_no_det/no_det_*.jpg`.
