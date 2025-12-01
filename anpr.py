# anpr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from ocr import PlateOCR


ImageType = Union[str, np.ndarray]


@dataclass
class DetectionResult:
    plate: Optional[str]
    det_conf: float
    ocr_conf: float
    bbox: Optional[Tuple[int, int, int, int]]


def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """
    Жёсткая предобработка маленького номера:
    - серое изображение
    - CLAHE (контраст)
    - bilateral filter (шум)
    - ресайз вверх
    - адаптивный порог
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Увеличиваем картинку, чтобы OCRу было проще
    h, w = gray.shape[:2]
    scale = max(2.0, 240.0 / max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Выравниваем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Сглаживаем шум, но сохраняем границы
    blur = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    # Бинаризация
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    # Для PaddleOCR делаем 3 канала
    proc = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    # Отладочные сохранения (можешь выключить)
    cv2.imwrite("debug_raw_crop.jpg", img)
    cv2.imwrite("debug_proc_crop.jpg", proc)

    return proc


class ANPR:
    """
    Общий движок:
    - YOLO детектит номерной знак
    - вырезаем кроп
    - предобработка
    - PaddleOCR + нормализация под KZ
    """

    def __init__(self, yolo_weights: str = "runs/detect/train4/weights/best.pt") -> None:
        """
        yolo_weights – путь к весам детектора номера.
        ОБРАТИ ВНИМАНИЕ: если у тебя файл называется иначе,
        просто поправь путь.
        """
        self.yolo = YOLO(yolo_weights)
        self.ocr = PlateOCR()

    def _load_image(self, img: ImageType) -> np.ndarray:
        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Cannot read image from path: {img}")
            return image
        if isinstance(img, np.ndarray):
            return img
        raise TypeError("img must be str path or numpy.ndarray")

    def infer(self, img: ImageType) -> Dict[str, Any]:
        """
        Основной метод:
        - img: путь к файлу или numpy-картинка
        - возвращает dict для JSON-ответа API
        """
        image = self._load_image(img)
        h, w = image.shape[:2]

        # 1. Детекция номерного знака YOLO
        det_result = self.yolo(image, verbose=False)[0]
        if det_result.boxes is None or len(det_result.boxes) == 0:
            # Ничего не нашли
            return DetectionResult(
                plate=None,
                det_conf=0.0,
                ocr_conf=0.0,
                bbox=None,
            ).__dict__

        # Берём бокс с максимальной уверенностью
        boxes = det_result.boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        det_conf = float(confs[best_idx])

        x1, y1, x2, y2 = best_box.tolist()

        # Ограничиваем координаты границами изображения
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return DetectionResult(
                plate=None,
                det_conf=det_conf,
                ocr_conf=0.0,
                bbox=(x1, y1, x2, y2),
            ).__dict__

        plate_crop = image[y1:y2, x1:x2]

        # 2. Предобработка кропа
        proc_crop = preprocess_plate(plate_crop)

        # 3. OCR
        plate, ocr_conf = self.ocr.recognize(proc_crop)

        result = DetectionResult(
            plate=plate,
            det_conf=det_conf,
            ocr_conf=ocr_conf,
            bbox=(x1, y1, x2, y2),
        )

        return result.__dict__


def test_anpr(path: str) -> None:
    """
    Утилитный запуск для локальной проверки:
    python -m anpr path/to/image.jpg
    """
    engine = ANPR()
    res = engine.infer(path)
    print(res)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m anpr path/to/image.jpg")
        sys.exit(1)
    test_anpr(sys.argv[1])
