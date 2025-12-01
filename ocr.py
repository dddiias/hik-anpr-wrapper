# ocr.py
from typing import Tuple, Optional

import cv2
from paddleocr import PaddleOCR

from plate_rules import normalize_plate


class PlateOCR:
    """
    Обёртка над PaddleOCR + нормализация под KZ форматы.
    """

    def __init__(self) -> None:
        # Важно: в твоей версии paddleocr нет параметра show_log,
        # поэтому оставляем только поддерживаемые аргументы.
        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=False,
            # show_log=False  # <-- убрали, он и ломал запуск
        )

    def recognize(self, img) -> Tuple[Optional[str], float]:
        """
        img: numpy-картинка (BGR или grayscale) – кроп номерного знака
        Возвращает: (plate: str | None, ocr_conf: float)
        """
        # PaddleOCR ожидает BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        result = self.ocr.ocr(img)
        if not result or not result[0]:
            # OCR вообще ничего не увидел
            return None, 0.0

        # Стандартный формат ответа: result[0][0] = [bbox, (text, score)]
        text, score = result[0][0][1]

        # Нормализуем строку под форматы гос-номеров РК
        plate = normalize_plate(text)

        if plate is None:
            # OCR что-то увидел, но формат не похож на гос-номер
            return None, float(score)

        return plate, float(score)
