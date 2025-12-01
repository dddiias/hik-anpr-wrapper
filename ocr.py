# ocr.py
from __future__ import annotations

from typing import Tuple

import numpy as np
from paddleocr import PaddleOCR


class PlateOCR:
    """
    Класс-обёртка над PaddleOCR, заточенный именно под распознавание номера.
    ВАЖНО: используем только rec (распознавание), детекцию текста отключаем.
    """

    def __init__(self) -> None:
        """
        Инициализация OCR-движка.
        """
        # Детектор текста нам не нужен, мы подаём уже вырезанный номер.
        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=False,
            det=False,   # детектор не нужен
            rec=True,
        )

    def recognize(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Распознать текст на уже обрезанном изображении номера.

        :param img: numpy-изображение (BGR или RGB — PaddleOCR сам разберётся).
        :return: (plate_text, confidence)
        """
        # ЯВНО отключаем детекцию и классификатор угла — только распознавание
        result = self.ocr.ocr(img, det=False, rec=True, cls=False)

        # Иногда PaddleOCR может вернуть пустой список, если ничего не увидел
        if not result:
            return "", 0.0

        first = result[0]

        # У разных конфигураций PaddleOCR структура может отличаться.
        # Сделаем универсальный поиск пары вида (str, float) внутри результата.
        def find_text_score(node):
            if isinstance(node, (list, tuple)):
                # Классический формат: ("TEXT", score)
                if (
                    len(node) == 2
                    and isinstance(node[0], str)
                    and isinstance(node[1], (float, int))
                ):
                    return node[0], float(node[1])
                # Рекурсивно обходим все вложенные элементы
                for child in node:
                    found = find_text_score(child)
                    if found is not None:
                        return found
            return None

        pair = find_text_score(first)
        if pair is None:
            # Ничего не смогли достать — считаем, что распознавания нет
            return "", 0.0

        text, score = pair

        # Нормализуем: избавляемся от пробелов, переводим в верхний регистр
        text = text.replace(" ", "").upper()

        return text, float(score)
