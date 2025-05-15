import os
import cv2
import json
import re
import pytesseract
from pytesseract import TesseractError
from ultralytics import YOLO

# ——— ПУТИ ——————————————————————————————————————————————
model_path   = '/content/best segment medium.pt'
image_path   = '/content/1551114833_Qkr8HprHdtA.jpg'
output_path  = '/content/out/output_tesseract_best.jpg'
debug_dir    = 'debug'

os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# ——— УСТАНАВЛИВАЕМ TESSDATA_PREFIX И ПРОВЕРЯЕМ RUS.TRAINEDDATA ———
# При необходимости скорректируйте путь к tessdata в вашей системе
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
tessdata_dir = os.environ['TESSDATA_PREFIX']
if not os.path.isfile(os.path.join(tessdata_dir, 'rus.traineddata')):
    raise FileNotFoundError(
        f"Не найден файл rus.traineddata в {tessdata_dir}. "
        "Установите пакет tesseract-ocr-rus или исправьте путь TESSDATA_PREFIX."
    )

# ——— КАРТЫ КЛАССОВ ————————————————————————————————————————
field_names = {
    0: "surname",
    1: "name",
    2: "patronymic",
    3: "gender",
    4: "dob",
    5: "birth_place",
    6: "issuing_auth",
    7: "issue_date",
    8: "division_code",
    9: "series_number",
   10: "mrz"
}

# ——— РЕГУЛЯРКИ И WHITELIST ДЛЯ ЧИСЛЕННЫХ ————————————————————
# ключ — cls_id, значение — (regex, whitelist_chars)
patterns = {
    7: (r'\d{2}[.\-]\d{2}[.\-]\d{4}', '0123456789.-'),  # DD.MM.YYYY или DD-MM-YYYY
    8: (r'\d{3}-\d{3}',               '0123456789.-'),  # 123-456
}

# ——— ЗАГРУЖАЕМ МОДЕЛЬ YOLO ——————————————————————————————————
model = YOLO(model_path)
image = cv2.imread(image_path)
results = model(image)[0]  # ultralytics Results

# ——— УТИЛИТА ДЛЯ РАСШИРЕННОГО КРОПА ———————————————————————
def expand_bbox(x1, y1, x2, y2, img, pad=10):
    h, w = img.shape[:2]
    return (
        max(x1 - pad, 0),
        max(y1 - pad, 0),
        min(x2 + pad, w),
        min(y2 + pad, h),
    )

# ——— ФУНКЦИЯ OCR С TESSERACT —————————————————————————
def recognize_tesseract(crop, field_id):
    """
    Берёт бинаризацию, сохраняет debug-картинку
    и запускает pytesseract с whitelist-ом для численных полей.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_fn = os.path.join(debug_dir, f"{field_names[field_id]}.jpg")
    cv2.imwrite(debug_fn, binary)

    # базовый config с указанием tessdata-dir
    config = f'--tessdata-dir "{tessdata_dir}" -l rus --psm 6'
    # ограничиваем символы для полей из patterns
    if field_id in patterns:
        whitelist = patterns[field_id][1]
        config += f' -c tessedit_char_whitelist={whitelist}'

    try:
        text = pytesseract.image_to_string(binary, config=config)
    except TesseractError as e:
        raise RuntimeError("Tesseract вернул ошибку:\n" + str(e))
    return text.strip()

# ——— ГРУППИРУЕМ БОКСЫ ПО КЛАССАМ ————————————————————————
boxes_by_cls = {}
for box in results.boxes:
    cls_id = int(box.cls[0])
    if cls_id == 10:  # пропускаем MRZ
        continue
    boxes_by_cls.setdefault(cls_id, []).append(box)

# ——— ИНИЦИАЛИЗИРУЕМ СЛОВАРЬ ПОД ЭКСТРАКТ ————————————————
extracted_data = {v: None for v in field_names.values()}

# ——— ОБРАБОТКА ПО КЛАССАМ ————————————————————————————
for cls_id, boxes in boxes_by_cls.items():
    field = field_names[cls_id]

    # 1) Специальная логика для issue_date (7) и division_code (8)
    if cls_id in patterns:
        regex, _ = patterns[cls_id]
        candidates = []
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if x2 - x1 < 20 or y2 - y1 < 10:
                continue
            x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, image)
            crop = image[y1e:y2e, x1e:x2e]
            text = recognize_tesseract(crop, cls_id)
            is_match = bool(re.search(regex, text))
            candidates.append({
                'conf': float(b.conf[0]),
                'text': text,
                'match': is_match,
                'coords': (x1e, y1e, x2e, y2e),
            })

        # сначала берём попавшие в regex, иначе — по max(conf)
        good = [c for c in candidates if c['match']]
        best = max(good if good else candidates, key=lambda c: c['conf'], default=None)
        if best:
            extracted_data[field] = best['text']
            x1e, y1e, x2e, y2e = best['coords']
            cv2.rectangle(image, (x1e, y1e), (x2e, y2e), (0,255,0), 2)
            cv2.putText(
                image,
                f"{field} ({best['conf']:.2f})",
                (x1e, y1e-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    # 2) Все остальные поля — берём бокс с max confidence
    else:
        best = max(boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        if x2 - x1 < 20 or y2 - y1 < 10:
            continue
        x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, image)
        crop = image[y1e:y2e, x1e:x2e]

        # для серии и номера (cls_id == 9) поворачиваем на 90°
        if cls_id == 9:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        text = recognize_tesseract(crop, cls_id)
        extracted_data[field] = text or None

        cv2.rectangle(image, (x1e, y1e), (x2e, y2e), (0,255,0), 2)
        cv2.putText(
            image,
            f"{field} ({best.conf[0]:.2f})",
            (x1e, y1e-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

# ——— СОХРАНЯЕМ И ВЫВОДИМ РЕЗУЛЬТАТЫ —————————————————————
cv2.imwrite(output_path, image)
print("\nFinal JSON:")
print(json.dumps(extracted_data, ensure_ascii=False, indent=2))
