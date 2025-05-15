import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import os

# ------- Параметры инференса -------
# Модель 1: детектор паспорта на фоне
DET_MODEL_PATH = '/content/ocr_test/ocr_models/stage1/best.pt'
# Модель 2: сегментатор/детектор полей внутри паспорта
SEG_MODEL_PATH = '/content/ocr_test/runs/detect/train/weights/best.pt'
# Входное изображение
IMAGE_PATH = '/content/161840081264.jpg'
# Папка, куда будут сохраняться вырезы
OUTPUT_BASE = Path('crops')
# Пороги для обоих моделей
DET_CONF = 0.25
SEG_CONF = 0.25
# -----------------------------------

for p in (DET_MODEL_PATH, SEG_MODEL_PATH):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Модель не найдена: {p}")
    if os.path.getsize(p) < 1_000_000:  # например, минимальный размер 1 МБ
        raise RuntimeError(f"Похоже, файл {p} повреждён или слишком маленький ({os.path.getsize(p)} байт)")

# Теперь безопасно грузим
try:
    det_model = YOLO(DET_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить детектор: {e}")
    

# Если OUTPUT_BASE уже существует, очищаем его
if OUTPUT_BASE.exists():
    shutil.rmtree(OUTPUT_BASE)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Загружаем модели
det_model = YOLO(DET_MODEL_PATH)
seg_model = YOLO(SEG_MODEL_PATH)

# Открываем исходное изображение
img = Image.open(IMAGE_PATH).convert('RGB')

# 1) Инференс первой модели: находим паспорта
det_results = det_model.predict(source=IMAGE_PATH, conf=DET_CONF, save=False)
passports = det_results[0].boxes.data.cpu().numpy()

for p_idx, pbox in enumerate(passports):
    x1, y1, x2, y2, det_conf, det_cls = pbox
    # вырезаем паспорт
    passport_crop = img.crop((int(x1), int(y1), int(x2), int(y2)))

    # 2) Инференс второй модели внутри выреза
    # Сохраняем временно
    tmp_path = OUTPUT_BASE / f'_tmp_passport_{p_idx}.png'
    passport_crop.save(tmp_path)
    seg_results = seg_model.predict(source=str(tmp_path), conf=SEG_CONF, save=False)
    fields = seg_results[0].boxes.data.cpu().numpy()

    # Удаляем временный файл
    tmp_path.unlink()

    # 3) Для каждого найденного поля: вырез и сохранение по классу
    for f_idx, fbox in enumerate(fields):
        fx1, fy1, fx2, fy2, fconf, fcls = fbox
        # приводим координаты к целочисленным в рамках выреза
        crop = passport_crop.crop((int(fx1), int(fy1), int(fx2), int(fy2)))
        # директория под класс
        cls_dir = OUTPUT_BASE / f'class_{int(fcls)}'
        cls_dir.mkdir(exist_ok=True)
        # имя файла
        fname = f'passport{p_idx:03d}_field{f_idx:03d}_cls{int(fcls)}_{fconf:.2f}.png'
        crop.save(cls_dir / fname)
        print(f"Saved: {cls_dir / fname}")

print("Inference completed. Crops sorted by class in 'crops' directory.")
