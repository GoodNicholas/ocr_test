import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import pytesseract
import re

# Полные пути к модели и тестовому изображению
MODEL_PATH = "/Users/krotovnikolay/PycharmProjects/mrz_detector/.venv/lib/python3.13/site-packages/fastmrz/model/mrz_seg.onnx"
DEFAULT_IMG_PATH = "/Users/krotovnikolay/PycharmProjects/Passport_generator/test/test_pic.png"
assert os.path.exists(MODEL_PATH), f"Не нашёл {MODEL_PATH}"

# Инициализация ONNX
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

# Регулярка для валидации MRZ: две строки A-Z0-9< длиной >=30
MRZ_RE = re.compile(r"^[A-Z0-9<]{30,}\s*[A-Z0-9<]{30,}$", re.MULTILINE)

def ocr_validate_mrz(crop_bgr):
    """Кроп в BGR, возвращает True, если в нём валидный MRZ."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # можно понизить разрешение, но лучше оставить
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    text = pytesseract.image_to_string(gray, config=config)
    print("=== OCR Text ===")
    print(text)
    print("=== End OCR ===")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    # берём последние две непустые строки
    candidate = lines[-2] + "\n" + lines[-1]
    return bool(MRZ_RE.match(candidate))

def detect_mrz_onnx(image_path,
                    sess,
                    inp_name,
                    out_name,
                    conf_thresh=0.4,
                    area_thresh=0.002,
                    aspect_ratio_range=(4.45, 50),
                    mean_prob_thresh=0.35,
                    mask_coverage_thresh=0.3,
                    ocr_margin=5):
    """
    Возвращает bbox MRZ или None.
    В самом конце — дополнительная OCR-валидация.
    """
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size

    # 1) resize + preprocess
    target_size = (256, 256)
    img_resized = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    inp = arr[np.newaxis, :, :, :]

    # 2) inference
    out = sess.run([out_name], {inp_name: inp})[0]
    if out.ndim == 4:
        prob_map = out[0,0] if out.shape[1]==1 else out[0,:,:,0]
    else:
        raise RuntimeError(f"Unexpected shape {out.shape}")

    # 3) binarize + find contours
    mask = (prob_map > conf_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[DEBUG] нет контуров выше conf_thresh =", conf_thresh)
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    print(f"[DEBUG] bbox (resized): x={x},y={y},w={w},h={h}")

    # 3) Площадь контура
    area_ratio = (w * h) / (256 * 256)
    print("[DEBUG] area_ratio =", area_ratio)
    if area_ratio < area_thresh:
        print(f"[DEBUG] отбрасываем: area_ratio {area_ratio:.4f} < area_thresh {area_thresh}")
        return None

    # 4) Соотношение сторон
    ar = w / (h + 1e-9)
    print("[DEBUG] aspect_ratio =", ar)
    if not (aspect_ratio_range[0] <= ar <= aspect_ratio_range[1]):
        print(f"[DEBUG] отбрасываем: aspect_ratio {ar:.2f} вне {aspect_ratio_range}")
        return None

    # 5) Средняя вероятность
    mean_prob = prob_map[y:y + h, x:x + w].mean()
    print("[DEBUG] mean_prob =", mean_prob)
    if mean_prob < mean_prob_thresh:
        print(f"[DEBUG] отбрасываем: mean_prob {mean_prob:.3f} < mean_prob_thresh {mean_prob_thresh}")
        return None

    # 6) Покрытие маской
    mask_crop = mask[y:y + h, x:x + w] // 255
    mask_coverage = mask_crop.sum() / float(w * h)
    print("[DEBUG] mask_coverage =", mask_coverage)
    if mask_coverage < mask_coverage_thresh:
        print(f"[DEBUG] отбрасываем: mask_coverage {mask_coverage:.3f} < mask_coverage_thresh {mask_coverage_thresh}")
        return None

    # Если дошли сюда — логируем, что сейчас пойдёт OCR
    print("[DEBUG] геометрия пройдена, запускаем OCR-валидацию")

    # 5) map to original
    sx, sy = w0/256, h0/256
    xmin = int(x*sx); ymin = int(y*sy)
    xmax = int((x+w)*sx); ymax = int((y+h)*sy)

    # 6) OCR-проверка
    img_bgr = cv2.imread(image_path)
    # небольшой margin, чтобы быть уверенными, что ловим всю строку
    ym0 = max(0, ymin-ocr_margin)
    ym1 = min(h0, ymax+ocr_margin)
    crop = img_bgr[ym0:ym1, xmin:xmax]
    if not ocr_validate_mrz(crop):
        return None

    return xmin, ymin, xmax, ymax

if __name__ == "__main__":
    bbox = detect_mrz_onnx(DEFAULT_IMG_PATH, sess, inp_name, out_name)
    img = cv2.imread(DEFAULT_IMG_PATH)
    if bbox:
        print("MRZ bbox:", bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
    else:
        print("MRZ не найдено")
    cv2.imwrite("mrz_out.png", img)
    print("Сохранено mrz_out.png")
