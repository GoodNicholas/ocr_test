import pytesseract
from PIL import Image

# Загружаем изображение
img = Image.open("/Users/krotovnikolay/PycharmProjects/final_passport_ocr/dataset_small/debug_images/000004_mrz.png")

# Применяем Tesseract для извлечения текста
text = pytesseract.image_to_string(img, config='--psm 6')  # psm 6 - для обработки текста в строках

print("Распознанный текст:", text)
