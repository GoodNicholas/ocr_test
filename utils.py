import cv2
import numpy as np

# Путь к изображению
input_image_path = '/Users/krotovnikolay/PycharmProjects/final_passport_ocr/data/1420521697_643414513.jpg'
output_image_path = '/Users/krotovnikolay/PycharmProjects/final_passport_ocr/data/extracted_passport_000010.png'

image = cv2.imread(input_image_path)

# Получаем размеры изображения
height, width, _ = image.shape

# Преобразуем изображение в цветовое пространство HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Определяем диапазоны для белого цвета в HSV
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# Создаем маску для белого фона
mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Инвертируем маску (чтобы фон был черным, а паспорт - белым)
mask_inv = cv2.bitwise_not(mask)

# Применяем маску к изображению
masked_image = cv2.bitwise_and(image, image, mask=mask_inv)

# Применяем морфологическую операцию для улучшения контуров
kernel = np.ones((50, 50), np.uint8)
processed_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernel)

# Теперь находим контуры
gray_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
_, binary_processed = cv2.threshold(gray_processed, 1, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Получаем максимальный контур
max_contour = max(contours, key=cv2.contourArea)

# Получаем ограничивающий прямоугольник
x, y, w, h = cv2.boundingRect(max_contour)

# Обрезаем изображение по найденному прямоугольнику
cropped_image = image[y:y+h, x:x+w]

# Сохраняем результат
cv2.imwrite(output_image_path, cropped_image)

print(f'Изображение сохранено по пути: {output_image_path}')