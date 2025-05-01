import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import os
import json

# Загрузка предобученной модели LayoutLMv3 с поддержкой русского языка
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Пример обработки данных
def process_data(images_dir, annotations_file):
    dataset = []
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    for img_info in annotations['images']:
        img_path = os.path.join(images_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        boxes = []
        labels = []
        texts = []
        for ann in annotations['annotations']:
            if ann['image_id'] == img_info['id']:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(1)  # Метка для извлечения данных
                texts.append(ann['text'])
        dataset.append({
            'image': img,
            'boxes': boxes,
            'labels': labels,
            'texts': texts
        })
    return dataset

# Загружаем данные
train_data = process_data("path_to_images", "annotations.json")

# Создание DataLoader для обучения
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# Определение оптимизатора
optimizer = AdamW(model.parameters(), lr=5e-5)

# Обучение модели
model.train()
for epoch in range(3):
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch in progress_bar:
        images = batch['image'].to(device)
        boxes = batch['boxes'].to(device)
        labels = batch['labels'].to(device)

        # Преобразование изображений и текста для подачи в LayoutLMv3
        encoding = processor(images=images, boxes=boxes, text=batch['texts'], padding="max_length", return_tensors="pt")
        outputs = model(**encoding, labels=labels)
        loss = outputs.loss

        # Обновление градиентов
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

# Сохранение обученной модели
model.save_pretrained("trained_layoutlmv3")
processor.save_pretrained("trained_layoutlmv3")
