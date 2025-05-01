import json, os, random, time, datetime
import torch
import torchvision
from PIL.ImageFile import ImageFile
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ----  Dataset  --------------------------------------------------------------
class PassportMRZDataset(Dataset):
    def __init__(self, images_dir, ann_path, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        with open(ann_path) as f:
            coco = json.load(f)
        # COCO helpers ---------------------------------------------------------
        self.imgs = {img["id"]: img for img in coco["images"]}
        self.anns = {}
        for ann in coco["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)                # единственный класс: MRZ

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

# ----  Аугментации  ----------------------------------------------------------
class ComposeTransforms:
    def __call__(self, img, target):
        # Случайный горизонтальный флип
        if random.random() < 0.5:
            img = F.hflip(img)
            w = img.width
            boxes = target["boxes"]
            boxes = boxes.clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes

        # Случайный поворот в пределах -30 до 30 градусов
        angle = random.uniform(-30, 30)
        img = F.rotate(img, angle)
        boxes = target["boxes"]
        boxes = boxes.clone()
        boxes = self.rotate_boxes(boxes, angle, img.width, img.height)
        target["boxes"] = boxes

        # Добавление случайного шума (не сильного)
        img = self.add_random_noise(img)

        # Добавление случайных бликов
        img = self.add_random_glare(img)

        # Легкое добавление потертости (по желанию)
        img = self.add_worn_effect(img)

        img = F.to_tensor(img)  # Преобразуем в тензор
        return img, target

    def rotate_boxes(self, boxes, angle, width, height):
        """Корректируем координаты боксов для поворота"""
        angle_rad = np.deg2rad(angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        new_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            new_x_min = cos_theta * (x_min - center_x) - sin_theta * (y_min - center_y) + center_x
            new_y_min = sin_theta * (x_min - center_x) + cos_theta * (y_min - center_y) + center_y
            new_x_max = cos_theta * (x_max - center_x) - sin_theta * (y_max - center_y) + center_x
            new_y_max = sin_theta * (x_max - center_x) + cos_theta * (y_max - center_y) + center_y
            new_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

        return torch.tensor(new_boxes, dtype=torch.float32)

    def add_random_noise(self, img):
        """Добавление случайного шума на изображение"""
        np_img = np.array(img)
        noise = np.random.normal(0, 5, np_img.shape)  # Небольшой шум
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def add_random_glare(self, img):
        """Добавление случайного блика на изображение"""
        width, height = img.size
        img_copy = img.copy()

        # Создаем пустое изображение для блика
        glare = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        # Задаем случайные размеры и положение блика
        glare_width = random.randint(50, 200)
        glare_height = random.randint(50, 200)
        glare_position = (random.randint(0, width - glare_width), random.randint(0, height - glare_height))

        # Генерация случайной прозрачности для блика
        opacity = random.randint(30, 100)  # Прозрачность блика (0-255)
        glare.paste((255, 255, 255, opacity), (0, 0, glare_width, glare_height))  # Размещение блика

        # Наложение блика на оригинальное изображение
        img_copy.paste(glare, glare_position, mask=glare)

        return img_copy

    def add_worn_effect(self, img):
        """Добавление легкой потертости (к примеру, через размытие)"""
        if random.random() < 0.3:  # Редко, но бывает
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))  # Легкое размытие
        return img


# ----  Модель  ---------------------------------------------------------------
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    return model

# ----  Обучение  -------------------------------------------------------------
def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=50):
    model.train()
    epoch_loss = 0
    for i, (imgs, targets) in enumerate(loader):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
        if i % print_freq == 0:
            print(f"Epoch {epoch}, step {i}, loss {losses.item():.4f}")
    return epoch_loss / len(loader)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

# ----  main  -----------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PassportMRZDataset(
        "dataset_small/images",
        "dataset_small/coco_annotations.json",
        transforms=ComposeTransforms())
    val_set = PassportMRZDataset(
        "dataset/val/images",
        "dataset/val/annotations.json",
        transforms=ComposeTransforms())      # для проверки можно убрать аугм.

    train_loader = DataLoader(
        train_set, batch_size=4, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(
        val_set, batch_size=2, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)))

    model = get_model()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    best_loss = float("inf")
    for epoch in range(1, 21):  # 20 эпох для начала
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"Epoch {epoch} finished. Train loss: {loss:.4f}")

        # простая проверка: посчитать loss на валидации
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                val_loss += sum(loss_dict.values()).item()
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_name = f"mrz_best_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt"
            save_checkpoint(model, ckpt_name)
            print(f"** Saved new best model to {ckpt_name} **")

if __name__ == "__main__":
    main()
