# requirements:
# pip install torch torchvision torchmetrics torchinfo tensorboard

import json
import os
import random
import datetime

import torch
import torchvision
import numpy as np

from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torchinfo import summary
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ----  Dataset  --------------------------------------------------------------
class PassportMRZDataset(Dataset):
    def __init__(self, images_dir, ann_path, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        with open(ann_path) as f:
            coco = json.load(f)
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
            labels.append(1)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

class ToTensorOnly:
    def __call__(self, img, target):
        img = F.to_tensor(img)
        return img, target

# ----  Аугментации  ----------------------------------------------------------
class ComposeTransforms:
    def __call__(self, img, target):
        # случайный горизонтальный флип
        if random.random() < 0.5:
            img = F.hflip(img)
            w = img.width
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes

        # добавить шум
        img = self.add_random_noise(img)
        # добавить блик
        img = self.add_random_glare(img)
        # легкая потертость
        img = self.add_worn_effect(img)

        img = F.to_tensor(img)
        return img, target

    def add_random_noise(self, img):
        np_img = np.array(img)
        noise = np.random.normal(0, 5, np_img.shape)
        noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def add_random_glare(self, img):
        w, h = img.size
        glare = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        gw, gh = random.randint(50,200), random.randint(50,200)
        pos = (random.randint(0, w-gw), random.randint(0, h-gh))
        opacity = random.randint(30,100)
        block = Image.new('RGBA', (gw, gh), (255,255,255,opacity))
        glare.paste(block, pos, block)
        out = img.copy()
        out.paste(glare, mask=glare)
        return out.convert("RGB")

    def add_worn_effect(self, img):
        if random.random() < 0.3:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
        return img


# ----  Модель  ---------------------------------------------------------------
def get_model(num_classes=2, input_size=(3,800,800)):
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

    # печать структуры и подсчёт параметров
    print("\n=== Model Summary ===")
    summary(model, input_size=[(1, *input_size)],
            col_names=["input_size", "output_size", "num_params", "trainable"])
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable params: {trainable:,}\n")

    return model


# ----  Обучение  -------------------------------------------------------------
def train_one_epoch(model, optimizer, loader, device, epoch, writer=None, print_freq=50):
    model.train()
    epoch_loss = 0.0
    for i, (imgs, targets) in enumerate(loader):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        step = epoch * len(loader) + i
        if writer:
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

        if i % print_freq == 0:
            print(f"Epoch {epoch}, step {i}, loss {loss.item():.4f}")

    return epoch_loss / len(loader)


def evaluate_map(model, loader, device, epoch=None, writer=None):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            preds, gts = [], []
            for out, tgt in zip(outputs, targets):
                preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu()
                })
                gts.append({
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu()
                })
            metric.update(preds, gts)

    stats = metric.compute()
    m50 = stats["map_50"].item()
    ma = stats["map"].item()
    print(f"Validation mAP@0.5: {m50:.4f}, mAP@[.5:.95]: {ma:.4f}")

    if writer and epoch is not None:
        writer.add_scalar("mAP/0.5", m50, epoch)
        writer.add_scalar("mAP/0.5:0.95", ma, epoch)

    return stats


# ----  main  -----------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir="runs/fasterrcnn_experiment")

    train_set = PassportMRZDataset(
        "../dataset_small/images",
        "../dataset_small/coco_annotations.json",
        transforms=ToTensorOnly()
    )
    val_set = PassportMRZDataset(
        "../dataset_small/images",
        "../dataset_small/coco_annotations.json",
        transforms=ComposeTransforms()
    )

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_set,   batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model()
    model.to(device)

    # если хотите залогировать граф:
    # dummy = torch.zeros((1,3,800,800), device=device)
    # writer.add_graph(model, dummy)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-4
    )

    best_map50 = 0.0
    for epoch in range(1, 21):
        print(f"\n=== Epoch {epoch} ===")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer)
        print(f"Train loss: {train_loss:.4f}")

        stats = evaluate_map(model, val_loader, device, epoch, writer)
        m50 = stats["map_50"].item()
        if m50 > best_map50:
            best_map50 = m50
            ckpt = f"mrz_best_map50_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"** Saved new best model to {ckpt} **")

    writer.close()


if __name__ == "__main__":
    main()
