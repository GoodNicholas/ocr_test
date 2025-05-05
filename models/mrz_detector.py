import json, os, random, datetime
import torch
import torchvision
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import faster_rcnn, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm
import numpy as np

# ----  Dataset  --------------------------------------------------------------
class PassportMRZDataset(Dataset):
    def __init__(self, images_dir, ann_path, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        with open(ann_path) as f:
            coco = json.load(f)
        self.imgs = {img['id']: img for img in coco['images']}
        self.anns = {}
        for ann in coco['annotations']:
            self.anns.setdefault(ann['image_id'], []).append(ann)
        self.ids = list(self.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.imgs[img_id]
        path = os.path.join(self.images_dir, info['file_name'])
        img = Image.open(path).convert('RGB')

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(1)
            areas.append(w*h)
            iscrowd.append(ann.get('iscrowd', 0))

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

# ----  Аугментации  ----------------------------------------------------------
class ComposeTransforms:
    def __init__(
        self,
        rotate_angle_range=(-30, 30), rotate_prob=1.0,
        flip_prob=0.5,
        noise_mean=0, noise_std=5,
        clip_min=0, clip_max=255,
        glare_size_range=(50, 200), glare_opacity_range=(30,100),
        glare_prob=1.0,
        worn_prob=0.3
    ):
        self.rotate_min, self.rotate_max = rotate_angle_range
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.noise_mean, self.noise_std = noise_mean, noise_std
        self.clip_min, self.clip_max = clip_min, clip_max
        self.glare_w_min, self.glare_w_max = glare_size_range
        self.glare_h_min, self.glare_h_max = glare_size_range
        self.glare_opacity_min, self.glare_opacity_max = glare_opacity_range
        self.glare_prob = glare_prob
        self.worn_prob = worn_prob

    def __call__(self, img, target):
        if random.random() < self.flip_prob:
            img, target = self._hflip(img, target)
        if random.random() < self.rotate_prob:
            angle = random.uniform(self.rotate_min, self.rotate_max)
            img, target = self._rotate(img, target, angle)
        img = self._add_noise(img)
        if random.random() < self.glare_prob:
            img = self._add_glare(img)
        if random.random() < self.worn_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
        img = F.to_tensor(img)
        return img, target

    def _hflip(self, img, target):
        w = img.width
        img2 = F.hflip(img)
        boxes = target['boxes'].clone()
        boxes[:, [0,2]] = w - boxes[:, [2,0]]
        target['boxes'] = boxes
        return img2, target

    def _rotate(self, img, target, angle):
        w, h = img.width, img.height
        img2 = img.rotate(angle, expand=True)
        boxes = target['boxes'].clone()
        new_boxes = []
        for box in boxes:
            pts = self._corners(box)
            pts_rot = [self._rotate_pt(p, angle, w, h) for p in pts]
            xs = [p[0] for p in pts_rot]; ys = [p[1] for p in pts_rot]
            new_boxes.append([min(xs), min(ys), max(xs), max(ys)])
        tb = torch.tensor(new_boxes, dtype=torch.float32)
        tb[:,[0,2]].clamp_(0, img2.width)
        tb[:,[1,3]].clamp_(0, img2.height)
        target['boxes'] = tb
        return img2.convert('RGB'), target

    @staticmethod
    def _corners(box):
        x1,y1,x2,y2 = box
        return [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]

    @staticmethod
    def _rotate_pt(pt, angle, w, h):
        cx, cy = w/2, h/2
        rad = torch.deg2rad(torch.tensor(angle))
        cos, sin = rad.cos(), rad.sin()
        x,y = pt
        x0, y0 = x-cx, y-cy
        xr = cos*x0 - sin*y0 + cx
        yr = sin*x0 + cos*y0 + cy
        return (xr.item(), yr.item())

    def _add_noise(self, img):
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(self.noise_mean, self.noise_std, arr.shape)
        arr = np.clip(arr + noise, self.clip_min, self.clip_max).astype(np.uint8)
        return Image.fromarray(arr)

    def _add_glare(self, img):
        w, h = img.size
        overlay = Image.new('RGBA', img.size, (0,0,0,0))
        gw = random.randint(self.glare_w_min, self.glare_w_max)
        gh = random.randint(self.glare_h_min, self.glare_h_max)
        ox = random.randint(0, w-gw); oy = random.randint(0, h-gh)
        opacity = random.randint(self.glare_opacity_min, self.glare_opacity_max)
        glare = Image.new('RGBA', (gw,gh), (255,255,255,opacity))
        overlay.paste(glare, (ox,oy), glare)
        composited = Image.alpha_composite(img.convert('RGBA'), overlay)
        return composited.convert('RGB')

# ----  Модель  ---------------------------------------------------------------
def get_model(num_classes=2):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = faster_rcnn.fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model

# ----  Обучение  -------------------------------------------------------------
def train_one_epoch(model, optimizer, loader, device, epoch, metrics_log):
    model.train()
    total_loss = 0
    for imgs, tgts in tqdm(loader, desc=f"Train Epoch {epoch}"):
        imgs = [img.to(device) for img in imgs]
        tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        losses = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        for k,v in loss_dict.items():
            metrics_log.setdefault(k, []).append(v.item())
    return total_loss / len(loader)

# ----  main  -----------------------------------------------------------------
def main(
    train_dir='dataset_small/images', train_ann='dataset_small/coco_annotations.json',
    val_dir='dataset/val/images', val_ann='dataset/val/annotations.json',
    batch_size=4, val_batch=2,
    lr=1e-4, weight_decay=1e-4,
    num_epochs=20
):
    # reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms_train = ComposeTransforms()
    train_set = PassportMRZDataset(train_dir, train_ann, transforms_train)
    val_set = PassportMRZDataset(val_dir, val_ann, transforms=None)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_set, batch_size=val_batch, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # метрика mAP для bbox
    metric = MeanAveragePrecision(iou_type='bbox')

    best_map50 = 0.0
    best_ckpt = None
    metrics_log = {}

    for epoch in range(1, num_epochs+1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, metrics_log)
        scheduler.step()
        # валидация и подсчёт mAP@0.5
        model.eval()
        metric.reset()
        with torch.no_grad():
            for imgs, tgts in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                imgs = [i.to(device) for i in imgs]
                outputs = model(imgs)
                # перевод таргетов в формат метрики
                targets = [{
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                } for t in tgts]
                preds = [{
                    'boxes': o['boxes'],
                    'scores': o['scores'],
                    'labels': o['labels']
                } for o in outputs]
                metric.update(preds, targets)
        res = metric.compute()
        map50 = res['map_50'].item()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} | Train loss: {train_loss:.4f}, mAP@0.5: {map50:.4f}, LR: {current_lr:.2e}")

        # сохраняем лучший по mAP50
        if map50 > best_map50:
            best_map50 = map50
            best_ckpt = f"mrz_best_map50_{map50:.3f}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt"
            torch.save(model.state_dict(), best_ckpt)

    last_ckpt = f"mrz_last_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt"
    torch.save(model.state_dict(), last_ckpt)
    print(f"Training complete. Best mAP@0.5: {best_map50:.4f} -> {best_ckpt}, Last: {last_ckpt}")

if __name__ == '__main__':
    main()
