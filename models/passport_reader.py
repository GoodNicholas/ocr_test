# passport_project/src/dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class PassportFieldDataset(Dataset):
    """
    Dataset для токен-классификации полей паспорта любой категории.
    Требует COCO-подобный JSON-файл с полями:
      - images: [{id, file_name, width, height}, ...]
      - annotations: [{image_id, bbox: [x,y,w,h], text, label}, ...]
      - categories (необязательно): [{id, name}, ...]

    Пример файла annotations.json:
    {
      "images": [
        {"id": 1, "file_name": "img_001.jpg", "width": 2480, "height": 3508},
        {"id": 2, "file_name": "img_002.jpg", "width": 2480, "height": 3508}
      ],
      "annotations": [
        {"image_id": 1, "bbox": [100, 200, 400, 50], "text": "IDNUMBER", "label": "PassportNumber"},
        {"image_id": 1, "bbox": [100, 260, 800, 60], "text": "JOHN DOE", "label": "Name"},
        {"image_id": 2, "bbox": [150, 300, 400, 50], "text": "123456789", "label": "PassportNumber"}
      ],
      "categories": [
        {"id": 1, "name": "PassportNumber"},
        {"id": 2, "name": "Name"},
        {"id": 3, "name": "MRZ"}
      ]
    }
    """
    def __init__(self, annotations_file, images_dir, processor, max_length: int = 512):
        self.processor = processor
        self.images_dir = images_dir
        self.max_length = max_length

        with open(annotations_file, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        # формируем список меток (label_list)
        if 'categories' in coco:
            labels = [cat['name'] for cat in coco['categories']]
        else:
            labels = sorted({ann.get('label', 'O') for ann in coco['annotations']})
        self.label_list = ['O'] + [l for l in labels if l != 'O']
        self.label2id = {lbl: i for i, lbl in enumerate(self.label_list)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        # группируем аннотации по image_id
        anns = {}
        for ann in coco['annotations']:
            anns.setdefault(ann['image_id'], []).append(ann)

        self.examples = []
        for img in coco['images']:
            items = anns.get(img['id'], [])
            if not items:
                continue
            words, boxes, word_labels = [], [], []
            for ann in items:
                txt = ann.get('text', '').strip()
                if not txt:
                    continue
                x, y, w, h = ann['bbox']
                words.append(txt)
                boxes.append([x, y, x+w, y+h])
                lbl = ann.get('label', 'O')
                word_labels.append(self.label2id.get(lbl, 0))
            self.examples.append({
                'image_path': os.path.join(images_dir, img['file_name']),
                'words': words,
                'boxes': boxes,
                'labels': word_labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = Image.open(ex['image_path']).convert('RGB')
        encoding = self.processor(
            image,
            ex['words'],
            boxes=ex['boxes'],
            word_labels=ex['labels'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # убираем batch-dim
        return {k: v.squeeze(0) for k, v in encoding.items()}

# passport_project/src/train.py
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, get_scheduler
from transformers import DataCollatorForTokenClassification
import evaluate
from tqdm.auto import tqdm

# ------------ Гиперпараметры ------------
CONFIG = {
    'ANNOTATIONS_FILE': 'path/to/annotations.json',
    'IMAGES_DIR': 'path/to/images',
    'OUTPUT_DIR': './model_output',
    'NUM_EPOCHS': 5,
    'TRAIN_BATCH_SIZE': 4,
    'LEARNING_RATE': 5e-5,
    'MAX_LENGTH': 512,
    'WARMUP_RATIO': 0.1
}
# ----------------------------------------

def train(config):
    # инициализация
    processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base')
    train_dataset = PassportFieldDataset(
        config['ANNOTATIONS_FILE'], config['IMAGES_DIR'], processor,
        max_length=config['MAX_LENGTH']
    )
    label_list = train_dataset.label_list

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        'microsoft/layoutlmv3-base',
        num_labels=len(label_list),
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)}
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # dataloader
    data_collator = DataCollatorForTokenClassification(processor, padding=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['TRAIN_BATCH_SIZE'],
        shuffle=True,
        collate_fn=data_collator
    )

    # optimizer и scheduler
    optimizer = AdamW(model.parameters(), lr=config['LEARNING_RATE'])
    num_steps = config['NUM_EPOCHS'] * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear', optimizer=optimizer,
        num_warmup_steps=int(config['WARMUP_RATIO'] * num_steps),
        num_training_steps=num_steps
    )

    # метрика seqeval
    metric = evaluate.load('seqeval')

    # обучение
    progress = tqdm(range(num_steps), desc='Training')
    model.train()
    for epoch in range(config['NUM_EPOCHS']):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress.update(1)
            progress.set_postfix({'loss': loss.item()})

    # сохранение
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    model.save_pretrained(config['OUTPUT_DIR'])
    processor.save_pretrained(config['OUTPUT_DIR'])

if __name__ == '__main__':
    train(CONFIG)
