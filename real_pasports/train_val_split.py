#!/usr/bin/env python3
# train_val_split.py ------------------------------------------------------------
"""
Раскладывает JPG + TXT по папкам train / val для Stage-2.

Пример:
    python train_val_split.py \
        --img_dir passports \
        --lbl_dir passports_annotations \
        --out    dataset_yolo_test/stage2 \
        --ratio  0.8 \
        --seed   42
"""
import argparse, pathlib, random, shutil, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', required=True, help='папка с JPG-картинками')
    p.add_argument('--lbl_dir', required=True, help='папка с *.txt-аннотациями')
    p.add_argument('--out', default='dataset_yolo_test/stage2',
                   help='корень датасета (будет создан)')
    p.add_argument('--ratio', type=float, default=0.8,
                   help='доля train (0…1), по умолчанию 0.8')
    p.add_argument('--seed',  type=int, default=42, help='random seed')
    p.add_argument('--copy',  action='store_true',
                   help='копировать вместо перемещения')
    return p.parse_args()

def ensure_dirs(root: pathlib.Path):
    folders = [root / p for p in (
        'images/train', 'images/val', 'labels/train', 'labels/val')]
    for d in folders: d.mkdir(parents=True, exist_ok=True)
    return folders[:2], folders[2:]      # (img_train, img_val), (lbl_train, lbl_val)

def main():
    a = parse_args()
    img_dir = pathlib.Path(a.img_dir); lbl_dir = pathlib.Path(a.lbl_dir)
    if not img_dir.exists() or not lbl_dir.exists():
        sys.exit('❌ img_dir или lbl_dir не существует')

    img_paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.jpeg'))
    if not img_paths: sys.exit('❌ В img_dir нет JPG-файлов')

    random.seed(a.seed); random.shuffle(img_paths)
    split = int(len(img_paths) * a.ratio)
    train_set = {p.stem for p in img_paths[:split]}

    (img_tr, img_val), (lbl_tr, lbl_val) = ensure_dirs(pathlib.Path(a.out))

    move = shutil.copy2 if a.copy else shutil.move
    for p in img_paths:
        dst_img = img_tr if p.stem in train_set else img_val
        move(p, dst_img / p.name)

        label_src = lbl_dir / f'{p.stem}.txt'
        dst_lbl  = lbl_tr if p.stem in train_set else lbl_val
        if label_src.exists():
            move(label_src, dst_lbl / label_src.name)
        else:
            print(f'⚠ нет аннотации для {p.name}')

    print('✅ Готово:')
    print(f'   train: {len(train_set)}   val: {len(img_paths)-len(train_set)}')
    print(f'   корень датасета → {a.out}')

if __name__ == '__main__':
    main()
