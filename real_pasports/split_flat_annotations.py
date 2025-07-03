#!/usr/bin/env python3
# split_flat_annotations.py -----------------------------------------------------
"""
Разбивает объединённую разметку (flatten.txt) на отдельные YOLO-файлы.

Пример:
    python split_flat_annotations.py \
        --flat flatten.txt \
        --out  passports_annotations
"""
import argparse, pathlib, collections

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--flat', required=True,
                   help='файл с объединённой разметкой')
    p.add_argument('--out',  required=True,
                   help='куда писать *.txt (будет создано)')
    p.add_argument('--overwrite', action='store_true',
                   help='переписывать существующие *.txt')
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = collections.defaultdict(list)
    with open(args.flat, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            img, *rest = ln.split()
            grouped[img].append(rest)

    for img, rows in grouped.items():
        rows.sort(key=lambda r: int(r[0]))          # по классу
        txt_path = out_dir / f'{pathlib.Path(img).stem}.txt'
        if txt_path.exists() and not args.overwrite:
            print(f'⚠ {txt_path} уже есть – пропускаю')
            continue
        txt_path.write_text('\n'.join(' '.join(r) for r in rows), encoding='utf-8')
        print(f'✔ {txt_path.name}')

if __name__ == '__main__':
    main()
