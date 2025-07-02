#!/usr/bin/env python3
"""
convert_to_jpg.py   –  рекурсивно конвертирует все изображения в каталоге
                       (и его подпапках) в JPG с тем же базовым именем.
                       Исходные файлы стираются.

Запуск:
    python convert_to_jpg.py /path/to/passports
Если путь не указан – берётся текущая директория.
"""

import sys
from pathlib import Path
from PIL import Image

PATH = "passports"

IMG_EXTS = {".png", ".webp", ".tiff", ".bmp", ".gif"}

def convert_one(path: Path):
    dst = path.with_suffix(".jpg")
    try:
        with Image.open(path) as im:
            # для PNG с альфа-каналом – заливаем белым
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, "white")
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
            im.save(dst, "JPEG", quality=95, subsampling=0)
        path.unlink()             # удаляем оригинал
        print(f"✔ {path.name} → {dst.name}")
    except Exception as e:
        print(f"⚠ Не удалось {path}: {e}")

def main(root: Path):
    for ext in IMG_EXTS:
        for fp in root.rglob(f"*{ext}"):
            convert_one(fp)

if __name__ == "__main__":
    folder = Path(PATH) if len(sys.argv) > 1 else Path.cwd()
    if not folder.exists():
        sys.exit(f"Нет такой папки: {folder}")
    main(folder)
