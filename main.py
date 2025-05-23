import os
import random
import itertools
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
from faker import Faker

# --------------------------------------------------------------------------------------
# --- 1. КОНФИГУРАЦИЯ YOLO-выхода и общие константы
# --------------------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
TEMPLATE   = ROOT / "templates" / "passport_template.png"
FONTS_DIR  = ROOT / "fonts"
YOLO_ROOT  = ROOT / "dataset_yolo"

# Stage1: полный паспорт
ST1_IMG_T  = YOLO_ROOT / "stage1" / "images" / "train"
ST1_IMG_V  = YOLO_ROOT / "stage1" / "images" / "val"
ST1_LBL_T  = YOLO_ROOT / "stage1" / "labels" / "train"
ST1_LBL_V  = YOLO_ROOT / "stage1" / "labels" / "val"
# Stage2: поля внутри паспорта
ST2_IMG_T  = YOLO_ROOT / "stage2" / "images" / "train"
ST2_IMG_V  = YOLO_ROOT / "stage2" / "images" / "val"
ST2_LBL_T  = YOLO_ROOT / "stage2" / "labels" / "train"
ST2_LBL_V  = YOLO_ROOT / "stage2" / "labels" / "val"

for d in (ST1_IMG_T, ST1_IMG_V, ST1_LBL_T, ST1_LBL_V,
          ST2_IMG_T, ST2_IMG_V, ST2_LBL_T, ST2_LBL_V):
    os.makedirs(d, exist_ok=True)

# Количество паспортов и доля тренировочного
N_PASSPORTS = 100
TRAIN_RATIO = 0.8

# Настройки шрифтов и генератора
RUS_FONTS    = list((FONTS_DIR).glob("*.TTF"))
MRZ_FONT     = FONTS_DIR / "ocr-b-regular.ttf"
FONT_SIZE    = 18
MRZ_FONT_SZ  = 18

fake   = Faker("ru_RU")
random.seed(42)

# Фиксированные координаты полей на шаблоне
COORDS = {
    "surname": (258, 405),
    "name":    (258, 460),
    "patronymic": (258, 490),
    "gender":     (198, 518),
    "dob":        (310, 517),
    "birth_place":    (220, 545),
    "issuing_auth":   (105, 73),
    "issue_date":     (100, 155),
    "division_code":  (328, 155),
    "series_number_top":    (490, 109),
    "series_number_bottom": (490, 445),
    "mrz1": (15, 650),
    "mrz2": (15, 680),
    "photo_box": (26, 435, 170, 630),
}

# Классы для полей (stage2)
FIELD_CLASSES = {
    "surname":0, "name":1, "patronymic":2, "gender":3,
    "dob":4, "birth_place":5, "issuing_auth":6,
    "issue_date":7, "division_code":8,
    "series_number_top":9, "series_number_bottom":9,
    "mrz1":10, "mrz2":10
}

# --------------------------------------------------------------------------------------
# --- 2. ХЕЛПЕРЫ
# --------------------------------------------------------------------------------------
def trim_whitespace(im: Image.Image) -> Image.Image:
    bg = im.getpixel((0, 0))
    diff = ImageChops.difference(im, Image.new(im.mode, im.size, bg))
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

def save_yolo_txt(path: Path, items, img_size):
    W, H = img_size
    lines = []
    for cls_id, (x1, y1, x2, y2) in items:
        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1: continue
        xc = ((x1 + x2) * 0.5) / W
        yc = ((y1 + y2) * 0.5) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))

def wrap_text(text, font, max_width, max_lines=3):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if font.getlength(test) <= max_width:
            line = test
        else:
            lines.append(line); line = w
            if len(lines) == max_lines - 1:
                lines.append(" ".join(words[words.index(w):]))
                return lines
    if line: lines.append(line)
    return lines[:max_lines]

def mrz_checksum(data: str) -> str:
    weights = [7, 3, 1]
    # A–Z → 10–35, 0–9 → 0–9, '<' → 0
    vals = {chr(i): i-55 for i in range(65,91)}
    vals.update({str(i): i for i in range(10)}); vals['<']=0
    total = sum(vals.get(c,0)*weights[i%3] for i,c in enumerate(data))
    return str(total%10)

def transliterate_gost(text: str) -> str:
    M = {'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ё':'2','Ж':'J','З':'Z',
         'И':'I','Й':'Q','К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P','Р':'R',
         'С':'S','Т':'T','У':'U','Ф':'F','Х':'H','Ц':'C','Ч':'3','Ш':'4','Щ':'W',
         'Ъ':'X','Ы':'Y','Ь':'9','Э':'6','Ю':'7','Я':'8',' ':'<'}
    return ''.join(M.get(c.upper(),'<') for c in text)

def random_date(start_year=1950, end_year=None):
    end_year = end_year or datetime.now().year
    s = datetime(start_year,1,1)
    e = datetime(end_year,12,31)
    return (s + timedelta(days=random.randint(0,(e-s).days))).strftime("%d.%m.%Y")

def random_division_code():
    return f"{random.randint(100,999)}-{random.randint(100,999)}"

def random_series_number():
    return random.randint(10,99), random.randint(10,99), random.randint(100000,999999)

def random_fullname(gender):
    if gender=="МУЖ":
        return (fake.last_name_male().upper(), fake.first_name_male().upper(), fake.middle_name_male().upper())
    return (fake.last_name_female().upper(), fake.first_name_female().upper(), fake.middle_name_female().upper())

def random_birth_place():
    city = fake.city().upper()
    region = random.choice(["МОСКОВСКАЯ ОБЛАСТЬ","САНКТ-ПЕТЕРБУРГ"])
    return random.choice([f"{city}", f"{city}, {region}", f"С. {city}, {region}", f"{city}, {region}, РФ"])

# --------------------------------------------------------------------------------------
# --- 3. КЛАСС ГЕНЕРАТОРА ПАСПОРТОВ
# --------------------------------------------------------------------------------------
class PassportGenerator:
    def __init__(self):
        tpl = Image.open(TEMPLATE).convert('RGB')
        self.template = trim_whitespace(tpl)
        self.fonts    = [ImageFont.truetype(str(fp), FONT_SIZE) for fp in RUS_FONTS]
        self.mrz_font = ImageFont.truetype(str(MRZ_FONT), MRZ_FONT_SZ)
        # высота строки для wrap_text
        self.line_h   = self.fonts[0].getbbox("Mg")[3] + 4

    def _draw_text(self, draw, pos, text, font, jitter=(0,0)):
        x, y = pos
        x += random.randint(-jitter[0], jitter[0])
        y += random.randint(-jitter[1], jitter[1])
        draw.text((x,y), text, font=font, fill="black")
        w, h = font.getbbox(text)[2:]
        return [x, y, x+w, y+h]

    def _draw_wrapped(self, draw, pos, text, font, max_w, max_lines):
        x0, y0 = pos
        bbs = []
        for i, line in enumerate(wrap_text(text, font, max_w, max_lines)):
            bb = self._draw_text(draw, (x0, y0 + i*self.line_h), line, font, jitter=(15,5))
            bbs.append(bb)
        xs = [v for bb in bbs for v in (bb[0], bb[2])]
        ys = [v for bb in bbs for v in (bb[1], bb[3])]
        return [min(xs), min(ys), max(xs), max(ys)]

    def generate_one(self, idx):
        """
        Рисует паспорт, возвращает готовый canvas,
        список annots=[{"label":..., "bbox":[...]}] и bbox полного паспорта.
        """
        img = self.template.copy()
        draw = ImageDraw.Draw(img)
        annots = []

        # --- случайные данные ---
        dob_text  = random_date(1960,2003)
        p1,p2,pn  = random_series_number()
        passport_num = f"{p1:02d}{p2:02d}{pn:06d}"
        gender_ru = random.choice(["МУЖ","ЖЕН"])
        nationality = "RUS"
        surname, name, patr = random_fullname(gender_ru)
        birth_place  = random_birth_place()
        issue_date   = random_date(2015)
        division_code= random_division_code()

        # --- текстовые поля ---
        font = random.choice(self.fonts)
        for key, text in [("surname", surname), ("name", name), ("patronymic", patr)]:
            annots.append({"label": key,
                           "bbox": self._draw_text(draw, COORDS[key], text, font, jitter=(15,5))})
        annots.append({"label": "gender",
                       "bbox": self._draw_text(draw, COORDS["gender"], gender_ru, font, jitter=(15,5))})
        annots.append({"label": "dob",
                       "bbox": self._draw_text(draw, COORDS["dob"], dob_text, font, jitter=(15,5))})
        annots.append({"label": "birth_place",
                       "bbox": self._draw_wrapped(draw, COORDS["birth_place"], birth_place, font,
                                                  img.width-COORDS["birth_place"][0]-20, 3)})
        annots.append({"label": "issuing_auth",
                       "bbox": self._draw_wrapped(draw, COORDS["issuing_auth"],
                                                  random.choice(["МВД РФ","ГУ МВД МО"]), font,
                                                  img.width-COORDS["issuing_auth"][0]-20, 2)})
        annots.append({"label": "issue_date",
                       "bbox": self._draw_text(draw, COORDS["issue_date"], issue_date, font, jitter=(15,5))})
        annots.append({"label": "division_code",
                       "bbox": self._draw_text(draw, COORDS["division_code"], division_code, font, jitter=(15,5))})

        # --- MRZ (всегда рисуем) ---
        # (упрощённо без проверки контрольных сумм)
        line1 = f"P<{nationality}{transliterate_gost(surname)}<<{transliterate_gost(name)}"
        line2 = passport_num + dob_text.replace(".","") + gender_ru[0] + issue_date.replace(".","")
        bb1 = self._draw_text(draw, COORDS["mrz1"], line1.ljust(44,'<'), self.mrz_font)
        bb2 = self._draw_text(draw, COORDS["mrz2"], line2.ljust(44,'<'), self.mrz_font)
        annots += [
            {"label":"mrz1","bbox":bb1},
            {"label":"mrz2","bbox":bb2},
        ]

        # --- финальный холст с отступами ---
        Mx, My = 200, 100
        cw, ch = img.width+Mx, img.height+My
        canvas = Image.new("RGB", (cw,ch), "white")
        dx, dy = random.randint(0,Mx), random.randint(0,My)
        canvas.paste(img, (dx,dy))
        # скорректировать bboxes
        for a in annots:
            x1,y1,x2,y2 = a["bbox"]
            a["bbox"] = [x1+dx,y1+dy,x2+dx,y2+dy]

        # bbox полного паспорта
        full_bbox = [dx,dy, dx+img.width, dy+img.height]
        return canvas, annots, full_bbox

# --------------------------------------------------------------------------------------
# --- 4. MAIN: ГЕНЕРАЦИЯ YOLO-датасетов
# --------------------------------------------------------------------------------------
def main():
    gen = PassportGenerator()

    for i in range(N_PASSPORTS):
        img, annots, full_bbox = gen.generate_one(i)
        W, H = img.size
        is_train = random.random() < TRAIN_RATIO

        # --- Stage 1: полный паспорт ---
        img_dir = ST1_IMG_T if is_train else ST1_IMG_V
        lbl_dir = ST1_LBL_T if is_train else ST1_LBL_V
        fname   = f"passport_{i:06d}.png"
        img.save(img_dir / fname)
        save_yolo_txt(lbl_dir / f"{fname[:-4]}.txt",
                      [(0, tuple(full_bbox))],
                      (W, H))

        # --- Stage 2: вырез полей ---
        x1,y1,x2,y2 = full_bbox
        crop = img.crop((x1,y1,x2,y2))
        pw, ph = crop.size
        c_img_dir = ST2_IMG_T if is_train else ST2_IMG_V
        c_lbl_dir = ST2_LBL_T if is_train else ST2_LBL_V
        crop_fname = fname
        crop.save(c_img_dir / crop_fname)

        # собрать аннотации полей внутри кропа
        items = []
        for a in annots:
            cls = FIELD_CLASSES[a["label"]]
            bx = a["bbox"]
            # смещение относительно x1,y1
            items.append((cls, (bx[0]-x1, bx[1]-y1, bx[2]-x1, bx[3]-y1)))
        save_yolo_txt(c_lbl_dir / f"{crop_fname[:-4]}.txt", items, (pw,ph))

        print(f"✅ YOLO datasets ready under {YOLO_ROOT}")

if __name__ == "__main__":
    main()
