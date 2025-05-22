import os
import shutil
import random
import json
import string
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
from faker import Faker
import itertools

# --------------------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "templates" / "passport_template.png"
FONTS_DIR = ROOT / "fonts"
OUT_DIR = ROOT / "dataset_small"
OUT_IMG_DIR = OUT_DIR / "images"
DEBUG_IMG_DIR = OUT_DIR / "debug_images"
ANNOT_PATH = OUT_DIR / "annotations.json"
COCO_ANNOT_PATH = OUT_DIR / "coco_annotations.json"

YOLO_ROOT = ROOT / "dataset_yolo"

ST1_IMG_T = YOLO_ROOT / "stage1" / "images" / "train"
ST1_IMG_V = YOLO_ROOT / "stage1" / "images" / "val"
ST1_LBL_T = YOLO_ROOT / "stage1" / "labels" / "train"
ST1_LBL_V = YOLO_ROOT / "stage1" / "labels" / "val"

ST2_IMG_T = YOLO_ROOT / "stage2" / "images" / "train"
ST2_IMG_V = YOLO_ROOT / "stage2" / "images" / "val"
ST2_LBL_T = YOLO_ROOT / "stage2" / "labels" / "train"
ST2_LBL_V = YOLO_ROOT / "stage2" / "labels" / "val"

for d in (ST1_IMG_T,ST1_IMG_V,ST1_LBL_T,ST1_LBL_V,
          ST2_IMG_T,ST2_IMG_V,ST2_LBL_T,ST2_LBL_V):
    d.mkdir(parents=True, exist_ok=True)

# Параметры
MARGIN_X = 200
MARGIN_Y = 100
MRZ_FONT_SIZE = 18
N_PASSPORTS = 1000
SERIES_FONT_SIZE = 20
GROUP_GAP = 25
FONT_SIZE = 18

# конфигурация дебага
DEBUG = True  # Здесь можно включать/выключать режим дебага

# фиксированные координаты полей на шаблоне (px)
COORDS = {
    "issuing_auth": (105, 73),
    "issue_date": (100, 155),
    "division_code": (328, 155),

    "surname": (258, 405),
    "name": (258, 460),
    "patronymic": (258, 490),
    "gender": (198, 518),
    "dob": (310, 517),
    "birth_place": (220, 545),

    "series_number_top": (490, 109),
    "series_number_bottom": (490, 445),

    "mrz1": (15, 650),
    "mrz2": (15, 680),

    "photo_box": (26, 435, 170, 630)
}

FIELD_CLASSES = {
    "surname":0, "name":1, "patronymic":2, "gender":3,
    "dob":4, "birth_place":5, "issuing_auth":6,
    "issue_date":7, "division_code":8,
    "series_number_top":9, "series_number_bottom":9,
    "mrz1":10, "mrz2":10
}
NC_STAGE1 = 1
NC_STAGE2 = max(FIELD_CLASSES.values())+1

# Инициализация
issuing_authorities = [
    "Министерством внутренних дел Российской Федерации",
    "ГУ МВД России по Московской области",
    "УМВД России по Республике Татарстан",
    "ОТДЕЛ МВД РОССИИ ПО ЛЕНИНСКОМУ РАЙОНУ Г. САМАРЫ",
    "ОТДЕЛЕНИЕ ПОЛИЦИИ № 3 ОТДЕЛА МВД РОССИИ ПО Г. ТУЛЕ",
    "МФЦ МОСКОВСКОЙ ОБЛАСТИ",
    "ГЕНЕРАЛЬНОЕ КОНСУЛЬСТВО РОССИИ В Г. НЬЮ-ЙОРКЕ"
]

authority_iterator = itertools.cycle(issuing_authorities)

# Настройка шрифтов
RUS_FONTS = [
    FONTS_DIR / "ARIAL.TTF",
    FONTS_DIR / "times.ttf",
    FONTS_DIR / "PTC55F.ttf",
]
MRZ_FONT = FONTS_DIR / "ocr-b-regular.ttf"
SERIES_FONT = ImageFont.truetype(str(random.choice(RUS_FONTS)), SERIES_FONT_SIZE)

# Инициализация генератора имён/дат
fake = Faker("ru_RU")
random.seed(42)

# Маппинг для транслитерации по ГОСТ
MAPPING_REV = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E',
    'Ё': '2', 'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Q', 'К': 'K',
    'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R',
    'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C',
    'Ч': '3', 'Ш': '4', 'Щ': 'W', 'Ъ': 'X', 'Ы': 'Y', 'Ь': '9',
    'Э': '6', 'Ю': '7', 'Я': '8', ' ': '<'
}


def transliterate_gost(text: str) -> str:
    return ''.join(MAPPING_REV.get(c.upper(), '<') for c in text.upper())


def get_next_issuing_authority():
    return next(authority_iterator)


# ----------------------------- HELPERS --------------------------------
def trim_whitespace(im: Image.Image) -> Image.Image:
    bg = im.getpixel((0, 0))
    diff = ImageChops.difference(im, Image.new(im.mode, im.size, bg))
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

def clamp_bbox(bb, W, H):
    x1,y1,x2,y2 = bb
    x1, y1 = max(0, x1),     max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return [x1, y1, x2, y2]


def save_yolo_txt(path, items, img_size):
    W, H = img_size
    lines = []
    for cls_id, (x1, y1, x2, y2) in items:

        x1, y1 = max(0, x1),      max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        xc = ((x1 + x2) * 0.5) / W
        yc = ((y1 + y2) * 0.5) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))

def prepare_stage2_items(annots, passport_bbox, crop_size, field_classes):
    px1, py1, px2, py2 = passport_bbox
    pw, ph = crop_size
    items = []
    for a in annots:
        label = a["label"]
        if label not in field_classes:
            continue
        cls = field_classes[label]
        fx1, fy1, fx2, fy2 = a["bbox"]
        # смещение относительно выреза
        rx1, ry1 = fx1 - px1, fy1 - py1
        rx2, ry2 = fx2 - px1, fy2 - py1
        # клэмпим по границам вырезанного паспорта
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(pw, rx2), min(ph, ry2)
        # пропускаем некорректные
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        items.append((cls, (rx1, ry1, rx2, ry2)))
    return items

def wrap_text(text, font, max_width, max_lines=3):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if font.getlength(test) <= max_width:
            line = test
        else:
            lines.append(line)
            line = w
            if len(lines) == max_lines - 1:
                lines.append(" ".join(words[words.index(w):]))
                return lines
    if line:
        lines.append(line)
    return lines[:max_lines]


def random_date(start_year=1950, end_year=None):
    end_year = end_year or datetime.now().year
    s = datetime(start_year, 1, 1)
    e = datetime(end_year, 12, 31)
    return (s + timedelta(days=random.randint(0, (e - s).days))).strftime("%d.%m.%Y")


def random_division_code(): return f"{random.randint(100, 999)}-{random.randint(100, 999)}"


def random_passport_series_number(): return random.randint(10, 99), random.randint(10, 99), random.randint(100000,
                                                                                                           999999)


def random_fullname(gender):
    if gender == "МУЖ":
        return (fake.last_name_male().upper(), fake.first_name_male().upper(), fake.middle_name_male().upper())
    return (fake.last_name_female().upper(), fake.first_name_female().upper(), fake.middle_name_female().upper())


def random_birth_place():
    city, region, village = fake.city().upper(), random.choice(
        ["МОСКОВСКАЯ ОБЛАСТЬ", "САНКТ‑ПЕТЕРБУРГ"]), fake.city().upper()
    template = random.choice(["{city}", "{city}, {region}", "С. {village}, {region}", "{city}, {region}, РФ"])
    return template.format(city=city, village=village, region=region)


def random_issuing_auth():
    org = get_next_issuing_authority()
    city, district, region = fake.city().upper(), random.choice(["ЛЕНИНСКОМУ", "ЦЕНТРАЛЬНОМУ"]), random.choice(
        ["МОСКОВСКАЯ ОБЛАСТЬ", "САНКТ‑ПЕТЕРБУРГ"])
    template = random.choice(["{org} по г. {city}", "{org} по {district} району г. {city}", "{org} по {region}"])
    return template.format(org=org, city=city, district=district, region=region)


def mrz_checksum(data: str) -> str:
    weights = [7, 3, 1]
    # A–Z → 10–35, 0–9 → 0–9, '<' → 0
    char_values = {chr(i): i - 55 for i in range(65, 91)}
    char_values.update({str(i): i for i in range(10)})
    char_values['<'] = 0

    total = sum(char_values.get(c, 0) * weights[i % 3] for i, c in enumerate(data))
    return str(total % 10)

# 2) основная функция
def gen_mrz(data: dict):
    """
    data = {
      'surname', 'given', 'patronymic',        # строки на кириллице
      'series': '1234',                         # 4 цифры
      'number': '123456',                       # 6 цифр
      'dob': 'YYMMDD',                          # дата рождения
      'expiry': 'YYMMDD',                       # срок действия
      'issue': 'YYMMDD',                        # дата выдачи
      'division_code': '123-456',               # код подразделения
      'nationality': 'RUS',                     # 3 буквы
      'gender': 'M' or 'F'
    }
    """

    # 2.1) строим имя
    surname = transliterate_gost(data['surname']).replace(' ', '<')
    given   = transliterate_gost(data['given']).replace(' ', '<')
    patron  = transliterate_gost(data['patronymic']).replace(' ', '<')

    name_field = f"{surname}<<{given}<{patron}"
    # удаляем все не A–Z и пробелы, заменяем на '<'
    name_field = ''.join(ch if 'A' <= ch <= 'Z' else '<' for ch in name_field)

    # 2.2) первая строка: P< + страна + имя, до 44 символов
    issuing = "RUS"  # код страны-эмитента
    line1 = f"P<{issuing}{name_field}"
    line1 = line1.ljust(44, '<')

    # 2.3) вторая строка: поля с чек-цифрами
    # паспорт: серия (3) + номер (6) = 9
    ser3 = data['series'][:3]
    num6 = data['number']
    doc_num = ser3 + num6
    c1 = mrz_checksum(doc_num)

    # дата рождения и чек
    bdate = data['dob']
    c2 = mrz_checksum(bdate)

    # срок действия и чек
    exp = data['expiry']
    c3 = mrz_checksum(exp)

    # персональные данные: 4-я цифра серии + дата выдачи + код подразделения
    last_ser = data['series'][3]
    div = data['division_code'].replace('-', '')
    pers = f"{last_ser}{data['issue']}{div}"
    pers = pers.ljust(14, '<')
    c4 = mrz_checksum(pers)

    # итоговая чек-цифра: по всем вышеперечисленным полям с их чек-цифрами
    concat = doc_num + c1 + bdate + c2 + data['gender'] + exp + c3 + pers + c4
    c5 = mrz_checksum(concat)

    # собираем вторую строку и заполняем до 44
    line2 = f"{doc_num}{c1}{data['nationality']}{bdate}{c2}{data['gender']}{exp}{c3}{pers}{c4}{c5}"
    line2 = line2.ljust(44, '<')

    return line1, line2



def insert_photo_noise(canvas: Image.Image, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    noise = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    noise_img = Image.fromarray(noise, mode='RGB')
    canvas.paste(noise_img, (x1, y1))


# ------------------------ PASSPORT GENERATOR --------------------------
class PassportGenerator:
    def __init__(self):
        tpl = Image.open(TEMPLATE_PATH).convert('RGB')
        self.template = trim_whitespace(tpl)
        self.fonts = [ImageFont.truetype(str(fp), FONT_SIZE) for fp in RUS_FONTS]
        self.mrz_font = ImageFont.truetype(str(MRZ_FONT), MRZ_FONT_SIZE)
        self.line_h = self.fonts[0].getbbox("Mg")[3] + 4

    def _draw_text(self, draw, pos, text, font, jitter=(0, 0)):
        x, y = pos
        x += random.randint(-jitter[0], jitter[0])
        y += random.randint(-jitter[1], jitter[1])
        draw.text((x, y), text, font=font, fill='black')
        w, h = font.getbbox(text)[2:]
        return [x, y, x + w, y + h]

    def _draw_wrapped(self, draw, pos, text, font, max_width, max_lines):
        x0, y0 = pos
        bboxes = []
        for i, line in enumerate(wrap_text(text, font, max_width, max_lines)):
            bb = self._draw_text(draw, (x0, y0 + i * self.line_h), line, font, jitter=(15, 5))
            bboxes.append(bb)
        xs = [v for bb in bboxes for v in (bb[0], bb[2])]
        ys = [v for bb in bboxes for v in (bb[1], bb[3])]
        return [min(xs), min(ys), max(xs), max(ys)]

    def generate_one(self, idx, offset=None, photo_box=None, include_mrz=True, debug=False):
        img = self.template.copy()
        draw = ImageDraw.Draw(img)
        annots = []

        # 1. Собираем данные паспорта
        dob_text = random_date(1960, 2003)
        dob_mrz = datetime.strptime(dob_text, "%d.%m.%Y").strftime("%y%m%d")
        exp_mrz = (datetime.now() + timedelta(days=365 * 10)).strftime("%y%m%d")
        p1, p2, pnum = random_passport_series_number()
        passport_num = f"{p1:02d}{p2:02d}{pnum:06d}"
        rus_gender = random.choice(["МУЖ", "ЖЕН"])
        gender_mrz = "M" if rus_gender == "МУЖ" else "F"
        surname, name, patr = random_fullname(rus_gender)
        birth_place = random_birth_place()
        issue_date = random_date(2015)
        division_code = random_division_code()
        issuing_auth = random_issuing_auth()
        nationality = "RUS"
        personal_num = ''.join(random.choices(string.digits, k=14))

        # 2. Рисуем обычные текстовые поля
        font = random.choice(self.fonts)
        for key, text in [("surname", surname), ("name", name), ("patronymic", patr)]:
            bb = self._draw_text(draw, COORDS[key], text, font, jitter=(15, 5))
            annots.append({"label": key, "bbox": bb, "text": text})

        bb = self._draw_text(draw, COORDS["gender"], rus_gender, font, jitter=(15, 5))
        annots.append({"label": "gender", "bbox": bb, "text": rus_gender})

        bb = self._draw_text(draw, COORDS["dob"], dob_text, font, jitter=(15, 5))
        annots.append({"label": "dob", "bbox": bb, "text": dob_text})

        bb = self._draw_wrapped(draw, COORDS["birth_place"], birth_place, font,
                                img.width - COORDS["birth_place"][0] - 20, 3)
        annots.append({"label": "birth_place", "bbox": bb, "text": birth_place})

        bb = self._draw_wrapped(draw, COORDS["issuing_auth"], issuing_auth, font,
                                img.width - COORDS["issuing_auth"][0] - 20, 2)
        annots.append({"label": "issuing_auth", "bbox": bb, "text": issuing_auth})

        bb = self._draw_text(draw, COORDS["issue_date"], issue_date, font, jitter=(15, 5))
        annots.append({"label": "issue_date", "bbox": bb, "text": issue_date})

        bb = self._draw_text(draw, COORDS["division_code"], division_code, font, jitter=(15, 5))
        annots.append({"label": "division_code", "bbox": bb, "text": division_code})

        # 3. MRZ (если нужно)
        if include_mrz:
            series_font = ImageFont.truetype(str(random.choice(RUS_FONTS)), SERIES_FONT_SIZE)
            parts = [f"{p1:02d}", f"{p2:02d}", f"{pnum:06d}"]

            widths = [int(series_font.getlength(p)) for p in parts]
            total_w = sum(widths) + GROUP_GAP * (len(parts) - 1)
            total_h = series_font.getbbox(parts[0])[3]
            sn_img = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0))
            sn_draw = ImageDraw.Draw(sn_img)
            x_cursor = 0
            for i, part in enumerate(parts):
                sn_draw.text((x_cursor, 0), part, font=series_font, fill=(192, 0, 0, 255))
                x_cursor += widths[i] + GROUP_GAP

            sn_rot = sn_img.rotate(270, expand=True)

            for key in ["series_number_top", "series_number_bottom"]:
                x0, y0 = COORDS[key]
                img.paste(sn_rot, (x0, y0), sn_rot)
                annots.append({
                    "label": key,
                    "bbox": [x0, y0, x0 + sn_rot.width, y0 + sn_rot.height],
                    "text": f"{parts[0]} {parts[1]} {parts[2]}"
                })

            issue_mrz = datetime.strptime(issue_date, "%d.%m.%Y").strftime("%y%m%d")

            # 4. MRZ
            mrz1, mrz2 = gen_mrz({
                "surname": surname,
                "given": name,
                "patronymic": patr,
                "series": f"{p1:02d}{p2:02d}",
                "number": f"{pnum:06d}",
                "dob": dob_mrz,
                "gender": gender_mrz,
                "expiry": exp_mrz,
                "issue": issue_mrz,  # формат YYMMDD
                "division_code": division_code.replace('-', ''),  # 6 цифр
                "nationality": "RUS"
            })

            bb1 = self._draw_text(draw, COORDS["mrz1"], mrz1, self.mrz_font)
            annots.append({"label": "mrz1", "bbox": bb1, "text": mrz1})
            bb2 = self._draw_text(draw, COORDS["mrz2"], mrz2, self.mrz_font)
            annots.append({"label": "mrz2", "bbox": bb2, "text": mrz2})

            # Debug: Сохранить область MRZ
            if debug:
                mrz_box = (bb1[0], bb1[1], bb2[2], bb2[3])  # Получаем bbox для MRZ
                mrz_img = img.crop(mrz_box)  # Вырезаем область MRZ
                DEBUG_IMG_DIR.mkdir(exist_ok=True, parents=True)
                mrz_img.save(DEBUG_IMG_DIR / f"{idx:06d}_mrz.png")  # Сохраняем с именем паспорта

        # 5. Вставка фотографии (шум)
        if photo_box:
            insert_photo_noise(img, photo_box)

        # Финальный холст и эффекты
        cw, ch = img.width + MARGIN_X, img.height + MARGIN_Y
        canvas = Image.new('RGB', (cw, ch), 'white')
        dx, dy = offset if offset is not None else (random.randint(0, MARGIN_X), random.randint(0, MARGIN_Y))
        canvas.paste(img, (dx, dy))
        for a in annots:
            x1, y1, x2, y2 = a['bbox']
            a['bbox'] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

        fname = f"passport_{idx:06d}.png"
        OUT_IMG_DIR.mkdir(exist_ok=True, parents=True)
        canvas.save(OUT_IMG_DIR / fname)

        # Структура аннотации
        mrz_coords = []
        if include_mrz:
            mrz_coords = [
                bb1[0] + dx, bb1[1] + dy,
                bb2[2] + dx, bb2[3] + dy
            ]

        annotation = {
            "image_id": fname,
            "mrz": {
                "exists": include_mrz,
                "coordinates": mrz_coords,
                "line1": mrz1 if include_mrz else "",
                "line2": mrz2 if include_mrz else ""
            },
            "photo": {
                "coordinates": photo_box if photo_box else []
            },
            "text": {
                "place_of_birth": birth_place,
                "issued_by": issuing_auth,
                "full_name": f"{surname} {name} {patr}",
                "gender": rus_gender,
                "dob": dob_text,
                "issue_date": issue_date,
                "division_code": division_code
            }
        }
        passport_bbox = [dx, dy, dx + img.width, dy + img.height]
        return fname, annotation, annots, passport_bbox


# ----------------------------- MAIN ----------------------------------
def main():
    # 1) Подготовка каталогов
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in (ST1_IMG_T, ST1_IMG_V, ST1_LBL_T, ST1_LBL_V,
              ST2_IMG_T, ST2_IMG_V, ST2_LBL_T, ST2_LBL_V):
        d.mkdir(parents=True, exist_ok=True)

    # 2) Инициализация генератора
    gen = PassportGenerator()
    ds = []
    coco_data = {
        "info": {
            "description": "RF internal passport MRZ dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "dolbaquaki",
            "date_created": "2025-05-01"
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC-BY-4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "mrz",
                "supercategory": "passport"
            }
        ]
    }

    # 3) Генерация
    for i in range(N_PASSPORTS):
        include_mrz = random.choice([True, False])
        fn, ann, annots, pbbox = gen.generate_one(
            i,
            offset=None,
            photo_box=COORDS["photo_box"],
            include_mrz=include_mrz,
            debug=DEBUG
        )

        img = Image.open(OUT_IMG_DIR / fn)
        W, H = img.size
        is_train = random.random() < 0.8

        # a) Сохраняем оригинальные JSON-аннотации
        ds.append({
            "image_id": fn,
            "mrz": ann["mrz"],
            "photo": ann["photo"],
            "text": ann["text"]
        })
        coco_data["images"].append({
            "id": i+1,
            "file_name": fn,
            "width": W,
            "height": H
        })
        if ann["mrz"]["exists"]:
            coco_data["annotations"].append({
                "id": i+1,
                "image_id": i+1,
                "category_id": 1,
                "bbox": ann["mrz"]["coordinates"],
                "area": (ann["mrz"]["coordinates"][2] - ann["mrz"]["coordinates"][0]) *
                        (ann["mrz"]["coordinates"][3] - ann["mrz"]["coordinates"][1]),
                "iscrowd": 0,
                "segmentation": []
            })

        # b) Stage1 YOLO: полный паспорт
        scan_dir = ST1_IMG_T if is_train else ST1_IMG_V
        lbl_dir  = ST1_LBL_T if is_train else ST1_LBL_V
        shutil.copy(OUT_IMG_DIR / fn, scan_dir / fn)
        save_yolo_txt(
            lbl_dir / f"{fn[:-4]}.txt",
            [(0, tuple(pbbox))],
            (W, H)
        )

        # c) Stage2 YOLO: поля внутри паспорта
        x1, y1, x2, y2 = pbbox
        crop = img.crop((x1, y1, x2, y2))
        pw, ph = crop.size
        crop_dir = ST2_IMG_T if is_train else ST2_IMG_V
        crop_lbl = ST2_LBL_T if is_train else ST2_LBL_V
        crop_fn  = f"passport_{i:06d}.png"
        crop.save(crop_dir / crop_fn)

        items = []
        for a in annots:
            cls = FIELD_CLASSES[a["label"]]
            fx1, fy1, fx2, fy2 = a["bbox"]
            rx1, ry1 = fx1 - x1, fy1 - y1
            rx2, ry2 = fx2 - x1, fy2 - y1
            items.append((cls, (rx1, ry1, rx2, ry2)))
        save_yolo_txt(
            crop_lbl / f"{crop_fn[:-4]}.txt",
            items,
            (pw, ph)
        )

    # 4) Сохранение JSON
    with open(ANNOT_PATH, 'w', encoding='utf-8') as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)
    with open(COCO_ANNOT_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"✅ {N_PASSPORTS} passports -> {OUT_IMG_DIR}")
    print(f"✅ COCO annotations saved to {COCO_ANNOT_PATH}")
    print(f"✅ YOLO datasets ready under {YOLO_ROOT}")

if __name__ == '__main__':
    main()
