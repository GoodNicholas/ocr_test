import os
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

# Параметры
MARGIN_X = 200
MARGIN_Y = 100
MRZ_FONT_SIZE = 18
N_PASSPORTS = 100
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
    char_values = {chr(i): i - 55 for i in range(65, 91)}  # A=10 ... Z=35
    char_values.update({str(i): i for i in range(10)})
    char_values['<'] = 0

    total = sum(char_values.get(c, 0) * weights[i % 3] for i, c in enumerate(data))
    return str(total % 10)


def gen_mrz(data: dict) -> tuple[str, str]:
    # 1-я строка
    surname = transliterate_gost(data['surname']).replace(' ', '<')
    given = transliterate_gost(data['given']).replace(' ', '<')
    patronymic = transliterate_gost(data['patronymic']).replace(' ', '<')
    name_field = f"{surname}<<{given}<{patronymic}".replace('<<', '<<')[:39]
    line1 = f"PNRUS{name_field}".ljust(44, '<')

    # 2-я строка
    series = data['series'][:3]
    last_digit = data['series'][3]
    passport_number = data['number']
    pd_field = series + passport_number
    pd_check = mrz_checksum(pd_field)

    birth = data['dob']
    birth_check = mrz_checksum(birth)

    expiry = data['expiry']
    expiry_check = mrz_checksum(expiry)

    opt_data = f"{last_digit}{data['issue']}{data['division_code']}".ljust(14, '<')
    opt_check = mrz_checksum(opt_data)

    line2 = f"{pd_field}{pd_check}{data['nationality']}{birth}{birth_check}{data['gender']}{expiry}{expiry_check}{opt_data}{opt_check}"
    line2 = line2[:43].ljust(44, '<')  # 44-й символ – можно добавить финальную контрольную цифру
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
        return fname, annotation


# ----------------------------- MAIN ----------------------------------
def main():
    OUT_IMG_DIR.mkdir(exist_ok=True, parents=True)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
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

    photo_box = (50, 100, 170, 270)  # Задаём координаты фото
    for i in range(N_PASSPORTS):
        # С вероятностью 50% генерируем с MRZ или без
        include_mrz = random.choice([True, False])
        fn, ann = gen.generate_one(i, offset=None, photo_box=COORDS["photo_box"], include_mrz=include_mrz, debug=DEBUG)
        # Оригинальные размеры изображения
        img = Image.open(OUT_IMG_DIR / fn)
        width, height = img.size

        # Сохраняем обычную аннотацию для каждого изображения
        ds.append({
            "image_id": fn,
            "mrz": ann['mrz'],
            "photo": ann['photo'],
            "text": ann['text']
        })

        coco_data["images"].append({
            "id": i + 1,
            "file_name": fn,
            "width": width,
            "height": height
        })

        # Только если MRZ есть, добавляем в COCO
        if ann['mrz']['exists']:
            coco_data["annotations"].append({
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": ann['mrz']['coordinates'],
                "area": (ann['mrz']['coordinates'][2] - ann['mrz']['coordinates'][0]) *
                        (ann['mrz']['coordinates'][3] - ann['mrz']['coordinates'][1]),
                "iscrowd": 0,
                "segmentation": []
            })

    # Сохранение обычных аннотаций
    with open(ANNOT_PATH, 'w', encoding='utf-8') as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)

    # Сохранение COCO аннотаций
    with open(COCO_ANNOT_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"✅ {N_PASSPORTS} passports -> {OUT_IMG_DIR}")
    print(f"✅ COCO annotations saved to {COCO_ANNOT_PATH}")
    print(f"✅ Regular annotations saved to {ANNOT_PATH}")


if __name__ == '__main__':
    main()
