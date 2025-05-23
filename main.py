#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор сканов российских паспортов
для двух-ступенчатого YOLO-датасета (stage 1 – весь паспорт,
stage 2 – поля внутри паспорта).

✓ класс PassportGenerator и все вспомогательные функции НЕ менялись;
✓ файл сам разбивает данные на train / val с заданным соотношением
  (TRAIN_RATIO) – для stage 1 и stage 2 одновременно.
"""

import random
import shutil
import string
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
from faker import Faker
import itertools

# ------------------------------------------------------------------------------
# Константы-параметры
# ------------------------------------------------------------------------------
N_PASSPORTS   = 1000      # сколько паспортов сгенерировать
TRAIN_RATIO   = 0.8       # доля train (0.8 → 80 % train, 20 % val)

MARGIN_X, MARGIN_Y = 200, 100
MRZ_FONT_SIZE      = 18
SERIES_FONT_SIZE   = 20
FONT_SIZE          = 18
GROUP_GAP          = 25

DEBUG = False             # сохранять или нет debug-PNG с MRZ

# ------------------------------------------------------------------------------
# Пути
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

TEMPLATE_PATH = ROOT / "templates" / "passport_template.png"
FONTS_DIR     = ROOT / "fonts"

OUT_DIR     = ROOT / "dataset_small"      # сюда кладём сгенерированные сканы
OUT_IMG_DIR = OUT_DIR / "images"
DEBUG_IMG_DIR = OUT_DIR / "debug_images"  # остаётся, код к нему обращается

YOLO_ROOT   = ROOT / "dataset_yolo"

ST1_IMG_T = YOLO_ROOT / "stage1" / "images" / "train"
ST1_IMG_V = YOLO_ROOT / "stage1" / "images" / "val"
ST1_LBL_T = YOLO_ROOT / "stage1" / "labels" / "train"
ST1_LBL_V = YOLO_ROOT / "stage1" / "labels" / "val"

ST2_IMG_T = YOLO_ROOT / "stage2" / "images" / "train"
ST2_IMG_V = YOLO_ROOT / "stage2" / "images" / "val"
ST2_LBL_T = YOLO_ROOT / "stage2" / "labels" / "train"
ST2_LBL_V = YOLO_ROOT / "stage2" / "labels" / "val"

for d in (OUT_IMG_DIR, DEBUG_IMG_DIR,
          ST1_IMG_T, ST1_IMG_V, ST1_LBL_T, ST1_LBL_V,
          ST2_IMG_T, ST2_IMG_V, ST2_LBL_T, ST2_LBL_V):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Координаты полей, классы, вспомогательные списки
# ------------------------------------------------------------------------------
COORDS = {
    "issuing_auth": (105, 73),
    "issue_date":   (100, 155),
    "division_code":(328, 155),

    "surname":   (258, 405),
    "name":      (258, 460),
    "patronymic":(258, 490),
    "gender":    (198, 518),
    "dob":       (310, 517),
    "birth_place":(220, 545),

    "series_number_top":    (490, 109),
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
NC_STAGE2 = max(FIELD_CLASSES.values()) + 1

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

RUS_FONTS = [
    FONTS_DIR / "ARIAL.TTF",
    FONTS_DIR / "times.ttf",
    FONTS_DIR / "PTC55F.ttf",
]
MRZ_FONT = FONTS_DIR / "ocr-b-regular.ttf"

fake = Faker("ru_RU")
random.seed(42)

# ------------------------------------------------------------------------------
# ------------------ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (без изменений) -------------------
# ------------------------------------------------------------------------------
MAPPING_REV = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E',
    'Ё': '2', 'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Q', 'К': 'K',
    'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R',
    'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C',
    'Ч': '3', 'Ш': '4', 'Щ': 'W', 'Ъ': 'X', 'Ы': 'Y', 'Ь': '9',
    'Э': '6', 'Ю': '7', 'Я': '8', ' ': '<'
}

def transliterate_gost(text): return ''.join(MAPPING_REV.get(c.upper(), '<') for c in text.upper())
def get_next_issuing_authority(): return next(authority_iterator)

def trim_whitespace(im):
    bg = im.getpixel((0,0))
    diff = ImageChops.difference(im, Image.new(im.mode, im.size, bg))
    return im.crop(diff.getbbox()) if diff.getbbox() else im

def save_yolo_txt(path, items, img_size):
    W, H = img_size
    lines = []
    for cls_id, (x1,y1,x2,y2) in items:
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(W,x2), min(H,y2)
        if x2<=x1 or y2<=y1: continue
        xc = ((x1+x2)*.5)/W;  yc = ((y1+y2)*.5)/H
        w  = (x2-x1)/W;      h  = (y2-y1)/H
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))

def wrap_text(text, font, max_width, max_lines=3):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line+" "+w).strip()
        if font.getlength(test) <= max_width:
            line = test
        else:
            lines.append(line); line = w
            if len(lines)==max_lines-1:
                lines.append(" ".join(words[words.index(w):])); return lines
    if line: lines.append(line)
    return lines[:max_lines]

def random_date(start_year=1950, end_year=None):
    end_year = end_year or datetime.now().year
    s = datetime(start_year,1,1); e = datetime(end_year,12,31)
    return (s + timedelta(days=random.randint(0,(e-s).days))).strftime("%d.%m.%Y")
def random_division_code(): return f"{random.randint(100,999)}-{random.randint(100,999)}"
def random_passport_series_number(): return random.randint(10,99), random.randint(10,99), random.randint(100000,999999)
def random_fullname(gender):
    if gender=="МУЖ": return (fake.last_name_male().upper(), fake.first_name_male().upper(), fake.middle_name_male().upper())
    return (fake.last_name_female().upper(), fake.first_name_female().upper(), fake.middle_name_female().upper())
def random_birth_place():
    city, region, village = fake.city().upper(), random.choice(["МОСКОВСКАЯ ОБЛАСТЬ","САНКТ-ПЕТЕРБУРГ"]), fake.city().upper()
    template = random.choice(["{city}", "{city}, {region}", "С. {village}, {region}", "{city}, {region}, РФ"])
    return template.format(city=city, village=village, region=region)
def random_issuing_auth():
    org=get_next_issuing_authority(); city=fake.city().upper()
    district=random.choice(["ЛЕНИНСКОМУ","ЦЕНТРАЛЬНОМУ"]); region=random.choice(["МОСКОВСКАЯ ОБЛАСТЬ","САНКТ-ПЕТЕРБУРГ"])
    return random.choice(["{org} по г. {city}", "{org} по {district} району г. {city}", "{org} по {region}"]).format(
        org=org, city=city, district=district, region=region)

def mrz_checksum(data):
    weights=[7,3,1]; vals={chr(i):i-55 for i in range(65,91)}|{str(i):i for i in range(10)}|{'<':0}
    return str(sum(vals.get(c,0)*weights[i%3] for i,c in enumerate(data))%10)

def gen_mrz(d):
    surname=transliterate_gost(d['surname']).replace(' ','<')
    given=transliterate_gost(d['given']).replace(' ','<')
    patron=transliterate_gost(d['patronymic']).replace(' ','<')
    name_field=''.join(ch if 'A'<=ch<='Z' else '<' for ch in f"{surname}<<{given}<{patron}")
    line1=f"P<{'RUS'}{name_field}".ljust(44,'<')

    ser3=d['series'][:3]; num6=d['number']; doc_num=ser3+num6; c1=mrz_checksum(doc_num)
    bdate=d['dob']; c2=mrz_checksum(bdate)
    exp=d['expiry']; c3=mrz_checksum(exp)
    pers=(d['series'][3]+d['issue']+d['division_code'].replace('-','')).ljust(14,'<'); c4=mrz_checksum(pers)
    c5=mrz_checksum(doc_num+c1+bdate+c2+d['gender']+exp+c3+pers+c4)
    line2=f"{doc_num}{c1}{d['nationality']}{bdate}{c2}{d['gender']}{exp}{c3}{pers}{c4}{c5}".ljust(44,'<')
    return line1,line2

def insert_photo_noise(canvas, box):
    x1,y1,x2,y2 = box; w,h = x2-x1, y2-y1
    noise=(np.random.rand(h,w,3)*255).astype(np.uint8)
    canvas.paste(Image.fromarray(noise,'RGB'),(x1,y1))

# ------------------------------------------------------------------------------
# --------------------------- PassportGenerator --------------------------------
# ------------------------------------------------------------------------------
class PassportGenerator:
    def __init__(self):
        tpl=Image.open(TEMPLATE_PATH).convert('RGB'); self.template=trim_whitespace(tpl)
        self.fonts=[ImageFont.truetype(str(f),FONT_SIZE) for f in RUS_FONTS]
        self.mrz_font=ImageFont.truetype(str(MRZ_FONT),MRZ_FONT_SIZE)
        self.line_h=self.fonts[0].getbbox("Mg")[3]+4

    def _draw_text(self,draw,pos,text,font,jitter=(0,0)):
        x,y=pos; x+=random.randint(-jitter[0],jitter[0]); y+=random.randint(-jitter[1],jitter[1])
        draw.text((x,y),text,font=font,fill='black'); w,h=font.getbbox(text)[2:]
        return [x,y,x+w,y+h]

    def _draw_wrapped(self,draw,pos,text,font,max_width,max_lines):
        x0,y0=pos; bbs=[]
        for i,line in enumerate(wrap_text(text,font,max_width,max_lines)):
            bbs.append(self._draw_text(draw,(x0,y0+i*self.line_h),line,font,jitter=(15,5)))
        xs=[v for bb in bbs for v in (bb[0],bb[2])]; ys=[v for bb in bbs for v in (bb[1],bb[3])]
        return [min(xs),min(ys),max(xs),max(ys)]

    def generate_one(self,idx,offset=None,photo_box=None,include_mrz=True,debug=False):
        img=self.template.copy(); draw=ImageDraw.Draw(img); annots=[]
        dob_txt=random_date(1960,2003); dob_mrz=datetime.strptime(dob_txt,"%d.%m.%Y").strftime("%y%m%d")
        exp_mrz=(datetime.now()+timedelta(days=365*10)).strftime("%y%m%d")
        p1,p2,pnum=random_passport_series_number(); rus_gender=random.choice(["МУЖ","ЖЕН"])
        gender_mrz="M" if rus_gender=="МУЖ" else "F"
        surname,name,patr=random_fullname(rus_gender); birth_place=random_birth_place()
        issue_date=random_date(2015); division_code=random_division_code(); issuing_auth=random_issuing_auth()

        font=random.choice(self.fonts)
        for key,text in [("surname",surname),("name",name),("patronymic",patr)]:
            annots.append({"label":key,"bbox":self._draw_text(draw,COORDS[key],text,font,(15,5)),"text":text})
        annots.append({"label":"gender","bbox":self._draw_text(draw,COORDS["gender"],rus_gender,font,(15,5)),"text":rus_gender})
        annots.append({"label":"dob","bbox":self._draw_text(draw,COORDS["dob"],dob_txt,font,(15,5)),"text":dob_txt})
        annots.append({"label":"birth_place","bbox":self._draw_wrapped(draw,COORDS["birth_place"],birth_place,font,img.width-COORDS["birth_place"][0]-20,3),"text":birth_place})
        annots.append({"label":"issuing_auth","bbox":self._draw_wrapped(draw,COORDS["issuing_auth"],issuing_auth,font,img.width-COORDS["issuing_auth"][0]-20,2),"text":issuing_auth})
        annots.append({"label":"issue_date","bbox":self._draw_text(draw,COORDS["issue_date"],issue_date,font,(15,5)),"text":issue_date})
        annots.append({"label":"division_code","bbox":self._draw_text(draw,COORDS["division_code"],division_code,font,(15,5)),"text":division_code})

        if include_mrz:
            sfont=ImageFont.truetype(str(random.choice(RUS_FONTS)),SERIES_FONT_SIZE)
            parts=[f"{p1:02d}",f"{p2:02d}",f"{pnum:06d}"]; ws=[int(sfont.getlength(p)) for p in parts]
            sn=Image.new('RGBA',(sum(ws)+GROUP_GAP*(len(parts)-1),sfont.getbbox(parts[0])[3]),(0,0,0,0))
            sn_draw=ImageDraw.Draw(sn); x=0
            for i,p in enumerate(parts):
                sn_draw.text((x,0),p,font=sfont,fill=(192,0,0,255)); x+=ws[i]+GROUP_GAP
            sn_rot=sn.rotate(270,expand=True)
            for key in ("series_number_top","series_number_bottom"):
                x0,y0=COORDS[key]; img.paste(sn_rot,(x0,y0),sn_rot)
                annots.append({"label":key,"bbox":[x0,y0,x0+sn_rot.width,y0+sn_rot.height],"text":" ".join(parts)})
            issue_mrz=datetime.strptime(issue_date,"%d.%m.%Y").strftime("%y%m%d")
            mrz1,mrz2=gen_mrz({"surname":surname,"given":name,"patronymic":patr,
                               "series":f"{p1:02d}{p2:02d}","number":f"{pnum:06d}",
                               "dob":dob_mrz,"gender":gender_mrz,"expiry":exp_mrz,
                               "issue":issue_mrz,"division_code":division_code.replace('-',''),
                               "nationality":"RUS"})
            bb1=self._draw_text(draw,COORDS["mrz1"],mrz1,self.mrz_font)
            annots.append({"label":"mrz1","bbox":bb1,"text":mrz1})
            bb2=self._draw_text(draw,COORDS["mrz2"],mrz2,self.mrz_font)
            annots.append({"label":"mrz2","bbox":bb2,"text":mrz2})
            if debug:
                mrz_box=(bb1[0],bb1[1],bb2[2],bb2[3]); DEBUG_IMG_DIR.mkdir(exist_ok=True,parents=True)
                img.crop(mrz_box).save(DEBUG_IMG_DIR/f"{idx:06d}_mrz.png")

        if photo_box: insert_photo_noise(img,photo_box)

        cw,ch=img.width+MARGIN_X,img.height+MARGIN_Y
        canvas=Image.new('RGB',(cw,ch),'white')
        dx,dy=offset if offset is not None else (random.randint(0,MARGIN_X),random.randint(0,MARGIN_Y))
        canvas.paste(img,(dx,dy))
        for a in annots:
            a['bbox']=[a['bbox'][0]+dx,a['bbox'][1]+dy,a['bbox'][2]+dx,a['bbox'][3]+dy]

        fname=f"passport_{idx:06d}.png"
        canvas.save(OUT_IMG_DIR/fname)
        passport_bbox=[dx,dy,dx+img.width,dy+img.height]
        return fname, annots, passport_bbox

# ------------------------------------------------------------------------------
# -------------------------------- MAIN ----------------------------------------
# ------------------------------------------------------------------------------
def main():
    gen=PassportGenerator()
    for i in range(N_PASSPORTS):
        include_mrz=random.choice([True,False])   # хотим/не хотим MRZ-блок
        fn, annots, pbbox = gen.generate_one(
            i, offset=None, photo_box=COORDS["photo_box"],
            include_mrz=include_mrz, debug=DEBUG
        )

        img=Image.open(OUT_IMG_DIR/fn); W,H=img.size
        is_train=random.random() < TRAIN_RATIO

        # ---------- Stage 1 ----------
        scan_img_dir = ST1_IMG_T if is_train else ST1_IMG_V
        scan_lbl_dir = ST1_LBL_T if is_train else ST1_LBL_V
        shutil.copy(OUT_IMG_DIR/fn, scan_img_dir/fn)
        save_yolo_txt(scan_lbl_dir/f"{fn[:-4]}.txt", [(0,tuple(pbbox))], (W,H))

        # ---------- Stage 2 ----------
        crop_img_dir = ST2_IMG_T if is_train else ST2_IMG_V
        crop_lbl_dir = ST2_LBL_T if is_train else ST2_LBL_V
        x1,y1,x2,y2 = pbbox
        crop = img.crop((x1,y1,x2,y2))
        crop_fn = f"passport_{i:06d}.png"
        crop.save(crop_img_dir/crop_fn)
        items=[]
        for a in annots:
            cls = FIELD_CLASSES[a["label"]]
            fx1,fy1,fx2,fy2 = a["bbox"]
            items.append((cls,(fx1-x1,fy1-y1,fx2-x1,fy2-y1)))
        save_yolo_txt(crop_lbl_dir/f"{crop_fn[:-4]}.txt", items, crop.size)

    print("✅ Датасет готов!")
    print(f"   Stage 1  → {ST1_IMG_T.parent}")
    print(f"   Stage 2  → {ST2_IMG_T.parent}")

if __name__=="__main__":
    main()
