# ─────────────────────────── ПЕРЕКЛЮЧАТЕЛИ ────────────────────────────
GENERATE_STAGE1 = True   # целый паспорт
GENERATE_STAGE2 = True   # внутренние поля
# ──────────────────────────────────────────────────────────────────────
if not (GENERATE_STAGE1 or GENERATE_STAGE2):
    raise RuntimeError("Оба стейджа выключены – генерировать нечего.")

# ─────────── БАЗОВЫЕ ПАРАМЕТРЫ (можно менять) ───────────
TOTAL       = 1000        # объём датасета
TRAIN_RATIO = 0.8
BASE_SEED   = 42
N_PROC      = __import__('os').cpu_count()   # процессов = ядер CPU
# ────────────────────────────────────────────────────────

import os, math, shutil, random, multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont, ImageChops

# ────────── РИСОВАЛЬНЫЕ КОНСТАНТЫ ──────────
MARGIN_X, MARGIN_Y = 200, 100
MRZ_FONT_SIZE      = 18
SERIES_FONT_SIZE   = 20
FONT_SIZE          = 18
GROUP_GAP          = 25
DEBUG              = False
BBOX_COLOR         = (0, 255, 0)   # зелёный для debug-bbox
BBOX_WIDTH         = 2
# ───────────────────────────────────────────

# ────────── ПУТИ И ДИРЕКТОРИИ ──────────
ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "templates" / "passport_template.png"
FONTS_DIR     = ROOT / "fonts"

OUT_DIR      = ROOT / "dataset_small"
OUT_IMG_DIR  = OUT_DIR / "images"
DEBUG_IMG_DIR = OUT_DIR / "debug_images"

YOLO_ROOT = ROOT / "dataset_yolo_test"
# Stage-1 (пути объявлены ВСЕГДА → NameError не случится)
ST1_IMG_T = YOLO_ROOT / "stage1" / "images" / "train"
ST1_IMG_V = YOLO_ROOT / "stage1" / "images" / "val"
ST1_LBL_T = YOLO_ROOT / "stage1" / "labels" / "train"
ST1_LBL_V = YOLO_ROOT / "stage1" / "labels" / "val"
# Stage-2
ST2_IMG_T = YOLO_ROOT / "stage2" / "images" / "train"
ST2_IMG_V = YOLO_ROOT / "stage2" / "images" / "val"
ST2_LBL_T = YOLO_ROOT / "stage2" / "labels" / "train"
ST2_LBL_V = YOLO_ROOT / "stage2" / "labels" / "val"

if DEBUG:
    DBG1_DIR = ROOT / "debug/stage1"
    DBG2_DIR = ROOT / "debug/stage2"
    for d in (DBG1_DIR, DBG2_DIR): d.mkdir(parents=True, exist_ok=True)

# создаём только нужные каталоги
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
if DEBUG:
    DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
if GENERATE_STAGE1:
    for d in (ST1_IMG_T, ST1_IMG_V, ST1_LBL_T, ST1_LBL_V):
        d.mkdir(parents=True, exist_ok=True)
if GENERATE_STAGE2:
    for d in (ST2_IMG_T, ST2_IMG_V, ST2_LBL_T, ST2_LBL_V):
        d.mkdir(parents=True, exist_ok=True)

# ────────── КООРДИНАТЫ ПОЛЕЙ И КЛАССЫ ──────────
COORDS = {
    "issuing_auth": (105,  73),
    "issue_date":   (100, 155),
    "division_code":(328, 155),

    "surname":      (258, 405),
    "name":         (258, 460),
    "patronymic":   (258, 490),
    "gender":       (198, 518),
    "dob":          (310, 517),
    "birth_place":  (220, 545),

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

# ────────── СЛОВАРИ ДЛЯ НОВЫХ ШАБЛОНОВ ──────────
AUTH_SHORT = [
    "ОВД Г. {city}",
    "УФМС {region}",
    "МФЦ {region}",
]
AUTH_MED = [
    "МВД России по {region}",
    "ГУ МВД России по г. {city}",
    "ОМВД России по {district} району г. {city}",
]
AUTH_LONG = [
    "ОТДЕЛЕНИЕМ МВД России по {district} району города {city} {region}",
    "УВМ ГУ МВД России по {region}",
    "ФКУ «ПП № 1» ГУ МВД России по г. {city} {region}",
]

BIRTH_SHORT = [
    "{city}",
    "{village}, {region}",
    "{city}, РФ",
]
BIRTH_MED = [
    "{city}, {region}",
    "С. {village}, {region}",
    "{city}, {region}, РСФСР",
]
BIRTH_LONG = [
    "{city}, {district} район, {region}, РФ",
    "Д. {village}, {district} р-н, {region}",
    "{city}, {region}, СССР",
]

# ────────── ПРОЧИЕ РЕСУРСЫ ──────────
issuing_authorities = [
    "Министерством внутренних дел Российской Федерации",
    "ГУ МВД России по Московской области",
    "УМВД России по Республике Татарстан",
    "ОТДЕЛ МВД РОССИИ ПО ЛЕНИНСКОМУ РАЙОНУ Г. САМАРЫ",
    "ОТДЕЛЕНИЕ ПОЛИЦИИ № 3 ОТДЕЛА МВД РОССИИ ПО Г. ТУЛЕ",
    "МФЦ МОСКОВСКОЙ ОБЛАСТИ",
    "ГЕНЕРАЛЬНОЕ КОНСУЛЬСТВО РОССИИ В Г. НЬЮ-ЙОРКЕ",
]
authority_cycle = iter(issuing_authorities)

RUS_FONTS = [
    FONTS_DIR / "ARIAL.TTF",
    FONTS_DIR / "times.ttf",
    FONTS_DIR / "PTC55F.ttf",
]
MRZ_FONT = FONTS_DIR / "ocr-b-regular.ttf"

fake = Faker("ru_RU")

# ────────── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (без изменений, кроме новых шаблонов) ──────────
MAPPING_REV = {       # ГОСТ-транслитерация к MRZ
    'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ё':'2','Ж':'J','З':'Z',
    'И':'I','Й':'Q','К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P','Р':'R',
    'С':'S','Т':'T','У':'U','Ф':'F','Х':'H','Ц':'C','Ч':'3','Ш':'4','Щ':'W',
    'Ъ':'X','Ы':'Y','Ь':'9','Э':'6','Ю':'7','Я':'8',' ':'<'
}
def transliterate_gost(t:str)->str:
    return ''.join(MAPPING_REV.get(c.upper(),'<') for c in t.upper())

def trim_whitespace(im:Image.Image)->Image.Image:
    bg = im.getpixel((0,0))
    diff = ImageChops.difference(im, Image.new(im.mode, im.size, bg))
    return im.crop(diff.getbbox()) if diff.getbbox() else im

def wrap_text(text, font, max_width, max_lines=3):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line+" "+w).strip()
        if font.getlength(test) <= max_width:
            line = test
        else:
            lines.append(line); line = w
            if len(lines) == max_lines-1:
                lines.append(" ".join(words[words.index(w):])); return lines
    if line: lines.append(line)
    return lines[:max_lines]

def save_yolo_txt(path:Path, items, img_size):
    W,H = img_size; lines=[]
    for cls,(x1,y1,x2,y2) in items:
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
        if x2<=x1 or y2<=y1: continue
        xc=((x1+x2)*.5)/W; yc=((y1+y2)*.5)/H
        w=(x2-x1)/W; h=(y2-y1)/H
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))

def random_date(start_year=1950, end_year=None):
    end_year = end_year or datetime.now().year
    s=datetime(start_year,1,1); e=datetime(end_year,12,31)
    return (s+timedelta(days=random.randint(0,(e-s).days))).strftime("%d.%m.%Y")

def random_division_code(): return f"{random.randint(100,999)}-{random.randint(100,999)}"
def random_passport_series_number(): return random.randint(10,99),random.randint(10,99),random.randint(100000,999999)
def random_fullname(sex):
    if sex=="МУЖ":
        return (fake.last_name_male().upper(),fake.first_name_male().upper(),fake.middle_name_male().upper())
    return (fake.last_name_female().upper(),fake.first_name_female().upper(),fake.middle_name_female().upper())

# ─── генерация текстов по количеству строк ───
def _gen_field_text(rng,font,max_w,target,variants):
    buckets={1:variants[0],2:variants[1],3:variants[2]}
    while True:
        tmpl=rng.choice(buckets[target])
        txt=tmpl.format(
            city=fake.city().upper(),
            region=rng.choice(["МОСКОВСКАЯ ОБЛАСТЬ","САНКТ-ПЕТЕРБУРГ"]),
            district=rng.choice(["ЛЕНИНСКОМУ","ЦЕНТРАЛЬНОМУ"]),
            village=fake.city().upper()
        )
        if len(wrap_text(txt,font,max_w,3))==target:
            return txt

def random_issuing_auth(rng,font,max_w):
    return _gen_field_text(rng,font,max_w,rng.choice([1,2,3]),
                           (AUTH_SHORT,AUTH_MED,AUTH_LONG))
def random_birth_place(rng,font,max_w):
    return _gen_field_text(rng,font,max_w,rng.choice([1,2,3]),
                           (BIRTH_SHORT,BIRTH_MED,BIRTH_LONG))

def mrz_checksum(data:str)->str:
    w=[7,3,1]; v={chr(i):i-55 for i in range(65,91)}|{str(i):i for i in range(10)}|{'<':0}
    return str(sum(v.get(c,0)*w[i%3] for i,c in enumerate(data))%10)

def gen_mrz(d:dict):
    surname=transliterate_gost(d['surname']).replace(' ','<')
    given  =transliterate_gost(d['given']).replace(' ','<')
    patron =transliterate_gost(d['patronymic']).replace(' ','<')
    name_field=''.join(ch if 'A'<=ch<='Z' else '<' for ch in f"{surname}<<{given}<{patron}")
    line1=f"P<{'RUS'}{name_field}".ljust(44,'<')

    doc_num=d['series'][:3]+d['number']; c1=mrz_checksum(doc_num)
    bdate=d['dob']; c2=mrz_checksum(bdate)
    exp=d['expiry']; c3=mrz_checksum(exp)
    pers=(d['series'][3]+d['issue']+d['division_code'].replace('-','')).ljust(14,'<'); c4=mrz_checksum(pers)
    c5=mrz_checksum(doc_num+c1+bdate+c2+d['gender']+exp+c3+pers+c4)
    line2=f"{doc_num}{c1}{d['nationality']}{bdate}{c2}{d['gender']}{exp}{c3}{pers}{c4}{c5}".ljust(44,'<')
    return line1,line2

def insert_photo_noise(img:Image.Image,box):
    x1,y1,x2,y2=box; h,w=y2-y1,x2-x1
    noise=np.random.randint(0,256,(h,w,3),dtype=np.uint8)
    img.paste(Image.fromarray(noise,'RGB'),(x1,y1))

# ────────── PassportGenerator (логика рисования не тронута, + новые поля) ──────────
class PassportGenerator:
    def __init__(self):
        tpl=Image.open(TEMPLATE_PATH).convert('RGB')
        self.template=trim_whitespace(tpl)
        self.fonts=[ImageFont.truetype(str(f),FONT_SIZE) for f in RUS_FONTS]
        self.mrz_font=ImageFont.truetype(str(MRZ_FONT),MRZ_FONT_SIZE)
        self.line_h=self.fonts[0].getbbox("Mg")[3]+4

    def _draw_text(self,draw,pos,text,font,jitter=(0,0)):
        x,y=pos; x+=random.randint(-jitter[0],jitter[0]); y+=random.randint(-jitter[1],jitter[1])
        draw.text((x,y),text,font=font,fill='black')
        w,h=font.getbbox(text)[2:]
        return [x,y,x+w,y+h]

    def _draw_wrapped(self,draw,pos,text,font,max_w,max_lines):
        x0,y0=pos; bbs=[]
        for i,line in enumerate(wrap_text(text,font,max_w,max_lines)):
            bbs.append(self._draw_text(draw,(x0,y0+i*self.line_h),line,font,(15,5)))
        xs=[v for b in bbs for v in (b[0],b[2])]; ys=[v for b in bbs for v in (b[1],b[3])]
        return [min(xs),min(ys),max(xs),max(ys)]

    def generate_one(self,idx,include_mrz=True,rng=None):
        rng=rng or random
        img=self.template.copy(); draw=ImageDraw.Draw(img); ann=[]
        dob_txt=random_date(1960,2003); dob_mrz=datetime.strptime(dob_txt,"%d.%m.%Y").strftime("%y%m%d")
        exp_mrz=(datetime.now()+timedelta(days=365*10)).strftime("%y%m%d")
        p1,p2,pnum=random_passport_series_number()
        sex=rng.choice(["МУЖ","ЖЕН"]); gender_mrz="M" if sex=="МУЖ" else "F"
        surname,name,patr=random_fullname(sex)

        font=rng.choice(self.fonts)
        for key,val in (("surname",surname),("name",name),("patronymic",patr)):
            ann.append({"label":key,"bbox":self._draw_text(draw,COORDS[key],val,font,(15,5))})
        ann.append({"label":"gender","bbox":self._draw_text(draw,COORDS["gender"],sex,font,(15,5))})
        ann.append({"label":"dob","bbox":self._draw_text(draw,COORDS["dob"],dob_txt,font,(15,5))})

        max_w_birth = img.width-COORDS["birth_place"][0]-20
        max_w_auth  = img.width-COORDS["issuing_auth"][0]-20
        birth_place  = random_birth_place(rng,font,max_w_birth)
        issuing_auth = random_issuing_auth(rng,font,max_w_auth)

        ann.append({"label":"birth_place",
                    "bbox":self._draw_wrapped(draw,COORDS["birth_place"],birth_place,font,max_w_birth,3)})
        ann.append({"label":"issuing_auth",
                    "bbox":self._draw_wrapped(draw,COORDS["issuing_auth"],issuing_auth,font,max_w_auth,3)})

        issue_date=random_date(2015); division_code=random_division_code()
        ann.append({"label":"issue_date","bbox":self._draw_text(draw,COORDS["issue_date"],issue_date,font,(15,5))})
        ann.append({"label":"division_code","bbox":self._draw_text(draw,COORDS["division_code"],division_code,font,(15,5))})

        if include_mrz:
            sfont=ImageFont.truetype(str(rng.choice(RUS_FONTS)),SERIES_FONT_SIZE)
            parts=[f"{p1:02d}",f"{p2:02d}",f"{pnum:06d}"]; ws=[int(sfont.getlength(p)) for p in parts]
            sn=Image.new('RGBA',(sum(ws)+GROUP_GAP*(len(parts)-1),sfont.getbbox(parts[0])[3]),(0,0,0,0))
            sn_draw=ImageDraw.Draw(sn); x=0
            for i,p in enumerate(parts):
                sn_draw.text((x,0),p,font=sfont,fill=(192,0,0,255))
                x+=ws[i]+GROUP_GAP
            sn_rot=sn.rotate(270,expand=True)
            for key in ("series_number_top","series_number_bottom"):
                x0,y0=COORDS[key]; img.paste(sn_rot,(x0,y0),sn_rot)
                ann.append({"label":key,"bbox":[x0,y0,x0+sn_rot.width,y0+sn_rot.height]})

            issue_mrz=datetime.strptime(issue_date,"%d.%m.%Y").strftime("%y%m%d")
            mrz1,mrz2=gen_mrz({
                "surname":surname,"given":name,"patronymic":patr,
                "series":f"{p1:02d}{p2:02d}","number":f"{pnum:06d}",
                "dob":dob_mrz,"gender":gender_mrz,"expiry":exp_mrz,
                "issue":issue_mrz,"division_code":division_code.replace('-',''),"nationality":"RUS"
            })
            ann.append({"label":"mrz1","bbox":self._draw_text(draw,COORDS["mrz1"],mrz1,self.mrz_font)})
            ann.append({"label":"mrz2","bbox":self._draw_text(draw,COORDS["mrz2"],mrz2,self.mrz_font)})

            if DEBUG:
                bb1,bb2=ann[-2]["bbox"],ann[-1]["bbox"]
                img.crop((bb1[0],bb1[1],bb2[2],bb2[3])).save(DEBUG_IMG_DIR/f"{idx:06d}_mrz.png")

        insert_photo_noise(img,COORDS["photo_box"])

        cw,ch=img.width+MARGIN_X, img.height+MARGIN_Y
        canvas=Image.new('RGB',(cw,ch),'white')
        dx,dy=rng.randint(0,MARGIN_X),rng.randint(0,MARGIN_Y)
        canvas.paste(img,(dx,dy))
        for a in ann: a['bbox']=[a['bbox'][0]+dx,a['bbox'][1]+dy,a['bbox'][2]+dx,a['bbox'][3]+dy]

        fname=f"passport_{idx:06d}.png"
        canvas.save(OUT_IMG_DIR/fname)
        passport_bbox=[dx,dy,dx+img.width,dy+img.height]
        return fname, ann, passport_bbox

# ────────── WORKER/POOL ──────────
def worker_init():
    global GEN
    GEN = PassportGenerator()

def worker(chunk_start, chunk_len, train_cut):
    rng = random.Random(BASE_SEED + chunk_start)
    res = []
    for idx in range(chunk_start, chunk_start + chunk_len):
        is_train = idx < train_cut
        fn, ann, pbbox = GEN.generate_one(
            idx, include_mrz=rng.choice([True, False]), rng=rng
        )
        res.append((fn, is_train, ann, pbbox))
    return res

def _wrap(args): return worker(*args)

# ────────── MAIN ──────────
def main():
    chunk = max(1, math.ceil(TOTAL / (N_PROC * 4)))
    train_cut = math.floor(TOTAL * TRAIN_RATIO)
    tasks = [(start, min(chunk, TOTAL - start), train_cut)
             for start in range(0, TOTAL, chunk)]

    with mp.Pool(processes=N_PROC, initializer=worker_init) as pool:
        for batch in pool.imap_unordered(_wrap, tasks, chunksize=1):
            for fn, is_train, ann, pbbox in batch:
                img_path = OUT_IMG_DIR / fn
                img = Image.open(img_path)
                W, H = img.size

                # ---------- stage-1 ----------
                if GENERATE_STAGE1:
                    dst_img = ST1_IMG_T if is_train else ST1_IMG_V
                    dst_lbl = ST1_LBL_T if is_train else ST1_LBL_V
                    shutil.move(img_path, dst_img / fn)
                    save_yolo_txt(dst_lbl / f"{fn[:-4]}.txt",
                                  [(0, tuple(pbbox))], (W, H))

                    if DEBUG:
                        dbg = img.copy()
                        draw = ImageDraw.Draw(dbg)
                        draw.rectangle(pbbox, outline=BBOX_COLOR, width=BBOX_WIDTH)
                        dbg.save(DBG1_DIR / fn)

                # ---------- stage-2 ----------
                if GENERATE_STAGE2:
                    dst_img = ST2_IMG_T if is_train else ST2_IMG_V
                    dst_lbl = ST2_LBL_T if is_train else ST2_LBL_V
                    x1, y1, x2, y2 = pbbox
                    crop = img.crop((x1, y1, x2, y2))
                    crop.save(dst_img / fn)

                    items = []
                    for a in ann:
                        cls = FIELD_CLASSES[a["label"]]
                        fx1, fy1, fx2, fy2 = a["bbox"]
                        items.append((cls, (fx1 - x1, fy1 - y1, fx2 - x1, fy2 - y1)))
                    save_yolo_txt(dst_lbl / f"{fn[:-4]}.txt", items, crop.size)

                    if DEBUG:
                        dbg = crop.copy()
                        draw = ImageDraw.Draw(dbg)
                        for _, (bx1, by1, bx2, by2) in items:
                            draw.rectangle([bx1, by1, bx2, by2],
                                           outline=BBOX_COLOR, width=BBOX_WIDTH)
                        dbg.save(DBG2_DIR / fn)

                # если stage-1 выключен — удаляем исходный скан
                if (not GENERATE_STAGE1) and img_path.exists():
                    img_path.unlink()

    print("✅ Датасет готов!")
    if GENERATE_STAGE1: print(f"   stage-1 → {ST1_IMG_T.parent}")
    if GENERATE_STAGE2: print(f"   stage-2 → {ST2_IMG_T.parent}")
    if DEBUG:           print(f"   debug   → {DBG1_DIR.parent}")

# ────────── entry ──────────
if __name__ == "__main__":
    main()
