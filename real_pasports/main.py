#!/usr/bin/env python3
"""
annotator.py   –  интерактивная разметка паспортов прямоугольниками.
• Prev / Next  (← →) — навигация.
• ЛКМ+drag             — создать бокс.
• Клик по боксу        — выбрать.
• Delete               — удалить (с подтверждением).
• Set class            — назначить класс выбранному боксу.
• Finish               — выйти; предложит слить все .txt в один.

Аннотации пишутся в подпапку   <passports_dir>_annotations   (YOLO-v5).

Запуск:
    python annotator.py /path/to/passports
Если путь не указан – берётся «passports» в текущей директории.
"""

import sys, os, glob
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

CLASSES = [
    "surname", "name", "patronymic", "gender", "dob", "birth_place",
    "issuing_auth", "issue_date", "division_code", "series_number", "mrz"
]
MIN_BOX = 5          # px – меньше не сохраняем
RESIZE_DELAY = 150   # ms – debounce окна

class Annotator(tk.Tk):
    def __init__(self, img_dir: Path):
        super().__init__()
        self.title("Passport Annotator")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # ---------- paths ----------
        self.img_dir = img_dir
        self.ann_dir = img_dir.parent / f"{img_dir.name}_annotations"
        self.ann_dir.mkdir(exist_ok=True)
        self.img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")))
        if not self.img_paths:
            messagebox.showerror("Нет картинок", f"В папке {img_dir} нет JPG-файлов.")
            self.destroy(); return
        self.index = 0

        # ---------- UI ----------
        self._build_ui()
        self.bind("<Left>",  lambda e: self.prev())
        self.bind("<Right>", lambda e: self.next())
        self.bind("<Configure>", self._on_resize)

        # ---------- state ----------
        self.boxes, self.selected = [], None
        self.start_pt, self.scale = None, 1.0
        self._resize_after = None
        self.img_orig = None          # будет позже

        # После того как окно «проявится», загрузим первое изображение
        self.after(50, self._load_image)

    # ───────────────────── UI ─────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Canvas
        self.canvas = tk.Canvas(self, bg="gray20", cursor="tcross")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Right panel
        frm = ttk.Frame(self, padding=5)
        frm.grid(row=0, column=1, sticky="ns")
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Classes").grid(row=0, column=0, pady=(0,5))
        self.cls_list = tk.Listbox(frm, height=len(CLASSES), exportselection=False)
        for i, name in enumerate(CLASSES): self.cls_list.insert("end", f"{i}: {name}")
        self.cls_list.select_set(0)
        self.cls_list.grid(row=1, column=0, sticky="n")
        self.cls_list.bind("<<ListboxSelect>>", lambda e: None)

        ttk.Separator(frm, orient="horizontal").grid(row=2, column=0, pady=10, sticky="ew")

        ttk.Button(frm, text="← Prev", command=self.prev).grid(row=3, column=0, sticky="ew", pady=2)
        ttk.Button(frm, text="Next →", command=self.next).grid(row=4, column=0, sticky="ew", pady=2)

        ttk.Button(frm, text="Set class", command=self.set_class).grid(row=5, column=0, sticky="ew", pady=(15,2))
        ttk.Button(frm, text="Delete",    command=self.delete_box).grid(row=6, column=0, sticky="ew", pady=2)

        ttk.Separator(frm, orient="horizontal").grid(row=7, column=0, pady=10, sticky="ew")
        ttk.Button(frm, text="Finish", command=self.finish).grid(row=8, column=0, sticky="ew", pady=2)

    # ───────────────────── event handlers ─────────────────────
    def _on_click(self, event):
        x, y = self._to_orig(event.x, event.y)
        for i,(x1,y1,x2,y2,_) in enumerate(self.boxes):
            if x1<=x<=x2 and y1<=y<=y2:
                self.selected = i; self._redraw_boxes(); return
        self.start_pt = (x, y); self.selected = None

    def _on_drag(self, event):
        if self.start_pt:
            self._redraw_boxes(temp_box=(*self.start_pt, *self._to_orig(event.x,event.y)))

    def _on_release(self, event):
        if not self.start_pt: return
        x1,y1 = self.start_pt; x2,y2 = self._to_orig(event.x,event.y)
        self.start_pt = None
        if abs(x2-x1)<MIN_BOX or abs(y2-y1)<MIN_BOX: self._redraw_boxes(); return
        if x2<x1: x1,x2 = x2,x1
        if y2<y1: y1,y2 = y2,y1
        cls = self.cls_list.curselection()[0]
        self.boxes.append((x1,y1,x2,y2,cls)); self.selected=len(self.boxes)-1
        self._redraw_boxes()

    def _on_resize(self, _e):
        if not self.img_orig: return
        if self._resize_after: self.after_cancel(self._resize_after)
        self._resize_after = self.after(RESIZE_DELAY, self._refresh_scale)

    # ───────────────────── nav ─────────────────────
    def prev(self):  self._nav(-1)
    def next(self):  self._nav(+1)

    def _nav(self, step):
        if not (0 <= self.index+step < len(self.img_paths)): return
        self._save_current(); self.index += step; self._load_image()

    # ───────────────────── edit ─────────────────────
    def delete_box(self):
        if self.selected is None: return
        if messagebox.askyesno("Delete", "Удалить выбранный бокс?"):
            self.boxes.pop(self.selected); self.selected=None; self._redraw_boxes()

    def set_class(self):
        if self.selected is None: return
        cls = self.cls_list.curselection()[0]
        x1,y1,x2,y2,_ = self.boxes[self.selected]
        self.boxes[self.selected]=(x1,y1,x2,y2,cls); self._redraw_boxes()

    # ───────────────────── I/O ─────────────────────
    def _ann_path(self, img_path:Path)->Path: return self.ann_dir / (img_path.stem+".txt")

    def _save_current(self):
        if not self.img_orig: return
        p = self._ann_path(self.img_paths[self.index])
        W,H = self.img_orig.size
        with open(p,"w") as f:
            for x1,y1,x2,y2,cls in self.boxes:
                xc, yc = ((x1+x2)/2)/W, ((y1+y2)/2)/H
                bw, bh = (x2-x1)/W, (y2-y1)/H
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    def _load_image(self):
        # оригинал
        p = self.img_paths[self.index]
        self.img_orig = Image.open(p).convert("RGB")
        # аннотации
        self.boxes.clear(); self.selected=None
        ann=self._ann_path(p)
        if ann.exists():
            W,H = self.img_orig.size
            for ln in ann.read_text().splitlines():
                cls, xc,yc,bw,bh = ln.split()
                cls=int(cls); xc=float(xc); yc=float(yc); bw=float(bw); bh=float(bh)
                x1,y1 = (xc-bw/2)*W, (yc-bh/2)*H
                x2,y2 = (xc+bw/2)*W, (yc+bh/2)*H
                self.boxes.append((x1,y1,x2,y2,cls))
        self._refresh_scale()
        self._redraw_boxes()

    # ───────────────────── scaling / drawing ─────────────────────
    def _refresh_scale(self):
        W,H = self.img_orig.size
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw<10 or ch<10: cw, ch = self.winfo_width(), self.winfo_height()
        if cw<10 or ch<10: cw,ch = W,H                # страховка
        self.scale = min(1.0, cw/W, ch/H)
        disp_w, disp_h = max(1,int(W*self.scale)), max(1,int(H*self.scale))
        disp = self.img_orig.resize((disp_w, disp_h), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.create_image(0,0,anchor="nw", image=self.tkimg)

    def _to_screen(self,x,y): return x*self.scale, y*self.scale
    def _to_orig(self,xs,ys): return xs/self.scale, ys/self.scale

    def _redraw_boxes(self,*,temp_box=None):
        self.canvas.delete("box")
        for i,(x1,y1,x2,y2,cls) in enumerate(self.boxes):
            xs,ys = self._to_screen(x1,y1); xe,ye = self._to_screen(x2,y2)
            color = "red" if i==self.selected else "lime"
            dash  = (4,2) if i==self.selected else None
            self.canvas.create_rectangle(xs,ys,xe,ye, outline=color,width=2,dash=dash,tags="box")
            self.canvas.create_text(xs+4,ys+4, anchor="nw", text=CLASSES[cls], fill=color,
                                    font=("Arial",10), tags="box")
        if temp_box:
            x1,y1,x2,y2 = temp_box
            xs,ys = self._to_screen(x1,y1); xe,ye = self._to_screen(x2,y2)
            self.canvas.create_rectangle(xs,ys,xe,ye, outline="yellow", dash=(2,2), tags="box")

    # ───────────────────── finish ─────────────────────
    def finish(self):
        self._save_current()
        if messagebox.askyesno("Merge", "Объединить все .txt в один файл?"):
            out = filedialog.asksaveasfilename(initialdir=self.ann_dir,
                                               defaultextension=".txt",
                                               filetypes=[("Text","*.txt")],
                                               title="Сохранить как…")
            if out:
                with open(out,"w") as fout:
                    for img in self.img_paths:
                        txt = self._ann_path(img)
                        if txt.exists():
                            for ln in txt.read_text().splitlines():
                                fout.write(f"{img.name} {ln}\n")
        self.destroy()

# ────────────────────── entry ──────────────────────
if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd()/ "passports"
    Annotator(root).mainloop()
