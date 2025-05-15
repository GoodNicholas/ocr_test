import csv
from rapidfuzz import process, fuzz

# ─── ЗАГРУЗКА СПИСКОВ ИЗ CSV ───────────────────────────────────────────
def load_names(filepath):
    """
    Читает файл, где в каждой строке — одно имя.
    Возвращает список имён в верхнем регистре.
    """
    with open(filepath, encoding='utf-8') as f:
        return [line.strip().upper() for line in f if line.strip()]

male_names   = load_names('/content/russian_male_names.csv')    # ~3000 строк
female_names = load_names('/content/russian_female_names.csv')  # ~2000 строк

# ─── ФУНКЦИЯ КОРРЕКЦИИ ИМЕНИ ───────────────────────────────────────────
def correct_name(query: str, gender: str, threshold: int = 70):
    """
    query   — OCR-имя;
    gender  — 'M'/'F' (или 'male'/'female', 'м'/'ж' и т. п.);
    threshold — минимальный % схожести.
    
    Возвращает (best_name, score) или (None, None), если score < threshold.
    """
    q = query.strip().upper()
    # выбираем нужный список
    if gender.lower() in ('m', 'male', 'м', 'муж'):
        choices = male_names
    else:
        choices = female_names

    # score_cutoff сразу прерывает, если нет шансов превысить threshold
    result = process.extractOne(
        q,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=threshold
    )
    if result:
        best, score, _ = result
        return best, score
    return None, None

# ─── ПРИМЕР ИСПОЛЬЗОВАНИЯ ─────────────────────────────────────────────
if __name__ == '__main__':
    ocr_name = "анннтон"    # что-то из OCR
    gender   = "M"          # или "F"
    
    corrected, score = correct_name(ocr_name, gender, threshold=70)
    if corrected:
        print(f"Исправлено на «{corrected}» (схожесть {score:.1f}%)")
    else:
        print(f"Нет кандидата ≥70% -> {ocr_name}")
