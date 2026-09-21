"""
Robi ze wskazanego CSV galerie HTML z miniaturami, zeby obejrzec wybrane zdjecia w przegladarce.

KOMENDY - caly blok mozna wkleic do konsoli w /workspace, linie z # sa ignorowane:

# 1. galeria dla 200 wybranych zdjec:
python src/scripts/preview_selection.py

# 2. galeria dla etapu pierwszego (2000 zdjec):
python src/scripts/preview_selection.py --csv afterfacilitylocation/selected_2000.csv

# 3. galeria + kopie zdjec w kolejnosci wyboru (do przegladania w Eksploratorze):
python src/scripts/preview_selection.py --copy

Plik HTML otwierasz podwojnym klikiem w Windowsie:
C:\\Users\\mateu\\Desktop\\kamera chwast\\RagwortDetection\\afterfacilitylocation\\preview_selected_200.html
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import shutil
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

DEFAULT_CSV = REPO_ROOT / "afterfacilitylocation" / "selected_200.csv"

PAGE = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; --bg:#fbfbfa; --fg:#1a1a18; --muted:#6b6b66; --card:#fff; --line:#e6e6e2; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#191917; --fg:#f0efec; --muted:#a3a29c; --card:#232320; --line:#33332e; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--fg);
         font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ max-width:1400px; margin:0 auto 20px; }}
  h1 {{ font-size:20px; margin:0 0 4px; font-weight:600; }}
  .sub {{ color:var(--muted); font-size:13px; }}
  .grid {{ max-width:1400px; margin:0 auto; display:grid; gap:14px;
           grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden;
           display:flex; flex-direction:column; }}
  .card img {{ width:100%; aspect-ratio:4/3; object-fit:cover; display:block; background:var(--line); }}
  .meta {{ padding:8px 10px 10px; font-size:12px; line-height:1.45; }}
  .ord {{ font-weight:600; }}
  .dim {{ color:var(--muted); }}
  .meta a {{ color:inherit; }}
  @media (max-width:520px) {{ .grid {{ grid-template-columns: repeat(auto-fill, minmax(140px,1fr)); }} }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
</header>
<div class="grid">
{cards}
</div>
</body>
</html>
"""

CARD = """  <div class="card">
    <a href="{url}" target="_blank" rel="noopener"><img src="{src}" alt="{alt}" loading="lazy"></a>
    <div class="meta">
      <div><span class="ord">#{order}</span> <span class="dim">gbifID {gid}</span></div>
      <div class="dim">{author}</div>
      <div class="dim">gap {gap} &middot; <a href="{url}" target="_blank" rel="noopener">oryginal</a></div>
    </div>
  </div>"""


def short_license(value: str) -> str:
    v = (value or "").lower()
    for code in ("by-nc-nd", "by-nc-sa", "by-nc", "by-sa", "by-nd", "cc0", "by"):
        if code in v:
            return "CC " + code.upper()
    return ""


def build(rows: List[dict], csv_path: Path, out_path: Path) -> None:
    cards = []
    missing = 0

    for row in rows:
        thumb = REPO_ROOT / row.get("thumb_path", "")
        if not thumb.exists():
            missing += 1
            continue

        src = os.path.relpath(thumb, out_path.parent).replace(os.sep, "/")
        author = row.get("rightsHolder") or row.get("creator") or ""
        lic = short_license(row.get("license", ""))
        cards.append(
            CARD.format(
                url=html.escape(row.get("identifier", ""), quote=True),
                src=html.escape(src, quote=True),
                alt=html.escape(row.get("gbifID", ""), quote=True),
                order=html.escape(str(row.get("pick_order", ""))),
                gid=html.escape(row.get("gbifID", "")),
                author=html.escape(" · ".join(x for x in (author, lic) if x)),
                gap=html.escape(str(row.get("gap_at_pick", "") or "-")),
            )
        )

    title = f"{csv_path.name} - {len(cards)} zdjec"
    subtitle = "kolejnosc = wynik k-center (#1 = medoid, dalej coraz mniejsze luki w pokryciu)"
    if missing:
        subtitle += f" &middot; pominieto {missing} bez miniatury"

    out_path.write_text(PAGE.format(title=html.escape(title), subtitle=subtitle, cards="\n".join(cards)), encoding="utf-8")
    print(f"[zapis] {out_path}  ({len(cards)} kafelkow)")


def copy_ordered(rows: List[dict], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in rows:
        thumb = REPO_ROOT / row.get("thumb_path", "")
        if not thumb.exists():
            continue
        dest = target_dir / f"{int(row['pick_order']):03d}_{row['gbifID']}.jpg"
        shutil.copy2(thumb, dest)
        n += 1
    print(f"[kopie] {n} zdjec w {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--copy", action="store_true", help="skopiuj zdjecia w kolejnosci wyboru do osobnego folderu")
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else (REPO_ROOT / args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Blad: nie ma pliku {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise SystemExit(f"Blad: {csv_path.name} jest pusty")

    out_path = args.out or (csv_path.parent / f"preview_{csv_path.stem}.html")
    build(rows, csv_path, out_path)

    if args.copy:
        copy_ordered(rows, csv_path.parent / f"images_{csv_path.stem}")

    print(f"Otworz plik w przegladarce: {out_path}")


if __name__ == "__main__":
    main()
