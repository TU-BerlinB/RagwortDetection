"""
Wlasna implementacja doboru zdjec metoda FACILITY LOCATION (wariant submodularny,
maksymalizacja pokrycia srednio, lazy greedy) + filtr jakosci zdjec.

Rozni sie od select_representative.py: tamten liczy p-center (minimalizuje NAJGORSZY
przypadek, przez co goni za outlierami - takze za smieciami), ten maksymalizuje
    F(S) = suma po wszystkich zdjeciach z max podobienstwa do wybranego zbioru S
czyli klasyczna funkcje facility location. Efekt: wybiera reprezentantow gestych
obszarow, a pojedyncze dziwactwa ignoruje.

KOMENDY:

python src/scripts/claude_facility_location.py --n-select 200
python src/scripts/claude_facility_location.py --n-select 200 --no-quality-filter

WEJSCIE:  afterfacilitylocation/embeddings.npz   (gotowe wektory DINOv3)
          afterfacilitylocation/thumbs/          (miniatury do statystyk jakosci)
          sample_10000.csv                       (metadane: gbifID, licencja, autor)
WYJSCIE:  afterfacilitylocation/claude_selected_<N>.csv
          afterfacilitylocation/claude_preview_<N>.html
          afterfacilitylocation/claude_quality.csv       statystyki jakosci wszystkich zdjec
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
WORK_DIR = REPO_ROOT / "afterfacilitylocation"

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def thumb_for(url: str) -> Path:
    return WORK_DIR / "thumbs" / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg")


def load_metadata(csv_path: Path) -> Dict[str, dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return {r["identifier"]: r for r in csv.DictReader(fh) if r.get("identifier")}


def image_stats(path: Path) -> dict:
    """Kilka tanich wskaznikow jakosci liczonych z miniatury."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        small = np.asarray(im.resize((96, 96)), dtype=np.float32) / 255.0

    mx = small.max(axis=2)
    mn = small.min(axis=2)
    saturation = float(np.mean(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)))

    lum = small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum_std = float(lum.std())
    sharpness = float(np.mean(np.abs(np.diff(lum, axis=0))) + np.mean(np.abs(np.diff(lum, axis=1))))
    dark = float((lum < 0.08).mean())
    bright = float((lum > 0.96).mean())

    return {
        "width": w,
        "height": h,
        "saturation": round(saturation, 4),
        "lum_std": round(lum_std, 4),
        "sharpness": round(sharpness, 4),
        "dark_frac": round(dark, 4),
        "bright_frac": round(bright, 4),
    }


def collect_stats(urls: List[str], workers: int) -> Dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor

    def one(url: str) -> Tuple[str, dict]:
        path = thumb_for(url)
        if not path.exists():
            return url, {}
        try:
            return url, image_stats(path)
        except Exception:
            return url, {}

    out: Dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (url, stats) in enumerate(pool.map(one, urls), start=1):
            out[url] = stats
            if i % 2000 == 0:
                print(f"[jakosc] {i}/{len(urls)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[jakosc] policzone dla {sum(1 for v in out.values() if v)} zdjec w {time.time() - t0:.0f}s", flush=True)
    return out


def quality_mask(urls: List[str], stats: Dict[str, dict]) -> Tuple[np.ndarray, dict]:
    """Odrzuca obrazy skrajnie rozmyte, bezbarwne albo przepalone.
    Progi sa percentylowe, wiec dopasowuja sie do zbioru."""
    sharp = np.array([stats[u].get("sharpness", 0.0) if stats[u] else 0.0 for u in urls])
    sat = np.array([stats[u].get("saturation", 0.0) if stats[u] else 0.0 for u in urls])
    lstd = np.array([stats[u].get("lum_std", 0.0) if stats[u] else 0.0 for u in urls])
    has = np.array([bool(stats[u]) for u in urls])

    t_sharp = float(np.percentile(sharp[has], 3))
    t_sat = float(np.percentile(sat[has], 2))
    t_lstd = float(np.percentile(lstd[has], 2))

    bad = (~has) | (sharp <= t_sharp) | (sat <= t_sat) | (lstd <= t_lstd)
    info = {
        "prog_sharpness": round(t_sharp, 4),
        "prog_saturation": round(t_sat, 4),
        "prog_lum_std": round(t_lstd, 4),
        "odrzucone": int(bad.sum()),
        "brak_miniatury": int((~has).sum()),
    }
    print(f"[filtr] odrzucone {info['odrzucone']} z {len(urls)} (progi: {info})", flush=True)
    return ~bad, info


def similarity_matrix(E: np.ndarray, block: int = 1024) -> np.ndarray:
    """Pelna macierz podobienstw kosinusowych, liczona blokami, obcieta do [0, 1]."""
    n = E.shape[0]
    S = np.empty((n, n), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, block):
        stop = min(start + block, n)
        np.clip(E[start:stop] @ E.T, 0.0, None, out=S[start:stop])
        print(f"[macierz] {stop}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return S


def facility_location_lazy(S: np.ndarray, k: int) -> Tuple[List[int], List[float], List[float]]:
    """Maksymalizuje F(S) = sum_i max_{j in S} sim(i,j) algorytmem lazy greedy.

    Zwraca: wybrane indeksy, przyrost pokrycia na kroku, srednie pokrycie po kroku.
    """
    n = S.shape[0]
    k = max(1, min(k, n))

    covered = np.zeros(n, dtype=np.float32)
    selected: List[int] = []
    gains: List[float] = []
    coverage: List[float] = []

    # kopiec: (-gorna granica zysku, indeks, numer iteracji, w ktorej zysk policzono)
    heap = [(-float(S[:, j].sum()), j, 0) for j in range(n)]
    heapq.heapify(heap)

    t0 = time.time()
    for step in range(k):
        while True:
            neg_gain, j, computed_at = heapq.heappop(heap)
            if computed_at == step:
                best = j
                best_gain = -neg_gain
                break
            true_gain = float(np.maximum(S[:, j] - covered, 0.0).sum())
            heapq.heappush(heap, (-true_gain, j, step))

        selected.append(best)
        np.maximum(covered, S[:, best], out=covered)
        gains.append(best_gain)
        coverage.append(float(covered.mean()))

        if (step + 1) % 50 == 0:
            print(f"[greedy] {step + 1}/{k}, pokrycie {coverage[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    return selected, gains, coverage


def evaluate(S: np.ndarray, idx: List[int]) -> dict:
    """Ocena zbioru: srednie pokrycie (cel facility location) i najgorszy przypadek (cel p-center)."""
    cov = S[:, idx].max(axis=1)
    return {
        "srednie_pokrycie": round(float(cov.mean()), 4),
        "mediana_pokrycia": round(float(np.median(cov)), 4),
        "najgorsze_pokrycie": round(float(cov.min()), 4),
        "ponizej_0.5": int((cov < 0.5).sum()),
    }


PAGE = """<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>
:root{{color-scheme:light dark;--bg:#fbfbfa;--fg:#1a1a18;--muted:#6b6b66;--card:#fff;--line:#e6e6e2}}
@media(prefers-color-scheme:dark){{:root{{--bg:#191917;--fg:#f0efec;--muted:#a3a29c;--card:#232320;--line:#33332e}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:24px 16px 48px;background:var(--bg);color:var(--fg);
font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{max-width:1400px;margin:0 auto 20px}}h1{{font-size:20px;margin:0 0 4px;font-weight:600}}
.sub{{color:var(--muted);font-size:13px;line-height:1.5}}
.grid{{max-width:1400px;margin:0 auto;display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(190px,1fr))}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.card img{{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:var(--line)}}
.meta{{padding:8px 10px 10px;font-size:12px;line-height:1.45}}.ord{{font-weight:600}}.dim{{color:var(--muted)}}
.meta a{{color:inherit}}@media(max-width:520px){{.grid{{grid-template-columns:repeat(auto-fill,minmax(140px,1fr))}}}}
</style></head><body><header><h1>{title}</h1><div class="sub">{subtitle}</div></header>
<div class="grid">
{cards}
</div></body></html>
"""


def write_html(rows: List[dict], out_path: Path, title: str, subtitle: str) -> None:
    cards = []
    for row in rows:
        src = os.path.relpath(REPO_ROOT / row["thumb_path"], out_path.parent).replace(os.sep, "/")
        cards.append(
            f'  <div class="card"><a href="{html.escape(row["identifier"], quote=True)}" target="_blank" rel="noopener">'
            f'<img src="{html.escape(src, quote=True)}" loading="lazy" alt=""></a><div class="meta">'
            f'<div><span class="ord">#{row["rank"]}</span> <span class="dim">gbifID {html.escape(row["gbifID"])}</span></div>'
            f'<div class="dim">{html.escape(row.get("rightsHolder") or row.get("creator") or "")}</div>'
            f'<div class="dim">zysk {row["gain"]} &middot; ostrosc {row["sharpness"]}</div></div></div>'
        )
    out_path.write_text(PAGE.format(title=html.escape(title), subtitle=subtitle, cards="\n".join(cards)), encoding="utf-8")
    print(f"[zapis] {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-select", type=int, default=200)
    parser.add_argument("--meta-csv", type=Path, default=REPO_ROOT / "sample_10000.csv")
    parser.add_argument("--compare-csv", type=Path, default=WORK_DIR / "selected_200.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-quality-filter", dest="quality_filter", action="store_false")
    args = parser.parse_args()

    print("=== Facility location (submodular, lazy greedy) - wlasna implementacja ===", flush=True)

    data = np.load(WORK_DIR / "embeddings.npz", allow_pickle=False)
    urls = [str(u) for u in data["urls"]]
    E = data["embeddings"].astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    print(f"[dane] {E.shape[0]} wektorow, wymiar {E.shape[1]}, model {str(data['model'])}", flush=True)

    meta = load_metadata(args.meta_csv)
    stats = collect_stats(urls, args.workers)

    with open(WORK_DIR / "claude_quality.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["identifier", "width", "height", "saturation", "lum_std", "sharpness", "dark_frac", "bright_frac"])
        for u in urls:
            s = stats.get(u) or {}
            writer.writerow([u, s.get("width", ""), s.get("height", ""), s.get("saturation", ""),
                             s.get("lum_std", ""), s.get("sharpness", ""), s.get("dark_frac", ""), s.get("bright_frac", "")])

    if args.quality_filter:
        keep_mask, filter_info = quality_mask(urls, stats)
    else:
        keep_mask, filter_info = np.ones(len(urls), dtype=bool), {"odrzucone": 0}

    keep_idx = np.flatnonzero(keep_mask)
    E_keep = E[keep_idx]
    urls_keep = [urls[i] for i in keep_idx]
    print(f"[pula] do wyboru z {len(urls_keep)} zdjec", flush=True)

    S = similarity_matrix(E_keep)
    print(f"[macierz] gotowa {S.shape}, srednie podobienstwo {float(S.mean()):.4f}", flush=True)

    selected, gains, coverage = facility_location_lazy(S, args.n_select)
    my_eval = evaluate(S, selected)
    print(f"[wynik] {len(selected)} zdjec, ocena: {my_eval}", flush=True)

    rows: List[dict] = []
    for rank, (local, gain) in enumerate(zip(selected, gains), start=1):
        url = urls_keep[local]
        m = dict(meta.get(url, {}))
        s = stats.get(url) or {}
        m.update({
            "rank": rank,
            "gain": round(gain, 3),
            "coverage_after": round(coverage[rank - 1], 5),
            "sharpness": s.get("sharpness", ""),
            "saturation": s.get("saturation", ""),
            "thumb_path": thumb_for(url).relative_to(REPO_ROOT).as_posix(),
            "identifier": url,
            "gbifID": m.get("gbifID", ""),
        })
        rows.append(m)

    out_csv = WORK_DIR / f"claude_selected_{args.n_select}.csv"
    columns = ["rank", "gbifID", "identifier", "gain", "coverage_after", "sharpness", "saturation",
               "license", "rightsHolder", "creator", "references", "created", "thumb_path"]
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[zapis] {out_csv}", flush=True)

    comparison = {}
    if args.compare_csv.exists():
        with open(args.compare_csv, "r", encoding="utf-8", newline="") as fh:
            theirs = [r["identifier"] for r in csv.DictReader(fh)]
        pos = {u: i for i, u in enumerate(urls_keep)}
        their_idx = [pos[u] for u in theirs if u in pos]
        comparison = {
            "p_center_skrypt": evaluate(S, their_idx),
            "facility_location_moje": my_eval,
            "czesc_wspolna": len(set(theirs) & {urls_keep[i] for i in selected}),
            "ich_zdjec_w_puli_po_filtrze": len(their_idx),
        }
        print(f"[porownanie] {json.dumps(comparison, ensure_ascii=False)}", flush=True)

    summary = {
        "model": str(data["model"]),
        "pula_wejsciowa": len(urls),
        "pula_po_filtrze": len(urls_keep),
        "filtr": filter_info,
        "wybrane": len(selected),
        "ocena": my_eval,
        "porownanie": comparison,
    }
    (WORK_DIR / f"claude_summary_{args.n_select}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_html(
        rows,
        WORK_DIR / f"claude_preview_{args.n_select}.html",
        f"Facility location (submodular) - {len(rows)} zdjec",
        f"pula {len(urls_keep)} z {len(urls)} po filtrze jakosci &middot; srednie pokrycie "
        f"{my_eval['srednie_pokrycie']} &middot; kolejnosc = malejacy zysk pokrycia",
    )
    print("KONIEC", flush=True)


if __name__ == "__main__":
    main()
