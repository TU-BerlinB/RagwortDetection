"""
Dobor zdjec do trenowania detektora starca: NAJPIERW trafnosc, POTEM roznorodnosc.

Dlaczego inaczej niz select_representative.py: czysty k-center (p-center) maksymalizuje
roznorodnosc, wiec z definicji wybiera zdjecia najbardziej odstajace od reszty. W zbiorze
z GBIF odstajace = ludzie, arkusze zielnikowe, mapy, zrzuty ekranu. Tutaj jest odwrotnie:
1) filtr trafnosci zostawia zdjecia TYPOWE dla zbioru i majace zolte kwiaty,
2) dopiero w tej puli dziala facility location, dbajac o roznorodnosc.

KOMENDY:

python src/scripts/claude_pick_ragwort.py
python src/scripts/claude_pick_ragwort.py --n-select 300 --keep-frac 0.5

WEJSCIE:  afterfacilitylocation/embeddings.npz, afterfacilitylocation/thumbs/, sample_10000.csv
WYJSCIE:  afterfacilitylocation/claude_selected_<N>.csv        wybrane zdjecia
          afterfacilitylocation/claude_preview_<N>.html        galeria wybranych
          afterfacilitylocation/claude_rejected.html           galeria odrzuconych (do kontroli)
          afterfacilitylocation/claude_scores.csv              oceny wszystkich zdjec
          afterfacilitylocation/claude_summary.json            podsumowanie i porownanie
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
WORK = REPO_ROOT / "afterfacilitylocation"

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def thumb_for(url: str) -> Path:
    return WORK / "thumbs" / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg")


def image_stats(path: Path) -> dict:
    """Tanie wskazniki z miniatury: udzial zoltych kwiatow, zieleni, ostrosc, kontrast."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        a = np.asarray(im.resize((96, 96)), dtype=np.float32) / 255.0

    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    chroma = mx - mn
    sat = np.where(mx > 0, chroma / np.maximum(mx, 1e-6), 0.0)

    # zolty: czerwony i zielony wysokie, niebieski wyraznie nizszy, nasycone i jasne
    yellow = (r > 0.45) & (g > 0.40) & (b < g - 0.12) & (r > b + 0.15) & (sat > 0.30) & (mx > 0.35)
    # zielen roslinna
    green = (g > r + 0.03) & (g > b + 0.05) & (sat > 0.12)

    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    return {
        "width": w,
        "height": h,
        "yellow": float(yellow.mean()),
        "green": float(green.mean()),
        "saturation": float(sat.mean()),
        "lum_std": float(lum.std()),
        "sharpness": float(np.abs(np.diff(lum, axis=0)).mean() + np.abs(np.diff(lum, axis=1)).mean()),
        "bright_frac": float((lum > 0.96).mean()),
    }


def collect_stats(urls: List[str], workers: int) -> List[dict]:
    def one(url: str) -> dict:
        path = thumb_for(url)
        if not path.exists():
            return {}
        try:
            return image_stats(path)
        except Exception:
            return {}

    t0 = time.time()
    out: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, s in enumerate(pool.map(one, urls), start=1):
            out.append(s)
            if i % 2500 == 0:
                print(f"[obrazy] {i}/{len(urls)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[obrazy] gotowe: {sum(1 for s in out if s)} z {len(urls)} w {time.time() - t0:.0f}s", flush=True)
    return out


def prototype_similarity(E: np.ndarray, rounds: int = 2, core_frac: float = 0.3) -> np.ndarray:
    """Podobienstwo do 'typowego zdjecia zbioru'. Prototyp liczony iteracyjnie:
    srednia -> najblizsze core_frac -> ich srednia -> ... Zdjecia nietypowe (ludzie,
    dokumenty, mapy) maja niskie podobienstwo do takiego prototypu."""
    proto = E.mean(axis=0)
    proto /= np.linalg.norm(proto)
    for _ in range(rounds):
        sim = E @ proto
        core = np.argsort(-sim)[: max(1, int(len(sim) * core_frac))]
        proto = E[core].mean(axis=0)
        proto /= np.linalg.norm(proto)
    return E @ proto


def similarity_matrix(E: np.ndarray, block: int = 1024) -> np.ndarray:
    n = E.shape[0]
    S = np.empty((n, n), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, block):
        stop = min(start + block, n)
        np.clip(E[start:stop] @ E.T, 0.0, None, out=S[start:stop])
    print(f"[macierz] {S.shape} w {time.time() - t0:.0f}s", flush=True)
    return S


def facility_location_lazy(S: np.ndarray, k: int) -> Tuple[List[int], List[float], List[float]]:
    """max F(S) = suma_i max_{j in S} sim(i,j), algorytm lazy greedy."""
    n = S.shape[0]
    k = max(1, min(k, n))

    covered = np.zeros(n, dtype=np.float32)
    selected: List[int] = []
    gains: List[float] = []
    coverage: List[float] = []

    heap = [(-float(S[:, j].sum()), j, 0) for j in range(n)]
    heapq.heapify(heap)

    t0 = time.time()
    for step in range(k):
        while True:
            neg, j, when = heapq.heappop(heap)
            if when == step:
                best, best_gain = j, -neg
                break
            heapq.heappush(heap, (-float(np.maximum(S[:, j] - covered, 0.0).sum()), j, step))

        selected.append(best)
        np.maximum(covered, S[:, best], out=covered)
        gains.append(best_gain)
        coverage.append(float(covered.mean()))

        if (step + 1) % 50 == 0:
            print(f"[greedy] {step + 1}/{k}, pokrycie {coverage[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    return selected, gains, coverage


def evaluate(S: np.ndarray, idx: List[int]) -> dict:
    cov = S[:, idx].max(axis=1)
    return {
        "srednie_pokrycie": round(float(cov.mean()), 4),
        "najgorsze_pokrycie": round(float(cov.min()), 4),
    }


PAGE = """<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
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
</style></head><body><header><h1>{title}</h1><div class="sub">{subtitle}</div></header><div class="grid">
{cards}
</div></body></html>
"""


def write_html(cards_data: List[dict], out_path: Path, title: str, subtitle: str) -> None:
    cards = []
    for c in cards_data:
        src = os.path.relpath(REPO_ROOT / c["thumb_path"], out_path.parent).replace(os.sep, "/")
        cards.append(
            f'  <div class="card"><a href="{html.escape(c["identifier"], quote=True)}" target="_blank" rel="noopener">'
            f'<img src="{html.escape(src, quote=True)}" loading="lazy" alt=""></a><div class="meta">'
            f'<div><span class="ord">{html.escape(str(c["label"]))}</span></div>'
            f'<div class="dim">{html.escape(c["note"])}</div></div></div>'
        )
    out_path.write_text(PAGE.format(title=html.escape(title), subtitle=subtitle, cards="\n".join(cards)), encoding="utf-8")
    print(f"[zapis] {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-select", type=int, default=200)
    ap.add_argument("--keep-frac", type=float, default=0.45, help="jaka czesc puli przejsc ma filtr trafnosci")
    ap.add_argument("--min-yellow", type=float, default=0.004, help="minimalny udzial zoltych pikseli")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--meta-csv", type=Path, default=REPO_ROOT / "sample_10000.csv")
    ap.add_argument("--compare-csv", type=Path, default=WORK / "selected_200.csv")
    args = ap.parse_args()

    print("=== Dobor zdjec starca: trafnosc + facility location ===", flush=True)

    data = np.load(WORK / "embeddings.npz", allow_pickle=False)
    urls = [str(u) for u in data["urls"]]
    E = data["embeddings"].astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    print(f"[dane] {E.shape[0]} zdjec, model {str(data['model'])}", flush=True)

    with open(args.meta_csv, "r", encoding="utf-8", newline="") as fh:
        meta = {r["identifier"]: r for r in csv.DictReader(fh) if r.get("identifier")}

    stats = collect_stats(urls, args.workers)
    proto = prototype_similarity(E)

    yellow = np.array([s.get("yellow", 0.0) if s else 0.0 for s in stats], dtype=np.float32)
    green = np.array([s.get("green", 0.0) if s else 0.0 for s in stats], dtype=np.float32)
    sharp = np.array([s.get("sharpness", 0.0) if s else 0.0 for s in stats], dtype=np.float32)
    bright = np.array([s.get("bright_frac", 1.0) if s else 1.0 for s in stats], dtype=np.float32)
    has = np.array([bool(s) for s in stats])

    def z(v: np.ndarray) -> np.ndarray:
        m = v[has].mean()
        s = v[has].std() or 1.0
        return (v - m) / s

    # ocena trafnosci: typowosc + zolte kwiaty + zielen + ostrosc
    score = 1.0 * z(proto) + 0.8 * z(np.sqrt(yellow)) + 0.3 * z(green) + 0.3 * z(sharp)
    score[~has] = -1e6

    hard_fail = (~has) | (yellow < args.min_yellow) | (bright > 0.35) | (sharp < np.percentile(sharp[has], 2))
    score[hard_fail] = -1e6

    n_keep = max(args.n_select, int(len(urls) * args.keep_frac))
    keep_idx = np.argsort(-score)[:n_keep]
    keep_idx = np.array([i for i in keep_idx if score[i] > -1e5])
    print(f"[filtr] zostaje {len(keep_idx)} z {len(urls)} "
          f"(odpadlo na twardych warunkach: {int(hard_fail.sum())})", flush=True)

    with open(WORK / "claude_scores.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["identifier", "score", "proto_sim", "yellow", "green", "sharpness", "kept"])
        kept = set(int(i) for i in keep_idx)
        for i, u in enumerate(urls):
            w.writerow([u, round(float(score[i]), 4), round(float(proto[i]), 4), round(float(yellow[i]), 5),
                        round(float(green[i]), 4), round(float(sharp[i]), 4), int(i in kept)])

    E_keep = E[keep_idx]
    urls_keep = [urls[i] for i in keep_idx]
    S = similarity_matrix(E_keep)

    selected, gains, coverage = facility_location_lazy(S, args.n_select)
    my_eval = evaluate(S, selected)
    print(f"[wynik] {len(selected)} zdjec, {my_eval}", flush=True)

    rows = []
    for rank, (local, gain) in enumerate(zip(selected, gains), start=1):
        gi = int(keep_idx[local])
        u = urls[gi]
        m = meta.get(u, {})
        rows.append({
            "rank": rank,
            "gbifID": m.get("gbifID", ""),
            "identifier": u,
            "score": round(float(score[gi]), 4),
            "proto_sim": round(float(proto[gi]), 4),
            "yellow": round(float(yellow[gi]), 5),
            "sharpness": round(float(sharp[gi]), 4),
            "gain": round(gain, 3),
            "license": m.get("license", ""),
            "rightsHolder": m.get("rightsHolder", ""),
            "creator": m.get("creator", ""),
            "references": m.get("references", ""),
            "thumb_path": thumb_for(u).relative_to(REPO_ROOT).as_posix(),
        })

    out_csv = WORK / f"claude_selected_{args.n_select}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[zapis] {out_csv}", flush=True)

    comparison = {}
    if args.compare_csv.exists():
        with open(args.compare_csv, "r", encoding="utf-8", newline="") as fh:
            theirs = [r["identifier"] for r in csv.DictReader(fh)]
        kept_set = {urls[int(i)] for i in keep_idx}
        comparison = {
            "ich_200_ktore_przeszlyby_filtr": sum(1 for u in theirs if u in kept_set),
            "czesc_wspolna_wyborow": len(set(theirs) & {r["identifier"] for r in rows}),
            "ich_srednia_typowosc": round(float(np.mean([proto[urls.index(u)] for u in theirs if u in meta])), 4),
            "moja_srednia_typowosc": round(float(np.mean([r["proto_sim"] for r in rows])), 4),
            "ich_srednia_zoltosc": round(float(np.mean([yellow[urls.index(u)] for u in theirs if u in meta])), 5),
            "moja_srednia_zoltosc": round(float(np.mean([r["yellow"] for r in rows])), 5),
        }
        print(f"[porownanie] {json.dumps(comparison, ensure_ascii=False)}", flush=True)

    write_html(
        [{"label": f"#{r['rank']}", "note": f"typowosc {r['proto_sim']} · zolty {r['yellow']} · {r['rightsHolder']}",
          "identifier": r["identifier"], "thumb_path": r["thumb_path"]} for r in rows],
        WORK / f"claude_preview_{args.n_select}.html",
        f"Wybrane do treningu - {len(rows)} zdjec",
        f"filtr trafnosci zostawil {len(keep_idx)} z {len(urls)}, potem facility location &middot; "
        f"srednie pokrycie {my_eval['srednie_pokrycie']}",
    )

    worst = np.argsort(score)[:60]
    write_html(
        [{"label": f"score {score[i]:.2f}" if score[i] > -1e5 else "odrzucone",
          "note": f"typowosc {proto[i]:.3f} · zolty {yellow[i]:.4f}",
          "identifier": urls[i], "thumb_path": thumb_for(urls[i]).relative_to(REPO_ROOT).as_posix()}
         for i in worst if thumb_for(urls[i]).exists()],
        WORK / "claude_rejected.html",
        "Odrzucone przez filtr - kontrola",
        "60 zdjec z najnizsza ocena trafnosci; tu powinny byc ludzie, zielniki, mapy i przepalone kadry",
    )

    (WORK / "claude_summary.json").write_text(json.dumps({
        "model": str(data["model"]),
        "pula": len(urls),
        "po_filtrze": int(len(keep_idx)),
        "odrzucone_twardo": int(hard_fail.sum()),
        "wybrane": len(rows),
        "ocena": my_eval,
        "porownanie": comparison,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("KONIEC", flush=True)


if __name__ == "__main__":
    main()
