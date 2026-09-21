"""
Filtr oparty na wzorcach: bierze zdjecia z top_view.zip jako "tak ma wygladac dobre zdjecie"
i wybiera z puli GBIF te najbardziej do nich podobne, a dopiero potem dba o roznorodnosc.

WAZNE: nie ma tu zadnego warunku o zoltych kwiatach. Trafnosc wynika wylacznie z podobienstwa
do wzorcow, wiec rosliny bez kwiatostanu przechodza, o ile sa w zbiorze wzorcowym.

Dwie oceny, domyslnie usredniane (--mode):
  knn   - srednia z k najwyzszych podobienstw do wzorcow (dobrze znosi rozne ujecia)
  pu    - regresja logistyczna: wzorce jako pozytywy, losowa probka puli jako tlo
  blend - srednia obu (domyslnie)

KOMENDY - uruchamiaj w kontenerze (potrzebny torch), z /workspace:

# 1. pelny przebieg: embeddingi wzorcow + filtr + wybor 200:
python src/scripts/seed_filter.py

# 2. szerzej lub wezej - ile zdjec przechodzi filtr przed etapem roznorodnosci:
python src/scripts/seed_filter.py --keep 4000 --n-select 300

# 3. tylko podobienstwo do wzorcow, bez klasyfikatora:
python src/scripts/seed_filter.py --mode knn

WEJSCIE:  top_view.zip                            wzorce (czytane wprost z archiwum)
          afterfacilitylocation/embeddings.npz    gotowe wektory puli 9944
WYJSCIE:  afterfacilitylocation/seed_embeddings.npz   cache wzorcow
          afterfacilitylocation/seed_scores.csv       oceny wszystkich 9944
          afterfacilitylocation/seed_selected_<N>.csv wybrane
          afterfacilitylocation/seed_preview_<N>.html galeria wybranych
          afterfacilitylocation/seed_rejected.html    60 najgorszych (kontrola)
          afterfacilitylocation/seed_summary.json     podsumowanie i porownania
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import html
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent
WORK = REPO_ROOT / "afterfacilitylocation"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def thumb_for(url: str) -> Path:
    return WORK / "thumbs" / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg")


def embed_seeds(zip_path: Path, model_name: str, batch_size: int, max_side: int) -> Tuple[np.ndarray, List[str]]:
    """Liczy embeddingi zdjec wzorcowych czytanych prosto z archiwum (bez rozpakowywania)."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    from models.dinov3 import DINOv3

    zf = zipfile.ZipFile(zip_path)
    names = sorted(n for n in zf.namelist()
                   if not n.endswith("/") and PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES)
    print(f"[wzorce] {len(names)} zdjec w {zip_path.name}", flush=True)

    model = DINOv3(model_name=model_name)
    print(f"[model] {model_name} na {model.device}", flush=True)

    vectors: List[np.ndarray] = []
    kept: List[str] = []

    for start in tqdm(range(0, len(names), batch_size), desc="Wzorce"):
        chunk = names[start : start + batch_size]
        images = []
        ok = []
        for name in chunk:
            try:
                img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
                img.thumbnail((max_side, max_side))  # ta sama skala co miniatury puli
                images.append(img)
                ok.append(name)
            except Exception as exc:
                print(f"[wzorce] pomijam {name}: {type(exc).__name__}", flush=True)

        if not images:
            continue

        with torch.no_grad():
            feats = model.extract_features(images)

        vectors.append(feats.mean(dim=1).cpu().numpy().astype(np.float32))
        kept.extend(ok)
        for img in images:
            img.close()

    return np.vstack(vectors), kept


def normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def knn_score(P: np.ndarray, S: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Srednia z k najwyzszych podobienstw kazdego kandydata do wzorcow + indeks najblizszego."""
    sims = P @ S.T
    k = min(k, sims.shape[1])
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1), sims.argmax(axis=1)


def pu_score(P: np.ndarray, S: np.ndarray, seed: int, n_bg: int, epochs: int = 300, lr: float = 0.5) -> np.ndarray:
    """Regresja logistyczna: wzorce = 1, losowa probka puli = 0 (tlo). Czysty numpy."""
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(P), size=min(n_bg, len(P)), replace=False)
    X = np.vstack([S, P[bg_idx]])
    y = np.concatenate([np.ones(len(S)), np.zeros(len(bg_idx))]).astype(np.float32)

    w = np.zeros(X.shape[1], dtype=np.float32)
    b = np.float32(0.0)
    pos_weight = np.float32(len(bg_idx) / max(1, len(S)))
    sample_w = np.where(y > 0, pos_weight, 1.0).astype(np.float32)
    sample_w /= sample_w.sum()

    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) * sample_w
        w -= lr * (X.T @ g + 1e-3 * w)
        b -= lr * g.sum()

    return P @ w + b


def facility_location_lazy(S: np.ndarray, k: int) -> Tuple[List[int], List[float], List[float]]:
    n = S.shape[0]
    k = max(1, min(k, n))
    covered = np.zeros(n, dtype=np.float32)
    selected: List[int] = []
    gains: List[float] = []
    coverage: List[float] = []

    heap = [(-float(S[:, j].sum()), j, 0) for j in range(n)]
    heapq.heapify(heap)

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

    return selected, gains, coverage


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


def write_html(cards: List[dict], out_path: Path, title: str, subtitle: str) -> None:
    body = []
    for c in cards:
        src = os.path.relpath(REPO_ROOT / c["thumb_path"], out_path.parent).replace(os.sep, "/")
        body.append(
            f'  <div class="card"><a href="{html.escape(c["identifier"], quote=True)}" target="_blank" rel="noopener">'
            f'<img src="{html.escape(src, quote=True)}" loading="lazy" alt=""></a><div class="meta">'
            f'<div><span class="ord">{html.escape(str(c["label"]))}</span></div>'
            f'<div class="dim">{html.escape(c["note"])}</div></div></div>'
        )
    out_path.write_text(PAGE.format(title=html.escape(title), subtitle=subtitle, cards="\n".join(body)), encoding="utf-8")
    print(f"[zapis] {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, default=REPO_ROOT / "top_view.zip")
    ap.add_argument("--n-select", type=int, default=200)
    ap.add_argument("--keep", type=int, default=3000, help="ile najbardziej podobnych zdjec przechodzi do etapu roznorodnosci")
    ap.add_argument("--mode", default="blend", choices=["knn", "pu", "blend"])
    ap.add_argument("--knn-k", type=int, default=3)
    ap.add_argument("--bg-sample", type=int, default=3000, help="ile zdjec puli jako tlo dla klasyfikatora")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--meta-csv", type=Path, default=REPO_ROOT / "sample_10000.csv")
    ap.add_argument("--recompute-seeds", action="store_true")
    args = ap.parse_args()

    print("=== Filtr na wzorcach (top_view) + facility location ===", flush=True)

    pool = np.load(WORK / "embeddings.npz", allow_pickle=False)
    urls = [str(u) for u in pool["urls"]]
    model_name = str(pool["model"])
    P = normalize(pool["embeddings"].astype(np.float32))
    print(f"[pula] {len(urls)} zdjec, model {model_name}", flush=True)

    seed_cache = WORK / "seed_embeddings.npz"
    if seed_cache.exists() and not args.recompute_seeds:
        sc = np.load(seed_cache, allow_pickle=False)
        if str(sc["model"]) != model_name:
            raise SystemExit(f"Blad: cache wzorcow policzony modelem {str(sc['model'])}, a pula {model_name}. "
                             f"Uruchom z --recompute-seeds.")
        Sv, seed_names = sc["embeddings"].astype(np.float32), [str(n) for n in sc["names"]]
        print(f"[wzorce] {len(seed_names)} z cache ({seed_cache.name})", flush=True)
    else:
        if not args.seeds.exists():
            raise SystemExit(f"Blad: nie ma pliku {args.seeds}")
        t0 = time.time()
        Sv, seed_names = embed_seeds(args.seeds, model_name, args.batch_size, args.max_side)
        np.savez(seed_cache, embeddings=Sv, names=np.array(seed_names), model=np.array(model_name))
        print(f"[wzorce] policzone {len(seed_names)} w {time.time() - t0:.0f}s -> {seed_cache}", flush=True)

    S = normalize(Sv)

    knn, nearest = knn_score(P, S, args.knn_k)
    print(f"[knn] podobienstwo do wzorcow: min {knn.min():.3f}, mediana {np.median(knn):.3f}, max {knn.max():.3f}", flush=True)

    if args.mode == "knn":
        score = knn
    else:
        pu = pu_score(P, S, args.seed, args.bg_sample)
        print(f"[pu] klasyfikator: min {pu.min():.2f}, mediana {np.median(pu):.2f}, max {pu.max():.2f}", flush=True)
        if args.mode == "pu":
            score = pu
        else:
            zk = (knn - knn.mean()) / (knn.std() or 1.0)
            zp = (pu - pu.mean()) / (pu.std() or 1.0)
            score = 0.5 * zk + 0.5 * zp

    keep_idx = np.argsort(-score)[: min(args.keep, len(urls))]
    print(f"[filtr] do etapu roznorodnosci przechodzi {len(keep_idx)} z {len(urls)}", flush=True)

    with open(args.meta_csv, "r", encoding="utf-8", newline="") as fh:
        meta = {r["identifier"]: r for r in csv.DictReader(fh) if r.get("identifier")}

    with open(WORK / "seed_scores.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["identifier", "score", "knn_sim", "najblizszy_wzorzec", "kept"])
        kept = set(int(i) for i in keep_idx)
        for i, u in enumerate(urls):
            w.writerow([u, round(float(score[i]), 4), round(float(knn[i]), 4),
                        seed_names[int(nearest[i])], int(i in kept)])
    print(f"[zapis] {WORK / 'seed_scores.csv'}", flush=True)

    P_keep = P[keep_idx]
    M = np.clip(P_keep @ P_keep.T, 0.0, None).astype(np.float32)
    selected, gains, coverage = facility_location_lazy(M, args.n_select)
    cov = M[:, selected].max(axis=1)
    print(f"[wynik] {len(selected)} zdjec, srednie pokrycie {cov.mean():.4f}, najgorsze {cov.min():.4f}", flush=True)

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
            "knn_sim": round(float(knn[gi]), 4),
            "najblizszy_wzorzec": seed_names[int(nearest[gi])],
            "gain": round(gain, 3),
            "license": m.get("license", ""),
            "rightsHolder": m.get("rightsHolder", ""),
            "references": m.get("references", ""),
            "thumb_path": thumb_for(u).relative_to(REPO_ROOT).as_posix(),
        })

    out_csv = WORK / f"seed_selected_{args.n_select}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[zapis] {out_csv}", flush=True)

    # cala pula po filtrze, posortowana malejaco po ocenie; kolumna 'selected' oznacza finalowa N
    sel_urls = {r["identifier"] for r in rows}
    pool_rows = []
    for rank, gi in enumerate(keep_idx, start=1):
        gi = int(gi)
        u = urls[gi]
        m = meta.get(u, {})
        pool_rows.append({
            "rank_score": rank,
            "gbifID": m.get("gbifID", ""),
            "identifier": u,
            "score": round(float(score[gi]), 4),
            "knn_sim": round(float(knn[gi]), 4),
            "najblizszy_wzorzec": seed_names[int(nearest[gi])],
            "selected": int(u in sel_urls),
            "license": m.get("license", ""),
            "rightsHolder": m.get("rightsHolder", ""),
            "references": m.get("references", ""),
            "thumb_path": thumb_for(u).relative_to(REPO_ROOT).as_posix(),
        })

    pool_csv = WORK / f"seed_pool_{len(pool_rows)}.csv"
    with open(pool_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pool_rows[0].keys()))
        w.writeheader()
        w.writerows(pool_rows)
    print(f"[zapis] {pool_csv}  ({len(pool_rows)} zdjec po filtrze)", flush=True)

    write_html(
        [{"label": f"{p['rank_score']}. podob. {p['knn_sim']}" + (" · WYBRANE" if p["selected"] else ""),
          "note": f"wzorzec {PurePosixPath(p['najblizszy_wzorzec']).name} · {p['rightsHolder']}",
          "identifier": p["identifier"], "thumb_path": p["thumb_path"]} for p in pool_rows],
        WORK / f"seed_pool_{len(pool_rows)}.html",
        f"Pula po filtrze wzorcow - {len(pool_rows)} zdjec",
        f"posortowane malejaco po podobienstwie do top_view &middot; {len(rows)} oznaczonych WYBRANE "
        f"to finalowy wybor facility location",
    )

    # porownanie z wczesniejszymi wyborami + ile wybranych jest BEZ kwiatow
    comparison: Dict[str, object] = {}
    pos = {u: i for i, u in enumerate(urls)}
    for label, path in (("p_center_skrypt", WORK / "selected_200.csv"), ("heurystyka_claude", WORK / "claude_selected_200.csv")):
        if path.exists():
            with open(path, "r", encoding="utf-8", newline="") as fh:
                other = [r["identifier"] for r in csv.DictReader(fh)]
            idx = [pos[u] for u in other if u in pos]
            comparison[label] = {
                "srednie_podobienstwo_do_wzorcow": round(float(knn[idx].mean()), 4),
                "czesc_wspolna_z_moim_wyborem": len(set(other) & {r["identifier"] for r in rows}),
            }

    flowerless = None
    qpath = WORK / "claude_scores.csv"
    if qpath.exists():
        with open(qpath, "r", encoding="utf-8", newline="") as fh:
            yellow = {r["identifier"]: float(r["yellow"]) for r in csv.DictReader(fh)}
        vals = [yellow.get(r["identifier"], 0.0) for r in rows]
        flowerless = {
            "wybrane_bez_wyraznych_kwiatow (yellow<0.01)": int(sum(1 for v in vals if v < 0.01)),
            "mediana_udzialu_zoltego": round(float(np.median(vals)), 5),
        }
        print(f"[kontrola kwiatow] {flowerless}", flush=True)

    write_html(
        [{"label": f"#{r['rank']}", "note": f"podob. {r['knn_sim']} · wzorzec {PurePosixPath(r['najblizszy_wzorzec']).name} · {r['rightsHolder']}",
          "identifier": r["identifier"], "thumb_path": r["thumb_path"]} for r in rows],
        WORK / f"seed_preview_{args.n_select}.html",
        f"Wybrane na podstawie wzorcow top_view - {len(rows)} zdjec",
        f"filtr zostawil {len(keep_idx)} z {len(urls)} &middot; tryb {args.mode} &middot; "
        f"srednie pokrycie {cov.mean():.4f} &middot; bez warunku o kwiatach",
    )

    worst = np.argsort(score)[:60]
    write_html(
        [{"label": f"score {score[i]:.2f}", "note": f"podob. do wzorcow {knn[i]:.3f}",
          "identifier": urls[i], "thumb_path": thumb_for(urls[i]).relative_to(REPO_ROOT).as_posix()}
         for i in worst if thumb_for(urls[i]).exists()],
        WORK / "seed_rejected.html",
        "Najmniej podobne do wzorcow - kontrola",
        "60 zdjec z najnizsza ocena; tu powinny wyladowac ludzie, zielniki, mapy i zblizenia z bliska",
    )

    (WORK / "seed_summary.json").write_text(json.dumps({
        "wzorce": {"plik": args.seeds.name, "liczba": len(seed_names)},
        "model": model_name,
        "tryb": args.mode,
        "pula": len(urls),
        "po_filtrze": int(len(keep_idx)),
        "wybrane": len(rows),
        "pokrycie": {"srednie": round(float(cov.mean()), 4), "najgorsze": round(float(cov.min()), 4)},
        "kontrola_kwiatow": flowerless,
        "porownanie": comparison,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("KONIEC", flush=True)


if __name__ == "__main__":
    main()
