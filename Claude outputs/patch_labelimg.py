"""Naprawa labelImg na Pythonie 3.10+ / nowym PyQt5: rzutowanie wspolrzednych na int.

Uzycie:
    python patch_labelimg.py [dodatkowy_katalog ...]

Domyslnie lata pakiet w site-packages aktywnego interpretera. Kazdy dodatkowy argument
to katalog, ktory tez zostanie przeszukany (np. folder srodowiska .labelimg).
"""

import pathlib
import re
import sys
import sysconfig

# metoda -> {liczba_argumentow: indeksy_do_rzutowania}
RULES = {
    "drawLine": {4: (0, 1, 2, 3)},
    "drawRect": {4: (0, 1, 2, 3)},
    "drawEllipse": {4: (0, 1, 2, 3)},
    "drawPoint": {2: (0, 1)},
    "drawText": {3: (0, 1)},       # trzeci argument to tekst - nie ruszamy
    "fillRect": {5: (0, 1, 2, 3)},  # piaty to pedzel/kolor - nie ruszamy
}

CALL = re.compile(r"^(\s*)([A-Za-z_][\w.]*)\.(" + "|".join(RULES) + r")\((.*)\)\s*$")


def split_args(text):
    args, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def patch_line(line):
    m = CALL.match(line)
    if not m:
        return line, False

    indent, obj, method, inner = m.groups()
    args = split_args(inner)
    idx = RULES[method].get(len(args))
    if idx is None:
        return line, False

    changed = False
    for i in idx:
        a = args[i]
        if a.startswith("int(") or a.isdigit():
            continue
        args[i] = f"int({a})"
        changed = True

    if not changed:
        return line, False
    return f"{indent}{obj}.{method}({', '.join(args)})\n", True


def main():
    roots = [pathlib.Path(sysconfig.get_paths()["purelib"])]
    roots += [pathlib.Path(p).resolve() for p in sys.argv[1:]]

    files = []
    for root in roots:
        if root.exists():
            for name in ("libs/canvas.py", "libs/shape.py"):
                files += list(root.rglob(name))

    files = list(dict.fromkeys(files))
    if not files:
        print("Nie znalazlem plikow labelImg (libs/canvas.py). Sprawdz, czy labelImg jest zainstalowany.")
        raise SystemExit(1)

    total = 0
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines(keepends=True)
        n = 0
        for i, line in enumerate(lines):
            new, changed = patch_line(line)
            if changed:
                lines[i] = new
                n += 1
                print(f"   linia {i + 1}: {line.strip()}  ->  {new.strip()}")
        if n:
            backup = f.with_suffix(".py.bak")
            if not backup.exists():
                backup.write_text("".join(f.read_text(encoding="utf-8")), encoding="utf-8")
            f.write_text("".join(lines), encoding="utf-8")
        print(f"{f} -> poprawione: {n}")
        total += n

    print(f"\nRazem poprawionych linii: {total}")
    if total == 0:
        print("Nic nie wymagalo zmiany - albo juz zalatane, albo inna wersja labelImg.")


if __name__ == "__main__":
    main()
