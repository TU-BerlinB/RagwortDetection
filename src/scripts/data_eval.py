import os
import shutil  # [Biblioteka do kopiowania i przenoszenia plików]


def oblicz_iou(box_a, box_b):
    """
    Oblicza IoU [Intersection over Union - procent pokrycia się dwóch ramek].
    Format ramek: [x_min, y_min, x_max, y_max]
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    pole_przeciecia = (x_right - x_left) * (y_bottom - y_top)
    pole_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    pole_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    pole_calkowite = float(pole_a + pole_b - pole_przeciecia)

    return pole_przeciecia / pole_calkowite


def ewaluacja_modelu(katalog_zdjec, predykcje, poprawne_dane, prog_iou=0.5):
    """
    Główna funkcja oceniająca model.
    """
    # Tworzymy foldery na błędy, żeby łatwo było je przeglądać
    katalog_fp = "bledy_False_Positives"  # [Model widzi starca, a go tam nie ma]
    katalog_fn = "bledy_False_Negatives"  # [Model przegapił starca]

    os.makedirs(katalog_fp, exist_ok=True)
    os.makedirs(katalog_fn, exist_ok=True)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Przechodzimy przez każde zdjęcie z poprawnymi danymi (Ground Truth)
    for nazwa_zdjecia, ramki_poprawne in poprawne_dane.items():
        ramki_modelu = predykcje.get(nazwa_zdjecia, [])

        znalezione_starce = 0

        # Sprawdzamy, jak dobrze model zgadł
        for ramka_m in ramki_modelu:
            trafienie = False
            for ramka_p in ramki_poprawne:
                iou = oblicz_iou(ramka_m, ramka_p)
                if iou >= prog_iou:
                    trafienie = True
                    znalezione_starce += 1
                    break

            if trafienie:
                true_positives += 1
            else:
                false_positives += 1
                # Kopiujemy zdjęcie z błędem do specjalnego folderu
                sciezka_zrodlo = os.path.join(katalog_zdjec, nazwa_zdjecia)
                if os.path.exists(sciezka_zrodlo):
                    shutil.copy(sciezka_zrodlo, os.path.join(katalog_fp, nazwa_zdjecia))

        # Sprawdzamy, ilu starców model w ogóle nie zauważył
        przegapione = len(ramki_poprawne) - znalezione_starce
        if przegapione > 0:
            false_negatives += przegapione
            sciezka_zrodlo = os.path.join(katalog_zdjec, nazwa_zdjecia)
            if os.path.exists(sciezka_zrodlo):
                shutil.copy(sciezka_zrodlo, os.path.join(katalog_fn, nazwa_zdjecia))

    # Obliczanie ostatecznych wyników
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print("--- WYNIKI EWALUACJI ---")
    print(f"Precyzja (Precision): {precision:.2f}")
    print(f"Czułość (Recall): {recall:.2f}")
    print(f"Zdjęcia z fałszywymi alarmami zapisano w: {katalog_fp}")
    print(f"Zdjęcia z przegapionymi starcami zapisano w: {katalog_fn}")


# --- Przykładowe użycie ---
# Ten fragment udaje Twoje dane. W prawdziwym życiu wczytasz je ze swojego modelu.
przykladowe_poprawne_dane = {
    "test_01.jpg": [[10, 10, 50, 50], [100, 100, 150, 150]]  # Dwa starce na zdjęciu
}
przykladowe_predykcje = {
    "test_01.jpg": [[12, 12, 48, 48], [200, 200, 250, 250]]  # Jeden trafiony, jeden wymyślony (Fałszywy alarm)
}

# Uruchomienie skryptu (wymaga folderu ze zdjęciami, np. 'moje_zdjecia')
# ewaluacja_modelu("moje_zdjecia", przykladowe_predykcje, przykladowe_poprawne_dane)