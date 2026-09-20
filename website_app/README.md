# 🌿 Ragwort Detection – Live Vision Web Dashboard
### TU Berlin Bootcamp (Group B) &bull; Hege Robot Vision Pipeline

Aplikacja webowa czasu rzeczywistego do wizualizacji detekcji starca jakubka (*Jacobaea vulgaris*) z kamery (np. Framos 435e / kamerki USB) za pomocą wytrenowanego modelu YOLOv8/YOLOv26.

---

## 🚀 Funkcjonalności

- **Transmisja wideo na żywo (MJPEG)** o niskim opóźnieniu bezpośrednio w przeglądarce (`/video_feed`).
- **Detekcja w czasie rzeczywistym**: Rysowanie bounding boxów, pewności detekcji oraz punktów centralnych roślin.
- **Sterowanie robotem Hege**: API `/api/telemetry` zwracające współrzędne $(X, Y)$ wykrytych roślin dla manipulatora usuwającego chwasty.
- **Dynamiczna kontrola progu**: Suwaki Confidence i IoU w interfejsie działające w locie bez restartu serwera.
- **Obsługa trybu awaryjnego (Fallback)**: Gdy fizyczna kamera nie jest podłączona, aplikacja automatycznie uruchamia symulację ze zdjęć testowych.

---

## 📁 Wagi modelu

Umieść wagi w katalogu:
```text
weights/yolo26n.pt
```
*(Jeśli plik nie istnieje, serwer automatycznie szuka `models/ragwort_yolov8_best.pt` lub pobiera bazowy `yolov8n.pt`)*.

---

## 🛠️ Uruchomienie lokalne

1. **Zainstaluj zależności**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Uruchom serwer**:
   ```bash
   python app.py
   ```

3. **Otwórz w przeglądarce**:
   ```text
   http://localhost:5000
   ```

---

## 📤 Jak wysłać ten kod do repozytorium GitHub Website?

W terminalu wewnątrz folderu `website_app`:
```bash
git init
git remote add origin https://github.com/TU-BerlinB/Website.git
git add .
git commit -m "feat: initial live camera detection website with yolo26n weights"
git branch -M main
git push -u origin main
```
