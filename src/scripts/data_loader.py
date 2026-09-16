import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class RagwortDataset(Dataset):
    def __init__(self, folder_ze_zdjeciami):
        # Zapisujemy, gdzie leżą oryginalne zdjęcia
        self.folder = folder_ze_zdjeciami
        self.lista_plikow = os.listdir(folder_ze_zdjeciami)

        self.moje_transformacje = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lista_plikow)

    def __getitem__(self, indeks):
        nazwa_pliku = self.lista_plikow[indeks]
        sciezka_do_pliku = os.path.join(self.folder, nazwa_pliku)

        zdjecie = Image.open(sciezka_do_pliku)

        gotowe_zdjecie = self.moje_transformacje(zdjecie)
        nazwa_pliku.removesuffix('.png')

        etykieta = int(nazwa_pliku)

        return gotowe_zdjecie, etykieta


moj_magazyn = RagwortDataset(folder_ze_zdjeciami="../../data")

# Uruchamiamy podajnik
moj_podajnik = DataLoader(
    dataset=moj_magazyn,
    batch_size=32,  # Kelner bierze 32 zdjęcia na jedną tacę
    shuffle=True,  # Tasowanie przed każdą nową epoką jest włączone
    num_workers=2  # Dwóch asystentów ładuje zdjęcia z dysku w tle
)