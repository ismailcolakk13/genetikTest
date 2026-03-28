# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:09 2025

@authors: İsmail Çolak, Mehmet Can Çalışkan, Yusuf Eren Aykurt
"""

from yardimcilar.gorsellestirici import gorsellestir_tasarim
from yardimcilar.yerlesimAnaliz import analiz_yap
from algoritmalar.ga import run_ga
from algoritmalar.pso import run_pso
from algoritmalar.nsga2 import run_nsga2
from modeller.aircraft import Aircraft
from modeller.komponent import Komponent

# --- UÇAK / KULLANICI DEĞİŞKENLERİ ---  Bunlar kullanıcıdan alınmalı, geçici olarak burada tanımlı.
GOVDE_UZUNLUK = 300.0 #CM
GOVDE_CAP = 60.0 #CM

TARGET_CG_X_MIN = 110.0
TARGET_CG_X_MAX = 130.0
TARGET_CG_Y = 0.0
TARGET_CG_Z = 0.0

MAX_YAKIT_AGIRLIGI = 50.0
TITRESIM_LIMITI = 50.0

KOMPONENTLER_DB = [ # Bunlar kullanıcıdan alınmalı, geçici olarak burada tanımlı.
    Komponent(id="Motor", agirlik=40.0, boyut=(60, 40, 40), sabit_bolge="BURUN", sabit_pos=(30, 0, 0), kilitli=True, titresim_hassasiyeti=False),
    Komponent(id="Batarya_Ana", agirlik=15.0, boyut=(20, 15, 10), sabit_bolge="GOVDE", kilitli=False, titresim_hassasiyeti=False),
    Komponent(id="Aviyonik_1", agirlik=5.0, boyut=(15, 15, 5), sabit_bolge="GOVDE", kilitli=False, titresim_hassasiyeti=True),
    Komponent(id="Aviyonik_2", agirlik=5.0, boyut=(15, 15, 5), sabit_bolge="GOVDE", kilitli=False, titresim_hassasiyeti=True),
    Komponent(id="Payload_Kam", agirlik=10.0, boyut=(20, 20, 20), sabit_bolge="ON_ALT", kilitli=False, titresim_hassasiyeti=True),
    Komponent(id="Yakit_Tanki", agirlik=40.0, boyut=(50, 40, 30), sabit_bolge="MERKEZ", kilitli=False, titresim_hassasiyeti=False),
    Komponent(id="Servo_Kuyruk", agirlik=2.0, boyut=(5, 5, 5), sabit_bolge="KUYRUK", kilitli=False, titresim_hassasiyeti=False),
]

# Aircraft modelini oluştur
aircraft = Aircraft(
    govde_uzunluk=GOVDE_UZUNLUK,
    govde_cap=GOVDE_CAP,
    target_cg_x_min=TARGET_CG_X_MIN,
    target_cg_x_max=TARGET_CG_X_MAX,
    target_cg_y=TARGET_CG_Y,
    target_cg_z=TARGET_CG_Z,
    max_yakit_agirligi=MAX_YAKIT_AGIRLIGI,
    titresim_limiti=TITRESIM_LIMITI,
    komponentler_db=KOMPONENTLER_DB
)

# --- SİMÜLASYON ---
POP_SIZE = 100
GENERATIONS = 50
print("\n--- SİMÜLASYON BAŞLATILIYOR ---")
print("Lütfen çalıştırmak istediğiniz algoritmayı seçin:")
print("1 - Genetik Algoritma (GA)")
print("2 - Parçacık Sürüsü Optimizasyonu (PSO)")
print("3 - NSGA-II (Çok Amaçlı Optimizasyon)")

while True:
    secim = input("Seçiminiz (1/2/3): ").strip()
    if secim == '1':
        ALGORITMA = "GA"
        break
    elif secim == '2':
        ALGORITMA = "PSO"
        break
    elif secim == '3':
        ALGORITMA = "NSGA2"
        break
    else:
        print("Geçersiz seçim! Lütfen 1, 2 veya 3 girin.")

if ALGORITMA == "PSO":
    en_iyi_tasarim, best_score, best_cg = run_pso(POP_SIZE, GENERATIONS, aircraft)
elif ALGORITMA == "NSGA2":
    en_iyi_tasarim, best_score, best_cg = run_nsga2(POP_SIZE, GENERATIONS, aircraft)
else:
    en_iyi_tasarim, best_score, best_cg = run_ga(POP_SIZE, GENERATIONS, aircraft)

# Terminaldeki analizi gösteren fonksiyon    
analiz_yap(en_iyi_tasarim, best_score, best_cg, aircraft, ALGORITMA) 

# Uçağın 3D modelini gösteren fonksiyon
gorsellestir_tasarim(en_iyi_tasarim, best_score, best_cg, aircraft, ALGORITMA) 