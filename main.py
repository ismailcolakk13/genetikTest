# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:09 2025

@authors: İsmail Çolak, Mehmet Can Çalışkan, Yusuf Eren Aykurt
"""

import os
import random

from yardimcilar.gorsellestirici import gorsellestir_tasarim
from yardimcilar.yerlesimAnaliz import analiz_yap
from algoritmalar.ga import run_ga
from algoritmalar.pso import run_pso
from algoritmalar.nsga2 import run_nsga2
from algoritmalar.nsga2_pso_hybrid import run_nsga2_pso_hybrid
from modeller.aircraft import Aircraft
from modeller.komponent import Komponent

# Pilot kilosu her simülasyon başında bir kez rastgele belirlenir (80–100 kg).
PILOT_AGIRLIGI = round(random.uniform(80.0, 100.0), 1)
print(f"\n[Pilot] Bu simülasyon için pilot kilosu: {PILOT_AGIRLIGI} kg")

# --- UÇAK / KULLANICI DEĞİŞKENLERİ ---  Bunlar kullanıcıdan alınmalı, geçici olarak burada tanımlı.
GOVDE_UZUNLUK = 300.0 #CM
GOVDE_CAP = 60.0 #CM

TARGET_CG_X_MIN = 90.0
TARGET_CG_X_MAX = 110.0
TARGET_CG_Y = 0.0
TARGET_CG_Z = 0.0

MAX_YAKIT_AGIRLIGI = 50.0
TITRESIM_LIMITI = 50.0
SICAKLIK_LIMITI = 30.0  # cm - Motor'a bu mesafeden yakın olan ısıya hassas parçalara ceza

KOMPONENTLER_DB = [ # Bunlar kullanıcıdan alınmalı, geçici olarak burada tanımlı.
    # Motor → BURUN'da sabit, kilitli
    Komponent(id="Motor",        agirlik=40.0, boyut=(60, 40, 40),
              izin_verilen_bolgeler=["BURUN"],
              sabit_pos=(30, 0, 0), kilitli=True,
              titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # Batarya → Ağır, denge için merkez/alt gövde
    Komponent(id="Batarya_Ana",  agirlik=15.0, boyut=(20, 15, 10),
              izin_verilen_bolgeler=["GOVDE", "TABAN"],
              kilitli=False, titresim_hassasiyeti=False, sicaklik_hassasiyeti=True),

    # Aviyonikler → Titreşim ve sıcaklıktan uzak, gövde üst bölgesi
    Komponent(id="Aviyonik_1",   agirlik=5.0,  boyut=(15, 15, 5),
              izin_verilen_bolgeler=["GOVDE", "TAVAN"],
              kilitli=False, titresim_hassasiyeti=True, sicaklik_hassasiyeti=True),

    Komponent(id="Aviyonik_2",   agirlik=5.0,  boyut=(15, 15, 5),
              izin_verilen_bolgeler=["GOVDE", "TAVAN"],
              kilitli=False, titresim_hassasiyeti=True, sicaklik_hassasiyeti=True),

    # Payload Kamera → Burun altında, görüş alanı için
    Komponent(id="Payload_Kam",  agirlik=10.0, boyut=(20, 20, 20),
              izin_verilen_bolgeler=["BURUN", "TABAN"],
              kilitli=False, titresim_hassasiyeti=True, sicaklik_hassasiyeti=False),

    # Yakıt Tankları → Kanat içi, span boyunca uzun kapsül (Cessna 172 wet wing)
    # boyut: (span_uzunlugu, chord_genisligi, kalinlik) — kanat geometrisini takip eder
    Komponent(id="Yakit_Tanki_Sol", agirlik=20.0, boyut=(110, 22, 6),
              izin_verilen_bolgeler=["GOVDE"],
              kilitli=False, titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    Komponent(id="Yakit_Tanki_Sag", agirlik=20.0, boyut=(110, 22, 6),
              izin_verilen_bolgeler=["GOVDE"],
              kilitli=False, titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # Servo → Kuyruk trim servosu (Cessna manuel kabloludur, küçük trim servosu temsili)
    Komponent(id="Servo_Kuyruk", agirlik=2.0,  boyut=(5, 5, 5),
              izin_verilen_bolgeler=["KUYRUK"],
              kilitli=False, titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # === KOLTUKLAR (Cessna 172: 2 ön + 2 arka) ===
    # Pilot (sol ön) → Motor arkası, kabin (sabit konum)
    Komponent(id="Koltuk_Pilot",    agirlik=8.0, boyut=(30, 15, 40),
              izin_verilen_bolgeler=["GOVDE", "TABAN"],
              sabit_pos=(80, -8, 0), kilitli=True,
              titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # Pilot (insan yükü) → Koltuk_Pilot üstünde, ağırlığı CG'yi belirgin etkiler
    Komponent(id="Pilot",           agirlik=PILOT_AGIRLIGI, boyut=(25, 12, 35),
              izin_verilen_bolgeler=["GOVDE", "TABAN"],
              sabit_pos=(80, -8, 0), kilitli=True,
              titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # Yardımcı pilot (sağ ön)
    Komponent(id="Koltuk_Yardimci", agirlik=8.0, boyut=(30, 15, 40),
              izin_verilen_bolgeler=["GOVDE", "TABAN"],
              sabit_pos=(80, 8, 0), kilitli=True,
              titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # Arka sol yolcu koltuğu
    Komponent(id="Koltuk_Arka_Sol", agirlik=7.0, boyut=(28, 15, 38),
              izin_verilen_bolgeler=["GOVDE", "TABAN"],
              sabit_pos=(120, -8, 0), kilitli=True,
              titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # Arka sağ yolcu koltuğu
    Komponent(id="Koltuk_Arka_Sag", agirlik=7.0, boyut=(28, 15, 38),
              izin_verilen_bolgeler=["GOVDE", "TABAN"],
              sabit_pos=(120, 8, 0), kilitli=True,
              titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),

    # Bagaj bölmesi → Arka koltukların arkasında (Cessna 172 baggage area)
    Komponent(id="Bagaj",           agirlik=15.0, boyut=(35, 30, 22),
              izin_verilen_bolgeler=["GOVDE"],
              sabit_pos=(160, 0, -2), kilitli=True,
              titresim_hassasiyeti=False, sicaklik_hassasiyeti=False),
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
    sicaklik_limiti=SICAKLIK_LIMITI,
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
print("4 - Karma (Hybrid) NSGA-II + PSO")

while True:
    secim = input("Seçiminiz (1/2/3/4): ").strip()
    if secim == '1':
        ALGORITMA = "GA"
        break
    elif secim == '2':
        ALGORITMA = "PSO"
        break
    elif secim == '3':
        ALGORITMA = "NSGA2"
        break
    elif secim == '4':
        ALGORITMA = "HYBRID_NSGA2_PSO"
        break
    else:
        print("Geçersiz seçim! Lütfen 1, 2, 3 veya 4 girin.")

if ALGORITMA == "PSO":
    en_iyi_tasarim, best_score, best_cg = run_pso(POP_SIZE, GENERATIONS, aircraft)
elif ALGORITMA == "NSGA2":
    en_iyi_tasarim, best_score, best_cg = run_nsga2(POP_SIZE, GENERATIONS, aircraft)
elif ALGORITMA == "HYBRID_NSGA2_PSO":
    en_iyi_tasarim, best_score, best_cg = run_nsga2_pso_hybrid(POP_SIZE, GENERATIONS, aircraft)
else:
    en_iyi_tasarim, best_score, best_cg = run_ga(POP_SIZE, GENERATIONS, aircraft)

# Terminaldeki analizi gösteren fonksiyon    
analiz_yap(en_iyi_tasarim, best_score, best_cg, aircraft, ALGORITMA) 

# Uçağın 3D modelini gösteren fonksiyon (toplu testlerde NO_VIZ=1 ile atlanır)
if os.environ.get("NO_VIZ") != "1":
    gorsellestir_tasarim(en_iyi_tasarim, best_score, best_cg, aircraft, ALGORITMA)