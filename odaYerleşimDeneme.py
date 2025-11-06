# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:09 2025

@author: ismai
"""
# MATEMATIKSEL KONVANSIYONA HESAPLAMALAR YAPILDI

import random

oda_plani = { #Bu örnek çıktıdır!!
    "mobilyalar": [
        {"ad": "yatak", "x": 0.5, "y": 1.0, "r": 0},  # 0 derece dönmüş
        {"ad": "masa", "x": 2.0, "y": 3.0, "r": 90},  # 90 derece dönmüş
    ]
}

ODA_GENISLIK = 3.52
ODA_DERINLIK = 5.0
YATAK_BOYUT = (2.1, 1.05)
MASA_BOYUT = (1.1, 0.6)
KAPI_ALANI = {"x1": 0.0, "y1": 0.0, "x2": 0.99, "y2": 0.8}
DOLAP_BOYUT = (2.0, 0.7)
SIFONYER_BOYUT = (0.63,0.47)
RANZA_BOYUT = (2.05,1.23)
KITAPLIK_BOYUT = (0.60,0.24)
PETEK_ALANI ={"x1":0,"y1":3.37,"x2":0.4,"y2":5}
CAM_ALANI = {"x1":0.46,"y1":4.35,"x2":3.06,"y2":5}

MOBILYA_LISTESI = [
    {"ad": "yatak", "boyut": YATAK_BOYUT},
    {"ad": "masa", "boyut": MASA_BOYUT},
    {"ad": "dolap", "boyut": DOLAP_BOYUT},
    {"ad": "ranza", "boyut": RANZA_BOYUT},
    {"ad": "kitaplik", "boyut": KITAPLIK_BOYUT},
    {"ad": "sifonyer", "boyut": SIFONYER_BOYUT},
]

AD_BOYUT_SOZLUGU = {m["ad"]: m["boyut"] for m in MOBILYA_LISTESI}

def dikdortgenler_cakisiyor(rect1, rect2):

    if rect1["x2"] <= rect2["x1"]:  # 1, 2'nin tamamen solunda
        return False
    if rect1["x1"] >= rect2["x2"]:  # 1, 2'nin tamamen sağında
        return False
    if rect1["y2"] <= rect2["y1"]:  # 1, 2'nin tamamen altında
        return False
    if rect1["y1"] >= rect2["y2"]:  # 1, 2'nin tamamen üstünde
        return False

    return True


def get_mobilya_koordinatlari(mobilya_dict):
    ad = mobilya_dict["ad"]
    x1 = mobilya_dict["x"]
    y1 = mobilya_dict["y"]

    boyut = AD_BOYUT_SOZLUGU[ad]

    if mobilya_dict["r"] == 0:
        genislik = boyut[0]
        derinlik = boyut[1]
    else:
        genislik = boyut[1]
        derinlik = boyut[0]

    x2 = x1 + genislik
    y2 = y1 + derinlik

    return {"ad": ad, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _hesapla_koordinatlar(plan):
    mobilya_koordinatlari_listesi = []
    for mob_dict in plan["mobilyalar"]:
        yeni_koordinat = get_mobilya_koordinatlari(mob_dict)
        mobilya_koordinatlari_listesi.append(yeni_koordinat)
        
    koordinat_sozlugu = {mob["ad"]: mob for mob in mobilya_koordinatlari_listesi}
    return mobilya_koordinatlari_listesi, koordinat_sozlugu

def _hesapla_statik_cezalar(mobilya_koordinatlari_listesi):
    puan=0
    for mob in mobilya_koordinatlari_listesi:
        if(mob["x1"]<0 or mob["x2"] > ODA_GENISLIK or
           mob["y1"]<0 or mob["y2"] >ODA_DERINLIK):
            puan -= 10000
            
        if dikdortgenler_cakisiyor(mob, KAPI_ALANI):
            puan -=5000
            
        if dikdortgenler_cakisiyor(mob, PETEK_ALANI):
            puan -=5000

        if dikdortgenler_cakisiyor(mob, CAM_ALANI):
            puan -=5000
            
    return puan

def _hesapla_dinamik_cezalar(mobilya_koordinatlari_listesi):
    puan=0
    for i in range(len(mobilya_koordinatlari_listesi)):
        for j in range(i+1,len(mobilya_koordinatlari_listesi)):
            mob1= mobilya_koordinatlari_listesi[i]
            mob2= mobilya_koordinatlari_listesi[j]
            if dikdortgenler_cakisiyor(mob1, mob2):
                puan -=5000
    return puan

def _hesapla_oduller(plan,koordinat_sozlugu):
    puan = 0
    TOLERANS = 0.05
    
    for mob_dict in plan["mobilyalar"]:
        ad = mob_dict["ad"]
        mob_koor = koordinat_sozlugu.get(ad)
        if not mob_koor: continue
    
        orjinal_genislik,orjinal_derinlik = AD_BOYUT_SOZLUGU[ad]
        
        if not (ad == "dolap" or ad == "ranza"):
            if(mob_koor["x1"]<TOLERANS) or (mob_koor["x2"] > ODA_GENISLIK-TOLERANS) or \
                (mob_koor["y1"] <TOLERANS) or (mob_koor["y2"]>ODA_DERINLIK-TOLERANS):
                    puan += 150
        else: #Dolap ve ranzanın uzun kenarı duvara dayalı mı
            uzun_kenar_duvarda=False
            rotasyon = mob_dict["r"]
            
            if orjinal_genislik > orjinal_derinlik:
                if rotasyon == 0:
                    if mob_koor["y1"] < TOLERANS or mob_koor["y2"] > ODA_DERINLIK - TOLERANS: uzun_kenar_duvarda = True
                else:
                    if mob_koor["x1"] < TOLERANS or mob_koor["x2"] > ODA_GENISLIK - TOLERANS: uzun_kenar_duvarda = True
            else:
                if rotasyon == 0:
                    if mob_koor["x1"] < TOLERANS or mob_koor["x2"] > ODA_GENISLIK - TOLERANS: uzun_kenar_duvarda = True
                else:
                    if mob_koor["y1"] < TOLERANS or mob_koor["y2"] > ODA_DERINLIK - TOLERANS: uzun_kenar_duvarda = True
                        
            if uzun_kenar_duvarda:
                puan += 300
            
            
    masa_koor = koordinat_sozlugu.get("masa")
    dolap_koor = koordinat_sozlugu.get("dolap")
    kitaplik_koor = koordinat_sozlugu.get("kitaplik")
    
    plan_dict_sozlugu = {mob["ad"]: mob for mob in plan["mobilyalar"]}
    masa_dict = plan_dict_sozlugu.get("masa")
    dolap_dict = plan_dict_sozlugu.get("dolap")
    kitaplik_dict = plan_dict_sozlugu.get("kitaplik")
    
    # Kural: "Masanın önü açık mı?" (1m)
    if masa_koor and masa_dict:
        # Masanın "önünü" tanımlayan bir tampon bölge oluştur
        if masa_dict["r"] == 0: # Masa 1.1(G) x 0.6(D) duruyor
            # Önü: Y ekseninde -1m (Alt tarafı)
            masa_onu = {"x1": masa_koor["x1"], "y1": masa_koor["y1"] - 1.0, 
                        "x2": masa_koor["x2"], "y2": masa_koor["y1"]}
        else: # Masa 0.6(G) x 1.1(D) duruyor
            # Önü: X ekseninde -1m (Sol tarafı) - (Sol tarafın 'ön' olduğunu varsayıyoruz)
            masa_onu = {"x1": masa_koor["x1"] - 1.0, "y1": masa_koor["y1"], 
                        "x2": masa_koor["x1"], "y2": masa_koor["y2"]}
        
        # Diğer mobilyalar bu "ön" bölgeye giriyor mu?
        for ad, koor in koordinat_sozlugu.items():
            if ad != "masa" and dikdortgenler_cakisiyor(koor, masa_onu):
                puan -= 200 # Masanın önünü kapattı!
    
    # Kural: "Dolabın önü açık mı?" (0.5m)
    if dolap_koor and dolap_dict:
        if dolap_dict["r"] == 0: # 2.0(G) x 0.7(D)
            dolap_onu = {"x1": dolap_koor["x1"], "y1": dolap_koor["y1"] - 0.5, 
                         "x2": dolap_koor["x2"], "y2": dolap_koor["y1"]}
        else: # 0.7(G) x 2.0(D)
            dolap_onu = {"x1": dolap_koor["x1"] - 0.5, "y1": dolap_koor["y1"], 
                         "x2": dolap_koor["x1"], "y2": dolap_koor["y2"]}
        
        for ad, koor in koordinat_sozlugu.items():
            if ad != "dolap" and dikdortgenler_cakisiyor(koor, dolap_onu):
                puan -= 200 # Dolabın önünü kapattı!

    # Kural: "Kitaplığın önü açık mı?" (0.5m)
    if kitaplik_koor and kitaplik_dict:
        if kitaplik_dict["r"] == 0: # 0.60(G) x 0.24(D)
            kitaplik_onu = {"x1": kitaplik_koor["x1"], "y1": kitaplik_koor["y1"] - 0.5, 
                            "x2": kitaplik_koor["x2"], "y2": kitaplik_koor["y1"]}
        else: # 0.24(G) x 0.60(D)
            kitaplik_onu = {"x1": kitaplik_koor["x1"] - 0.5, "y1": kitaplik_koor["y1"], 
                            "x2": kitaplik_koor["x1"], "y2": kitaplik_koor["y2"]}

        for ad, koor in koordinat_sozlugu.items():
            if ad != "kitaplik" and dikdortgenler_cakisiyor(koor, kitaplik_onu):
                puan -= 200 # Kitaplığın önünü kapattı!
    
    return puan
    
            
def calculate_fitness(plan):
    total_puan = 0
    
    koor_listesi,koor_sozlugu = _hesapla_koordinatlar(plan)
    total_puan += _hesapla_statik_cezalar(koor_listesi)
    total_puan += _hesapla_dinamik_cezalar(koor_listesi)
    total_puan += _hesapla_oduller(plan, koor_sozlugu)
    
    return total_puan


def create_random_plan():
    plan = {"mobilyalar": []}

    for mob in MOBILYA_LISTESI:
        plan["mobilyalar"].append({
            "ad": mob["ad"],
            "x": random.uniform(0, ODA_GENISLIK),
            "y": random.uniform(0, ODA_DERINLIK),
            "r": random.choice([0, 90])
        })
    return plan

import copy

def crossover(parent1, parent2):
    child_plan = {"mobilyalar": []}
    
    # Mobilya listesi uzunluğu kadar dön
    for i in range(len(MOBILYA_LISTESI)):
        # %50 ihtimalle P1'den, %50 ihtimalle P2'den al
        if random.random() < 0.5:
            # Ebeveyn 1'den bu mobilyayı (index i) kopyala
            child_plan["mobilyalar"].append(
                copy.deepcopy(parent1["mobilyalar"][i])
            )
        else:
            # Ebeveyn 2'den bu mobilyayı (index i) kopyala
            child_plan["mobilyalar"].append(
                copy.deepcopy(parent2["mobilyalar"][i])
            )
    return child_plan

def mutate(plan, rate=0.2, amount=0.5):
    if random.random() < rate:
        mob_index = random.randint(0, len(plan["mobilyalar"]) - 1)

        degisim_tipi = random.choice(["x", "y", "r"])
        
        mob = plan["mobilyalar"][mob_index]

        if degisim_tipi == "x":
            yeni_x=mob["x"] + random.uniform(-amount, amount)
            mob["x"] = max(0,min(yeni_x,ODA_GENISLIK))
        elif degisim_tipi == "y":
            yeni_y=mob["y"] + random.uniform(-amount, amount)
            mob["y"] = max(0,min(yeni_y,ODA_DERINLIK))
        else:
            mob["r"] = 90 if mob["r"] == 0 else 0 

    return plan

POPULASYON_BUYUKLUGU = 1000
NESIL_SAYISI = 50
ELITIZM_ORANI = 0.1
MUTASYON_ORANI = 0.2
MUTASYON_ETKI_MIKTARI = 0.5

populasyon = [create_random_plan() for _ in range(POPULASYON_BUYUKLUGU)]

print("Evrim başlıyor")

for nesil in range(NESIL_SAYISI):
    puanli_populasyon = []
    for plan in populasyon:
        puan = calculate_fitness(plan)
        puanli_populasyon.append((puan, plan))

    puanli_populasyon.sort(key=lambda x: x[0], reverse=True)

    en_iyi_puan = puanli_populasyon[0][0]
    print(f"Nesil {nesil}: En iyi puan = {en_iyi_puan:.2f}")

    yeni_populasyon = []

    elit_sayisi = int(POPULASYON_BUYUKLUGU * ELITIZM_ORANI)
    elitler = puanli_populasyon[:elit_sayisi]

    for puan, plan in elitler:
        yeni_populasyon.append(plan)

    kalan_sayi = POPULASYON_BUYUKLUGU - elit_sayisi

    for _ in range(kalan_sayi):
        parent1 = random.choice(
            puanli_populasyon[:POPULASYON_BUYUKLUGU // 2])[1]
        parent2 = random.choice(
            puanli_populasyon[:POPULASYON_BUYUKLUGU // 2])[1]

        child = crossover(parent1, parent2)

        child = mutate(child, rate=MUTASYON_ORANI,
                       amount=MUTASYON_ETKI_MIKTARI)

        yeni_populasyon.append(child)

    populasyon = yeni_populasyon


puanli_populasyon = []
for plan in populasyon:
    puanli_populasyon.append((calculate_fitness(plan), plan))

puanli_populasyon.sort(key=lambda x: x[0], reverse=True)

en_iyi_plan = puanli_populasyon[0][1]
en_iyi_puan = puanli_populasyon[0][0]

print(f"Bulunan En İyi Puan: {en_iyi_puan:.2f}")
print("En İyi Plan:")
print(en_iyi_plan)


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- GÖRSELLEŞTİRME FONKSİYONU ---
def visualize_room_plan(plan, title="En İyi Oda Yerleşimi"):
    fig, ax = plt.subplots(1, figsize=(ODA_GENISLIK * 1.5, ODA_DERINLIK * 1.5)) # Boyutu biraz büyüttük

    # Odanın sınırlarını çiz
    ax.add_patch(patches.Rectangle((0, 0), ODA_GENISLIK, ODA_DERINLIK,
                                   edgecolor='black', facecolor='lightgray', linewidth=2))

    # Kapı alanını çiz
    ax.add_patch(patches.Rectangle((KAPI_ALANI["x1"], KAPI_ALANI["y1"]),
                                   KAPI_ALANI["x2"] - KAPI_ALANI["x1"],
                                   KAPI_ALANI["y2"] - KAPI_ALANI["y1"],
                                   edgecolor='blue', facecolor='skyblue', linestyle='--', linewidth=1))
    ax.text(KAPI_ALANI["x1"] + 0.1, KAPI_ALANI["y1"] + 0.1, "Kapı", color='blue', fontsize=8)


    # --- YENİ KOD: Petek Alanını Çiz ---
    ax.add_patch(patches.Rectangle((PETEK_ALANI["x1"], PETEK_ALANI["y1"]),
                                   PETEK_ALANI["x2"] - PETEK_ALANI["x1"],
                                   PETEK_ALANI["y2"] - PETEK_ALANI["y1"],
                                   edgecolor='red', facecolor='mistyrose', linestyle='--', linewidth=1, alpha=0.8))
    ax.text(PETEK_ALANI["x1"] + 0.1, PETEK_ALANI["y1"] + 0.1, "Petek", color='red', fontsize=8)
    
    # --- YENİ KOD: Cam Alanını Çiz ---
    ax.add_patch(patches.Rectangle((CAM_ALANI["x1"], CAM_ALANI["y1"]),
                                   CAM_ALANI["x2"] - CAM_ALANI["x1"],
                                   CAM_ALANI["y2"] - CAM_ALANI["y1"],
                                   edgecolor='cyan', facecolor='lightcyan', linestyle='-', linewidth=2, alpha=0.7))
    ax.text(CAM_ALANI["x1"] + 0.1, CAM_ALANI["y1"] - 0.2, "Cam", color='darkcyan', fontsize=8)

    # Mobilyaları çiz
    for mob_dict in plan["mobilyalar"]:
        mob_koor = get_mobilya_koordinatlari(mob_dict) # Yeni mob_koor fonksiyonu
        
        # Mobilya için bir dikdörtgen yaması oluştur
        rect = patches.Rectangle((mob_koor["x1"], mob_koor["y1"]), 
                                 mob_koor["x2"] - mob_koor["x1"], 
                                 mob_koor["y2"] - mob_koor["y1"],
                                 edgecolor='darkgreen', facecolor='lightgreen', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        
        # Mobilya adını ortasına yaz
        center_x = (mob_koor["x1"] + mob_koor["x2"]) / 2
        center_y = (mob_koor["y1"] + mob_koor["y2"]) / 2
        ax.text(center_x, center_y, mob_dict["ad"].capitalize(), 
                color='darkgreen', weight='bold', fontsize=10, ha='center', va='center')

    # Eksen etiketleri ve başlık
    ax.set_xlabel("X (metre)")
    ax.set_ylabel("Y (metre)")
    ax.set_title(title)
    
    # Eksen limitlerini ayarla (oda boyutundan biraz daha geniş)
    ax.set_xlim(-0.5, ODA_GENISLIK + 0.5)
    ax.set_ylim(-0.5, ODA_DERINLIK + 0.5)
    
    # Oranları koru
    ax.set_aspect('equal', adjustable='box')
    
    # Izgara ekle
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.show()


visualize_room_plan(en_iyi_plan, f"Optimize Edilmiş Oda Planı (Puan: {en_iyi_puan:.2f})")