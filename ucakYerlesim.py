# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:09 2025

@authors: İsmail Çolak, Mehmet Can Çalışkan, Yusuf Eren Aykurt
"""
import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np

GOVDE_UZUNLUK=300.0 #CM (x ekseni)
GOVDE_CAP=60.0 # CM (y ve z ekseni genişliği)
GOVDE_YARICAP=GOVDE_CAP/2

TARGET_CG_X_MIN=110.0
TARGET_CG_X_MAX=130.0
TARGET_CG_Y=0.0
TARGET_CG_Z=0.0

DOLULUK_ORANLARI = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_YAKIT_AGIRLIGI = 50.0
TITRESIM_LIMITI = 50.0

# BÖLGE TANIMLARI (Sınırlar)
BOLGE_BURUN_SON = 40.0
BOLGE_KUYRUK_BAS = GOVDE_UZUNLUK - 40.0 # 260.0'dan sonrası kuyruk ucu

KOMPONENTLER_DB = [
    {"id": "Motor",       "agirlik": 40.0, "boyut": (60, 40, 40), "sabit_bolge": "BURUN", "sabit_pos": (30, 0, 0), "kilitli": True, "titresim_hassasiyeti": False},
    {"id": "Batarya_Ana", "agirlik": 15.0, "boyut": (20, 15, 10), "sabit_bolge": "GOVDE", "kilitli": False, "titresim_hassasiyeti": False},
    {"id": "Aviyonik_1",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "GOVDE",  "kilitli": False, "titresim_hassasiyeti": True},
    {"id": "Aviyonik_2",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "GOVDE",  "kilitli": False, "titresim_hassasiyeti": True},
    {"id": "Payload_Kam", "agirlik": 10.0, "boyut": (20, 20, 20), "sabit_bolge": "ON_ALT", "kilitli": False, "titresim_hassasiyeti": True},
    {"id": "Yakit_Tanki", "agirlik": 40.0, "boyut": (50, 40, 30), "sabit_bolge": "MERKEZ", "kilitli": False, "titresim_hassasiyeti": False},
    {"id": "Servo_Kuyruk","agirlik": 2.0,  "boyut": (5, 5, 5),    "sabit_bolge": "KUYRUK", "kilitli": False, "titresim_hassasiyeti": False},
]

#Çakışma kontrolü
def kutular_cakisiyor_mu(pos1,dim1,pos2,dim2):
    min1=[pos1[0]-dim1[0]/2,pos1[1]-dim1[1]/2,pos1[2]-dim1[2]/2]
    max1=[pos1[0]+dim1[0]/2,pos1[1]+dim1[1]/2,pos1[2]+dim1[2]/2]
    
    min2=[pos2[0]-dim2[0]/2,pos2[1]-dim2[1]/2,pos2[2]-dim2[2]/2]
    max2=[pos2[0]+dim2[0]/2,pos2[1]+dim2[1]/2,pos2[2]+dim2[2]/2]
    
    return (
        min1[0]<max2[0] and max1[0]>min2[0] and
        min1[1]<max2[1] and max1[1]>min2[1] and
        min1[2]<max2[2] and max1[2]>min2[2]
        )

# Gövde Geometrisi Tanımı (Genelleştirilebilir Yapı)
def get_fuselage_radius(x):
    """
    Verilen X konumundaki gövde yarıçapını döndürür.
    Farklı uçak tipleri için bu fonksiyon değiştirilebilir.
    Şu anki model: Burun kavisli, orta düz, kuyruk incelen.
    """
    if x < 0: return 0.0
    if x > GOVDE_UZUNLUK: return 0.0
    
    if x < BOLGE_BURUN_SON: 
        # Burun kısmı (Parabolik artış)
        return (x/BOLGE_BURUN_SON)**0.5 * GOVDE_YARICAP
    elif x < 180:  
        # Orta gövde (Sabit silindir)
        return GOVDE_YARICAP
    else: 
        # Kuyruk kısmı (Lineer incelme)
        # 180'den 300'e giderken yarıçap %100'den %20'ye düşüyor
        ratio = (x - 180) / (GOVDE_UZUNLUK - 180)
        return GOVDE_YARICAP * (1 - ratio * 0.8)

#Gövdeden taşma kontrolü (Genelleştirilmiş)
def govde_icinde_mi(pos, dim):
    x, y, z = pos
    dx, dy, dz = dim
    
    # 1. Boylamasına (X ekseni) kontrol
    x_min = x - dx/2
    x_max = x + dx/2
    
    if x_min < 0 or x_max > GOVDE_UZUNLUK:
        return False
    
    # 2. Radyal (Kesit) kontrolü
    
    # Parçanın kesit köşegeni (Merkezden en uzak nokta)
    part_radial_dist = ((abs(y) + dy/2)**2 + (abs(z) + dz/2)**2)**0.5
    
    # Kontrol edilecek noktalar: Ön, Orta, Arka
    check_points = [x_min, x, x_max]
    
    for cx in check_points:
        allowed_radius = get_fuselage_radius(cx)
        if part_radial_dist > allowed_radius:
            return False
            
    return True
    
#Genetik alg. sınıfları

class TasarimBireyi:
    def __init__(self):
        self.yerlesim = {}
        
    def rastgele_yerlestir(self):
        for komp in KOMPONENTLER_DB:
            # Eğer parça kilitliyse sabit pozisyonunu al ve geç
            if komp.get("kilitli", False):
                self.yerlesim[komp["id"]] = komp["sabit_pos"]
                continue
            
            bolge=komp["sabit_bolge"]
            
            if bolge=="BURUN":
                x=random.uniform(0,BOLGE_BURUN_SON)
            elif bolge=="KUYRUK":
                x=random.uniform(BOLGE_KUYRUK_BAS,GOVDE_UZUNLUK)
            elif bolge=="MERKEZ":
                center_x = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2
                x=random.uniform(center_x-30, center_x+30)
            elif bolge=="GOVDE":
                # Burun ile Kuyruk arasındaki ana hacim
                x=random.uniform(BOLGE_BURUN_SON, BOLGE_KUYRUK_BAS)
            else:
                x=random.uniform(0, GOVDE_UZUNLUK)
                
            y=random.uniform(-GOVDE_YARICAP/2,GOVDE_YARICAP/2)
            
            if bolge=="ON_ALT":
                z=-GOVDE_YARICAP/2
            else:
                z=random.uniform(-GOVDE_YARICAP/2, GOVDE_YARICAP/2)
                
            self.yerlesim[komp["id"]]=(x,y,z)
            
def calculate_fitness_design(birey):
    puan=0
    
    #çakışma
    cakisma_sayisi=0
    keys=list(birey.yerlesim.keys())
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            k1_id=keys[i]
            k2_id=keys[j]
            
            dim1=next(item for item in KOMPONENTLER_DB if item["id"]==k1_id)["boyut"]
            dim2=next(item for item in KOMPONENTLER_DB if item["id"]==k2_id)["boyut"]
            
            pos1=birey.yerlesim[k1_id]
            pos2=birey.yerlesim[k2_id]
            
            if(kutular_cakisiyor_mu(pos1, dim1, pos2, dim2)):
                cakisma_sayisi+=1
    
    puan-=cakisma_sayisi*10000
    
    #gövdeden taşma
    tasma_sayisi=0
    for k_id,pos in birey.yerlesim.items():
        dim=next(item for item in KOMPONENTLER_DB if item["id"]==k_id)["boyut"]
        if not govde_icinde_mi(pos, dim):
            tasma_sayisi+=1
            
    puan-=tasma_sayisi*5000
    
    # TİTREŞİM KONTROLÜ
    # Motoru bul (Titreşim kaynağı)
    pos_motor = birey.yerlesim["Motor"] 
    
    for k_id, pos in birey.yerlesim.items():
        # DB'den parça özelliklerini çek
        parca_db = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)
        
        # Eğer parça hassassa kontrol et
        if parca_db.get("titresim_hassasiyeti") == True:
            # Motora olan mesafeyi hesapla
            mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
            
            # Limitten yakınsa ceza kes
            if mesafe < TITRESIM_LIMITI:
                ihlâl = TITRESIM_LIMITI - mesafe
                puan -= (ihlâl ** 2) * 50

    # 4. CG (Ağırlık Merkezi) Hesabı
    toplam_cg_hatasi = 0

    dolu_cg_coords = (0,0,0)
    # Her bir doluluk senaryosu için ayrı CG hesapla
    for doluluk in DOLULUK_ORANLARI:
        total_mass = 0
        moment_x = 0
        moment_y = 0
        moment_z = 0
    
        for k_id, pos in birey.yerlesim.items():
            db_item = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)
            mass = db_item["agirlik"]

            if k_id == "Yakit_Tanki":
                mass += MAX_YAKIT_AGIRLIGI * doluluk

            total_mass += mass
            moment_x += mass * pos[0]
            moment_y += mass * pos[1]
            moment_z += mass * pos[2]
        
        cg_x = moment_x / total_mass
        cg_y = moment_y / total_mass
        cg_z = moment_z / total_mass
        

    # Eğer doluluk 1.0 ise bu koordinatları raporlama için sakla
        if doluluk == 1.0:
            dolu_cg_coords = (cg_x, cg_y, cg_z)

        target_x_center = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2

        # Hedef CG'ye olan mesafe hatası
        dist_error = ((cg_x - target_x_center)**2 + (cg_y - TARGET_CG_Y)**2 + (cg_z - TARGET_CG_Z)**2)**0.5
        toplam_cg_hatasi += dist_error

    # Ortalama hatayı puandan düş (Ceza yöntemi)
    puan -= (toplam_cg_hatasi / len(DOLULUK_ORANLARI)) * 1000

    return puan, dolu_cg_coords
  
#genetik işlemler
def crossover_design(parent1, parent2):
    child = TasarimBireyi()
    for k_id in KOMPONENTLER_DB:
        key = k_id["id"]
        if random.random() < 0.5:
            child.yerlesim[key] = parent1.yerlesim[key]
        else:
            child.yerlesim[key] = parent2.yerlesim[key]
    return child

def mutate_design(birey, rate=0.1):
    for k_id in birey.yerlesim:
        # Kilitli parçaları mutasyona uğratma
        comp_info = next((item for item in KOMPONENTLER_DB if item["id"] == k_id), None)
        if comp_info and comp_info.get("kilitli", False):
            continue

        if random.random() < rate:
            x, y, z = birey.yerlesim[k_id]
            # Küçük kaydırma
            x += random.uniform(-10, 10)
            y += random.uniform(-5, 5)
            z += random.uniform(-5, 5)
            birey.yerlesim[k_id] = (x, y, z)
    return birey

#simülasyon
POP_SIZE=100
GENERATIONS=50
populasyon=[]

for _ in range(POP_SIZE):
    b=TasarimBireyi()
    b.rastgele_yerlestir()
    populasyon.append(b)
    
print("optimizasyon başlıyor...")

best_cg=(0,0,0)

for gen in range(GENERATIONS):
    puanli_pop=[]
    for ind in populasyon:
        score,cg=calculate_fitness_design(ind)
        puanli_pop.append((score,ind,cg))
        
    puanli_pop.sort(key=lambda x:x[0],reverse=True)
    
    best_score=puanli_pop[0][0]
    best_cg=puanli_pop[0][2]
    
    if gen%10==0:
        print(f"Nesil {gen}: Puan {best_score:.0f} | CG X: {best_cg[0]:.1f} (Hedef: {TARGET_CG_X_MIN}-{TARGET_CG_X_MAX})")
        
    yeni_pop=[x[1]for x in puanli_pop[:10]]
    
    while len(yeni_pop)<POP_SIZE:
        parent1=random.choice(puanli_pop[:30])[1]
        parent2=random.choice(puanli_pop[:30])[1]
        child=crossover_design(parent1, parent2)
        child=mutate_design(child)
        yeni_pop.append(child)
        
    populasyon=yeni_pop
    
en_iyi_tasarim=puanli_pop[0][1]

#3d görsel
def kutu_ciz(pos, dim, color, name):
    # Plotly için bir kutunun köşe noktalarını ve yüzeylerini oluşturur
    x, y, z = pos
    dx, dy, dz = dim
    
    # 8 Köşe noktası
    x_vals = [x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2]
    y_vals = [y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2]
    z_vals = [z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2]
    
    return go.Mesh3d(
        x=x_vals, y=y_vals, z=z_vals,
        color=color,
        opacity=0.8,
        alphahull=0, # Convex hull kullanarak kutu oluşturur
        name=name,
        hoverinfo='name'
    )

def ucak_govdesi_olustur():
    """
    Cessna benzeri basit bir uçak geometrisi (Mesh) oluşturur.
    Gövde Uzunluğu: 300 birim (Global değişkenden alınır)
    """
    traces = []
    
    # 1. GÖVDE (FUSELAGE) - Aerodinamik bir tüp
    # Burun (0) -> Kabin (50-150) -> Kuyruk (300)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, GOVDE_UZUNLUK, 40)
    u, v = np.meshgrid(u, v)
    
    # Gövde şeklini belirleyen yarıçap fonksiyonu (Tapering)
    # Artık merkezi fonksiyonu kullanıyoruz
    
    r_values = np.array([get_fuselage_radius(x) for x in v.flatten()]).reshape(v.shape)
    
    x_govde = v
    y_govde = r_values * np.cos(u)
    z_govde = r_values * np.sin(u) * 1.2 # Yükseklik biraz daha eliptik olsun
    
    # Gövdeyi Yarı Şeffaf Çiz
    traces.append(go.Surface(
        x=x_govde, y=y_govde, z=z_govde,
        opacity=0.15, colorscale='Greys', showscale=False, name='Gövde', hoverinfo='skip'
    ))
    
    # Gövde Tel Kafes (Wireframe) çizgileri (Daha teknik görünüm için)
    # Sadece belli kesitleri çiz
    for i in range(0, 40, 4): 
        traces.append(go.Scatter3d(
            x=x_govde[i], y=y_govde[i], z=z_govde[i],
            mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'
        ))

    # 2. KANATLAR (WINGS) - High Wing Cessna Tipi
    kanat_x_bas = 80
    kanat_genislik = 40 # Chord
    kanat_uzunluk = 360 # Span (Gövdeden taşan)
    z_kanat = GOVDE_YARICAP * 1.1 # Gövdenin üstünde
    
    x_w = [kanat_x_bas, kanat_x_bas+kanat_genislik, kanat_x_bas+kanat_genislik, kanat_x_bas]
    y_w = [-kanat_uzunluk/2, -kanat_uzunluk/2, kanat_uzunluk/2, kanat_uzunluk/2]
    z_w = [z_kanat, z_kanat, z_kanat, z_kanat]
    
    traces.append(go.Mesh3d(
        x=x_w, y=y_w, z=z_w,
        color='lightblue', opacity=0.5, name='Kanat',
        i=[0, 0], j=[1, 2], k=[2, 3] # Yüzey örme indeksleri
    ))

    # 3. KUYRUK TAKIMI (TAIL)
    # Yatay Stabilize
    tail_x = GOVDE_UZUNLUK - 40
    h_stab_span = 120
    x_h = [tail_x, GOVDE_UZUNLUK, GOVDE_UZUNLUK, tail_x]
    y_h = [-h_stab_span/2, -h_stab_span/2, h_stab_span/2, h_stab_span/2]
    z_h = [0, 0, 0, 0]
    
    traces.append(go.Mesh3d(
        x=x_h, y=y_h, z=z_h, color='lightblue', opacity=0.5, name='Yatay Kuyruk',
        i=[0, 0], j=[1, 2], k=[2, 3]
    ))
    
    # Dikey Stabilize (Rudder)
    x_v = [tail_x, GOVDE_UZUNLUK, GOVDE_UZUNLUK, tail_x+10]
    y_v = [0, 0, 0, 0]
    z_v = [0, 0, 50, 50] # 50 birim yukarı
    
    traces.append(go.Mesh3d(
        x=x_v, y=y_v, z=z_v, color='lightblue', opacity=0.5, name='Dikey Kuyruk',
        i=[0, 0], j=[1, 2], k=[2, 3]
    ))

    return traces

def parca_kutusu_ciz(pos, dim, color, name):
    """Komponentleri katı kutular olarak çizer"""
    x, y, z = pos
    dx, dy, dz = dim
    
    # Küp Köşeleri
    x_k = [x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2]
    y_k = [y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2]
    z_k = [z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2]
    
    return go.Mesh3d(
        x=x_k, y=y_k, z=z_k,
        color=color, opacity=1.0, name=name,
        # Küp yüzey tanımları (index based)
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        hoverinfo='name'
    )

# --- GÖRSELLEŞTİRME BAŞLATIYORUZ ---
fig = go.Figure()

# 1. Uçak Gövdesini Çiz
ucak_parcalari = ucak_govdesi_olustur()
for parca in ucak_parcalari:
    fig.add_trace(parca)

# 2. Optimize Edilen Komponentleri Çiz
colors = ['red', 'blue', 'orange', 'purple', 'green', 'brown', 'cyan']
idx = 0
print("\n--- TASARIM ANALİZİ ---")

# 1. CG Hedefe Yakınlık Kontrolü
cg_x, cg_y, cg_z = best_cg
# X ekseninde hedef aralığa göre sapma hesabı
if cg_x < TARGET_CG_X_MIN:
    dx = TARGET_CG_X_MIN - cg_x
elif cg_x > TARGET_CG_X_MAX:
    dx = cg_x - TARGET_CG_X_MAX
else:
    dx = 0.0

# Toplam mesafe hatası (X aralığı, Y=0 ve Z=0 hedeflerine göre)
dist_error = (dx**2 + (cg_y - TARGET_CG_Y)**2 + (cg_z - TARGET_CG_Z)**2)**0.5

if dist_error < 2.0:
    print(f"✅ CG hedefe çok yakın (Sapma: {dist_error:.2f} cm)")
elif dist_error < 15.0:
    print(f"⚠️ CG hedefe orta mesafede (Sapma: {dist_error:.2f} cm)")
else:
    print(f"❌ CG hedeften uzak (Sapma: {dist_error:.2f} cm)")

# 2. Yakıt Tankı Etkisi Kontrolü
# Yakıt tankı ağırlık merkezinden (CG) ne kadar uzaksa, yakıt azaldıkça uçağın dengesi o kadar bozulur.
yakit_pos = en_iyi_tasarim.yerlesim.get("Yakit_Tanki", (0, 0, 0))
hedef_merkez_x = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2

if abs(yakit_pos[0] - hedef_merkez_x) > 10.0:
    print(f"⛽ Yakıt tankının X konumu ({yakit_pos[0]:.1f}) ideal merkezden uzak. Yakıt tüketimi CG'yi ETKİLEYECEK.")
else:
    print(f"⛽ Yakıt tankı ideal merkeze çok yakın. Yakıt tüketiminin dengeye etkisi MİNİMUM.")

# 3. Genel Skor Yorumu
# Ceza sistemi olduğu için skor 0'a ne kadar yakınsa (negatif değerler) o kadar iyidir.
if best_score > -4000:
    print(f"🏆 Tasarım çok iyi (Skor: {best_score:.0f})")
elif best_score > -6000:
    print(f"👍 Tasarım kabul edilebilir (Skor: {best_score:.0f})")
else:
    print(f"🚫 Tasarım zayıf (Skor: {best_score:.0f})")
print("\n--- DENGE ANALİZİ (CG DRIFT) ---")

# Denge Analizi Hesaplamaları (Sadece X ekseni için)
bos_agirlik = 0
bos_moment_x = 0
dolu_agirlik = 0
dolu_moment_x = 0

for k_id, pos in en_iyi_tasarim.yerlesim.items():
    db_item = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)
    mass = db_item["agirlik"]

    # Bos depo için moment (Yakıt = 0)
    bos_agirlik += mass
    bos_moment_x += mass * pos[0]

    # Dolu depo için moment (Yakıt = MAX)
    if k_id == "Yakit_Tanki":
        dolu_agirlik += (mass + MAX_YAKIT_AGIRLIGI)
        dolu_moment_x += (mass + MAX_YAKIT_AGIRLIGI) * pos[0]
    else:
        dolu_agirlik += mass
        dolu_moment_x += mass * pos[0]

cg_bos_x = bos_moment_x / bos_agirlik
cg_dolu_x = dolu_moment_x / dolu_agirlik
cg_kaymasi = abs(cg_dolu_x - cg_bos_x)

yakit_pos_x = en_iyi_tasarim.yerlesim.get("Yakit_Tanki", (0, 0, 0))[0]

print(f"Yakit Tanki Konumu (X): {yakit_pos_x:.2f} cm")
print(f"CG (Dolu Depo)        : {cg_dolu_x:.2f} cm")
print(f"CG (Bos Depo)         : {cg_bos_x:.2f} cm")
print(f"CG Kaymasi (Drift)    : {cg_kaymasi:.2f} cm")

# Uyarı Mekanizması
if cg_kaymasi > 5.0:
    print("❌ KRİTİK: Yakıt tüketimi CG'yi çok fazla kaydırıyor! Uçuş stabilitesi tehlikede.")
elif cg_kaymasi > 2.0:
    print("⚠️ DİKKAT: Yakıt tüketimi dengeyi etkiliyor. Trim ayarı gerekecek.")
else:
    print("✅ MÜKEMMEL: Yakıt tankı ideal konumda. Yakıt tüketiminin dengeye etkisi minimum.")
print("-----------------------\n")
print("\n--- YERLEŞİM DETAYLARI ---")
motor_pos = en_iyi_tasarim.yerlesim["Motor"] # Motor referansı

for k_id, pos in en_iyi_tasarim.yerlesim.items():
    # Boyut bilgisini DB'den çek
    boyut = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)["boyut"]
    
    # Kutuyu çiz
    fig.add_trace(parca_kutusu_ciz(pos, boyut, colors[idx % len(colors)], k_id))
    
    # Etiket ekle (Havada asılı yazı)
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2] + boyut[2]/1.5], # Kutunun biraz üstüne
        mode='text', text=[k_id], textposition="top center",
        textfont=dict(size=10, color="black"), showlegend=False
    ))
    
    print(f"📍 {k_id}: Gövde Başından {pos[0]:.1f} cm geride.")
    idx += 1

# 3. Ağırlık Merkezi (CG) Göstergeleri
# Hedef CG Aralığı (Altın Sarısı Yarı Şeffaf Kutu - Yakıt tankıyla karışmasın diye)
# Görünür olması için gövde çapından biraz daha geniş çiziyoruz.
box_r = GOVDE_YARICAP + 5 # Yarıçaptan 5cm daha geniş
fig.add_trace(go.Mesh3d(
    x=[TARGET_CG_X_MIN, TARGET_CG_X_MAX, TARGET_CG_X_MAX, TARGET_CG_X_MIN, TARGET_CG_X_MIN, TARGET_CG_X_MAX, TARGET_CG_X_MAX, TARGET_CG_X_MIN],
    y=[-box_r, -box_r, box_r, box_r, -box_r, -box_r, box_r, box_r],
    z=[-box_r, -box_r, -box_r, -box_r, box_r, box_r, box_r, box_r],
    color='gold', opacity=0.3, name='HEDEF CG ARALIĞI',
    alphahull=0
))

# Hesaplanan (Sonuç) CG - Görünürlük için yukarı taşıyoruz
viz_z = GOVDE_YARICAP + 40 # Gövdenin üstünde, her zaman görünür olması için

fig.add_trace(go.Scatter3d(
    x=[best_cg[0]], y=[best_cg[1]], z=[viz_z],
    mode='markers+text', marker=dict(size=12, color='black', symbol='diamond'),
    name='HESAPLANAN CG', text=["HESAPLANAN CG"], textposition="top center"
))

# Gerçek CG noktasına dikey çizgi (Drop line)
fig.add_trace(go.Scatter3d(
    x=[best_cg[0], best_cg[0]], y=[best_cg[1], best_cg[1]], z=[best_cg[2], viz_z],
    mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'
))

# Gerçek CG noktası (İçeride kalan küçük nokta)
fig.add_trace(go.Scatter3d(
    x=[best_cg[0]], y=[best_cg[1]], z=[best_cg[2]],
    mode='markers', marker=dict(size=5, color='black'), 
    name='Gerçek CG Konumu'
))

# Çizgi Çek (Hata payını görselleştirmek için - En yakın sınıra)
target_x_visual = best_cg[0]
if best_cg[0] < TARGET_CG_X_MIN: target_x_visual = TARGET_CG_X_MIN
elif best_cg[0] > TARGET_CG_X_MAX: target_x_visual = TARGET_CG_X_MAX

fig.add_trace(go.Scatter3d(
    x=[target_x_visual, best_cg[0]], y=[TARGET_CG_Y, best_cg[1]], z=[TARGET_CG_Z, best_cg[2]],
    mode='lines', line=dict(color='red', width=4, dash='dot'), name='CG Hatası'
))

# --- AYARLAR VE SAHNE DÜZENİ ---
camera = dict(
    eye=dict(x=2.0, y=-2.0, z=1.0) # Kamerayı çaprazdan baktır
)

fig.update_layout(
    title="Ön Tasarım: Uçak İçi Sistem Yerleşimi Optimizasyonu",
    scene=dict(
        xaxis=dict(title='Uzunluk (cm)', range=[0, GOVDE_UZUNLUK], backgroundcolor="rgb(240, 240, 240)"),
        yaxis=dict(title='Genişlik (cm)', range=[-200, 200]), # Kanatları kapsasın diye geniş
        zaxis=dict(title='Yükseklik (cm)', range=[-100, 100]),
        aspectmode='data', # Gerçek oranları koru (Uçak basık görünmesin)
        camera=camera
    ),
    margin=dict(r=0, l=0, b=0, t=50) # Kenar boşluklarını azalt
)

fig.show()