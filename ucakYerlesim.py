# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:09 2025

@author: ismai
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

TARGET_CG_X=120.0
TARGET_CG_Y=0.0
TARGET_CG_Z=0.0

MAX_YAKIT_AGIRLIGI = 35.0 # Tank tam doluyken eklenecek ekstra ağırlık
DOLULUK_ORANLARI = [1.0, 0.0] # 1.0 = Dolu, 0.0 = Boş

TITRESIM_LIMITI = 80.0 # cm (Hassas parçalar motora bundan daha yakın olmamalı)

# (titresim_hassasiyeti eklendi)
KOMPONENTLER_DB = [
    # Motor titreşim kaynağıdır, hassas değildir (False)
    {"id": "Motor",       "agirlik": 80.0, "boyut": (60, 40, 40), "sabit_bolge": "BURUN", "titresim_hassasiyeti": False}, 
    {"id": "Batarya_Ana", "agirlik": 15.0, "boyut": (20, 15, 10), "sabit_bolge": "SERBEST", "titresim_hassasiyeti": False},
    # Aviyonikler hassastır (True)
    {"id": "Aviyonik_1",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "SERBEST", "titresim_hassasiyeti": True},
    {"id": "Aviyonik_2",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "SERBEST", "titresim_hassasiyeti": True},
    {"id": "Yakit_Tanki", "agirlik": 40.0, "boyut": (50, 40, 30), "sabit_bolge": "MERKEZ", "titresim_hassasiyeti": False},
    {"id": "Servo_Kuyruk","agirlik": 2.0,  "boyut": (5, 5, 5),    "sabit_bolge": "KUYRUK", "titresim_hassasiyeti": False},
    # Kamera görüntü titrememeli, hassastır (True)
    {"id": "Payload_Kam", "agirlik": 10.0, "boyut": (20, 20, 20), "sabit_bolge": "ON_ALT", "titresim_hassasiyeti": True}, 
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

#Gövdeden taşma kontrolü
def govde_icinde_mi(pos,dim):
    x_ok=(pos[0]-dim[0]/2 >=0) and (pos[0]+dim[0]/2 <=GOVDE_UZUNLUK)
    
    dist_y=abs(pos[1])+dim[1]/2
    dist_z=abs(pos[2])+dim[2]/2
    radial_ok=(dist_y**2 + dist_z**2)**0.5 <=GOVDE_YARICAP
    
    return x_ok and radial_ok
    
#Genetik alg. sınıfları

class TasarimBireyi:
    def __init__(self):
        self.yerlesim = {}
        
    def rastgele_yerlestir(self):
        for komp in KOMPONENTLER_DB:
            bolge=komp["sabit_bolge"]
            
            if bolge=="BURUN":
                x=random.uniform(0,40)
            elif bolge=="KUYRUK":
                x=random.uniform(GOVDE_UZUNLUK-40,GOVDE_UZUNLUK)
            elif bolge=="MERKEZ":
                x=random.uniform(TARGET_CG_X-30, TARGET_CG_X+30)
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
    
    # 1. Çakışma Cezası
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
    
    # 2. Gövdeden Taşma Cezası
    tasma_sayisi=0
    for k_id,pos in birey.yerlesim.items():
        dim=next(item for item in KOMPONENTLER_DB if item["id"]==k_id)["boyut"]
        if not govde_icinde_mi(pos, dim):
            tasma_sayisi+=1
            
    puan-=tasma_sayisi*5000
    
    # YENİ EKLENEN: TİTREŞİM KONTROLÜ ---
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
                puan -= (ihlâl ** 2) * 50 # Karesel ceza uyguluyoruz ki hızla uzaklaşsın

    # 4. CG (Ağırlık Merkezi) Hesabı
    toplam_cg_hatasi = 0

    # Her bir doluluk senaryosu için ayrı CG hesapla
    for doluluk in DOLULUK_ORANLARI:
        total_mass = 0
        moment_x = 0
        moment_y = 0
        moment_z = 0
    
        for k_id, pos in birey.yerlesim.items():
            db_item = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)
            mass = db_item["agirlik"]

            # Yakıt tankı ise doluluk oranına göre ağırlık ekle
            if k_id == "Yakit_Tanki":
                mass += MAX_YAKIT_AGIRLIGI * doluluk

            total_mass += mass
            moment_x += mass * pos[0]
            moment_y += mass * pos[1]
            moment_z += mass * pos[2]
        
        cg_x = moment_x / total_mass
        cg_y = moment_y / total_mass
        cg_z = moment_z / total_mass
        
        # Hedef CG'ye olan mesafe hatası
        dist_error = ((cg_x - TARGET_CG_X)**2 + (cg_y - TARGET_CG_Y)**2 + (cg_z - TARGET_CG_Z)**2)**0.5
        toplam_cg_hatasi += dist_error

    # Ortalama hatayı puandan düş (Ceza yöntemi)
    puan -= (toplam_cg_hatasi / len(DOLULUK_ORANLARI)) * 1000

    return puan, (cg_x, cg_y, cg_z)


#genetik işlemler
def crossover_design(parent1, parent2):
    child = TasarimBireyi()
    # Her komponent için ebeveynlerden birini seç
    for k_id in KOMPONENTLER_DB:
        key = k_id["id"]
        if random.random() < 0.5:
            child.yerlesim[key] = parent1.yerlesim[key]
        else:
            child.yerlesim[key] = parent2.yerlesim[key]
    return child

def mutate_design(birey, rate=0.1):
    for k_id in birey.yerlesim:
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
        print(f"Nesil {gen}: Puan {best_score:.0f} | CG X: {best_cg[0]:.1f} (Hedef: {TARGET_CG_X})")
        
    yeni_pop=[x[1]for x in puanli_pop[:10]]
    
    while len(yeni_pop)<POP_SIZE:
        parent1=random.choice(puanli_pop[:30])[1]
        parent2=random.choice(puanli_pop[:30])[1]
        child=crossover_design(parent1, parent2)
        child=mutate_design(child)
        yeni_pop.append(child)
        
    populasyon=yeni_pop
    
en_iyi_tasarim=puanli_pop[0][1]
# Analiz: Yakıt boşalırken CG ne kadar oynuyor?
tank_pos = en_iyi_tasarim.yerlesim["Yakit_Tanki"]
print(f"⛽ Yakıt Tankı Konumu: {tank_pos[0]:.1f} cm")

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
    # Burun sivri başlar, kabinde genişler, kuyrukta incelir
    # Matematiksel bir "sigara" şekli oluşturuyoruz
    def r_func(x):
        if x < 50: return (x/50)**0.5 * GOVDE_YARICAP  # Burun kavisli
        elif x < 180: return GOVDE_YARICAP             # Kabin düz
        else: return (1 - (x-180)/(GOVDE_UZUNLUK-180)) * GOVDE_YARICAP * 0.8 # Kuyruk
    
    r_values = np.array([r_func(x) for x in v.flatten()]).reshape(v.shape)
    
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

print("\n--- YERLEŞİM DETAYLARI ---")
motor_pos = en_iyi_tasarim.yerlesim["Motor"] # Motor referansı

for k_id, pos in en_iyi_tasarim.yerlesim.items():
    # Boyut bilgisini DB'den çek
    db_item = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)
    boyut = db_item["boyut"]
    
    # Kutuyu çiz
    fig.add_trace(parca_kutusu_ciz(pos, boyut, colors[idx % len(colors)], k_id))
    
    # Etiket ekle (Havada asılı yazı)
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2] + boyut[2]/1.5], # Kutunun biraz üstüne
        mode='text', text=[k_id], textposition="top center",
        textfont=dict(size=10, color="black"), showlegend=False
    ))
    
    # Motora Uzaklığı Yazdır (Görsel kontrol için)
    dist_motor = ((pos[0]-motor_pos[0])**2 + (pos[1]-motor_pos[1])**2 + (pos[2]-motor_pos[2])**2)**0.5
    uyari = " (!)" if db_item["titresim_hassasiyeti"] and dist_motor < TITRESIM_LIMITI else ""
    print(f"📍 {k_id}: X={pos[0]:.1f} | Motora Uzaklık: {dist_motor:.1f} cm {uyari}")
    
    idx += 1

print("\n--- DENGE ANALİZİ ---")
# En iyi birey için Dolu ve Boş durumları tekrar hesaplayalım
tank_yer = en_iyi_tasarim.yerlesim["Yakit_Tanki"]
cg_dolu = calculate_fitness_design(en_iyi_tasarim)[1] # Bu aslında son değeri döndürdüğü için yanıltabilir, manuel hesaplayalım:

# Manuel doğrulama fonksiyonu (Hızlıca)
def get_cg_for_fuel(birey, fuel_ratio):
    t_mass, m_x = 0, 0
    for k_id, pos in birey.yerlesim.items():
        mass = next(i for i in KOMPONENTLER_DB if i["id"]==k_id)["agirlik"]
        if k_id == "Yakit_Tanki": mass += MAX_YAKIT_AGIRLIGI * fuel_ratio
        t_mass += mass
        m_x += mass * pos[0]
    return m_x / t_mass

cg_dolu_x = get_cg_for_fuel(en_iyi_tasarim, 1.0)
cg_bos_x = get_cg_for_fuel(en_iyi_tasarim, 0.0)

print(f"Yakit Tanki Konumu (X): {tank_yer[0]:.2f} cm")
print(f"CG (Dolu Depo)        : {cg_dolu_x:.2f} cm")
print(f"CG (Bos Depo)         : {cg_bos_x:.2f} cm")
print(f"CG Kaymasi (Drift)    : {abs(cg_dolu_x - cg_bos_x):.2f} cm")

if abs(cg_dolu_x - cg_bos_x) < 2.0:
    print("✅ MÜKEMMEL SONUÇ: Yakıt tüketimi dengeyi bozmuyor!")
else:
    print("⚠️ DİKKAT: Yakıt tüketimi dengeyi etkiliyor.")

# 3. Ağırlık Merkezi (CG) Göstergeleri
# Hedef CG
fig.add_trace(go.Scatter3d(
    x=[TARGET_CG_X], y=[TARGET_CG_Y], z=[TARGET_CG_Z],
    mode='markers+text', marker=dict(size=8, color='red', symbol='cross'),
    name='HEDEF CG', text=["HEDEF"], textposition="bottom center"
))

# Hesaplanan (Sonuç) CG
fig.add_trace(go.Scatter3d(
    x=[best_cg[0]], y=[best_cg[1]], z=[best_cg[2]],
    mode='markers+text', marker=dict(size=12, color='black', symbol='diamond'),
    name='SONUÇ CG', text=["SONUÇ"], textposition="top center"
))

# Çizgi Çek (Hata payını görselleştirmek için)
fig.add_trace(go.Scatter3d(
    x=[TARGET_CG_X, best_cg[0]], y=[TARGET_CG_Y, best_cg[1]], z=[TARGET_CG_Z, best_cg[2]],
    mode='lines', line=dict(color='red', width=4, dash='dot'), name='CG Hatası'
))

# --- AYARLAR VE SAHNE DÜZENİ ---
camera = dict(
    eye=dict(x=2.0, y=-2.0, z=1.0) # Kamerayı çaprazdan baktır
)

fig.update_layout(
    title="Ön Tasarım: Uçak İçi Sistem Yerleşimi Optimizasyonu (Titreşim Korumalı)",
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