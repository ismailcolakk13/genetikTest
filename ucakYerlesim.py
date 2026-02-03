# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:09 2025

@author: ismai
"""
# MATEMATIKSEL KONVANSIYONA GORE HESAPLAMALAR YAPILDI

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

KOMPONENTLER_DB = [
    {"id": "Motor",       "agirlik": 80.0, "boyut": (60, 40, 40), "sabit_bolge": "BURUN"}, # Motor önde olur
    {"id": "Batarya_Ana", "agirlik": 15.0, "boyut": (20, 15, 10), "sabit_bolge": "SERBEST"},
    {"id": "Aviyonik_1",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "SERBEST"},
    {"id": "Aviyonik_2",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "SERBEST"},
    {"id": "Yakit_Tanki", "agirlik": 40.0, "boyut": (50, 40, 30), "sabit_bolge": "MERKEZ"}, # Genellikle CG yakını
    {"id": "Servo_Kuyruk","agirlik": 2.0,  "boyut": (5, 5, 5),    "sabit_bolge": "KUYRUK"},
    {"id": "Payload_Kam", "agirlik": 10.0, "boyut": (20, 20, 20), "sabit_bolge": "ON_ALT"}, # Kamera altta olur
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
    
    if x < 50: 
        # Burun kısmı (Parabolik artış)
        return (x/50)**0.5 * GOVDE_YARICAP
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
    # Gövde kesiti X'e göre değiştiği için, parçanın
    # hem başı hem sonu hem de ortası gövde sınırları içinde kalmalı.
    
    # Parçanın kesit köşegeni (Merkezden en uzak nokta)
    # Eğer bu mesafe izin verilen yarıçaptan küçükse parça sığar.
    # Not: Kare/Dikdörtgen kesit varsayımıyla köşegen alıyoruz.
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
            bolge=komp["sabit_bolge"]
            
            if bolge=="BURUN":
                x=random.uniform(0,40)
            elif bolge=="KUYRUK":
                x=random.uniform(GOVDE_UZUNLUK-40,GOVDE_UZUNLUK)
            elif bolge=="MERKEZ":
                center_x = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2
                x=random.uniform(center_x-30, center_x+30)
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
    
    #cg hesaplama
    total_mass=0
    moment_x=0
    moment_y=0
    moment_z=0
    
    for k_id,pos in birey.yerlesim.items():
        mass=next(item for item in KOMPONENTLER_DB if item["id"]==k_id)["agirlik"]
        total_mass+=mass
        moment_x+=mass*pos[0]
        moment_y+=mass*pos[1]
        moment_z+=mass*pos[2]
        
    cg_x=moment_x/total_mass
    cg_y=moment_y/total_mass
    cg_z=moment_z/total_mass
    
    # CG X ekseninde aralık kontrolü
    if cg_x < TARGET_CG_X_MIN:
        dx = TARGET_CG_X_MIN - cg_x
    elif cg_x > TARGET_CG_X_MAX:
        dx = cg_x - TARGET_CG_X_MAX
    else:
        dx = 0.0 # Aralık içindeyse X hatası yok

    dist_error=(dx**2 + (cg_y-TARGET_CG_Y)**2 + (cg_z-TARGET_CG_Z)**2)**0.5
    
    puan-=dist_error*100
    
    return puan,(cg_x,cg_y,cg_z)


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

print("\n--- YERLEŞİM DETAYLARI ---")
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
fig.add_trace(go.Mesh3d(
    x=[TARGET_CG_X_MIN, TARGET_CG_X_MAX, TARGET_CG_X_MAX, TARGET_CG_X_MIN, TARGET_CG_X_MIN, TARGET_CG_X_MAX, TARGET_CG_X_MAX, TARGET_CG_X_MIN],
    y=[-5, -5, 5, 5, -5, -5, 5, 5],
    z=[-5, -5, -5, -5, 5, 5, 5, 5],
    color='gold', opacity=0.4, name='HEDEF CG ARALIĞI',
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