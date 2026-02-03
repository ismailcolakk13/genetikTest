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

# UÇAK GENEL ÖLÇÜLERİ (Cessna 172 Benzeri Oranlar)
GOVDE_UZUNLUK = 300.0 # CM (x ekseni)
# Referans maksimum genişlik/yükseklik (Rastgele üretim sınırları için)
MAX_GOVDE_YARI_GENISLIK = 30.0 # Y ekseni (Toplam 60cm)
MAX_GOVDE_YARI_YUKSEKLIK = 36.0 # Z ekseni (Toplam 72cm - Biraz daha yüksek kabin)

TARGET_CG_X_MIN = 110.0
TARGET_CG_X_MAX = 130.0
TARGET_CG_Y = 0.0
TARGET_CG_Z = 0.0

# BÖLGE TANIMLARI (Sınırlar)
BOLGE_BURUN_SON = 50.0 # Motor ve burun bombesi (Biraz uzattık)
BOLGE_KUYRUK_BAS = 180.0 # Kabin bitişi, incelme başlangıcı

KOMPONENTLER_DB = [
    # Motor KESİN SABİT (Locked).
    {"id": "Motor",       "agirlik": 80.0, "boyut": (60, 40, 40), "sabit_bolge": "BURUN", "sabit_pos": (30, 0, 0), "kilitli": True}, 
    {"id": "Batarya_Ana", "agirlik": 15.0, "boyut": (20, 15, 10), "sabit_bolge": "GOVDE", "kilitli": False},
    {"id": "Aviyonik_1",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "GOVDE",  "kilitli": False},
    {"id": "Aviyonik_2",  "agirlik": 5.0,  "boyut": (15, 15, 5),  "sabit_bolge": "GOVDE",  "kilitli": False},
    {"id": "Yakit_Tanki", "agirlik": 40.0, "boyut": (50, 40, 30), "sabit_bolge": "MERKEZ", "kilitli": False},
    {"id": "Servo_Kuyruk","agirlik": 2.0,  "boyut": (5, 5, 5),    "sabit_bolge": "KUYRUK", "kilitli": False},
    {"id": "Payload_Kam", "agirlik": 10.0, "boyut": (20, 20, 20), "sabit_bolge": "ON_ALT", "kilitli": False},
]

# Çakışma kontrolü
def kutular_cakisiyor_mu(pos1, dim1, pos2, dim2):
    min1 = [pos1[0]-dim1[0]/2, pos1[1]-dim1[1]/2, pos1[2]-dim1[2]/2]
    max1 = [pos1[0]+dim1[0]/2, pos1[1]+dim1[1]/2, pos1[2]+dim1[2]/2]
    
    min2 = [pos2[0]-dim2[0]/2, pos2[1]-dim2[1]/2, pos2[2]-dim2[2]/2]
    max2 = [pos2[0]+dim2[0]/2, pos2[1]+dim2[1]/2, pos2[2]+dim2[2]/2]
    
    return (
        min1[0] < max2[0] and max1[0] > min2[0] and
        min1[1] < max2[1] and max1[1] > min2[1] and
        min1[2] < max2[2] and max1[2] > min2[2]
        )

# GÖVDE GEOMETRİSİ (Aerodinamik / Superellipse Kesit)
def get_fuselage_section(x):
    """
    Daha modern ve akıcı bir gövde formu.
    Kesit: Superellipse (Kare ile Daire arası, 'Squircle')
    Profil: Sinüs dalgaları ile yumuşak geçişler.
    """
    if x < 0 or x > GOVDE_UZUNLUK: return (0.0, 0.0)
    
    # Geçiş Yumuşatma Faktörleri
    # 0 -> 1 arasında değerler üretir
    
    # 1. BURUN (0 - 60cm): Mermi şekli (Ellipsoid burun)
    if x < 60:
        # 0'dan 1'e giden sinüs eğrisi (Daha küt ama aerodinamik)
        ratio = np.sin((x / 60) * (np.pi / 2))
        ry = MAX_GOVDE_YARI_GENISLIK * ratio
        rz = MAX_GOVDE_YARI_YUKSEKLIK * ratio
        return (ry, rz)
        
    # 2. KABİN (60 - 180cm): Sabit genişlik, hafif bombeli
    elif x < 180:
        return (MAX_GOVDE_YARI_GENISLIK, MAX_GOVDE_YARI_YUKSEKLIK)
        
    # 3. KUYRUK (180 - 300cm): Akıcı sönümleme (Sigmoid/Cosine)
    else:
        # 180'de 1, 300'de ~0.2 olacak şekilde süzülme
        L_tail = GOVDE_UZUNLUK - 180
        pos = x - 180
        
        # Kosinüs ile yumuşak düşüş (Linear'den daha organik)
        # 0 radyan -> pi/2 radyan arası değil, 0 -> 1 arası azalma
        ratio = 0.5 * (1 + np.cos((pos / L_tail) * np.pi * 0.8)) # 0.8 ile tam kapatmıyoruz
        
        # Kuyruk ucunda minimum kalınlık kalsın (tam sıfırlanmasın)
        base_ry = 2.0
        base_rz = 4.0
        
        ry = (MAX_GOVDE_YARI_GENISLIK - base_ry) * ratio + base_ry
        rz = (MAX_GOVDE_YARI_YUKSEKLIK - base_rz) * ratio + base_rz
        
        return (ry, rz)

def superellipse_point(u, ry, rz, n=4):
    """
    Superellipse formülü: |x/a|^n + |y/b|^n = 1
    n=2 -> Daire/Elips
    n=4 -> Squircle (Yuvarlatılmış kare - Kabin için ideal)
    """
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    
    # Sign koruyarak üs alma
    x = ry * np.sign(cos_u) * (np.abs(cos_u) ** (2/n))
    y = rz * np.sign(sin_u) * (np.abs(sin_u) ** (2/n))
    return x, y

# Gövdeden taşma kontrolü (Superellipse Bazlı)
def govde_icinde_mi(pos, dim):
    x, y, z = pos
    dx, dy, dz = dim
    
    x_min, x_max = x - dx/2, x + dx/2
    if x_min < 0 or x_max > GOVDE_UZUNLUK: return False
    
    check_x_points = [x_min, x, x_max]
    check_y_offsets = [-dy/2, dy/2]
    check_z_offsets = [-dz/2, dz/2]
    
    # Kesit şekli parametresi (n)
    # Kabin kısmında (60-180) n=4 (karemsi), uçlarda n=2.5 (daha yuvarlak)
    # Basitlik için ortalama n=3.5 kullanabiliriz veya x'e göre değiştirebiliriz
    
    for cx in check_x_points:
        max_ry, max_rz = get_fuselage_section(cx)
        if max_ry <= 1.0: return False
        
        # Bölgeye göre şekil faktörü
        n_shape = 4.0 if (cx > 60 and cx < 180) else 2.5
        
        for off_y in check_y_offsets:
            for off_z in check_z_offsets:
                p_y = abs(y + off_y)
                p_z = abs(z + off_z)
                
                # Formül: (y/ry)^n + (z/rz)^n <= 1
                val = (p_y / max_ry)**n_shape + (p_z / max_rz)**n_shape
                if val > 1.0: return False
    return True
    
# Genetik alg. sınıfları
class TasarimBireyi:
    def __init__(self):
        self.yerlesim = {}
        
    def rastgele_yerlestir(self):
        for komp in KOMPONENTLER_DB:
            # Kilitli parça kontrolü
            if komp.get("kilitli", False):
                self.yerlesim[komp["id"]] = komp["sabit_pos"]
                continue
            
            bolge = komp["sabit_bolge"]
            
            # X Pozisyonu Seçimi
            if bolge == "BURUN":
                x = random.uniform(0, BOLGE_BURUN_SON)
            elif bolge == "KUYRUK":
                x = random.uniform(BOLGE_KUYRUK_BAS, GOVDE_UZUNLUK)
            elif bolge == "MERKEZ":
                center_x = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2
                x = random.uniform(center_x - 30, center_x + 30)
            elif bolge == "GOVDE":
                x = random.uniform(BOLGE_BURUN_SON, BOLGE_KUYRUK_BAS)
            else: # SERBEST
                x = random.uniform(0, GOVDE_UZUNLUK)
            
            # Y ve Z Pozisyonu (Kaba bir aralık, sonra optimize edilecek)
            y = random.uniform(-MAX_GOVDE_YARI_GENISLIK + 5, MAX_GOVDE_YARI_GENISLIK - 5)
            z = random.uniform(-MAX_GOVDE_YARI_YUKSEKLIK + 5, MAX_GOVDE_YARI_YUKSEKLIK - 5)
            
            # Özel: Kamera altta olsun
            if bolge == "ON_ALT":
                z = -MAX_GOVDE_YARI_YUKSEKLIK / 2
                
            self.yerlesim[komp["id"]] = (x, y, z)
            
def calculate_fitness_design(birey):
    puan = 0
    
    # 1. Çakışma Cezası
    cakisma_sayisi = 0
    keys = list(birey.yerlesim.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1_id = keys[i]
            k2_id = keys[j]
            
            dim1 = next(item for item in KOMPONENTLER_DB if item["id"] == k1_id)["boyut"]
            dim2 = next(item for item in KOMPONENTLER_DB if item["id"] == k2_id)["boyut"]
            
            pos1 = birey.yerlesim[k1_id]
            pos2 = birey.yerlesim[k2_id]
            
            if kutular_cakisiyor_mu(pos1, dim1, pos2, dim2):
                cakisma_sayisi += 1
    
    puan -= cakisma_sayisi * 10000
    
    # 2. Gövdeden Taşma Cezası
    tasma_sayisi = 0
    for k_id, pos in birey.yerlesim.items():
        dim = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)["boyut"]
        if not govde_icinde_mi(pos, dim):
            tasma_sayisi += 1
            
    puan -= tasma_sayisi * 10000 # Çakışma ile eşit ceza
    
    # 3. CG Hesaplama
    total_mass = 0
    moment_x = 0
    moment_y = 0
    moment_z = 0
    
    for k_id, pos in birey.yerlesim.items():
        mass = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)["agirlik"]
        total_mass += mass
        moment_x += mass * pos[0]
        moment_y += mass * pos[1]
        moment_z += mass * pos[2]
        
    cg_x = moment_x / total_mass
    cg_y = moment_y / total_mass
    cg_z = moment_z / total_mass
    
    # CG X ekseninde aralık kontrolü
    if cg_x < TARGET_CG_X_MIN:
        dx = TARGET_CG_X_MIN - cg_x
    elif cg_x > TARGET_CG_X_MAX:
        dx = cg_x - TARGET_CG_X_MAX
    else:
        dx = 0.0 # Aralık içindeyse X hatası yok

    dist_error = (dx**2 + (cg_y - TARGET_CG_Y)**2 + (cg_z - TARGET_CG_Z)**2)**0.5
    
    puan -= dist_error * 100
    
    return puan, (cg_x, cg_y, cg_z)


# Genetik Operatörler
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
        # Kilitli parçaları pass geç
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

# Simülasyon Döngüsü
POP_SIZE = 100
GENERATIONS = 50
populasyon = []

for _ in range(POP_SIZE):
    b = TasarimBireyi()
    b.rastgele_yerlestir()
    populasyon.append(b)
    
print("Cessna 172 Modeli Optimizasyon Başlıyor...")

best_cg = (0, 0, 0)
en_iyi_tasarim = None

for gen in range(GENERATIONS):
    puanli_pop = []
    for ind in populasyon:
        score, cg = calculate_fitness_design(ind)
        puanli_pop.append((score, ind, cg))
        
    puanli_pop.sort(key=lambda x: x[0], reverse=True)
    
    best_score = puanli_pop[0][0]
    best_cg = puanli_pop[0][2]
    
    if gen % 10 == 0:
        print(f"Nesil {gen}: Puan {best_score:.0f} | CG X: {best_cg[0]:.1f} (Hedef: {TARGET_CG_X_MIN}-{TARGET_CG_X_MAX})")
        
    yeni_pop = [x[1] for x in puanli_pop[:10]]
    
    while len(yeni_pop) < POP_SIZE:
        parent1 = random.choice(puanli_pop[:30])[1]
        parent2 = random.choice(puanli_pop[:30])[1]
        child = crossover_design(parent1, parent2)
        child = mutate_design(child)
        yeni_pop.append(child)
        
    populasyon = yeni_pop
    
en_iyi_tasarim = puanli_pop[0][1]

# 3D GÖRSELLEŞTİRME
def ucak_govdesi_olustur():
    traces = []
    
    # 1. GÖVDE (Variable Section Mesh - Superellipse)
    u = np.linspace(0, 2 * np.pi, 32) # Daha yüksek çözünürlük
    v = np.linspace(0, GOVDE_UZUNLUK, 80) # Daha pürüzsüz boyuna geçiş
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Grid noktalarını hesapla
    x_grid = np.zeros_like(u_grid)
    y_grid = np.zeros_like(u_grid)
    z_grid = np.zeros_like(u_grid)
    
    rows, cols = u_grid.shape
    for i in range(rows):
        current_x = v[i]
        ry, rz = get_fuselage_section(current_x)
        
        # Kesit şekli parametresi
        n_shape = 4.0 if (current_x > 60 and current_x < 180) else 2.5
        if current_x < 10 or current_x > 290: n_shape = 2.0 # Uçlarda tam dairesel olsun
        
        for j in range(cols):
            angle = u[j]
            # Superellipse noktasını bul
            py, pz = superellipse_point(angle, ry, rz, n_shape)
            
            x_grid[i, j] = current_x
            y_grid[i, j] = py
            z_grid[i, j] = pz
            
    # Gövdeyi Çiz (Daha estetik bir materyal)
    traces.append(go.Surface(
        x=x_grid, y=y_grid, z=z_grid,
        opacity=0.3, 
        colorscale='Blues', # Daha modern mavi ton
        showscale=False, 
        name='Aero-Gövde', 
        hoverinfo='skip',
        lighting=dict(ambient=0.5, diffuse=0.8, fresnel=0.5, specular=0.5, roughness=0.5)
    ))
    
    # Tel Kafes (Wireframe) - Daha seyrek ve temiz
    for i in range(0, 80, 8): 
        traces.append(go.Scatter3d(
            x=x_grid[i], y=y_grid[i], z=z_grid[i],
            mode='lines', line=dict(color='darkblue', width=1.5), showlegend=False, hoverinfo='skip'
        ))

    # KANAT VE KUYRUK YÜZEYLERİ (Aerodinamik Profiller)
    
    def naca4_symmetric(c, t, n_points=20):
        """Basit simetrik airfoil (NACA 00t benzeri)"""
        x = np.linspace(0, c, n_points)
        # NACA 00xx kalınlık dağılımı (basitleştirilmiş)
        # yt = 5 * t * c * (0.2969*sqrt(x/c) - ...)
        xc = x/c
        yt = 5 * t * c * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
        return x, yt

    def create_lifting_surface(x_start, y_center, z_start, span, chore_root, chord_tip, sweep_angle_deg, name, is_vertical=False):
        """
        Kanat/Kuyruk oluşturucu.
        is_vertical=True ise Dikey Stabilize (Rudder) gibi davranır (Z ekseninde yükselir).
        """
        n_u = 20 # Profil çözünürlüğü
        n_v = 10 # Kanat açıklığı çözünürlüğü
        
        # Kanat yarıçapı/yüksekliği boyunca noktalar
        if is_vertical:
            v = np.linspace(0, span, n_v) # Z ekseni boyunca
        else:
            v = np.linspace(-span/2, span/2, n_v) # Y ekseni boyunca
            
        x_mesh = np.zeros((n_u*2, n_v)) # Üst ve alt yüzey için
        y_mesh = np.zeros((n_u*2, n_v))
        z_mesh = np.zeros((n_u*2, n_v))
        
        sweep_rad = np.radians(sweep_angle_deg)
        
        for i, pos_v in enumerate(v):
            # Taper ratio (Uçlara doğru incelme)
            if is_vertical:
                ratio = 1 - (pos_v / span) * 0.5 # Ucuna doğru %50 incel
                current_chord = chore_root * (1 - (pos_v/span)*(1 - chord_tip/chore_root))
                offset_x = pos_v * np.tan(sweep_rad) # Geriye kaçış
            else:
                ratio = 1.0 # Dikdörtgen kanat (Cessna stili)
                current_chord = chore_root
                offset_x = abs(pos_v) * np.tan(sweep_rad)
            
            # Profil oluştur (Üst ve Alt birleşik)
            px, yt = naca4_symmetric(current_chord, 0.15) # %15 kalınlık
            
            # Koordinat dönüşümleri
            # Airfoil X ekseninde 0..chord
            # Bunu gövde koordinatlarına taşıyoruz
            
            # Üst Yüzey
            x_mesh[:n_u, i] = x_start + offset_x + px
            if is_vertical:
                y_mesh[:n_u, i] = y_center + yt
                z_mesh[:n_u, i] = z_start + pos_v
            else:
                y_mesh[:n_u, i] = y_center + pos_v
                z_mesh[:n_u, i] = z_start + yt
            
            # Alt Yüzey (Geri dönüş)
            x_mesh[n_u:, i] = x_start + offset_x + px[::-1]
            if is_vertical:
                y_mesh[n_u:, i] = y_center - yt[::-1]
                z_mesh[n_u:, i] = z_start + pos_v
            else:
                y_mesh[n_u:, i] = y_center + pos_v
                z_mesh[n_u:, i] = z_start - yt[::-1]
                
        traces.append(go.Surface(
            x=x_mesh, y=y_mesh, z=z_mesh,
            colorscale='Blues', showscale=False, opacity=0.8, name=name,
            hoverinfo='skip'
        ))

    # 2. KANATLAR (High Wing - HAFİF Swept & Tapered)
    kanat_x = 70.0
    kanat_z = MAX_GOVDE_YARI_YUKSEKLIK + 2 # Gövdeye saplanan
    # Tip Chord 45 -> 25 düşürüldü (İncelen Kanat)
    create_lifting_surface(kanat_x, 0, kanat_z, 400.0, 45.0, 25.0, 4.0, "Kanat")
    
    # Destek Dikmeleri (Cessna Struts)
    # Basit çizgiler
    traces.append(go.Scatter3d(
        x=[kanat_x+10, kanat_x+10], y=[-20, -100], z=[-MAX_GOVDE_YARI_YUKSEKLIK+5, kanat_z],
        mode='lines', line=dict(color='darkblue', width=3), name='Dikme L'
    ))
    traces.append(go.Scatter3d(
        x=[kanat_x+10, kanat_x+10], y=[20, 100], z=[-MAX_GOVDE_YARI_YUKSEKLIK+5, kanat_z],
        mode='lines', line=dict(color='darkblue', width=3), name='Dikme R'
    ))

    # 3. KUYRUK TAKIMI
    tail_x = GOVDE_UZUNLUK - 40
    
    # Yatay (Stabilizer) - Belirgin Taper (Sivri Uç)
    # Tip Chord 25 -> 12 düşürüldü
    create_lifting_surface(tail_x, 0, 0, 140.0, 35.0, 12.0, 20.0, "Yatay Kuyruk")
    
    # Dikey (Fin) - Swept Back & Sivri
    # Tip Chord 20 -> 10 düşürüldü, Sweep artırıldı
    create_lifting_surface(tail_x, 0, 5, 70.0, 40.0, 10.0, 45.0, "Dikey Kuyruk", is_vertical=True)

    return traces

def parca_kutusu_ciz(pos, dim, color, name):
    x, y, z = pos
    dx, dy, dz = dim
    
    x_k = [x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2]
    y_k = [y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2]
    z_k = [z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2]
    
    return go.Mesh3d(
        x=x_k, y=y_k, z=z_k, color=color, opacity=1.0, name=name,
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        hoverinfo='name'
    )

# --- Çizim ---
fig = go.Figure()

for parca in ucak_govdesi_olustur():
    fig.add_trace(parca)

colors = ['red', 'blue', 'orange', 'purple', 'green', 'brown', 'cyan']
idx = 0

print("\n--- YERLEŞİM DETAYLARI ---")
for k_id, pos in en_iyi_tasarim.yerlesim.items():
    boyut = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)["boyut"]
    fig.add_trace(parca_kutusu_ciz(pos, boyut, colors[idx % len(colors)], k_id))
    
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2] + boyut[2]/1.5],
        mode='text', text=[k_id], textposition="top center",
        textfont=dict(size=10, color="black"), showlegend=False
    ))
    print(f"📍 {k_id}: X={pos[0]:.1f}")
    idx += 1

# CG Görselleri
# Hedef Zarfı (Gövde max genişliğinden referans alarak)
box_r_y = MAX_GOVDE_YARI_GENISLIK + 5
box_r_z = MAX_GOVDE_YARI_YUKSEKLIK + 5

fig.add_trace(go.Mesh3d(
    x=[TARGET_CG_X_MIN, TARGET_CG_X_MAX, TARGET_CG_X_MAX, TARGET_CG_X_MIN, TARGET_CG_X_MIN, TARGET_CG_X_MAX, TARGET_CG_X_MAX, TARGET_CG_X_MIN],
    y=[-box_r_y, -box_r_y, box_r_y, box_r_y, -box_r_y, -box_r_y, box_r_y, box_r_y],
    z=[-box_r_z, -box_r_z, -box_r_z, -box_r_z, box_r_z, box_r_z, box_r_z, box_r_z],
    color='gold', opacity=0.3, name='HEDEF CG ARALIĞI', alphahull=0
))

viz_z = MAX_GOVDE_YARI_YUKSEKLIK + 40
fig.add_trace(go.Scatter3d(
    x=[best_cg[0]], y=[best_cg[1]], z=[viz_z],
    mode='markers+text', marker=dict(size=12, color='black', symbol='diamond'),
    name='HESAPLANAN CG', text=["HESAPLANAN CG"], textposition="top center"
))

fig.add_trace(go.Scatter3d(
    x=[best_cg[0], best_cg[0]], y=[best_cg[1], best_cg[1]], z=[best_cg[2], viz_z],
    mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'
))

fig.add_trace(go.Scatter3d(
    x=[best_cg[0]], y=[best_cg[1]], z=[best_cg[2]],
    mode='markers', marker=dict(size=5, color='black'), name='Gerçek CG Konumu'
))

# Hata Çizgisi
target_x_visual = best_cg[0]
if best_cg[0] < TARGET_CG_X_MIN: target_x_visual = TARGET_CG_X_MIN
elif best_cg[0] > TARGET_CG_X_MAX: target_x_visual = TARGET_CG_X_MAX

fig.add_trace(go.Scatter3d(
    x=[target_x_visual, best_cg[0]], y=[TARGET_CG_Y, best_cg[1]], z=[TARGET_CG_Z, best_cg[2]],
    mode='lines', line=dict(color='red', width=4, dash='dot'), name='CG Hatası'
))

camera = dict(eye=dict(x=2.0, y=-2.0, z=1.0))
fig.update_layout(
    title="Cessna 172 Stili: Uçak İçi Sistem Yerleşimi",
    scene=dict(
        xaxis=dict(title='Uzunluk (cm)', range=[0, GOVDE_UZUNLUK], backgroundcolor="rgb(240, 240, 240)"),
        yaxis=dict(title='Genişlik (cm)', range=[-200, 200]),
        zaxis=dict(title='Yükseklik (cm)', range=[-100, 100]),
        aspectmode='data', 
        camera=camera
    ),
    margin=dict(r=0, l=0, b=0, t=50)
)

fig.show()