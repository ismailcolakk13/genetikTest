# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:06:09 2025

@authors: İsmail Çolak, Yusuf Eren Aykurt, Mehmet Can Çalışkan
"""
import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
import trimesh

# CESSNA 172 REALISTIC DIMENSIONS (cm) and WEIGHTS (kg)
GOVDE_UZUNLUK = 828.0 # CM (x ekseni, gerçeğe yakın uzunluk)
GOVDE_CAP = 120.0 # CM (Kabin genişliği)
GOVDE_YARICAP = GOVDE_CAP / 2

# Cessna CG referansı genellikle firewall veya pervaneden alınır. 
# Kanatların ortalarına denk gelmesi için X = 230 - 270 cm arası idealdir.
TARGET_CG_X_MIN = 230.0
TARGET_CG_X_MAX = 270.0
TARGET_CG_Y = 0.0
TARGET_CG_Z = 0.0

DOLULUK_ORANLARI = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_YAKIT_AGIRLIGI = 150.0 # Yaklaşık 56 galon (212 litre) avgas
TITRESIM_LIMITI = 100.0 # Gerçek boyutta mesafe arttı

# BÖLGE TANIMLARI (Sınırlar)
BOLGE_BURUN_SON = 120.0 # Motor kabini (Cowling) payı
BOLGE_KUYRUK_BAS = 680.0 # Gövdenin arka daralma noktası

KOMPONENTLER_DB = [
    # Gerçek boyutlara, ağırlıklara kalibre edildi
    {"id": "Motor",       "agirlik": 130.0,"boyut": (80, 60, 60), "sabit_bolge": "BURUN", "sabit_pos": (40, 0, 0), "kilitli": True, "titresim_hassasiyeti": False}, 
    {"id": "Batarya_Ana", "agirlik": 15.0, "boyut": (25, 20, 20), "sabit_bolge": "BURUN", "kilitli": False, "titresim_hassasiyeti": False}, # Genelde firewall arkasında
    {"id": "Aviyonik_1",  "agirlik": 10.0, "boyut": (30, 30, 20), "sabit_bolge": "GOVDE",  "kilitli": False, "titresim_hassasiyeti": True},
    {"id": "Aviyonik_2",  "agirlik": 10.0, "boyut": (30, 30, 20), "sabit_bolge": "GOVDE",  "kilitli": False, "titresim_hassasiyeti": True},
    {"id": "Payload_Kam", "agirlik": 50.0, "boyut": (50, 50, 50), "sabit_bolge": "ON_ALT", "kilitli": False, "titresim_hassasiyeti": True}, # Büyük bir gözetleme kamerası
    {"id": "Yakit_Tanki_Sol", "agirlik": 15.0, "boyut": (80, 120, 14),"sabit_bolge": "KANAT_SOL", "kilitli": False, "titresim_hassasiyeti": False}, # Sol kanat içi yakıt (dikdörtgen prizma)
    {"id": "Yakit_Tanki_Sag", "agirlik": 15.0, "boyut": (80, 120, 14),"sabit_bolge": "KANAT_SAG", "kilitli": False, "titresim_hassasiyeti": False}, # Sağ kanat içi yakıt (dikdörtgen prizma)
    {"id": "Servo_Kuyruk","agirlik": 5.0,  "boyut": (10, 10, 10), "sabit_bolge": "KUYRUK", "kilitli": False, "titresim_hassasiyeti": False},
]

KOMPONENTLER_DICT = {comp["id"]: comp for comp in KOMPONENTLER_DB}

# --- GLOBAL 3D MODEL YÜKLEME VE ÇARPIŞMA (COLLISION) MOTORU HIZIRLIĞI ---
UCAK_MESH = None
UCAK_COLLISION_MANAGER = None
UCAK_KESIT_BOUNDS = {} # {x_int: (min_y, max_y, min_z, max_z)}

try:
    print("3D Model yükleniyor (Çarpışma Motoru ve UI için)...")
    _scene = trimesh.load('cessna-172.glb', force='scene')
    
    rot_x = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    _scene.apply_transform(rot_x)
    
    b_min, b_max = _scene.bounds
    scale_factor = GOVDE_UZUNLUK / (b_max[0] - b_min[0])
    
    matrix = np.eye(4)
    matrix[:3, :3] *= scale_factor
    
    nb_min = b_min * scale_factor
    nb_max = b_max * scale_factor
    
    matrix[0, 3] = -nb_min[0]
    matrix[1, 3] = -(nb_min[1] + nb_max[1]) / 2
    
    # Modelin bounding box merkezini orijine oturtmak kabini Z=0 çevresine hizalar.
    # Bu sayede Z ekseninde [-72, 72] aralığında yer alan matematiksel parçalar uçağın içinde kalır.
    matrix[2, 3] = -(nb_min[2] + nb_max[2]) / 2 
    
    _scene.apply_transform(matrix)
    UCAK_MESH = _scene.to_geometry()
    
    UCAK_COLLISION_MANAGER = trimesh.collision.CollisionManager()
    UCAK_COLLISION_MANAGER.add_object('Ucak_Govde', UCAK_MESH)
    print("✅ Çarpışma Motoru (FCL) ve Raycast Motoru Aktif: Gerçekçi duvar temasları test edilecek.")
    
    # ----------------------------------------------------
    # BÖLGE MASKELERİ (Plotly Rengi ve Yerleşim İçin Ortak)
    # ----------------------------------------------------
    vertices = UCAK_MESH.vertices
    faces = UCAK_MESH.faces
    
    centroids = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0
    cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    
    # Plotly'de uçağı boyamak için kullandığımız matematiksel bölgesel ayrım maskeleri
    MASK_WINGS = (np.abs(cy) > 60) & (cz > 30)
    MASK_BOTTOM = (cz < -5) & (~MASK_WINGS)
    MASK_FRONT = (cx < 120) & (~MASK_WINGS) & (~MASK_BOTTOM)
    MASK_BACK = (cx > 500) & (~MASK_WINGS) & (~MASK_BOTTOM)
    MASK_FUSELAGE = (~MASK_WINGS) & (~MASK_BOTTOM) & (~MASK_FRONT) & (~MASK_BACK)
    
    UCAK_BOLGELER = {
        "KANAT_SOL": MASK_WINGS & (cy < 0),
        "KANAT_SAG": MASK_WINGS & (cy > 0),
        "ON_ALT": MASK_BOTTOM,
        "BURUN": MASK_FRONT,
        "KUYRUK": MASK_BACK,
        "GOVDE": MASK_FUSELAGE,
        "MERKEZ": MASK_FUSELAGE # Merkez de gövdenin içindedir
    }
    
    UCAK_BOLGE_SINIRLARI = {}
    for bolge_adi, mask in UCAK_BOLGELER.items():
        if not np.any(mask): continue
        # Bu maskeye ait face'lerin sahip olduğu vertex'leri bul
        v_idx = np.unique(faces[mask].flatten())
        v = vertices[v_idx]
        UCAK_BOLGE_SINIRLARI[bolge_adi] = {
            "x_min": np.min(v[:,0]), "x_max": np.max(v[:,0]),
            "y_min": np.min(v[:,1]), "y_max": np.max(v[:,1]),
            "z_min": np.min(v[:,2]), "z_max": np.max(v[:,2])
        }
        
except Exception as e:
    print(f"Uyarı: 3D Model çarpışma için yüklenemedi: {e}")


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

# Gerçek Model Boyutlarına Göre İç Hacim Kontrolü (Hızlı Raycast)
def govde_icinde_mi(pos, dim):
    """
    Merkez noktasından 6 yöne ışın atarak parçanın uçağın iç hacminde olup olmadığını kontrol eder.
    Uçağın içindeki bir noktadan atılan ışınlar her yönde duvara çarpmalıdır.
    En az 5/6 ışının çarpması gerekir (eski eşik 3/6 idi, çok gevşekti).
    """
    if UCAK_MESH is None:
        return True
        
    x, y, z = pos
    dx, dy, dz = dim
    
    if x - dx/2.0 < 0 or x + dx/2.0 > GOVDE_UZUNLUK:
        return False
    
    # Merkez noktasından 6 yöne ışın at
    ray_origins = np.array([[x, y, z]] * 6)
    ray_directions = np.array([
        [0, 0, 1],  # Yukarı
        [0, 0, -1], # Aşağı
        [0, 1, 0],  # Sağ
        [0, -1, 0], # Sol
        [1, 0, 0],  # Ön
        [-1, 0, 0]  # Arka
    ])
    hits = UCAK_MESH.ray.intersects_any(
        ray_origins=ray_origins, ray_directions=ray_directions
    )
    
    # En az 5/6 ışın duvara çarpmalı (eski eşik 3 idi → dışarıdaki noktalar geçiyordu)
    if np.sum(hits) < 5:
        return False
    return True
    
#Genetik alg. sınıfları

class TasarimBireyi:
    def __init__(self):
        self.yerlesim = {}
        
    def rastgele_yerlestir(self):
        for komp in KOMPONENTLER_DB:
            # Eğer parça kilitliyse (örn: Motor), sabit pozisyonunu al ve geç
            if komp.get("kilitli", False):
                self.yerlesim[komp["id"]] = komp["sabit_pos"]
                continue
            
            bolge=komp["sabit_bolge"]
            dx, dy, dz = komp["boyut"]
            
            x, y, z = 0, 0, 0
            yerlesti = False
            
            # Makul bir rastgele konum bul ve CAD modeli içinde olduğundan emin ol
            for _ in range(1000):  # Daha sıkı kontrol → daha fazla deneme gerekli
                # Uçağın Plotly şekli ile oluşturduğu gerçek sınırları (Masked Bounding Box) kullan
                if bolge in UCAK_BOLGE_SINIRLARI:
                    b_sinir = UCAK_BOLGE_SINIRLARI[bolge]
                    
                    # Merkez bileşeni hedefe daha sıkı tutunmalı
                    if bolge == "MERKEZ":
                        center_x = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2
                        x = random.uniform(center_x - 40, center_x + 40)
                    else:
                        x = random.uniform(b_sinir["x_min"] + dx/2, b_sinir["x_max"] - dx/2)
                        
                    y = random.uniform(b_sinir["y_min"] + dy/2, b_sinir["y_max"] - dy/2)
                    z = random.uniform(b_sinir["z_min"] + dz/2, b_sinir["z_max"] - dz/2)
                    
                else: 
                    # Varsayılan
                    x = random.uniform(0, GOVDE_UZUNLUK)
                    y = random.uniform(-50, 50)
                    z = random.uniform(-30, 40)
                
                # Mesh iç-hacim kontrolü (contains veya geliştirilmiş raycast)
                if govde_icinde_mi((x,y,z), (dx,dy,dz)):
                    yerlesti = True
                    break
                    
            if not yerlesti:
                # Bölgenin bounding box merkezini güvenli varsayılan olarak kullan
                if bolge in UCAK_BOLGE_SINIRLARI:
                    b = UCAK_BOLGE_SINIRLARI[bolge]
                    x = (b["x_min"] + b["x_max"]) / 2
                    y = (b["y_min"] + b["y_max"]) / 2
                    z = (b["z_min"] + b["z_max"]) / 2
                else:
                    x, y, z = GOVDE_UZUNLUK / 2, 0, 0
                
            self.yerlesim[komp["id"]]=(x,y,z)
            
def calculate_fitness_design(birey):
    puan = 0.0
    
    #çakışma
    cakisma_sayisi=0
    keys=list(birey.yerlesim.keys())
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            k1_id=keys[i]
            k2_id=keys[j]
            
            dim1=KOMPONENTLER_DICT[k1_id]["boyut"]
            dim2=KOMPONENTLER_DICT[k2_id]["boyut"]
            
            pos1=birey.yerlesim[k1_id]
            pos2=birey.yerlesim[k2_id]
            
            if(kutular_cakisiyor_mu(pos1, dim1, pos2, dim2)):
                cakisma_sayisi+=1
    
    puan-=cakisma_sayisi*15000  # Hard constraint: Ağır ceza
    
    #gövdeden taşma (Basit matematiksel kontrol - Parça uçağın tamamen dışında mı?)
    # Kanat tankları gövde dışındadır, bu hesaba katılmazlar.
    tasma_sayisi=0
    for k_id,pos in birey.yerlesim.items():
        if "Yakit_Tanki" in k_id:
            continue
        dim=KOMPONENTLER_DICT[k_id]["boyut"]
        if not govde_icinde_mi(pos, dim):
            tasma_sayisi+=1
            
    puan-=tasma_sayisi*15000    # Hard constraint: Ağır ceza
    
    # 3D MODEL DUVAR TEMASI (Clipping) KONTROLÜ (Gerçekçi FCL Çarpışma)
    if UCAK_COLLISION_MANAGER is not None:
        duvar_temas_sayisi = 0
        for k_id, pos in birey.yerlesim.items():
            dim = KOMPONENTLER_DICT[k_id]["boyut"]
            
            # Kutuyu oluştur ve pozisyona taşı
            transform = np.eye(4)
            transform[:3, 3] = pos
            box = trimesh.creation.box(extents=dim, transform=transform)
            
            # Uçak zırhı (mesh) ile kesişiyor mu?
            if UCAK_COLLISION_MANAGER.in_collision_single(box):
                duvar_temas_sayisi += 1
                
        puan -= duvar_temas_sayisi * 15000 # Hard constraint: Duvara değen parça reddedilir
    
    # Sabit Bölge (Region) İhlali Kontrolü (Hard Constraint)
    bolge_ihlali_sayisi = 0
    for k_id, pos in birey.yerlesim.items():
        parca_db = KOMPONENTLER_DICT[k_id]
        bolge = parca_db.get("sabit_bolge", "")
        
        x, y, z = pos
        if bolge == "BURUN" and (x < 10 or x > 100):
            bolge_ihlali_sayisi += 1
        elif bolge == "KUYRUK" and x < BOLGE_KUYRUK_BAS:
            bolge_ihlali_sayisi += 1
        elif bolge == "MERKEZ":
            center_x = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2
            if abs(x - center_x) > 40:
                bolge_ihlali_sayisi += 1
        elif bolge == "GOVDE" and (x < 120 or x > 500): # Kabin dışı
            bolge_ihlali_sayisi += 1
        elif bolge == "ON_ALT" and (x < 100 or x > 400 or z > -20): 
            bolge_ihlali_sayisi += 1
        elif bolge == "KANAT_SOL":
            if y > -60 or x < 220 or x > 320 or z < 40 or z > 90:
                bolge_ihlali_sayisi += 1
        elif bolge == "KANAT_SAG":
            if y < 60 or x < 220 or x > 320 or z < 40 or z > 90:
                bolge_ihlali_sayisi += 1
            
    puan -= bolge_ihlali_sayisi * 15000  # Hard constraint

    # YENİ EKLENEN: TİTREŞİM KONTROLÜ ---
    # Motoru bul (Titreşim kaynağı)
    pos_motor = birey.yerlesim["Motor"] 
    
    for k_id, pos in birey.yerlesim.items():
        # DB'den parça özelliklerini çek
        parca_db = KOMPONENTLER_DICT[k_id]
        
        # Eğer parça hassassa kontrol et
        if parca_db.get("titresim_hassasiyeti") == True:
            # Motora olan mesafeyi hesapla
            mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
            
            # Limitten yakınsa ceza kes
            if mesafe < TITRESIM_LIMITI:
                ihlâl = TITRESIM_LIMITI - mesafe
                puan -= (ihlâl ** 2) * 5 # Soft ceza (Gradient)

    # 4. CG (Ağırlık Merkezi) Hesabı
    toplam_cg_hatasi = 0
    # Sadece raporlama için kullanılacak değişken
    dolu_cg_coords = (0,0,0)
    
    # Optimizasyon: Sabit kütle ve momentleri döngü dışına çıkar
    static_mass = 0
    static_moment_x = 0
    static_moment_y = 0
    static_moment_z = 0
    yakit_pos_sol = (0, 0, 0)
    yakit_pos_sag = (0, 0, 0)
    
    for k_id, pos in birey.yerlesim.items():
        db_item = KOMPONENTLER_DICT[k_id]
        mass = db_item["agirlik"]
        
        if k_id == "Yakit_Tanki_Sol":
            yakit_pos_sol = pos
        elif k_id == "Yakit_Tanki_Sag":
            yakit_pos_sag = pos
        else:
            static_mass += mass
            static_moment_x += mass * pos[0]
            static_moment_y += mass * pos[1]
            static_moment_z += mass * pos[2]

    # Her bir doluluk senaryosu için ayrı CG hesapla
    for doluluk in DOLULUK_ORANLARI:
        yakit_mass_per_tank = (MAX_YAKIT_AGIRLIGI / 2) * doluluk
        
        if "Yakit_Tanki_Sol" in birey.yerlesim and "Yakit_Tanki_Sag" in birey.yerlesim:
            m_sol = KOMPONENTLER_DICT["Yakit_Tanki_Sol"]["agirlik"] + yakit_mass_per_tank
            m_sag = KOMPONENTLER_DICT["Yakit_Tanki_Sag"]["agirlik"] + yakit_mass_per_tank
            total_mass = static_mass + m_sol + m_sag
            moment_x = static_moment_x + m_sol * yakit_pos_sol[0] + m_sag * yakit_pos_sag[0]
            moment_y = static_moment_y + m_sol * yakit_pos_sol[1] + m_sag * yakit_pos_sag[1]
            moment_z = static_moment_z + m_sol * yakit_pos_sol[2] + m_sag * yakit_pos_sag[2]
        else:
            total_mass = static_mass
            moment_x = static_moment_x
            moment_y = static_moment_y
            moment_z = static_moment_z
        
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

    # Ortalama hatayı puandan düş (Soft ceza)
    puan -= (toplam_cg_hatasi / len(DOLULUK_ORANLARI)) * 100

    return puan, dolu_cg_coords
  
   


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
        # Kilitli parçaları mutasyona uğratma
        comp_info = KOMPONENTLER_DICT.get(k_id)
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
    Cessna 172 Global Modelini (UCAK_MESH) Plotly Mesh3d nesnelerine çevirir.
    Gövdeyi görsel olarak çarpışma-yerleşim bölgelerine ayırıp renklendirir.
    """
    traces = []
    if UCAK_MESH is not None:
        vertices = UCAK_MESH.vertices
        faces = UCAK_MESH.faces
        
        # Bölgelerin Plotly Trace eşleşmeleri
        bolgeler = [
            (MASK_FRONT, 'orange', 'Burun (Motor)'),
            (MASK_BACK, 'purple', 'Kuyruk (Servo)'),
            (MASK_WINGS, 'lightgreen', 'Kanatlar (Yakıt)'),
            (MASK_BOTTOM, 'darkgray', 'Alt Gövde (Kamera)'),
            (MASK_FUSELAGE, 'lightblue', 'Ana Kabin (Aviyonik)')
        ]
        
        for mask_arr, color, name in bolgeler:
            if not np.any(mask_arr):
                continue
            section_faces = faces[mask_arr]
            traces.append(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=section_faces[:, 0], j=section_faces[:, 1], k=section_faces[:, 2],
                color=color, opacity=0.3, name=name,
                hoverinfo='skip', flatshading=True
            ))
            
    return traces

def komponent_mesh_ciz(pos, dim, color, k_id):
    """
    Komponentlerin tipine göre daha gerçekçi 3D geometriler (Silindir, Kutu, vb.) oluşturur.
    Önceden sadece keskin köşeli bir Box çiziliyordu.
    """
    x, y, z = pos
    dx, dy, dz = dim
    
    # Parçaya özel geometri seçimi
    if k_id == "Motor":
        # Motor genelde yuvarlak bir bloğa benzer (Silindir)
        radius = (dy + dz) / 4
        mesh = trimesh.creation.cylinder(radius=radius, height=dx)
        # trimesh silindiri Z ekseni boyunca oluşturur. Motor yatay durmalı (X yönünde).
        rot = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        mesh.apply_transform(rot)
        
    elif k_id == "Payload_Kam":
        # Kamera altına bakan bir küremsi dome (kapsül)
        mesh = trimesh.creation.capsule(height=dx/2, radius=min(dy, dz)/2)
        
    else:
        # Yakıt tankı, Batarya, Aviyonik ve Servolar için kutu (dikdörtgen prizma)
        mesh = trimesh.creation.box(extents=dim)
    
    # Geometriyi konumuna taşı
    transform = np.eye(4)
    transform[:3, 3] = pos
    mesh.apply_transform(transform)
    
    # Plotly'e gönder
    return go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        color=color, opacity=1.0, name=k_id,
        hoverinfo='name', flatshading=True
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
# Yakıt tankları ağırlık merkezinden (CG) ne kadar uzaksa, yakıt azaldıkça uçağın dengesi o kadar bozulur.
yakit_pos_x = 0
if "Yakit_Tanki_Sol" in en_iyi_tasarim.yerlesim:
    yakit_pos_x = en_iyi_tasarim.yerlesim["Yakit_Tanki_Sol"][0]
    
hedef_merkez_x = (TARGET_CG_X_MIN + TARGET_CG_X_MAX) / 2

if abs(yakit_pos_x - hedef_merkez_x) > 10.0:
    print(f"⛽ Yakıt tanklarının X konumu ({yakit_pos_x:.1f}) ideal merkezden uzak. Yakıt tüketimi CG'yi ETKİLEYECEK.")
else:
    print(f"⛽ Yakıt tankları ideal merkeze çok yakın. Yakıt tüketiminin dengeye etkisi MİNİMUM.")

# 3. Genel Skor Yorumu
# Ceza sistemi olduğu için skor 0'a ne kadar yakınsa (negatif değerler) o kadar iyidir.
if best_score > -2000:
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
    db_item = KOMPONENTLER_DICT[k_id]
    mass = db_item["agirlik"]

    # Bos depo için moment (Yakıt = 0)
    bos_agirlik += mass
    bos_moment_x += mass * pos[0]

    # Dolu depo için moment (Yakıt = MAX)
    if "Yakit_Tanki" in k_id:
        dolu_agirlik += (mass + MAX_YAKIT_AGIRLIGI/2)
        dolu_moment_x += (mass + MAX_YAKIT_AGIRLIGI/2) * pos[0]
    else:
        dolu_agirlik += mass
        dolu_moment_x += mass * pos[0]

cg_bos_x = bos_moment_x / bos_agirlik
cg_dolu_x = dolu_moment_x / dolu_agirlik
cg_kaymasi = abs(cg_dolu_x - cg_bos_x)

print(f"Yakit Tanklari Konumu (X): {yakit_pos_x:.2f} cm")
print(f"CG (Dolu Depo)        : {cg_dolu_x:.2f} cm")
print(f"CG (Bos Depo)         : {cg_bos_x:.2f} cm")
print(f"CG Kaymasi (Drift)    : {cg_kaymasi:.2f} cm")

# Uyarı Mekanizması
if cg_kaymasi > 5.0:
    print("❌ KRİTİK: Yakıt tüketimi CG'yi çok fazla kaydırıyor! Uçuş stabilitesi tehlikede.")
elif cg_kaymasi > 2.0:
    print("⚠️ DİKKAT: Yakıt tüketimi dengeyi etkiliyor. Trim ayarı gerekecek.")
else:
    print("✅ MÜKEMMEL: Yakıt tankları ideal konumda. Yakıt tüketiminin dengeye etkisi minimum.")
print("-----------------------\n")
print("\n--- YERLEŞİM DETAYLARI ---")
motor_pos = en_iyi_tasarim.yerlesim["Motor"] # Motor referansı

for k_id, pos in en_iyi_tasarim.yerlesim.items():
    # Boyut bilgisini DB'den çek
    boyut = next(item for item in KOMPONENTLER_DB if item["id"] == k_id)["boyut"]
    
    # Kutuyu (veya silindiri) çiz
    fig.add_trace(komponent_mesh_ciz(pos, boyut, colors[idx % len(colors)], k_id))
    
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
        xaxis=dict(title='Uzunluk (cm)', backgroundcolor="rgb(240, 240, 240)"),
        yaxis=dict(title='Genişlik (cm)'),
        zaxis=dict(title='Yükseklik (cm)'),
        aspectmode='data', # Gerçek oranları koru (Uçak basık görünmesin)
        camera=camera
    ),
    margin=dict(r=0, l=0, b=0, t=50) # Kenar boşluklarını azalt
)

fig.show()