import random

# Çakışma kontrolü
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

# --- Temel Sınıflar ---
class TasarimBireyi:
    def __init__(self):
        self.yerlesim = {}
        
    def rastgele_yerlestir(self, aircraft):
        for komp in aircraft.komponentler_db:
            # Eğer parça kilitliyse sabit pozisyonunu al ve geç
            if komp.kilitli:
                self.yerlesim[komp.id] = komp.sabit_pos
                continue
            
            bolge=komp.sabit_bolge
            
            if bolge=="BURUN":
                x=random.uniform(0, aircraft.bolge_burun_son)
            elif bolge=="KUYRUK":
                x=random.uniform(aircraft.bolge_kuyruk_bas, aircraft.govde_uzunluk)
            elif bolge=="MERKEZ":
                center_x = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2
                x=random.uniform(center_x-30, center_x+30)
            elif bolge=="GOVDE":
                # Burun ile Kuyruk arasındaki ana hacim
                x=random.uniform(aircraft.bolge_burun_son, aircraft.bolge_kuyruk_bas)
            else:
                x=random.uniform(0, aircraft.govde_uzunluk)
                
            y=random.uniform(-aircraft.govde_yaricap/2, aircraft.govde_yaricap/2)
            
            if bolge=="ON_ALT":
                z=-aircraft.govde_yaricap/2
            else:
                z=random.uniform(-aircraft.govde_yaricap/2, aircraft.govde_yaricap/2)
                
            self.yerlesim[komp.id]=(x,y,z)

def calculate_fitness_design(birey, aircraft):
    puan=0

    # Çakışma Kontrolü
    cakisma_sayisi=0
    keys=list(birey.yerlesim.keys())
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            k1_id=keys[i]
            k2_id=keys[j]
            
            dim1=next(item for item in aircraft.komponentler_db if item.id==k1_id).boyut
            dim2=next(item for item in aircraft.komponentler_db if item.id==k2_id).boyut
            
            pos1=birey.yerlesim[k1_id]
            pos2=birey.yerlesim[k2_id]
            
            if(kutular_cakisiyor_mu(pos1, dim1, pos2, dim2)):
                cakisma_sayisi+=1
    
    puan-=cakisma_sayisi*10000
    
    # Gövdeden Taşma Kontrolü
    tasma_sayisi=0
    for k_id,pos in birey.yerlesim.items():
        dim=next(item for item in aircraft.komponentler_db if item.id==k_id).boyut
        if not aircraft.govde_icinde_mi(pos, dim):
            tasma_sayisi+=1
            
    puan-=tasma_sayisi*5000
    
    # TİTREŞİM KONTROLÜ
    # Motoru bul (Titreşim kaynağı)
    pos_motor = birey.yerlesim.get("Motor") 
    
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            # DB'den parça özelliklerini çek
            parca_db = next(item for item in aircraft.komponentler_db if item.id == k_id)
            
            # Eğer parça hassassa kontrol et
            if parca_db.titresim_hassasiyeti:
                # Motora olan mesafeyi hesapla
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                
                # Limitten yakınsa ceza kes
                if mesafe < aircraft.titresim_limiti:
                    ihlâl = aircraft.titresim_limiti - mesafe
                    puan -= (ihlâl ** 2) * 50

    # 4. CG (Ağırlık Merkezi) Hesabı
    toplam_cg_hatasi = 0

    dolu_cg_coords = (0,0,0)
    # Her bir doluluk senaryosu için ayrı CG hesapla
    for doluluk in aircraft.doluluk_oranlari:
        total_mass = 0
        moment_x = 0
        moment_y = 0
        moment_z = 0
    
        for k_id, pos in birey.yerlesim.items():
            db_item = next(item for item in aircraft.komponentler_db if item.id == k_id)
            mass = db_item.agirlik

            if k_id == "Yakit_Tanki":
                mass += aircraft.max_yakit_agirligi * doluluk

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

        target_x_center = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2

        # Hedef CG'ye olan mesafe hatası
        dist_error = ((cg_x - target_x_center)**2 + (cg_y - aircraft.target_cg_y)**2 + (cg_z - aircraft.target_cg_z)**2)**0.5
        toplam_cg_hatasi += dist_error

    # Ortalama hatayı puandan düş (Ceza yöntemi)
    puan -= (toplam_cg_hatasi / len(aircraft.doluluk_oranlari)) * 1000

    return puan, dolu_cg_coords

def calculate_fitness_nsga2(birey, aircraft):
    """
    NSGA-II için iki ayrı obje/hedef (objective) döndürür:
    1. Ceza Puanı (Minimize edilecek): Çakışma, Taşma ve Titreşim cezaları
    2. CG Hatası (Minimize edilecek): Ağırlık merkezinin hedeften sapması
    """
    ceza_puani = 0.0

    # Çakışma Kontrolü
    cakisma_sayisi = 0
    keys = list(birey.yerlesim.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1_id = keys[i]
            k2_id = keys[j]
            dim1 = next(item for item in aircraft.komponentler_db if item.id == k1_id).boyut
            dim2 = next(item for item in aircraft.komponentler_db if item.id == k2_id).boyut
            pos1 = birey.yerlesim[k1_id]
            pos2 = birey.yerlesim[k2_id]
            if kutular_cakisiyor_mu(pos1, dim1, pos2, dim2):
                cakisma_sayisi += 1
                
    ceza_puani += cakisma_sayisi * 10000

    # Gövdeden Taşma Kontrolü
    tasma_sayisi = 0
    for k_id, pos in birey.yerlesim.items():
        dim = next(item for item in aircraft.komponentler_db if item.id == k_id).boyut
        if not aircraft.govde_icinde_mi(pos, dim):
            tasma_sayisi += 1
            
    ceza_puani += tasma_sayisi * 5000

    # TİTREŞİM KONTROLÜ
    pos_motor = birey.yerlesim.get("Motor")
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            parca_db = next(item for item in aircraft.komponentler_db if item.id == k_id)
            if parca_db.titresim_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                if mesafe < aircraft.titresim_limiti:
                    ihlâl = aircraft.titresim_limiti - mesafe
                    ceza_puani += (ihlâl ** 2) * 50

    # CG (Ağırlık Merkezi) Miktarı
    toplam_cg_hatasi = 0
    dolu_cg_coords = (0, 0, 0)

    for doluluk in aircraft.doluluk_oranlari:
        total_mass = 0
        moment_x = 0
        moment_y = 0
        moment_z = 0
        for k_id, pos in birey.yerlesim.items():
            db_item = next(item for item in aircraft.komponentler_db if item.id == k_id)
            mass = db_item.agirlik
            if k_id == "Yakit_Tanki":
                mass += aircraft.max_yakit_agirligi * doluluk
            total_mass += mass
            moment_x += mass * pos[0]
            moment_y += mass * pos[1]
            moment_z += mass * pos[2]

        cg_x = moment_x / total_mass
        cg_y = moment_y / total_mass
        cg_z = moment_z / total_mass

        if doluluk == 1.0:
            dolu_cg_coords = (cg_x, cg_y, cg_z)

        target_x_center = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2
        dist_error = ((cg_x - target_x_center)**2 + (cg_y - aircraft.target_cg_y)**2 + (cg_z - aircraft.target_cg_z)**2)**0.5
        toplam_cg_hatasi += dist_error

    cg_hatasi = toplam_cg_hatasi / len(aircraft.doluluk_oranlari)

    return ceza_puani, cg_hatasi, dolu_cg_coords

