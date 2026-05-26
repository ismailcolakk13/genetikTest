import random
import math

# ---------------------------------------------------------------------------
# BÖLGE YARDIMCI FONKSİYONLARI
# ---------------------------------------------------------------------------

def _secili_bolge_sinirlari(komp, aircraft):
    """
    Komponentin izin_verilen_bolgeler listesinden bölmeleri alarak
    geçerli X ve Z aralıklarını hesaplar.

    Döner: (x_min, x_max, z_min, z_max, secili_bolge)
    - Birden fazla bölge varsa ilk rastgele yerleştirme için
      stochastic kullanılır; fitness/clamp için UNION kullanılır.
    """
    bolgeler = komp.izin_verilen_bolgeler
    if not bolgeler:
        r = aircraft.govde_yaricap
        return (0.0, aircraft.govde_uzunluk, -r, r, "SERBEST")

    # Tüm izin verilen bölgelerin birleşim (UNION) sınırları
    x_mins, x_maxs, z_mins, z_maxs = [], [], [], []
    for b in bolgeler:
        x_min_b, x_max_b = aircraft.get_bolge_x_siniri(b)
        z_min_b, z_max_b = aircraft.get_bolge_z_siniri(b)
        x_mins.append(x_min_b)
        x_maxs.append(x_max_b)
        z_mins.append(z_min_b)
        z_maxs.append(z_max_b)

    return (min(x_mins), max(x_maxs), min(z_mins), max(z_maxs), "/".join(bolgeler))


def _rastgele_bolge_sec(komp, aircraft):
    """
    İzin verilen bölgelerden birini rastgele seçer ve o bölgenin
    (x_min, x_max, z_min, z_max) sınırını döndürür.
    """
    bolgeler = komp.izin_verilen_bolgeler
    if not bolgeler:
        r = aircraft.govde_yaricap
        return (0.0, aircraft.govde_uzunluk, -r, r)

    secilen = random.choice(bolgeler)
    x_min, x_max = aircraft.get_bolge_x_siniri(secilen)
    z_min, z_max = aircraft.get_bolge_z_siniri(secilen)
    return (x_min, x_max, z_min, z_max)


def bolge_x_sinirlari(komp, aircraft):
    """Geriye dönük uyumluluk için — sadece X sınırını döndürür."""
    x_min, x_max, _, _, _ = _secili_bolge_sinirlari(komp, aircraft)
    return (x_min, x_max)


def clamp_x_bolge(komp, x, aircraft):
    """X koordinatını komponentin bölge union sınırları içine çeker."""
    x_min, x_max = bolge_x_sinirlari(komp, aircraft)
    return max(x_min, min(x, x_max))


def clamp_xz_bolge(komp, x, z, aircraft):
    """X ve Z koordinatlarını birlikte bölge sınırına çeker."""
    x_min, x_max, z_min, z_max, _ = _secili_bolge_sinirlari(komp, aircraft)
    return max(x_min, min(x, x_max)), max(z_min, min(z, z_max))


def clamp_yz_fuselage(komp, x, y, z, aircraft):
    """
    Y ve Z koordinatlarını o X istasyonundaki fuselage dairesi içine çeker.
    Komponen boyutunun en geniş tarafı kadar güvenlik marjı bırakır.
    """
    r_fus = aircraft.get_fuselage_radius(x)
    _, dy, dz = komp.boyut
    margin = max(dy / 2, dz / 2)
    r_max = max(0.0, r_fus - margin)
    radial = math.sqrt(y ** 2 + z ** 2)
    if radial > r_max and radial > 0:
        scale = r_max / radial
        y *= scale
        z *= scale
    return y, z


# ---------------------------------------------------------------------------
# ÇAKIŞMA KONTROLÜ
# ---------------------------------------------------------------------------

def kutular_cakisiyor_mu(pos1, dim1, pos2, dim2, tolerance=1.5):
    # tolerance kadar kutuların sınırlarını daraltıyoruz (esneklik payı).
    # Örn: tolerance=1.5 ise, parçalar 1.5 cm birbirine girebilir, ceza yemez.
    min1 = [pos1[0] - dim1[0]/2 + tolerance, pos1[1] - dim1[1]/2 + tolerance, pos1[2] - dim1[2]/2 + tolerance]
    max1 = [pos1[0] + dim1[0]/2 - tolerance, pos1[1] + dim1[1]/2 - tolerance, pos1[2] + dim1[2]/2 - tolerance]
    
    min2 = [pos2[0] - dim2[0]/2 + tolerance, pos2[1] - dim2[1]/2 + tolerance, pos2[2] - dim2[2]/2 + tolerance]
    max2 = [pos2[0] + dim2[0]/2 - tolerance, pos2[1] + dim2[1]/2 - tolerance, pos2[2] + dim2[2]/2 - tolerance]
    
    return (
        min1[0] < max2[0] and max1[0] > min2[0] and
        min1[1] < max2[1] and max1[1] > min2[1] and
        min1[2] < max2[2] and max1[2] > min2[2]
    )


# ---------------------------------------------------------------------------
# TASARIM BİREYİ
# ---------------------------------------------------------------------------

class TasarimBireyi:
    def __init__(self):
        self.yerlesim = {}

    def rastgele_yerlestir(self, aircraft):
        """
        Her komponenti izin_verilen_bolgeler listesinden rastgele
        seçilen bir bölgeye, o bölgenin X/Z sınırları içinde yerleştirir.
        YZ düzleminde kutupsal koordinat kullanılarak fuselage içinde
        gerçekçi 3D dağılım sağlanır.
        """
        for komp in aircraft.komponentler_db:
            # Kilitli parçalar sabit pozisyonlarında
            if komp.kilitli:
                self.yerlesim[komp.id] = komp.sabit_pos
                continue

            x_min, x_max, z_min, z_max = _rastgele_bolge_sec(komp, aircraft)

            # X pozisyonu
            x = random.uniform(x_min, x_max)

            # O X kesitindeki fuselage yarıçapı
            r_fus = aircraft.get_fuselage_radius(x)

            # Komponentin taşmaması için maksimum kullanılabilir iç yarıçap
            dx_k, dy, dz = komp.boyut
            # Kesit dışına taşmamak için y ve z yarı genişliklerinin en büyüğü
            margin = max(dy / 2, dz / 2)
            r_max = max(0.0, r_fus - margin)

            # Z bölge kısıtlarına göre açı aralığını belirle
            # TAVAN: z>0 → theta ∈ [0, π], TABAN: z<0 → theta ∈ [π, 2π], diğer: tam daire
            if z_min >= 0:
                # TAVAN bölgesi — yalnızca üst yarı daire
                theta = random.uniform(0, math.pi)
            elif z_max <= 0:
                # TABAN bölgesi — yalnızca alt yarı daire
                theta = random.uniform(math.pi, 2 * math.pi)
            else:
                # Tam daire serbestlik
                theta = random.uniform(0, 2 * math.pi)

            # Kutupsal → kartezyen  (dairesel disk üzerinde uniform dağılım)
            r_place = math.sqrt(random.uniform(0, 1)) * r_max
            y = r_place * math.cos(theta)
            z = r_place * math.sin(theta)

            # Z bölge sınırına sıkıştır
            z = max(z_min, min(z, z_max))

            self.yerlesim[komp.id] = (x, y, z)


# ---------------------------------------------------------------------------
# FİTNESS — Tek Amaçlı (GA / PSO)
# ---------------------------------------------------------------------------

def calculate_fitness_design(birey, aircraft):
    puan = 0
    # O(1) component lookup via pre-built map
    kmap = aircraft.komponentler_map

    # 1. ÇAKIŞMA KONTROLÜ
    # Muafiyetler:
    #  - İki komponent de kilitli (sabit_pos) ise (örn. Pilot ↔ Koltuk_Pilot).
    #  - Kanat içi tanklar (Yakit_Tanki_*): fiziksel olarak gövdenin dışında
    #    kanat içinde konumlanıyor; AABB kabin parçalarıyla yapay çakışıyor.
    cakisma_sayisi = 0
    keys = list(birey.yerlesim.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1_id, k2_id = keys[i], keys[j]
            # Kanat tankı ↔ gövde parçası: farklı hacimde → yapay çakışma, atla
            # Ancak iki yakıt tankı birbiriyle çakışıyorsa kontrol et
            is_tank1 = k1_id.startswith("Yakit_Tanki")
            is_tank2 = k2_id.startswith("Yakit_Tanki")
            if is_tank1 != is_tank2:
                continue
            k1 = kmap[k1_id]
            k2 = kmap[k2_id]
            if k1.kilitli and k2.kilitli:
                continue
            pos1 = birey.yerlesim[k1_id]
            pos2 = birey.yerlesim[k2_id]
            if kutular_cakisiyor_mu(pos1, k1.boyut, pos2, k2.boyut):
                cakisma_sayisi += 1
    puan -= cakisma_sayisi * 10000

    # 2. GÖVDEDEN TAŞMA KONTROLÜ (kanat içi tanklar ve kilitli parçalar muaf)
    # Kilitli parçalar (Motor, koltuklar vb.) sabit konumlara yerleştirilmiş;
    # geometrik taşmaları bilinçli bir tasarım kararıdır, optimizasyonu etkilemez.
    tasma_sayisi = 0
    for k_id, pos in birey.yerlesim.items():
        comp = kmap[k_id]
        if k_id.startswith("Yakit_Tanki") or comp.kilitli:
            continue
        if not aircraft.govde_icinde_mi(pos, comp.boyut):
            tasma_sayisi += 1
    puan -= tasma_sayisi * 5000

    # 3. TİTREŞİM KONTROLÜ
    pos_motor = birey.yerlesim.get("Motor")
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            parca_db = kmap[k_id]
            if parca_db.titresim_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                if mesafe < aircraft.titresim_limiti:
                    ihlal = aircraft.titresim_limiti - mesafe
                    puan -= (ihlal ** 2) * 50

    # 4. SICAKLIK PROFİLİ KONTROLÜ
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            parca_db = kmap[k_id]
            if parca_db.sicaklik_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                if mesafe < aircraft.sicaklik_limiti:
                    ihlal = aircraft.sicaklik_limiti - mesafe
                    puan -= (ihlal ** 2) * 60

    # 5. BÖLGE İHLALİ KONTROLÜ (X ve Z)
    for k_id, pos in birey.yerlesim.items():
        parca_db = kmap[k_id]
        if parca_db.kilitli:
            continue
        x_min, x_max, z_min, z_max, _ = _secili_bolge_sinirlari(parca_db, aircraft)
        px, _, pz = pos
        if px < x_min:
            puan -= (x_min - px) * 100
        elif px > x_max:
            puan -= (px - x_max) * 100
        if pz < z_min:
            puan -= (z_min - pz) * 100
        elif pz > z_max:
            puan -= (pz - z_max) * 100

    # 6. CG (AĞIRLIK MERKEZİ) HESABI
    toplam_cg_hatasi = 0
    dolu_cg_coords = (0, 0, 0)

    for doluluk in aircraft.doluluk_oranlari:
        total_mass = 0
        moment_x = moment_y = moment_z = 0

        for k_id, pos in birey.yerlesim.items():
            db_item = kmap[k_id]
            mass = db_item.agirlik
            if k_id in ("Yakit_Tanki_Sol", "Yakit_Tanki_Sag"):
                mass += aircraft.max_yakit_agirligi * doluluk * 0.5  # Her tank yarı yakıt taşır
            total_mass += mass
            moment_x += mass * pos[0]
            moment_y += mass * pos[1]
            moment_z += mass * pos[2]

        # Guard against degenerate zero-mass configurations
        if total_mass == 0:
            continue

        cg_x = moment_x / total_mass
        cg_y = moment_y / total_mass
        cg_z = moment_z / total_mass

        if doluluk == 1.0:
            dolu_cg_coords = (cg_x, cg_y, cg_z)

        # X için envelope: aralık içinde ceza yok, dışına çıkarsa mesafe kadar.
        if cg_x < aircraft.target_cg_x_min:
            cg_x_err = aircraft.target_cg_x_min - cg_x
        elif cg_x > aircraft.target_cg_x_max:
            cg_x_err = cg_x - aircraft.target_cg_x_max
        else:
            cg_x_err = 0.0
        cg_y_err = abs(cg_y - aircraft.target_cg_y)
        cg_z_err = abs(cg_z - aircraft.target_cg_z)
        dist_error = (cg_x_err**2 + cg_y_err**2 + cg_z_err**2)**0.5
        toplam_cg_hatasi += dist_error

    # 7. YAKIT TANKI KONUM CEZASI (CG Driftini Minimize Etmek İçin)
    # Yakıt tankı CG'den ne kadar uzaksa, yakıt tüketimi dengeyi o kadar bozar.
    for tank_id in ("Yakit_Tanki_Sol", "Yakit_Tanki_Sag"):
        yakit_pos = birey.yerlesim.get(tank_id)
        if yakit_pos:
            target_x_center = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2
            yakit_drift_cezasi = abs(yakit_pos[0] - target_x_center)
            puan -= (yakit_drift_cezasi ** 2) * 25  # iki tank, toplam ağırlık aynı

    n_doluluk = len(aircraft.doluluk_oranlari)
    puan -= (toplam_cg_hatasi / n_doluluk) * 1000 if n_doluluk else 0

    # --- ÖDÜL MEKANİZMALARI ---
    # Not: Ödüller, cezalara (ör: çakışma için -10000) kıyasla çok daha küçüktür (maks 50-150 puan).
    # Böylece algoritma ödül almak için bile bile çakışmaya sebep olmaz, sadece eşitler arasında en iyi dizilimi seçer.

    # Ödül 1: Kompaktlık ve Kablolama (Avionik 1 ve 2 yakınlığı)
    pos_avi1 = birey.yerlesim.get("Aviyonik_1")
    pos_avi2 = birey.yerlesim.get("Aviyonik_2")
    if pos_avi1 and pos_avi2:
        dist_avi = ((pos_avi1[0]-pos_avi2[0])**2 + (pos_avi1[1]-pos_avi2[1])**2 + (pos_avi1[2]-pos_avi2[2])**2)**0.5
        if dist_avi < 50:
            puan += (50 - dist_avi) * 2
            
    # Ödül 2: Eylemsizlik (Ağır parçaların -Batarya- uçağın merkezine yakınlığı)
    pos_bat = birey.yerlesim.get("Batarya_Ana")
    if pos_bat:
        target_x_center = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2
        dist_bat = ((pos_bat[0] - target_x_center)**2 + (pos_bat[1] - aircraft.target_cg_y)**2 + (pos_bat[2] - aircraft.target_cg_z)**2)**0.5
        if dist_bat < 100:
            puan += (100 - dist_bat) * 1.5
            
    # Ödül 3: Ekstra Güvenlik Marjı
    pos_motor = birey.yerlesim.get("Motor")
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            parca_db = kmap[k_id]
            if parca_db.titresim_hassasiyeti or parca_db.sicaklik_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                lim = max(aircraft.titresim_limiti, aircraft.sicaklik_limiti)
                if mesafe > lim:
                    # Sınırı aşıp uzaklaştığı için ödül (Sonsuza kaçmaması için min ile limitlendi)
                    ekstra = mesafe - lim
                    puan += min(ekstra, 50) * 1.0

    # Ödül 4: Kusursuz Simetri (Yakıt Tankları)
    pos_tank_sol = birey.yerlesim.get("Yakit_Tanki_Sol")
    pos_tank_sag = birey.yerlesim.get("Yakit_Tanki_Sag")
    if pos_tank_sol and pos_tank_sag:
        x_diff = abs(pos_tank_sol[0] - pos_tank_sag[0])
        y_sym = abs(pos_tank_sol[1] + pos_tank_sag[1])  # biri +, biri - olmalı, toplam 0 olmalı
        z_diff = abs(pos_tank_sol[2] - pos_tank_sag[2])
        total_diff = x_diff + y_sym + z_diff
        if total_diff < 15:
            puan += (15 - total_diff) * 10

    return puan, dolu_cg_coords


# ---------------------------------------------------------------------------
# FİTNESS — Çok Amaçlı (NSGA-II)
# ---------------------------------------------------------------------------

def calculate_fitness_nsga2(birey, aircraft):
    """
    NSGA-II için iki hedef:
    1. Ceza Puanı (Minimize): Çakışma, Taşma, Titreşim, Sıcaklık, Bölge ihlali
    2. CG Hatası  (Minimize): Ağırlık merkezinin hedeften sapması
    """
    ceza_puani = 0.0
    # O(1) component lookup via pre-built map
    kmap = aircraft.komponentler_map

    # 1. ÇAKIŞMA
    # Muafiyetler: (a) iki komponent de kilitli (Pilot ↔ Koltuk_Pilot gibi),
    # (b) kanat içi tanklar (Yakit_Tanki_*) — gövde AABB'si dışındalar.
    cakisma_sayisi = 0
    keys = list(birey.yerlesim.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1_id, k2_id = keys[i], keys[j]
            # Kanat tankı ↔ gövde parçası: farklı hacimde → yapay çakışma, atla
            # Ancak iki yakıt tankı birbiriyle çakışıyorsa kontrol et
            is_tank1 = k1_id.startswith("Yakit_Tanki")
            is_tank2 = k2_id.startswith("Yakit_Tanki")
            if is_tank1 != is_tank2:
                continue
            k1 = kmap[k1_id]
            k2 = kmap[k2_id]
            if k1.kilitli and k2.kilitli:
                continue
            pos1 = birey.yerlesim[k1_id]
            pos2 = birey.yerlesim[k2_id]
            if kutular_cakisiyor_mu(pos1, k1.boyut, pos2, k2.boyut):
                cakisma_sayisi += 1
    ceza_puani += cakisma_sayisi * 10000

    # 2. TAŞMA (kanat içi tanklar ve kilitli parçalar muaf)
    tasma_sayisi = 0
    for k_id, pos in birey.yerlesim.items():
        comp = kmap[k_id]
        if k_id.startswith("Yakit_Tanki") or comp.kilitli:
            continue
        if not aircraft.govde_icinde_mi(pos, comp.boyut):
            tasma_sayisi += 1
    ceza_puani += tasma_sayisi * 5000

    # 3. TİTREŞİM
    pos_motor = birey.yerlesim.get("Motor")
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            parca_db = kmap[k_id]
            if parca_db.titresim_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                if mesafe < aircraft.titresim_limiti:
                    ihlal = aircraft.titresim_limiti - mesafe
                    ceza_puani += (ihlal ** 2) * 50

    # 4. SICAKLIK
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            parca_db = kmap[k_id]
            if parca_db.sicaklik_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                if mesafe < aircraft.sicaklik_limiti:
                    ihlal = aircraft.sicaklik_limiti - mesafe
                    ceza_puani += (ihlal ** 2) * 60

    # 5. BÖLGE İHLALİ (X ve Z)
    for k_id, pos in birey.yerlesim.items():
        parca_db = kmap[k_id]
        if parca_db.kilitli:
            continue
        x_min, x_max, z_min, z_max, _ = _secili_bolge_sinirlari(parca_db, aircraft)
        px, _, pz = pos
        if px < x_min:
            ceza_puani += (x_min - px) * 100
        elif px > x_max:
            ceza_puani += (px - x_max) * 100
        if pz < z_min:
            ceza_puani += (z_min - pz) * 100
        elif pz > z_max:
            ceza_puani += (pz - z_max) * 100

    # 6. CG HATASI
    toplam_cg_hatasi = 0
    dolu_cg_coords = (0, 0, 0)

    for doluluk in aircraft.doluluk_oranlari:
        total_mass = 0
        moment_x = moment_y = moment_z = 0
        for k_id, pos in birey.yerlesim.items():
            db_item = kmap[k_id]
            mass = db_item.agirlik
            if k_id in ("Yakit_Tanki_Sol", "Yakit_Tanki_Sag"):
                mass += aircraft.max_yakit_agirligi * doluluk * 0.5
            total_mass += mass
            moment_x += mass * pos[0]
            moment_y += mass * pos[1]
            moment_z += mass * pos[2]

        # Guard against degenerate zero-mass configurations
        if total_mass == 0:
            continue

        cg_x = moment_x / total_mass
        cg_y = moment_y / total_mass
        cg_z = moment_z / total_mass

        if doluluk == 1.0:
            dolu_cg_coords = (cg_x, cg_y, cg_z)

        # X için envelope: aralık içinde ceza yok, dışına taşma mesafesi kadar.
        if cg_x < aircraft.target_cg_x_min:
            cg_x_err = aircraft.target_cg_x_min - cg_x
        elif cg_x > aircraft.target_cg_x_max:
            cg_x_err = cg_x - aircraft.target_cg_x_max
        else:
            cg_x_err = 0.0
        cg_y_err = abs(cg_y - aircraft.target_cg_y)
        cg_z_err = abs(cg_z - aircraft.target_cg_z)
        dist_error = (cg_x_err**2 + cg_y_err**2 + cg_z_err**2)**0.5
        toplam_cg_hatasi += dist_error

    # 7. YAKIT TANKI KONUM CEZASI
    for tank_id in ("Yakit_Tanki_Sol", "Yakit_Tanki_Sag"):
        yakit_pos = birey.yerlesim.get(tank_id)
        if yakit_pos:
            target_x_center = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2
            yakit_drift_cezasi = abs(yakit_pos[0] - target_x_center)
            ceza_puani += (yakit_drift_cezasi ** 2) * 25

    n_doluluk = len(aircraft.doluluk_oranlari)
    cg_hatasi = (toplam_cg_hatasi / n_doluluk) if n_doluluk else 0.0

    # --- ÖDÜL MEKANİZMALARI ---
    # NSGA-II'de ceza_puani minimize edildiği için, ödüller ceza_puani'ndan çıkarılır (negatif etki).

    # Ödül 1: Kompaktlık ve Kablolama (Avionik 1 ve 2 yakınlığı)
    pos_avi1 = birey.yerlesim.get("Aviyonik_1")
    pos_avi2 = birey.yerlesim.get("Aviyonik_2")
    if pos_avi1 and pos_avi2:
        dist_avi = ((pos_avi1[0]-pos_avi2[0])**2 + (pos_avi1[1]-pos_avi2[1])**2 + (pos_avi1[2]-pos_avi2[2])**2)**0.5
        if dist_avi < 50:
            ceza_puani -= (50 - dist_avi) * 2
            
    # Ödül 2: Eylemsizlik (Ağır parçaların -Batarya- uçağın merkezine yakınlığı)
    pos_bat = birey.yerlesim.get("Batarya_Ana")
    if pos_bat:
        target_x_center = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2
        dist_bat = ((pos_bat[0] - target_x_center)**2 + (pos_bat[1] - aircraft.target_cg_y)**2 + (pos_bat[2] - aircraft.target_cg_z)**2)**0.5
        if dist_bat < 100:
            ceza_puani -= (100 - dist_bat) * 1.5
            
    # Ödül 3: Ekstra Güvenlik Marjı
    pos_motor = birey.yerlesim.get("Motor")
    if pos_motor:
        for k_id, pos in birey.yerlesim.items():
            parca_db = kmap[k_id]
            if parca_db.titresim_hassasiyeti or parca_db.sicaklik_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                lim = max(aircraft.titresim_limiti, aircraft.sicaklik_limiti)
                if mesafe > lim:
                    ekstra = mesafe - lim
                    ceza_puani -= min(ekstra, 50) * 1.0

    # Ödül 4: Kusursuz Simetri (Yakıt Tankları)
    pos_tank_sol = birey.yerlesim.get("Yakit_Tanki_Sol")
    pos_tank_sag = birey.yerlesim.get("Yakit_Tanki_Sag")
    if pos_tank_sol and pos_tank_sag:
        x_diff = abs(pos_tank_sol[0] - pos_tank_sag[0])
        y_sym = abs(pos_tank_sol[1] + pos_tank_sag[1])
        z_diff = abs(pos_tank_sol[2] - pos_tank_sag[2])
        total_diff = x_diff + y_sym + z_diff
        if total_diff < 15:
            ceza_puani -= (15 - total_diff) * 10

    # Tek-amaçlı skor (GA fitness ile eşdeğer): score = -(ceza) - cg*1000
    score = -ceza_puani - cg_hatasi * 1000

    return ceza_puani, cg_hatasi, dolu_cg_coords, score