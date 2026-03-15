class Aircraft:
    def __init__(self, govde_uzunluk, govde_cap, 
                 target_cg_x_min, target_cg_x_max, target_cg_y, target_cg_z, 
                 max_yakit_agirligi, titresim_limiti, komponentler_db):
        
        self.govde_uzunluk = govde_uzunluk
        self.govde_cap = govde_cap
        self.govde_yaricap = govde_cap / 2
        
        self.target_cg_x_min = target_cg_x_min
        self.target_cg_x_max = target_cg_x_max
        self.target_cg_y = target_cg_y
        self.target_cg_z = target_cg_z
        
        self.max_yakit_agirligi = max_yakit_agirligi
        self.titresim_limiti = titresim_limiti
        self.komponentler_db = komponentler_db
        
        self.doluluk_oranlari = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # BÖLGE TANIMLARI (Sınırlar)
        self.bolge_burun_son = 40.0
        self.bolge_kuyruk_bas = self.govde_uzunluk - 40.0 
        
    def get_fuselage_radius(self, x):
        """
        Verilen X konumundaki gövde yarıçapını döndürür.
        Farklı uçak tipleri için bu fonksiyon değiştirilebilir.
        Şu anki model: Burun kavisli, orta düz, kuyruk incelen.
        """
        if x < 0: return 0.0
        if x > self.govde_uzunluk: return 0.0
        
        if x < self.bolge_burun_son: 
            # Burun kısmı (Parabolik artış)
            return (x/self.bolge_burun_son)**0.5 * self.govde_yaricap
        elif x < 180:  
            # Orta gövde (Sabit silindir)
            return self.govde_yaricap
        else: 
            # Kuyruk kısmı (Lineer incelme)
            # 180'den 300'e giderken yarıçap %100'den %20'ye düşüyor
            ratio = (x - 180) / (self.govde_uzunluk - 180)
            return self.govde_yaricap * (1 - ratio * 0.8)

    def govde_icinde_mi(self, pos, dim):
        x, y, z = pos
        dx, dy, dz = dim
        
        # 1. Boylamasına (X ekseni) kontrol
        x_min = x - dx/2
        x_max = x + dx/2
        
        if x_min < 0 or x_max > self.govde_uzunluk:
            return False
        
        # 2. Radyal (Kesit) kontrolü
        # Parçanın kesit köşegeni (Merkezden en uzak nokta)
        part_radial_dist = ((abs(y) + dy/2)**2 + (abs(z) + dz/2)**2)**0.5
        
        # Kontrol edilecek noktalar: Ön, Orta, Arka
        check_points = [x_min, x, x_max]
        
        for cx in check_points:
            allowed_radius = self.get_fuselage_radius(cx)
            if part_radial_dist > allowed_radius:
                return False
                
        return True
