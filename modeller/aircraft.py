class Aircraft:
    def __init__(self, govde_uzunluk, govde_cap,
                 target_cg_x_min, target_cg_x_max, target_cg_y, target_cg_z,
                 max_yakit_agirligi, titresim_limiti, sicaklik_limiti, komponentler_db):

        self.govde_uzunluk = govde_uzunluk
        self.govde_cap = govde_cap
        self.govde_yaricap = govde_cap / 2

        self.target_cg_x_min = target_cg_x_min
        self.target_cg_x_max = target_cg_x_max
        self.target_cg_y = target_cg_y
        self.target_cg_z = target_cg_z

        self.max_yakit_agirligi = max_yakit_agirligi
        self.titresim_limiti = titresim_limiti
        self.sicaklik_limiti = sicaklik_limiti
        self.komponentler_db = komponentler_db

        # O(1) component lookup — avoids repeated O(n) list scans in fitness
        self.komponentler_map: dict = {k.id: k for k in komponentler_db}

        # X position where the cylindrical mid-section transitions into the
        # tapering tail.  Derived from fuselage length so it works for any size.
        self._taper_start_x = govde_uzunluk * 0.6

        self.doluluk_oranlari = [0.0, 0.25, 0.5, 0.75, 1.0]

        # BÖLGE X SINIR TANIMLARI
        self.bolge_burun_son   = 45.0                           # BURUN:  0 → 45
        self.bolge_govde_bas   = 45.0
        self.bolge_govde_son   = self.govde_uzunluk - 45.0      # GOVDE: 45 → 255
        self.bolge_kuyruk_bas  = self.govde_uzunluk - 45.0      # KUYRUK: 255 → 300
        # Aviyonik bölge (eskiyle uyumluluk için korunuyor)
        self.bolge_aviyonik_bas = 80.0
        self.bolge_aviyonik_son = 180.0

    # ------------------------------------------------------------------
    # Bölge Tanımları
    # ------------------------------------------------------------------
    BOLGE_ISIMLERI = ["BURUN", "GOVDE", "KUYRUK", "TAVAN", "TABAN"]

    # TAVAN/TABAN bölgeleri tüm X boyunca geçerli; sadece Z (yükseklik)
    # eksenini kısıtlar. BURUN/GOVDE/KUYRUK ise X eksenini kısıtlar.
    # Bir bileşen hem X hem Z bölgesinde tanımlanabilir (örn. BURUN+TABAN).

    def get_bolge_x_siniri(self, bolge_adi):
        """
        Belirtilen bölgenin X eksenindeki (x_min, x_max) sınırını döndürür.
        TAVAN/TABAN için X tüm uzunluktur.
        """
        if bolge_adi == "BURUN":
            return (0.0, self.bolge_burun_son)
        elif bolge_adi == "GOVDE":
            return (self.bolge_govde_bas, self.bolge_govde_son)
        elif bolge_adi == "KUYRUK":
            return (self.bolge_kuyruk_bas, self.govde_uzunluk)
        elif bolge_adi in ("TAVAN", "TABAN"):
            # TAVAN/TABAN X'te serbest (GOVDE aralığını baz al)
            return (self.bolge_govde_bas, self.bolge_govde_son)
        else:
            return (0.0, self.govde_uzunluk)

    def get_bolge_z_siniri(self, bolge_adi):
        """
        Belirtilen bölgenin Z eksenindeki (z_min, z_max) sınırını döndürür.
        TAVAN → z > 0, TABAN → z < 0, diğerleri tüm Z serbest.
        """
        r = self.govde_yaricap
        if bolge_adi == "TAVAN":
            return (0.0, r)
        elif bolge_adi == "TABAN":
            return (-r, 0.0)
        else:
            return (-r, r)

    def get_bolge_siniri(self, bolge_adi):
        """
        (x_min, x_max, z_min, z_max) döndürür.
        """
        x_min, x_max = self.get_bolge_x_siniri(bolge_adi)
        z_min, z_max = self.get_bolge_z_siniri(bolge_adi)
        return (x_min, x_max, z_min, z_max)

    # ------------------------------------------------------------------

    def get_fuselage_radius(self, x):
        """
        Verilen X konumundaki gövde yarıçapını döndürür.
        Burun kavisli, orta düz, kuyruk incelen.
        """
        if x < 0: return 0.0
        if x > self.govde_uzunluk: return 0.0

        if x < self.bolge_burun_son:
            # Burun kısmı (Parabolik artış)
            return (x / self.bolge_burun_son) ** 0.5 * self.govde_yaricap
        elif x < self._taper_start_x:
            # Orta gövde (Sabit silindir)
            return self.govde_yaricap
        else:
            # Kuyruk kısmı (Lineer incelme)
            ratio = (x - self._taper_start_x) / (self.govde_uzunluk - self._taper_start_x)
            return self.govde_yaricap * (1 - ratio * 0.8)

    def govde_icinde_mi(self, pos, dim, tolerance=2.0):
        x, y, z = pos
        dx, dy, dz = dim

        # 1. Boylamasına (X ekseni) kontrol - Tolerans ile
        x_min = x - dx / 2
        x_max = x + dx / 2

        if x_min < -tolerance or x_max > self.govde_uzunluk + tolerance:
            return False

        # 2. Radyal (Kesit) kontrolü
        # Parçanın merkezden en uzak köşesinin mesafesi
        part_radial_dist = ((abs(y) + dy / 2) ** 2 + (abs(z) + dz / 2) ** 2) ** 0.5

        # Kontrol noktaları: Ön, Orta, Arka (X değerlerini gövde içinde kalacak şekilde sınırla)
        check_points = [max(0.0, x_min), min(self.govde_uzunluk, x), min(self.govde_uzunluk, x_max)]

        for cx in check_points:
            allowed_radius = self.get_fuselage_radius(cx)
            # Tolerans (hata payı) kadar gövdenin dışına çıkmaya müsaade et
            if part_radial_dist > (allowed_radius + tolerance):
                return False

        return True