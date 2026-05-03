class Komponent:
    """
    Uçak içi sistem bileşenini temsil eder.

    izin_verilen_bolgeler: Bu komponentin yerleştirilebileceği bölgeler listesi.
        Geçerli bölgeler: "BURUN", "GOVDE", "KUYRUK", "TAVAN", "TABAN"
        Birden fazla bölge verilebilir → algoritma bu bölgelerden birini seçer.
        None verilirse tüm gövde serbest kabul edilir.
    """
    def __init__(self, id, agirlik, boyut,
                 izin_verilen_bolgeler=None,
                 sabit_pos=None, kilitli=False,
                 titresim_hassasiyeti=False, sicaklik_hassasiyeti=False):
        self.id = id
        self.agirlik = agirlik
        self.boyut = boyut
        # ["BURUN"], ["GOVDE", "TABAN"], ["TAVAN"] vb.
        self.izin_verilen_bolgeler = izin_verilen_bolgeler or []
        self.sabit_pos = sabit_pos
        self.kilitli = kilitli
        self.titresim_hassasiyeti = titresim_hassasiyeti
        self.sicaklik_hassasiyeti = sicaklik_hassasiyeti