class Komponent:
    def __init__(self, id, agirlik, boyut, sabit_bolge=None, sabit_pos=None, kilitli=False, titresim_hassasiyeti=False, sicaklik_hassasiyeti=False):
        self.id = id
        self.agirlik = agirlik
        self.boyut = boyut
        self.sabit_bolge = sabit_bolge
        self.sabit_pos = sabit_pos
        self.kilitli = kilitli
        self.titresim_hassasiyeti = titresim_hassasiyeti
        self.sicaklik_hassasiyeti = sicaklik_hassasiyeti