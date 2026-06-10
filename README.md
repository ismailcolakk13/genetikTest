# AeroOpt - Uçak İçi Sistem Yerleştirme Optimizasyonu

Uçak gövdesi içindeki bileşenlerin (motor, batarya, aviyonik, yakıt tankları, koltuklar vb.) optimal konumlarını belirlemek için geliştirilmiş, evrimsel ve sürü tabanlı optimizasyon algoritmaları kullanan bir bitirme tezi projesidir.

Sistem; ağırlık merkezi (CG) dengesini, fiziksel kısıtlamaları (çakışma, taşma), sıcaklık ve titreşim güvenliği gibi mühendislik hedeflerini eş zamanlı olarak optimize eder. Sonuçlar hem terminal üzerinden detaylı analiz çıktısı olarak hem de tarayıcı tabanlı interaktif 3D görselleştirme ile sunulur.

**Yazarlar:** İsmail Çolak, Mehmet Can Çalışkan, Yusuf Eren Aykurt

---

## İçindekiler

1. [Proje Hakkında](#proje-hakkında)
2. [Özellikler](#özellikler)
3. [Sistem Gereksinimleri](#sistem-gereksinimleri)
4. [Kurulum](#kurulum)
5. [Çalıştırma](#çalıştırma)
   - [Terminal (CLI) Modu](#terminal-cli-modu)
   - [Web Arayüzü](#web-arayüzü)
6. [Proje Yapısı](#proje-yapısı)
7. [Algoritmalar](#algoritmalar)
8. [Fitness Fonksiyonu ve Ceza Sistemi](#fitness-fonksiyonu-ve-ceza-sistemi)
9. [Mühendislik Analizi](#mühendislik-analizi)
10. [Test Paketi](#test-paketi)
11. [Konfigürasyon ve Parametreler](#konfigürasyon-ve-parametreler)
12. [Bilinen Kısıtlamalar](#bilinen-kısıtlamalar)

---

## Proje Hakkında

Bu proje, bir uçağın gövde içi sistem yerleştirme problemini çok amaçlı bir optimizasyon problemi olarak ele alır. Gerçek bir uçak tasarım sürecinde, motor, batarya, aviyonik kutular, yakıt tankları, koltuklar ve diğer bileşenlerin gövde içine yerleştirilmesi; uçuş güvenliği, yapısal bütünlük ve performans açısından kritik bir mühendislik kararıdır.

Problem şunları içerir:
- Her bileşen belirli bir bölgeye (burun, gövde, kuyruk, tavan, taban) yerleştirilebilir
- Bazı bileşenler sabit konumdadır ve yerinden oynatılamaz (örneğin motor, pilot koltukları)
- Bileşenler birbirleriyle çakışmamalı ve gövde dışına taşmamalıdır
- Isıya ve titreşime hassas parçalar motordan yeterince uzak olmalıdır
- Ağırlık merkezi (CG) belirlenen hedef aralıkta kalmalıdır
- Yakıt tüketildikçe CG kayması (drift) minimum olmalıdır

Cessna 172 tipi bir hafif uçak referans alınarak varsayılan bileşen seti oluşturulmuştur; ancak kullanıcı kendi bileşen ve gövde parametrelerini tanımlayabilir.

---

## Özellikler

- Dört farklı optimizasyon algoritması: GA, PSO, NSGA-II ve Hibrit NSGA-II + PSO
- Terminal tabanlı interaktif çalıştırma (main.py)
- Web tabanlı kullanıcı arayüzü (FastAPI + statik HTML/JS/CSS)
- İnteraktif 3D Plotly görselleştirme (uçak gövdesi, bölgeler, bileşenler ve CG konumu)
- Detaylı mühendislik analizi (CG sapma, yakıt drift, sıcaklık profili, titreşim profili)
- Hazır Cessna 172 preset konfigürasyonu
- Otomatik test paketi (her algoritma için 10'ar çalıştırma, sonuçları CSV'ye kaydetme)
- Çoklu yakıt doluluk seviyelerinde CG hesabı (boş, çeyrek, yarı, üçte üç, tam)

---

## Sistem Gereksinimleri

- **Python 3.9 veya üzeri** (3.10+ tavsiye edilir)
- **pip** (Python paket yöneticisi)
- Modern bir web tarayıcısı (Chrome, Firefox, Edge — 3D görselleştirme için)
- İşletim sistemi: Windows, macOS veya Linux

---

## Kurulum

Projeyi bilgisayarınıza indirdikten sonra aşağıdaki adımları takip edin.

### 1. Projeyi Klonlayın (veya ZIP olarak indirin)

```bash
git clone https://github.com/ismailcolakk13/genetikTest.git
cd genetikTest
```

ZIP olarak indirdiyseniz, arşivi bir klasöre çıkartın ve terminal ile o klasöre girin.

### 2. Sanal Ortam Oluşturun (Tavsiye Edilir)

Sanal ortam kullanmak, projenin bağımlılıklarını sisteminizin diğer Python paketlerinden izole eder. Zorunlu değil ama şiddetle tavsiye edilir.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Sanal ortam aktif olduğunuzda terminal satırının başında `(.venv)` göreceksiniz.

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

Bu komut FastAPI, Uvicorn, NumPy, Matplotlib, Plotly ve diğer gerekli kütüphaneleri kuracaktır.

Kurulum tamamlandıktan sonra projeyi kullanmaya hazırsınız.

---

## Çalıştırma

Projeyi iki farklı şekilde kullanabilirsiniz: terminal üzerinden veya web arayüzü üzerinden.

### Terminal (CLI) Modu

Terminal modu, kodun doğrudan Python ile çalıştırılmasını sağlar. Varsayılan Cessna 172 bileşen seti ile hızlı bir simülasyon yapmak için uygundur.

```bash
python main.py
```

Çalıştırdığınızda şu ekranla karşılaşacaksınız:

```
[Pilot] Bu simülasyon için pilot kilosu: 92.3 kg

--- SİMÜLASYON BAŞLATILIYOR ---
Lütfen çalıştırmak istediğiniz algoritmayı seçin:
1 - Genetik Algoritma (GA)
2 - Parçacık Sürüsü Optimizasyonu (PSO)
3 - NSGA-II (Çok Amaçlı Optimizasyon)
4 - Karma (Hybrid) NSGA-II + PSO
Seçiminiz (1/2/3/4):
```

Algoritmalardan birinin numarasını girip Enter'a basın. Simülasyon başlar ve nesil nesil ilerleme durumunu terminalde gösterir. Bittikten sonra:

1. Terminalde detaylı mühendislik analizi bastırılır (CG sapma, yakıt drift, sıcaklık/titreşim profili)
2. Otomatik olarak 3D Plotly görselleştirmesi tarayıcınızda açılır

3D görselleştirmeyi açmak istemiyorsanız (örneğin sunucu ortamında çalıştırıyorsanız):

```bash
NO_VIZ=1 python main.py
```

### Web Arayüzü

Web arayüzü, parametreleri görsel olarak ayarlamanıza, farklı konfigürasyonları denemenize ve sonuçları interaktif bir panoda görmenize olanak tanır.

**Sunucuyu başlatmak için:**

```bash
python app.py
```

veya alternatif olarak:

```bash
uvicorn app:app --reload
```

Sunucu başlatıldıktan sonra tarayıcınızda şu adrese gidin:

```
http://localhost:8000
```

Web arayüzünde yapabilecekleriniz:

- Sol panelden gövde uzunluğu, çap, hedef CG aralığı, yakıt kapasitesi, sıcaklık/titreşim limitleri gibi global parametreleri ayarlayın
- Hazır Cessna 172 preset'ini seçin veya kendi bileşen listenizi oluşturun
- Bileşen tablosunda her parçayı düzenleyin: isim, ağırlık, boyut, izin verilen bölge, sabit konum, kilitli/titreşim hassas/sıcaklık hassas bayrakları
- Algoritma, popülasyon büyüklüğü ve nesil sayısını belirleyin
- "Run Optimization Sequence" butonuna tıklayın

Simülasyon tamamlandığında sayfa üzerinde:
- En iyi fitness skoru, hesaplanan CG koordinatları ve kullanılan algoritma gösterilir
- Mühendislik analizi (CG, fiziksel ihlal, sıcaklık, titreşim, denge) kart şeklinde sunulur
- İnteraktif 3D uçak görselleştirmesi iframe içinde yüklenir (fare ile döndürebilir, yakınlaştırabilirsiniz)

---

## Proje Yapısı

```
genetikTest/
│
├── app.py                    # FastAPI web sunucusu (API endpointleri + statik dosya servisi)
├── main.py                   # Terminal tabanlı çalıştırma noktası (CLI)
├── requirements.txt          # Python bağımlılıkları
├── test_suite.py             # Otomatik test paketi (4 algoritma x 10 çalıştırma)
│
├── algoritmalar/             # Optimizasyon algoritmaları
│   ├── __init__.py
│   ├── ga.py                 # Genetik Algoritma (GA)
│   ├── pso.py                # Parçacık Sürüsü Optimizasyonu (PSO)
│   ├── nsga2.py              # NSGA-II (Çok Amaçlı)
│   └── nsga2_pso_hybrid.py   # Hibrit NSGA-II + PSO
│
├── modeller/                 # Veri modelleri
│   ├── __init__.py
│   ├── aircraft.py           # Uçak gövde modeli (bölgeler, fuselage geometrisi, CG)
│   └── komponent.py          # Bileşen (komponent) veri sınıfı
│
├── yardimcilar/              # Yardımcı fonksiyonlar
│   ├── __init__.py
│   ├── yardimciFonksiyonlar.py  # Fitness hesabı, çakışma kontrolü, bölge sınır yönetimi
│   ├── gorsellestirici.py       # Plotly 3D görselleştirme (uçak gövdesi, bölgeler, parçalar)
│   └── yerlesimAnaliz.py        # Mühendislik analizi (CG, drift, sıcaklık, titreşim)
│
└── static/                   # Web arayüzü frontend dosyaları
    ├── index.html            # Ana sayfa HTML
    ├── style.css             # Stil dosyası
    └── script.js             # Frontend JavaScript (API iletişimi, tablo yönetimi)
```

---

## Algoritmalar

Projede dört farklı optimizasyon algoritması uygulanmıştır. Her biri aynı fitness fonksiyonunu kullanır ancak çözüm arama stratejileri farklıdır.

### 1. Genetik Algoritma (GA)

Klasik evrimsel bir yaklaşımdır. Bir popülasyon oluşturulur, her nesilde en iyi bireyler seçilir (elitizm), çaprazlama ve mutasyon ile yeni bireyler üretilir.

- Seçim yöntemi: Turnuva seçimi (k=3)
- Elitizm: Popülasyonun en iyi yüzde 20'si bir sonraki nesle doğrudan aktarılır
- Adaptif mutasyon: Erken nesillerde geniş keşif (oran=0.3), geç nesillerde hassas iyileştirme (oran=0.1)

### 2. Parçacık Sürüsü Optimizasyonu (PSO)

Her bileşen için hız ve konum vektörleri tutan parçacıklardan oluşan bir sürü kullanır. Her parçacık kendi en iyi konumuna (pbest) ve sürünün global en iyi konumuna (gbest) doğru çekilir.

- Adaptif atalet katsayısı: 0.9'dan (keşif) 0.4'e (sömürü) doğrusal azalma
- Sıkışma tespiti: Global skor çok düşükse, sürünün en kötü yüzde 30'u sıfırdan başlatılır
- Maksimum hız sınırı: Her eksende 20 cm/nesil

### 3. NSGA-II (Çok Amaçlı)

İki hedefi eş zamanlı olarak minimize eder:
- Hedef 1: Ceza puanı (çakışma, taşma, titreşim, sıcaklık ihlalleri)
- Hedef 2: CG hatası (ağırlık merkezinin hedeften sapması)

Non-dominated sıralama ve kalabalık mesafesi (crowding distance) kullanılarak Pareto frontu oluşturulur. Son seçim için normalize edilmiş ideal noktaya en yakın çözüm belirlenir.

### 4. Hibrit NSGA-II + PSO (Varsayılan)

NSGA-II'nin çok amaçlı sıralama mekanizmasını, PSO'nun hız tabanlı arama yeteneği ile birleştirir. Ek olarak GA'dan çaprazlama ve mutasyon operatörleri de kullanılır.

- Global non-dominated arşiv (maks 200 çözüm)
- PSO hareketi: Parçacıklar, arşivdeki lider çözümlere doğru çekilir
- GA enjeksiyonu: Her nesilde popülasyonun üçte biri çaprazlama çocuğu ile değiştirilir
- Sıkışma mekanizması: Sürü sıkışmışsa en kötü yüzde 30 sıfırdan başlatılır, mutasyon oranı artırılır

Bu algoritma, diğer üç algoritmaya kıyasla genel olarak en kararlı ve en yüksek kaliteli sonuçları üretmektedir.

---

## Fitness Fonksiyonu ve Ceza Sistemi

Her bileşen yerleştirmesi şu kriterlere göre değerlendirilir:

**Ceza Kalemleri (Skoru Düşürenler):**

| Kriter | Ceza Miktarı | Açıklama |
|--------|-------------|----------|
| Bileşen çakışması | -10.000 / çift | İki parçanın kutu geometrileri üst üste geldiğinde |
| Gövdeden taşma | -5.000 / parça | Fuselage dairesel kesitinin dışına çıkan parçalar |
| Titreşim ihlali | -(ihlal²) x 50 | Motora yakın hassas parçalar (mesafe < limit) |
| Sıcaklık ihlali | -(ihlal²) x 60 | Motora yakın ısıya hassas parçalar |
| Bölge ihlali | -100 / cm | Parçaların tanımlı bölge dışına çıkması |
| CG sapması | -1000 x ortalama hata | Beş yakıt doluluk seviyesi üzerinden ortalama |
| Yakıt tankı drift | -(sapma²) x 25 | Tankın CG hedef merkezinden uzaklığı |

**Ödül Kalemleri (Skoru Yükseltenler):**

| Kriter | Ödül Miktarı | Açıklama |
|--------|-------------|----------|
| Aviyonik yakınlığı | +2 / cm (maks 100) | Aviyonik 1 ve 2'nin birbirine yakınlığı (kablolama) |
| Batarya merkeziliği | +1.5 / cm (maks 150) | Ana bataryanın uçak merkezine yakınlığı |
| Ekstra güvenlik marjı | +1 / cm (maks 50) | Hassas parçaların limitin ötesinde uzaklığı |
| Yakıt tankı simetrisi | +10 / cm (maks 150) | Sol ve sağ tankın simetrik yerleşimi |

---

## Mühendislik Analizi

Simülasyon sonuçlarında şunlar raporlanır:

**CG Hedef Kontrolü:** Hesaplanan ağırlık merkezinin X, Y, Z koordinatlarının hedef aralığa olan uzaklığını değerlendirir. 2 cm'den az sapma "çok iyi", 15 cm'den az "kabul edilebilir" olarak nitelendirilir.

**Yakıt Tankı Etkisi:** Yakıt tanklarının hedef CG merkezine olan uzaklığını hesaplar. Tanklar merkezden uzaksa, yakıt tüketildikçe denge bozulur.

**Fiziksel İhlal:** Çakışma ve taşma durumlarını doğrudan kontrol eder. Çakışma veya taşma varsa tasarım "Zayıf", yoksa skora göre "Çok İyi" veya "Kabul Edilebilir" olarak sınıflandırılır.

**Sıcaklık Profili:** Isıya hassas parçaların (batarya, aviyonikler) motora olan mesafesini ölçer ve sıcaklık limitiyle karşılaştırır.

**Titreşim Profili:** Titreşime hassas parçaların (aviyonikler, kamera) motora olan mesafesini ölçer.

**Denge Analizi (CG Drift):** Yakıt tankları dolu ve boş iken CG'nin X eksenindeki kaymasını hesaplar. 2 cm'den az kayma "mükemmel", 5 cm'den fazla kayma "kritik" olarak değerlendirilir.

---

## Test Paketi

Test paketi, dört algoritmayı her biri 10 kez çalıştırarak istatistiksel karşılaştırma yapar.

```bash
python test_suite.py
```

Bu komut aşağıdakileri yapar:

1. Her algoritmayı (GA, PSO, NSGA-II, Hibrit) 10'ar kez çalıştırır
2. Her çalıştırma için skoru, tasarım durumunu, sıcaklık ve titreşim risklerini kaydeder
3. Sonuçları `test_results.csv` ve `test_results.txt` dosyalarına yazar
4. Terminalde algoritma bazında özet tablo gösterir (ortalama skor, başarı oranı)

Test paketi çalıştırma sırasında 3D görselleştirmeyi otomatik olarak devre dışı bırakır (NO_VIZ=1).

Test süresinin uzun olabileceğini göz önünde bulundurun: 4 algoritma x 10 çalıştırma x 50 nesil x 100 birey demek toplamda yaklaşık 200.000 fitness değerlendirmesi anlamına gelir.

---

## Konfigürasyon ve Parametreler

### Gövde Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| Gövde Uzunluğu | 300 cm | Uçağın burundan kuyrğa toplam uzunluğu |
| Gövde Çapı | 60 cm | Gövdenin en geniş noktasındaki dış çap |
| Hedef CG X (Min/Max) | 90-110 cm | Ağırlık merkezinin kabul edilen X aralığı |
| Hedef CG Y | 0 cm | Ağırlık merkezinin hedef Y (simetri ekseni) |
| Hedef CG Z | 0 cm | Ağırlık merkezinin hedef Z (orta yükseklik) |
| Maks Yakıt Ağırlığı | 50 kg | Tam depodaki toplam yakıt kütlesi |
| Sıcaklık Limiti | 30 cm | Isıya hassas parçaların motora min. mesafesi |
| Titreşim Limiti | 50 cm | Titreşime hassas parçaların motora min. mesafesi |

### Algoritma Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| Popülasyon Büyüklüğü | 100 | Her nesildeki birey sayısı |
| Nesil Sayısı | 50 | Algoritmanın kaç nesil çalışacağı |

Popülasyonu artırmak çözüm kalitesini iyileştirebilir ancak çalıştırma süresini uzatır. 100 birey ve 50 nesil, çoğu senaryo için iyi bir denge noktasıdır.

### Bileşen Parametreleri

Her bileşen şu özelliklere sahiptir:

- **ID:** Benzersiz tanıcı isim (örneğin "Motor", "Batarya_Ana")
- **Ağırlık:** Kilogram cinsinden kütle
- **Boyut:** (X, Y, Z) cinsinden cm olarak dış boyutlar
- **İzin Verilen Bölgeler:** Parçaların yerleştirilebileceği bölgeler (BURUN, GOVDE, KUYRUK, TAVAN, TABAN). Birden fazla bölge verilebilir
- **Sabit Pozisyon:** Kilitli parçalar için (X, Y, Z) koordinatı
- **Kilitli:** Bu parça yerinden oynatılmaz (True/False)
- **Titreşim Hassasiyeti:** Motordan uzak tutulması gereken parça (True/False)
- **Sıcaklık Hassasiyeti:** Motor ısısına karşı korunması gereken parça (True/False)

### Gövde Bölgeleri

Uçak gövdesi beş bölgeye ayrılmıştır:

| Bölge | X Aralığı | Z Aralığı | Tipik Parçalar |
|-------|-----------|-----------|----------------|
| BURUN | 0 - 45 cm | Tam daire | Motor, kamera |
| GOVDE | 45 - 255 cm | Tam daire | Batarya, koltuklar, bagaj, yakıt tankları |
| KUYRUK | 255 - 300 cm | Tam daire | Servo |
| TAVAN | 45 - 255 cm | Üst yarı (z > 0) | Aviyonikler |
| TABAN | 45 - 255 cm | Alt yarı (z < 0) | Ağır bileşenler |

---

## Bilinen Kısıtlamalar

- Gövde geometrisi basitleştirilmiş bir silindirik + konik model kullanır; gerçek uçak gövdelerindeki kanat geçiş bölgeleri, kapı açıklıkları gibi detaylar modellenmemiştir.
- Yakıt tankları kanat içinde konumlanır; ancak mevcut model gövde AABB sınırları içinde değerlendirilir. Bu nedenle yakıt tankları ile gövde parçaları arasındaki çakışma kontrolü bilerek devre dışı bırakılmıştır.
- Kilitli parçalar arasındaki çakışma (örneğin pilot ile pilot koltuğunun üst üste gelmesi) bilerek muaf tutulmuştur, çünkü fiziksel olarak aynı alanı paylaşırlar.
- Pilot ağırlığı her çalıştırmada 80-100 kg arasında rastgele belirlenir. Bu, gerçek dünyada farklı pilotların denge üzerindeki etkisini simüle eder ancak deterministik sonuç alınmasını zorlaştırır.
- Web arayüzünde aynı anda birden fazla simülasyon çalıştırmak desteklenmez; sunucu tek sonuç kaydını tutar.

---

## API Referansı

Web sunucusu iki ana endpoint sağlar:

**POST /api/run-simulation**
Simülasyon parametrelerini JSON olarak alır, seçilen algoritmayı çalıştırır ve sonuçları döndürür. İstek gövdesi gövde parametreleri, bileşen listesi, algoritma seçimi ve solver ayarlarını içerir.

**GET /api/get-3d-view**
Son çalıştırılan simülasyonun 3D Plotly görselleştirmesini tam sayfa HTML olarak döndürür. Henüz simülasyon çalıştırılmadıysa 404 döndürür.

---

## Lisans

Bu proje bir bitirme tezi kapsamında geliştirilmiştir. Akademik amaçlı kullanım serbesttir.
