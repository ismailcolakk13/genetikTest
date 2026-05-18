import copy
import random
from yardimcilar.yardimciFonksiyonlar import TasarimBireyi, calculate_fitness_nsga2, calculate_fitness_design, clamp_xz_bolge, clamp_yz_fuselage
from algoritmalar.ga import mutate_design
from algoritmalar.nsga2 import fast_non_dominated_sort, calculate_crowding_distance

class HybridParticle(TasarimBireyi):
    def __init__(self):
        super().__init__()
        self.hiz = {}
        self.pbest_yerlesim = None
        self.pbest_obj1 = float('inf')
        self.pbest_obj2 = float('inf')
        
    def rastgele_yerlestir(self, aircraft):
        super().rastgele_yerlestir(aircraft)
        for k_id in self.yerlesim:
            self.hiz[k_id] = (0.0, 0.0, 0.0)
        self.pbest_yerlesim = copy.deepcopy(self.yerlesim)

def run_nsga2_pso_hybrid(pop_size, generations, aircraft):
    print("NSGA-II + PSO Karma (Hybrid) Optimizasyonu başlıyor (Multi-Objective)...")
    swarm = []
    
    # 1. Başlangıç Popülasyonu Oluşturma
    for _ in range(pop_size):
        p = HybridParticle()
        p.rastgele_yerlestir(aircraft)
        
        # NSGA-II hedeflerini hesapla
        obj1, obj2, cg = calculate_fitness_nsga2(p, aircraft)
        p.obj1 = obj1 # Minimize edilmek istenen hedef 1 (Ceza Puanı)
        p.obj2 = obj2 # Minimize edilmek istenen hedef 2 (CG Hatası)
        p.cg = cg
        p.score = calculate_fitness_design(p, aircraft)[0]
        
        # PBest başlangıç ataması
        p.pbest_obj1 = obj1
        p.pbest_obj2 = obj2
        p.pbest_yerlesim = copy.deepcopy(p.yerlesim)
        swarm.append(p)
        
    # PSO Parametreleri
    w = 0.7  # Ataleti (Inertia)
    c1 = 1.5 # Bilişsel Katsayı (Cognitive - PBest)
    c2 = 1.5 # Sosyal Katsayı (Social - GBest)
    
    for gen in range(generations):
        # 2. NSGA-II Sıralaması ve Kalabalık Mesafesi
        fronts = fast_non_dominated_sort(swarm)
        for front in fronts:
            calculate_crowding_distance(front)
            
        # Liderlerin seçileceği pareto cephesi (Front 0)
        best_front = fronts[0] 
        
        # 3. PSO Hareketi ve Güncelleme
        for p in swarm:
            # Lider Seçimi (Global Best - GBest): 
            # En iyi Pareto cephesinden rastgele 2 birey seçip kalabalık mesafesi yüksek olanı lider olarak ata.
            # Bu işlem NSGA-II'deki çeşitliliği PSO'ya taşır.
            leader1 = random.choice(best_front)
            leader2 = random.choice(best_front)
            if leader1.distance > leader2.distance:
                gbest_yerlesim = leader1.yerlesim
            else:
                gbest_yerlesim = leader2.yerlesim
                
            # Parçacığın bileşenleri için Hız ve Pozisyon Güncelleme
            for k_id in list(p.yerlesim.keys()):
                comp_info = next((item for item in aircraft.komponentler_db if item.id == k_id), None)
                if comp_info and comp_info.kilitli:
                    continue # Kilitli parçalar (örn. Motor) hareket etmez
                    
                x, y, z = p.yerlesim[k_id]
                vx, vy, vz = p.hiz[k_id]
                pbx, pby, pbz = p.pbest_yerlesim[k_id]
                gbx, gby, gbz = gbest_yerlesim[k_id]
                
                r1, r2 = random.random(), random.random()
                
                # Hız denklemi
                new_vx = w * vx + c1 * r1 * (pbx - x) + c2 * r2 * (gbx - x)
                new_vy = w * vy + c1 * r1 * (pby - y) + c2 * r2 * (gby - y)
                new_vz = w * vz + c1 * r1 * (pbz - z) + c2 * r2 * (gbz - z)
                
                # Hız Sınırlandırması (Explosion'ı önlemek için)
                max_v = 20.0
                new_vx = max(-max_v, min(max_v, new_vx))
                new_vy = max(-max_v, min(max_v, new_vy))
                new_vz = max(-max_v, min(max_v, new_vz))
                
                p.hiz[k_id] = (new_vx, new_vy, new_vz)
                
                # Pozisyon Güncellemesi
                new_x = x + new_vx
                new_y = y + new_vy
                new_z = z + new_vz
                
                # Uçağın fiziksel sınırlarına sıkıştırma (Clamp)
                new_x, new_z = clamp_xz_bolge(comp_info, new_x, new_z, aircraft)
                new_y, new_z = clamp_yz_fuselage(comp_info, new_x, new_y, new_z, aircraft)
                
                p.yerlesim[k_id] = (new_x, new_y, new_z)
                
            # 4. GA'dan Gelen Mutasyon Desteği
            # Sürünün yerel bir noktada sıkışmasını engellemek için NSGA-II taktiği olarak mutasyon ekliyoruz (%10 ihtimal)
            if random.random() < 0.10:
                p = mutate_design(p, aircraft)
                
            # 5. Yeni Durum için Fitness Değerlendirmesi
            obj1, obj2, cg = calculate_fitness_nsga2(p, aircraft)
            p.obj1 = obj1
            p.obj2 = obj2
            p.cg = cg
            p.score = calculate_fitness_design(p, aircraft)[0] # Görselleştirici için tekil skor
            
            # 6. PBest (Kişisel En İyi) Güncellemesi (Pareto Dominance Kuralı)
            # Eğer yeni pozisyon eski pbest'i her iki hedefte de domine ediyorsa kesinlikle değiştir.
            if (p.obj1 <= p.pbest_obj1 and p.obj2 <= p.pbest_obj2) and (p.obj1 < p.pbest_obj1 or p.obj2 < p.pbest_obj2):
                p.pbest_obj1 = p.obj1
                p.pbest_obj2 = p.obj2
                p.pbest_yerlesim = copy.deepcopy(p.yerlesim)
            # Eğer iki çözüm birbirini domine edemiyorsa (biri birinde iyi, diğeri ötekinde), rastgele karar ver.
            elif not ((p.pbest_obj1 <= p.obj1 and p.pbest_obj2 <= p.obj2) and (p.pbest_obj1 < p.obj1 or p.pbest_obj2 < p.obj2)):
                if random.random() < 0.5:
                    p.pbest_obj1 = p.obj1
                    p.pbest_obj2 = p.obj2
                    p.pbest_yerlesim = copy.deepcopy(p.yerlesim)
                    
        # Log Yazdırma (10 Nesilde Bir)
        if gen % 10 == 0:
            # Sadece ekrana bilgi vermek için en iyi çözümü seçme
            best_ind = best_front[0]
            for ind in best_front:
                # Öncelikle ceza puanı düşük olanı, eşitse cg hatası düşük olanı öne çıkar
                if ind.obj1 < best_ind.obj1 or (ind.obj1 == best_ind.obj1 and ind.obj2 < best_ind.obj2):
                    best_ind = ind
            print(f"Nesil {gen}: [Hibrit Lideri] Ceza: {best_ind.obj1:.0f}, CG Hatası: {best_ind.obj2:.2f} | CG X: {best_ind.cg[0]:.1f}")

    # 7. Son Durum Hesaplaması ve En İyi Çözümün Seçilmesi
    # Parçacıklar son jenerasyonda kötü bir konuma sürüklenmiş olabilir.
    # Bu yüzden her parçacığın tüm süreç boyunca bulduğu "Kendi En İyisi" (PBest) verilerinden bir arşiv oluşturuyoruz.
    archive = []
    for p in swarm:
        best_p = copy.deepcopy(p)
        best_p.yerlesim = copy.deepcopy(p.pbest_yerlesim)
        best_p.obj1 = p.pbest_obj1
        best_p.obj2 = p.pbest_obj2
        
        # PBest'e göre genel skoru ve cg koordinatını tekrar hesapla
        best_p.score = calculate_fitness_design(best_p, aircraft)[0]
        _, _, best_cg = calculate_fitness_nsga2(best_p, aircraft)
        best_p.cg = best_cg
        archive.append(best_p)

    fronts = fast_non_dominated_sort(archive)
    best_front = fronts[0]
    
    # Pareto cephesinden GA fitness skoru en yüksek olanı seç
    # (ceza + CG cezası + ödüller bileşkesi → analiz ve görsel ile tutarlı).
    best_ind = max(best_front, key=lambda x: x.score)

    print(f"\nNSGA-II + PSO Karma (Hybrid) Optimizasyon Tamamlandı.")
    print(f"Pareto Front (Rank 1) Çözüm Sayısı: {len(best_front)}")
    print(f"Seçilen Hibrit Tasarım -> Ceza Puanı: {best_ind.obj1:.0f}, CG Hatası: {best_ind.obj2:.2f}, Skor: {best_ind.score:.0f}")
    print(f"CG Koordinatları: {best_ind.cg}")
    return best_ind, best_ind.score, best_ind.cg
