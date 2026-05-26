import copy
import random
from yardimcilar.yardimciFonksiyonlar import TasarimBireyi, calculate_fitness_nsga2, clamp_xz_bolge, clamp_yz_fuselage
from algoritmalar.ga import crossover_design, mutate_design
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
        for comp_id in self.yerlesim:
            self.hiz[comp_id] = (0.0, 0.0, 0.0)
        # Shallow copy is sufficient: yerlesim values are immutable tuples
        self.pbest_yerlesim = dict(self.yerlesim)


def _domine_eder(a_obj1, a_obj2, b_obj1, b_obj2):
    """a, b'yi domine ediyor mu? (Her iki hedefte eşit veya daha iyi, en az birinde kesin daha iyi)"""
    return (a_obj1 <= b_obj1 and a_obj2 <= b_obj2) and (a_obj1 < b_obj1 or a_obj2 < b_obj2)


def _archive_snapshot(birey):
    """
    Create a lightweight archive entry — copies only the fields needed for
    dominance comparison and final output.  Avoids deepcopy of velocity dicts
    and pbest state which are not needed once a solution enters the archive.
    """
    from yardimcilar.yardimciFonksiyonlar import TasarimBireyi
    snap = TasarimBireyi()
    snap.yerlesim = dict(birey.yerlesim)  # shallow copy: values are immutable tuples
    snap.obj1 = birey.obj1
    snap.obj2 = birey.obj2
    snap.cg = birey.cg
    snap.score = birey.score
    return snap


def _arsive_ekle(archive, birey, max_archive_size=200):
    """
    Bireyi global arşive ekler (non-dominated archive).
    - Birey arşivdeki herhangi bir elemanı domine ediyorsa, o eleman çıkarılır.
    - Birey arşivdeki herhangi bir eleman tarafından domine ediliyorsa, eklenmez.
    - Arşiv doluysa, kalabalık mesafesi (crowding distance) en küçük eleman çıkarılır.
    """
    # Arşivdeki elemanlardan bu bireyi domine eden var mı?
    for a in archive:
        if _domine_eder(a.obj1, a.obj2, birey.obj1, birey.obj2):
            return  # Birey domine ediliyor, ekleme

    # Bu birey tarafından domine edilen arşiv elemanlarını çıkar
    archive[:] = [a for a in archive if not _domine_eder(birey.obj1, birey.obj2, a.obj1, a.obj2)]

    # Ekle
    archive.append(_archive_snapshot(birey))

    # Arşiv taşarsa, kalabalık mesafesi en düşük olanı çıkar
    if len(archive) > max_archive_size:
        calculate_crowding_distance(archive)
        archive.sort(key=lambda x: x.distance)
        archive.pop(0)  # En küçük distance'ı çıkar


def run_nsga2_pso_hybrid(pop_size, generations, aircraft):
    print("NSGA-II + PSO Karma (Hybrid) Optimizasyonu başlıyor (Multi-Objective)...")
    swarm = []
    archive = []  # Global Non-Dominated Arşiv
    
    # 1. Başlangıç Popülasyonu Oluşturma
    for _ in range(pop_size):
        p = HybridParticle()
        p.rastgele_yerlestir(aircraft)
        
        obj1, obj2, cg, score = calculate_fitness_nsga2(p, aircraft)
        p.obj1 = obj1
        p.obj2 = obj2
        p.cg = cg
        p.score = score
        
        p.pbest_obj1 = obj1
        p.pbest_obj2 = obj2
        p.pbest_yerlesim = copy.deepcopy(p.yerlesim)
        swarm.append(p)
        
        # Arşive ekle
        _arsive_ekle(archive, p)
        
    # PSO Parametreleri
    c1 = 1.5  # Bilişsel Katsayı (Cognitive - PBest)
    c2 = 1.5  # Sosyal Katsayı (Social - GBest)
    
    for gen in range(generations):
        # 2. NSGA-II Sıralaması ve Kalabalık Mesafesi
        fronts = fast_non_dominated_sort(swarm)
        for front in fronts:
            calculate_crowding_distance(front)
            
        # Adaptif atalet katsayısı: erken keşif (0.9) → geç sömürü (0.4)
        w = 0.9 - 0.5 * (gen / max(1, generations - 1))
        
        # 3. PSO Hareketi — Lider seçiminde düşük cezalı arşiv üyelerini tercih et
        feasible_leaders = [a for a in archive if a.obj1 < 5000]
        if len(feasible_leaders) >= 2:
            leader_pool = feasible_leaders
        elif len(archive) >= 2:
            leader_pool = archive
        else:
            leader_pool = fronts[0]
        
        # Sürü sıkışma tespiti: en iyi ceza hâlâ yüksekse
        best_penalty_in_swarm = min(p.obj1 for p in swarm)
        stuck = best_penalty_in_swarm > 5000
        
        # Sıkışma durumunda: sürünün en kötü %30'unu sıfırdan rastgele başlat
        # Bu, fiziksel ihlalli bölgeden kaçışı sağlar
        if stuck and gen > 0 and gen % 5 == 0:
            swarm.sort(key=lambda x: x.score)
            reset_count = max(1, pop_size // 3)
            for i in range(reset_count):
                fresh = HybridParticle()
                fresh.rastgele_yerlestir(aircraft)
                obj1, obj2, cg, score = calculate_fitness_nsga2(fresh, aircraft)
                fresh.obj1 = obj1
                fresh.obj2 = obj2
                fresh.cg = cg
                fresh.score = score
                fresh.pbest_obj1 = obj1
                fresh.pbest_obj2 = obj2
                swarm[i] = fresh
                _arsive_ekle(archive, fresh)
        
        for p in swarm:
            # Lider Seçimi: havuzdan rastgele 2 birey, skoru yüksek olan lider
            if len(leader_pool) >= 2:
                leader1, leader2 = random.sample(leader_pool, 2)
            else:
                leader1 = leader2 = leader_pool[0]
            
            # Skora göre lider seç (daha pratik — ceza+CG dengesini yansıtır)
            gbest_yerlesim = leader1.yerlesim if leader1.score > leader2.score else leader2.yerlesim
                
            # Parçacığın bileşenleri için Hız ve Pozisyon Güncelleme
            for comp_id in list(p.yerlesim.keys()):
                comp_info = aircraft.komponentler_map.get(comp_id)
                if comp_info and comp_info.kilitli:
                    continue
                    
                x, y, z = p.yerlesim[comp_id]
                vx, vy, vz = p.hiz[comp_id]
                pbx, pby, pbz = p.pbest_yerlesim[comp_id]
                gbx, gby, gbz = gbest_yerlesim[comp_id]
                
                r1, r2 = random.random(), random.random()
                
                new_vx = w * vx + c1 * r1 * (pbx - x) + c2 * r2 * (gbx - x)
                new_vy = w * vy + c1 * r1 * (pby - y) + c2 * r2 * (gby - y)
                new_vz = w * vz + c1 * r1 * (pbz - z) + c2 * r2 * (gbz - z)
                
                max_v = 20.0
                new_vx = max(-max_v, min(max_v, new_vx))
                new_vy = max(-max_v, min(max_v, new_vy))
                new_vz = max(-max_v, min(max_v, new_vz))
                
                p.hiz[comp_id] = (new_vx, new_vy, new_vz)
                
                new_x = x + new_vx
                new_y = y + new_vy
                new_z = z + new_vz
                
                new_x, new_z = clamp_xz_bolge(comp_info, new_x, new_z, aircraft)
                new_y, new_z = clamp_yz_fuselage(comp_info, new_x, new_y, new_z, aircraft)
                
                p.yerlesim[comp_id] = (new_x, new_y, new_z)
                
            # 4. GA Mutasyon Desteği (sıkışınca %30, normal %15)
            mutation_chance = 0.30 if stuck else 0.15
            if random.random() < mutation_chance:
                adaptive_rate = 0.4 if stuck else (0.3 - 0.2 * (gen / max(1, generations - 1)))
                p = mutate_design(p, aircraft, rate=adaptive_rate)
                
            # 5. Fitness Değerlendirmesi
            obj1, obj2, cg, score = calculate_fitness_nsga2(p, aircraft)
            p.obj1 = obj1
            p.obj2 = obj2
            p.cg = cg
            p.score = score
            
            # 6. PBest Güncellemesi (Pareto Dominance)
            if _domine_eder(p.obj1, p.obj2, p.pbest_obj1, p.pbest_obj2):
                p.pbest_obj1 = p.obj1
                p.pbest_obj2 = p.obj2
                p.pbest_yerlesim = dict(p.yerlesim)  # shallow copy: tuples are immutable
            elif not _domine_eder(p.pbest_obj1, p.pbest_obj2, p.obj1, p.obj2):
                # Birbirini domine edemiyor → rastgele karar
                if random.random() < 0.5:
                    p.pbest_obj1 = p.obj1
                    p.pbest_obj2 = p.obj2
                    p.pbest_yerlesim = dict(p.yerlesim)  # shallow copy: tuples are immutable
            
            # 7. Global arşivi güncelle
            _arsive_ekle(archive, p)
        
        # 8. NSGA-II Çaprazlama (Genetik çeşitlilik enjeksiyonu)
        crossover_count = pop_size // 3  # %33 çaprazlama çocuğu
        for _ in range(crossover_count):
            # Turnuva seçimi (rank yoksa obj1+obj2 toplamına göre)
            a, b = random.choice(swarm), random.choice(swarm)
            a_rank, b_rank = getattr(a, 'rank', 999), getattr(b, 'rank', 999)
            parent1 = a if (a_rank < b_rank or (a_rank == b_rank and getattr(a, 'distance', 0) > getattr(b, 'distance', 0))) else b
            c, d = random.choice(swarm), random.choice(swarm)
            c_rank, d_rank = getattr(c, 'rank', 999), getattr(d, 'rank', 999)
            parent2 = c if (c_rank < d_rank or (c_rank == d_rank and getattr(c, 'distance', 0) > getattr(d, 'distance', 0))) else d
            
            child = HybridParticle()
            temp = crossover_design(parent1, parent2, aircraft)
            adaptive_rate = 0.3 - 0.2 * (gen / max(1, generations - 1))
            temp = mutate_design(temp, aircraft, rate=adaptive_rate)
            child.yerlesim = temp.yerlesim
            for k_id in child.yerlesim:
                child.hiz[k_id] = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2))
            child.pbest_yerlesim = dict(child.yerlesim)  # shallow copy: tuples are immutable
            
            obj1, obj2, cg, score = calculate_fitness_nsga2(child, aircraft)
            child.obj1 = obj1
            child.obj2 = obj2
            child.cg = cg
            child.score = score
            child.pbest_obj1 = obj1
            child.pbest_obj2 = obj2
            
            # Arşive ekle
            _arsive_ekle(archive, child)
            
            # 9. Sürüdeki en kötü bireyi çaprazlama çocuğuyla değiştir
            # Score karşılaştırması kullan (penalty+CG dengesini yansıtır)
            worst = min(swarm, key=lambda x: x.score)
            if child.score > worst.score:
                idx = swarm.index(worst)
                swarm[idx] = child
                    
        # Log (10 Nesilde Bir)
        if gen % 10 == 0:
            if archive:
                best_ind = min(archive, key=lambda x: x.obj1 + x.obj2 * 1000)
                print(f"Nesil {gen}: [Hibrit Lideri] Ceza: {best_ind.obj1:.0f}, CG Hatası: {best_ind.obj2:.2f} | CG X: {best_ind.cg[0]:.1f}")

    # 10. Son Seçim: Arşivden en iyi çözüm
    if not archive:
        archive = swarm
    
    # Önce düşük cezalı çözümleri filtrele (fiziksel ihlal yok/az olanlar)
    # 5000 eşiği: tek bir taşma = 5000, tek bir çakışma = 10000
    low_penalty = [p for p in archive if p.obj1 < 5000]
    selection_pool = low_penalty if low_penalty else archive
    
    # Normalize edilmiş ideal noktaya en yakın çözümü seç
    ideal_obj1 = min(p.obj1 for p in selection_pool)
    ideal_obj2 = min(p.obj2 for p in selection_pool)
    range1 = max(p.obj1 for p in selection_pool) - ideal_obj1
    range2 = max(p.obj2 for p in selection_pool) - ideal_obj2
    if range1 == 0: range1 = 1.0
    if range2 == 0: range2 = 1.0
    best_ind = min(selection_pool, key=lambda p:
        ((p.obj1 - ideal_obj1) / range1)**2 + ((p.obj2 - ideal_obj2) / range2)**2)

    print(f"\nNSGA-II + PSO Karma (Hybrid) Optimizasyon Tamamlandı.")
    print(f"Arşiv (Non-Dominated) Çözüm Sayısı: {len(archive)}")
    print(f"Seçilen Hibrit Tasarım -> Ceza Puanı: {best_ind.obj1:.0f}, CG Hatası: {best_ind.obj2:.2f}, Skor: {best_ind.score:.0f}")
    print(f"CG Koordinatları: {best_ind.cg}")
    return best_ind, best_ind.score, best_ind.cg
