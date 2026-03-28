import random
import math
from yardimcilar.yardimciFonksiyonlar import TasarimBireyi, calculate_fitness_nsga2, calculate_fitness_design
from algoritmalar.ga import crossover_design, mutate_design

def fast_non_dominated_sort(P):
    fronts = [[]]
    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            # Objective 1: ceza_puani (Minimize)
            # Objective 2: cg_hatasi (Minimize)
            if (p.obj1 <= q.obj1 and p.obj2 <= q.obj2) and (p.obj1 < q.obj1 or p.obj2 < q.obj2):
                p.S.append(q)
            elif (q.obj1 <= p.obj1 and q.obj2 <= p.obj2) and (q.obj1 < p.obj1 or q.obj2 < p.obj2):
                p.n += 1
        if p.n == 0:
            p.rank = 0
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in p.S:
                q.n -= 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts[:-1]

def calculate_crowding_distance(front):
    l = len(front)
    for p in front:
        p.distance = 0
    
    if l <= 2:
        for p in front:
            p.distance = float('inf')
        return
    
    # Sort by obj1 (Ceza Puanı)
    front.sort(key=lambda x: x.obj1)
    front[0].distance = float('inf')
    front[-1].distance = float('inf')
    m_obj1 = front[-1].obj1 - front[0].obj1
    if m_obj1 == 0: m_obj1 = 1
    
    for i in range(1, l - 1):
        front[i].distance += (front[i+1].obj1 - front[i-1].obj1) / m_obj1
        
    # Sort by obj2 (CG Hatası)
    front.sort(key=lambda x: x.obj2)
    front[0].distance = float('inf')
    front[-1].distance = float('inf')
    m_obj2 = front[-1].obj2 - front[0].obj2
    if m_obj2 == 0: m_obj2 = 1
    
    for i in range(1, l - 1):
        front[i].distance += (front[i+1].obj2 - front[i-1].obj2) / m_obj2

def make_new_pop(pop, aircraft, pop_size):
    new_pop = []
    while len(new_pop) < pop_size:
        # Tournament selection based on rank and crowding distance
        a = random.choice(pop)
        b = random.choice(pop)
        if a.rank < b.rank or (a.rank == b.rank and a.distance > b.distance):
            parent1 = a
        else:
            parent1 = b
            
        c = random.choice(pop)
        d = random.choice(pop)
        if c.rank < d.rank or (c.rank == d.rank and c.distance > d.distance):
            parent2 = c
        else:
            parent2 = d
            
        child = crossover_design(parent1, parent2, aircraft)
        child = mutate_design(child, aircraft)
        new_pop.append(child)
    return new_pop

def run_nsga2(pop_size, generations, aircraft):
    print("NSGA-II optimizasyonu başlıyor (Multi-Objective)...")
    populasyon = []
    for _ in range(pop_size):
        b = TasarimBireyi()
        b.rastgele_yerlestir(aircraft)
        # Çoklu hedefleri hesapla
        obj1, obj2, cg = calculate_fitness_nsga2(b, aircraft)
        b.obj1 = obj1 # Ceza Puanı
        b.obj2 = obj2 # CG Hatası
        b.cg = cg
        b.score = calculate_fitness_design(b, aircraft)[0] # Görselleştirici ile uyum için genel skor
        populasyon.append(b)

    for gen in range(generations):
        fronts = fast_non_dominated_sort(populasyon)
        for front in fronts:
            calculate_crowding_distance(front)
        
        # Popülasyonu rank ve distance'a göre sırala (Rank düşük olan iyi, distance büyük olan iyi)
        populasyon.sort(key=lambda x: (x.rank, -x.distance))
        
        # Elitizm: En iyi pop_size kadarını al
        parents = populasyon[:pop_size]
        
        if gen % 10 == 0:
            best_ind = parents[0] 
            print(f"Nesil {gen}: [Pareto 1. Eleman] Ceza: {best_ind.obj1:.0f}, CG Hatası: {best_ind.obj2:.2f} | CG X: {best_ind.cg[0]:.1f}")
        
        children = make_new_pop(parents, aircraft, pop_size)
        
        for c in children:
            obj1, obj2, cg = calculate_fitness_nsga2(c, aircraft)
            c.obj1 = obj1
            c.obj2 = obj2
            c.cg = cg
            c.score = calculate_fitness_design(c, aircraft)[0]
            
        populasyon = parents + children
    
    # Son popülasyon içerisinden en iyi sonucu bulma
    fronts = fast_non_dominated_sort(populasyon)
    best_front = fronts[0]
    
    # NSGA-II'de en iyi tek bir çözüm yoktur, Pareto Front vardır.
    # Uyumlu olması için öncelikle "Ceza Puanı" en düşük, sonra CG hatası en düşük olanı seçiyoruz.
    best_front.sort(key=lambda x: (x.obj1, x.obj2))
    best_ind = best_front[0]
    
    print(f"\nNSGA-II Tamamlandı. Pareto Front (Rank 1) Çözüm Sayısı: {len(best_front)}")
    print(f"Seçilen Tasarım -> Ceza: {best_ind.obj1:.0f}, CG Hatası: {best_ind.obj2:.2f}")
    print(f"CG Koordinatları: {best_ind.cg}")
    return best_ind, best_ind.score, best_ind.cg
