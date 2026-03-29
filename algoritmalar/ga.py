import random
from yardimcilar.yardimciFonksiyonlar import TasarimBireyi, calculate_fitness_design, clamp_x_bolge

def crossover_design(parent1, parent2, aircraft):
    child = TasarimBireyi()
    for k_id in aircraft.komponentler_db:
        key = k_id.id
        if random.random() < 0.5:
            child.yerlesim[key] = parent1.yerlesim[key]
        else:
            child.yerlesim[key] = parent2.yerlesim[key]
    return child

def mutate_design(birey, aircraft, rate=0.1):
    for k_id in birey.yerlesim:
        # Kilitli parçaları mutasyona uğratma
        comp_info = next((item for item in aircraft.komponentler_db if item.id == k_id), None)
        if comp_info and comp_info.kilitli:
            continue

        x, y, z = birey.yerlesim[k_id]

        if random.random() < rate:
            # Küçük kaydırma
            x += random.uniform(-10, 10)
            y += random.uniform(-5, 5)
            z += random.uniform(-5, 5)

        # Bölge sınırına clamp (her zaman uygula)
        x = clamp_x_bolge(comp_info, x, aircraft)
        birey.yerlesim[k_id] = (x, y, z)
    return birey

def run_ga(pop_size, generations, aircraft):
    print("GA optimizasyonu başlıyor...")
    populasyon=[]
    for _ in range(pop_size):
        b=TasarimBireyi()
        b.rastgele_yerlestir(aircraft)
        populasyon.append(b)
        
    best_cg=(0,0,0)
    best_score=-float('inf')
    en_iyi_tasarim=None

    for gen in range(generations):
        puanli_pop=[]
        for ind in populasyon:
            score,cg=calculate_fitness_design(ind, aircraft)
            puanli_pop.append((score,ind,cg))
            
        puanli_pop.sort(key=lambda x:x[0],reverse=True)
        
        best_score=puanli_pop[0][0]
        best_cg=puanli_pop[0][2]
        
        if gen%10==0:
            print(f"Nesil {gen}: Puan {best_score:.0f} | CG X: {best_cg[0]:.1f} (Hedef: {aircraft.target_cg_x_min}-{aircraft.target_cg_x_max})")
            
        yeni_pop=[x[1]for x in puanli_pop[:10]]
        
        # Elitlere de bölge clamp uygula
        for ind in yeni_pop:
            for k_id in ind.yerlesim:
                comp_info = next((item for item in aircraft.komponentler_db if item.id == k_id), None)
                if comp_info and not comp_info.kilitli:
                    x, y, z = ind.yerlesim[k_id]
                    x = clamp_x_bolge(comp_info, x, aircraft)
                    ind.yerlesim[k_id] = (x, y, z)
        
        while len(yeni_pop)<pop_size:
            parent1=random.choice(puanli_pop[:30])[1]
            parent2=random.choice(puanli_pop[:30])[1]
            child=crossover_design(parent1, parent2, aircraft)
            child=mutate_design(child, aircraft)
            yeni_pop.append(child)
            
        populasyon=yeni_pop
    
    en_iyi_tasarim=puanli_pop[0][1]
    return en_iyi_tasarim, best_score, best_cg