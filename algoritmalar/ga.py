import random
from yardimcilar.yardimciFonksiyonlar import TasarimBireyi, calculate_fitness_design, clamp_xz_bolge, clamp_yz_fuselage


def _tournament_select(puanli_pop, k=3):
    """Select best individual from a random sample of k candidates (tournament selection)."""
    candidates = random.sample(puanli_pop, min(k, len(puanli_pop)))
    return max(candidates, key=lambda x: x[0])[1]


def crossover_design(parent1, parent2, aircraft):
    child = TasarimBireyi()
    for komp in aircraft.komponentler_db:
        key = komp.id  # named `komp` (a Komponent object) to distinguish from string IDs
        if random.random() < 0.5:
            child.yerlesim[key] = parent1.yerlesim[key]
        else:
            child.yerlesim[key] = parent2.yerlesim[key]
    return child


def mutate_design(birey, aircraft, rate=0.1):
    kmap = aircraft.komponentler_map
    # Scale the spatial step with the mutation rate:
    # rate=0.3 (early) → ±30 cm (wide exploration)
    # rate=0.1 (late)  → ±10 cm (fine-tuning)
    max_delta = 100.0 * rate
    for comp_id in birey.yerlesim:
        comp_info = kmap.get(comp_id)
        if comp_info and comp_info.kilitli:
            continue

        x, y, z = birey.yerlesim[comp_id]

        if random.random() < rate:
            x += random.uniform(-max_delta, max_delta)
            y += random.uniform(-max_delta, max_delta)
            z += random.uniform(-max_delta, max_delta)

        # X ve Z bölge sınırına clamp
        x, z = clamp_xz_bolge(comp_info, x, z, aircraft)
        # YZ fuselage dairesel sınırına clamp
        y, z = clamp_yz_fuselage(comp_info, x, y, z, aircraft)
        birey.yerlesim[comp_id] = (x, y, z)
    return birey


def run_ga(pop_size, generations, aircraft):
    print("GA optimizasyonu başlıyor...")
    populasyon = []
    for _ in range(pop_size):
        b = TasarimBireyi()
        b.rastgele_yerlestir(aircraft)
        populasyon.append(b)

    best_cg = (0, 0, 0)
    best_score = -float('inf')
    en_iyi_tasarim = populasyon[0] if populasyon else TasarimBireyi()

    # Elite count: keep top 20% of population each generation
    elite_count = max(2, pop_size // 5)

    for gen in range(generations):
        puanli_pop = []
        for ind in populasyon:
            score, cg = calculate_fitness_design(ind, aircraft)
            puanli_pop.append((score, ind, cg))

        puanli_pop.sort(key=lambda x: x[0], reverse=True)

        best_score = puanli_pop[0][0]
        best_cg = puanli_pop[0][2]
        en_iyi_tasarim = puanli_pop[0][1]

        if gen % 10 == 0:
            print(f"Nesil {gen}: Puan {best_score:.0f} | CG X: {best_cg[0]:.1f} "
                  f"(Hedef: {aircraft.target_cg_x_min}-{aircraft.target_cg_x_max})")

        # Elitizm: En iyi %20'yi koru
        yeni_pop = [x[1] for x in puanli_pop[:elite_count]]

        # Elitlere de X+Z bölge clamp uygula
        kmap = aircraft.komponentler_map
        for ind in yeni_pop:
            for comp_id in ind.yerlesim:
                comp_info = kmap.get(comp_id)
                if comp_info and not comp_info.kilitli:
                    x, y, z = ind.yerlesim[comp_id]
                    x, z = clamp_xz_bolge(comp_info, x, z, aircraft)
                    y, z = clamp_yz_fuselage(comp_info, x, y, z, aircraft)
                    ind.yerlesim[comp_id] = (x, y, z)

        # Adaptif mutasyon: erken nesillerde yüksek keşif (0.3), geç nesillerde hassas iyileştirme (0.1)
        adaptive_rate = 0.3 - 0.2 * (gen / max(1, generations - 1))

        while len(yeni_pop) < pop_size:
            # Tournament selection — better than random choice from fixed top-N
            parent1 = _tournament_select(puanli_pop, k=3)
            parent2 = _tournament_select(puanli_pop, k=3)
            child = crossover_design(parent1, parent2, aircraft)
            child = mutate_design(child, aircraft, rate=adaptive_rate)
            yeni_pop.append(child)

        populasyon = yeni_pop

    return en_iyi_tasarim, best_score, best_cg