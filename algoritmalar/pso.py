import copy
import random
from yardimcilar.yardimciFonksiyonlar import TasarimBireyi, calculate_fitness_design, clamp_xz_bolge, clamp_yz_fuselage

class PsoParticle(TasarimBireyi):
    def __init__(self):
        super().__init__()
        self.hiz = {}
        self.best_yerlesim = None
        self.best_score = -float('inf')
        self.best_cg = (0, 0, 0)

    def rastgele_yerlestir(self, aircraft):
        super().rastgele_yerlestir(aircraft)
        for comp_id in self.yerlesim:
            self.hiz[comp_id] = (0.0, 0.0, 0.0)
        # Shallow copy is sufficient: yerlesim values are immutable tuples
        self.best_yerlesim = dict(self.yerlesim)

def run_pso(pop_size, generations, aircraft):
    print("PSO optimizasyonu başlıyor...")
    swarm = []

    global_best_yerlesim = None
    global_best_score = -float('inf')
    global_best_cg = (0, 0, 0)
    global_best_birey = None

    for _ in range(pop_size):
        p = PsoParticle()
        p.rastgele_yerlestir(aircraft)
        score, cg = calculate_fitness_design(p, aircraft)
        p.best_score = score
        p.best_cg = cg
        swarm.append(p)

        if score > global_best_score:
            global_best_score = score
            global_best_yerlesim = dict(p.yerlesim)  # shallow copy: tuples are immutable
            global_best_cg = cg
            global_best_birey = copy.copy(p)
            global_best_birey.yerlesim = dict(p.yerlesim)  # decouple dict — copy.copy shares it

    # Guard: nothing to optimise if the swarm is empty
    if not swarm:
        empty = PsoParticle()
        return empty, global_best_score, global_best_cg

    c1 = 1.5
    c2 = 1.5
    kmap = aircraft.komponentler_map

    for gen in range(generations):
        # Adaptif atalet katsayısı: erken keşif (0.9) → geç sömürü (0.4)
        w = 0.9 - 0.5 * (gen / max(1, generations - 1))

        # Sıkışma tespiti: global_best skoru hâlâ çok kötüyse (fiziksel ihlal var)
        # Skor -5000'den düşükse muhtemelen çakışma/taşma var
        stuck = global_best_score < -5500

        # Sıkışma durumunda: sürünün en kötü %30'unu sıfırdan rastgele başlat
        if stuck and gen > 0 and gen % 5 == 0:
            swarm.sort(key=lambda x: x.best_score, reverse=True)
            reset_count = max(1, pop_size // 3)
            for i in range(pop_size - reset_count, pop_size):
                fresh = PsoParticle()
                fresh.rastgele_yerlestir(aircraft)
                score, cg = calculate_fitness_design(fresh, aircraft)
                fresh.best_score = score
                fresh.best_cg = cg
                swarm[i] = fresh
                if score > global_best_score:
                    global_best_score = score
                    global_best_yerlesim = dict(fresh.yerlesim)
                    global_best_cg = cg
                    global_best_birey = copy.copy(fresh)

        for p in swarm:
            for comp_id in list(p.yerlesim.keys()):
                comp_info = kmap.get(comp_id)
                if comp_info and comp_info.kilitli:
                    continue

                x, y, z = p.yerlesim[comp_id]
                vx, vy, vz = p.hiz[comp_id]
                pbx, pby, pbz = p.best_yerlesim[comp_id]
                gbx, gby, gbz = global_best_yerlesim[comp_id]

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

                # X ve Z bölge sınırına clamp
                new_x, new_z = clamp_xz_bolge(comp_info, new_x, new_z, aircraft)
                # YZ fuselage dairesel sınırına clamp
                new_y, new_z = clamp_yz_fuselage(comp_info, new_x, new_y, new_z, aircraft)

                p.yerlesim[comp_id] = (new_x, new_y, new_z)
            
            score, cg = calculate_fitness_design(p, aircraft)
            
            if score > p.best_score:
                p.best_score = score
                p.best_yerlesim = dict(p.yerlesim)  # shallow copy: tuples are immutable
                p.best_cg = cg
                
            if score > global_best_score:
                global_best_score = score
                global_best_yerlesim = dict(p.yerlesim)
                global_best_cg = cg
                global_best_birey = copy.copy(p)
                global_best_birey.yerlesim = dict(p.yerlesim)  # decouple dict — copy.copy shares it
                
        if gen % 10 == 0:
            print(f"Nesil {gen}: Puan {global_best_score:.0f} | CG X: {global_best_cg[0]:.1f} (Hedef: {aircraft.target_cg_x_min}-{aircraft.target_cg_x_max})")

    # Fallback: if no best was ever found (e.g. pop_size=0), return first particle
    if global_best_birey is None:
        global_best_birey = swarm[0] if swarm else PsoParticle()

    return global_best_birey, global_best_score, global_best_cg
