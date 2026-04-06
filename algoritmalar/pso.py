import copy
import random
from yardimcilar.yardimciFonksiyonlar import TasarimBireyi, calculate_fitness_design, clamp_xz_bolge

class PsoParticle(TasarimBireyi):
    def __init__(self):
        super().__init__()
        self.hiz = {}
        self.best_yerlesim = None
        self.best_score = -float('inf')
        self.best_cg = (0, 0, 0)

    def rastgele_yerlestir(self, aircraft):
        super().rastgele_yerlestir(aircraft)
        for k_id in self.yerlesim:
            self.hiz[k_id] = (0.0, 0.0, 0.0)
        self.best_yerlesim = copy.deepcopy(self.yerlesim)

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
            global_best_yerlesim = copy.deepcopy(p.yerlesim)
            global_best_cg = cg
            global_best_birey = copy.deepcopy(p)

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    for gen in range(generations):
        for p in swarm:
            for k_id in list(p.yerlesim.keys()):
                comp_info = next((item for item in aircraft.komponentler_db if item.id == k_id), None)
                if comp_info and comp_info.kilitli:
                    continue

                x, y, z = p.yerlesim[k_id]
                vx, vy, vz = p.hiz[k_id]
                pbx, pby, pbz = p.best_yerlesim[k_id]
                gbx, gby, gbz = global_best_yerlesim[k_id]

                r1, r2 = random.random(), random.random()

                new_vx = w * vx + c1 * r1 * (pbx - x) + c2 * r2 * (gbx - x)
                new_vy = w * vy + c1 * r1 * (pby - y) + c2 * r2 * (gby - y)
                new_vz = w * vz + c1 * r1 * (pbz - z) + c2 * r2 * (gbz - z)

                max_v = 20.0
                new_vx = max(-max_v, min(max_v, new_vx))
                new_vy = max(-max_v, min(max_v, new_vy))
                new_vz = max(-max_v, min(max_v, new_vz))

                p.hiz[k_id] = (new_vx, new_vy, new_vz)

                new_x = x + new_vx
                new_y = y + new_vy
                new_z = z + new_vz

                # X ve Z bölge sınırına clamp
                new_x, new_z = clamp_xz_bolge(comp_info, new_x, new_z, aircraft)

                p.yerlesim[k_id] = (new_x, new_y, new_z)
            
            score, cg = calculate_fitness_design(p, aircraft)
            
            if score > p.best_score:
                p.best_score = score
                p.best_yerlesim = copy.deepcopy(p.yerlesim)
                p.best_cg = cg
                
            if score > global_best_score:
                global_best_score = score
                global_best_yerlesim = copy.deepcopy(p.yerlesim)
                global_best_cg = cg
                global_best_birey = copy.deepcopy(p)
                
        if gen % 10 == 0:
            print(f"Nesil {gen}: Puan {global_best_score:.0f} | CG X: {global_best_cg[0]:.1f} (Hedef: {aircraft.target_cg_x_min}-{aircraft.target_cg_x_max})")
            
    return global_best_birey, global_best_score, global_best_cg
