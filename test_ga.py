# -*- coding: utf-8 -*-
"""GA v2 final verification."""
from backend.models.aircraft import load_aircraft_type, load_components
from backend.optimization.genetic_algorithm import GeneticAlgorithm, GAConfig

ac = load_aircraft_type('aircraft_data/aircraft_types/fighter_5gen.json')
comps = load_components('aircraft_data/component_libraries/fighter_components.json')

print(f"MAC LE: {ac.cg_target.mac_leading_edge_x}")
print(f"MAC length: {ac.cg_target.mac_length}")
print(f"Target X: {ac.cg_target.target_x_min:.0f} - {ac.cg_target.target_x_max:.0f} cm")
print(f"Target center: {ac.cg_target.target_x_center:.0f} cm")
print()

cfg = GAConfig(population_size=200, generations=100, mutation_rate_start=0.35)
ga = GeneticAlgorithm(ac, comps, cfg)
best = ga.run()

fit = best.fitness
print(f'=== GA v2 Final ===')
print(f'CG MAC: {fit.cg_mac_percent:.1f}% (hedef: 25-35%)')
print(f'CG X: {fit.cg_x:.0f} cm (hedef: {ac.cg_target.target_x_min:.0f}-{ac.cg_target.target_x_max:.0f})')
print(f'CG Y: {fit.cg_y:.1f}, Z: {fit.cg_z:.1f}')
print(f'Violations: {fit.violation_count}')
print(f'Total Score: {fit.total_score:.0f}')
print(f'  CG Score: {fit.cg_score:.0f}')
print(f'  Constraint: {fit.constraint_score:.0f}')
print(f'  Drift: {fit.drift_score:.0f}')
print(f'Elapsed: {ga.progress.elapsed_seconds:.1f}s')
print()
# Every 10th gen
for h in ga.progress.history:
    if h['generation'] % 20 == 0 or h['generation'] <= 5:
        g = h['generation']
        s = h['best_score']
        c = h['cg_mac']
        v = h['violations']
        mr = h['mutation_rate']
        print(f'  Gen {g:3d}: score={s:>10.0f}  CG={c:>5.1f}%  viol={v}  mut={mr:.3f}')
