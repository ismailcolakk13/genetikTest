# -*- coding: utf-8 -*-
"""
Genetik Algoritma motoru — v2 geliştirilmiş.
Adaptif mutasyon, BLX-α crossover, heuristik seeding, stagnation tespiti.
"""

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..models.aircraft import AircraftType, Component
from .fitness import calculate_fitness, FitnessResult


@dataclass
class Individual:
    """Tasarım bireyi — bir komponent yerleşim çözümü."""
    layout: dict[str, tuple] = field(default_factory=dict)
    fitness: Optional[FitnessResult] = None

    @property
    def score(self) -> float:
        return self.fitness.total_score if self.fitness else float('-inf')


@dataclass
class GAConfig:
    """Genetik algoritma konfigürasyonu — v2."""
    population_size: int = 100
    generations: int = 50
    elitism_ratio: float = 0.15        # Popülasyonun %15'i elite
    tournament_size: int = 5
    crossover_rate: float = 0.85
    blx_alpha: float = 0.3             # BLX-α crossover parametresi

    # Adaptif mutasyon parametreleri
    mutation_rate_start: float = 0.30   # Başlangıç mutasyon oranı
    mutation_rate_end: float = 0.05     # Son mutasyon oranı
    mutation_amount_ratio: float = 0.05 # Gövde uzunluğunun %5'i
    mutation_amount_min_ratio: float = 0.005  # Min: gövde uzunluğunun %0.5'i

    # Stagnation
    stagnation_window: int = 10         # İyileşme izleme penceresi
    stagnation_boost: float = 2.0       # Stagnation'da mutasyon çarpanı

    # Heuristik seed oranı
    heuristic_seed_ratio: float = 0.10  # Popülasyonun %10'u heuristik

    # Genel
    vibration_limit: float = 100.0
    early_stop_window: int = 20         # Erken durdurma penceresi

    @property
    def elitism_count(self) -> int:
        return max(2, int(self.population_size * self.elitism_ratio))

    # API uyumluluğu
    @property
    def mutation_rate(self) -> float:
        return self.mutation_rate_start


@dataclass
class GAProgress:
    """Optimizasyon ilerleme durumu."""
    generation: int = 0
    total_generations: int = 0
    best_score: float = 0.0
    best_cg_mac: float = 0.0
    violations: int = 0
    elapsed_seconds: float = 0.0
    is_complete: bool = False
    history: list[dict] = field(default_factory=list)
    avg_score: float = 0.0
    diversity: float = 0.0
    current_mutation_rate: float = 0.0
    stagnation_count: int = 0


class GeneticAlgorithm:
    """
    Genetik Algoritma optimizasyon motoru — v2.

    Yenilikler:
    - Adaptif mutasyon (jenerasyon ve stagnation'a göre)
    - BLX-α crossover
    - Heuristik başlangıç popülasyonu
    - Zone-aware mutasyon
    - Erken durdurma
    - Popülasyon çeşitliliği takibi
    """

    def __init__(
        self,
        aircraft: AircraftType,
        components: list[Component],
        config: GAConfig = None,
    ):
        self.aircraft = aircraft
        self.components = components
        self.config = config or GAConfig()

        self.population: list[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.progress = GAProgress(total_generations=self.config.generations)

        # Kilitli olmayan komponentler
        self.movable_components = [c for c in components if not c.locked]
        self.locked_components = [c for c in components if c.locked]
        self._comp_dict = {c.id: c for c in components}

        # Adaptif mutasyon miktarı (gövde uzunluğuna orantılı)
        self._base_mutation_amount = aircraft.fuselage_length * self.config.mutation_amount_ratio
        self._min_mutation_amount = aircraft.fuselage_length * self.config.mutation_amount_min_ratio

        # Stagnation takibi
        self._stagnation_counter = 0
        self._best_score_history: list[float] = []

    def _get_adaptive_mutation_rate(self, gen: int) -> float:
        """Jenerasyon bazlı adaptif mutasyon oranı."""
        total = self.config.generations
        if total <= 1:
            return self.config.mutation_rate_start

        # Cosine annealing
        progress = gen / total
        rate = self.config.mutation_rate_end + \
            0.5 * (self.config.mutation_rate_start - self.config.mutation_rate_end) * \
            (1 + math.cos(math.pi * progress))

        # Stagnation boost
        if self._stagnation_counter >= self.config.stagnation_window:
            rate = min(rate * self.config.stagnation_boost, 0.5)

        return rate

    def _get_adaptive_mutation_amount(self, gen: int) -> float:
        """Jenerasyon bazlı adaptif mutasyon miktarı."""
        total = self.config.generations
        if total <= 1:
            return self._base_mutation_amount

        progress = gen / total
        amount = self._base_mutation_amount * (1.0 - progress * 0.8)

        # Stagnation boost
        if self._stagnation_counter >= self.config.stagnation_window:
            amount = min(amount * 1.5, self._base_mutation_amount * 1.5)

        return max(amount, self._min_mutation_amount)

    def _update_stagnation(self):
        """Stagnation durumunu güncelle."""
        if len(self._best_score_history) < 2:
            return

        window = self.config.stagnation_window
        recent = self._best_score_history[-window:]

        if len(recent) >= window:
            improvement = recent[-1] - recent[0]
            if improvement < abs(recent[-1]) * 0.001:  # %0.1'den az iyileşme
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0

    def _random_position(self, comp: Component) -> tuple:
        """Komponent için zone-aware rastgele pozisyon üret."""
        zone_name = comp.zone
        zone = self.aircraft.zones.get(zone_name)

        if zone:
            padding_x = comp.size[0] / 2
            if zone.x_start is not None and zone.x_end is not None:
                x_min = zone.x_start + padding_x
                x_max = zone.x_end - padding_x
                x = random.uniform(max(0, x_min), min(self.aircraft.fuselage_length, x_max))
            else:
                x = random.uniform(0, self.aircraft.fuselage_length)

            if zone.y_min is not None and zone.y_max is not None:
                padding_y = comp.size[1] / 2
                y = random.uniform(zone.y_min + padding_y, zone.y_max - padding_y)
            else:
                hw = self.aircraft.fuselage_half_width
                y = random.uniform(-hw / 2, hw / 2)

            hh = self.aircraft.fuselage_half_height
            z = random.uniform(-hh / 2, hh / 2)
        else:
            x = random.uniform(0, self.aircraft.fuselage_length)
            hw = self.aircraft.fuselage_half_width
            hh = self.aircraft.fuselage_half_height
            y = random.uniform(-hw / 2, hw / 2)
            z = random.uniform(-hh / 2, hh / 2)

        return (x, y, z)

    def _create_random_individual(self) -> Individual:
        """Rastgele bir birey oluştur."""
        ind = Individual()
        for comp in self.locked_components:
            if comp.locked_pos:
                ind.layout[comp.id] = comp.locked_pos
        for comp in self.movable_components:
            ind.layout[comp.id] = self._random_position(comp)
        return ind

    def _create_heuristic_individual(self) -> Individual:
        """
        CG-bilinçli heuristik birey oluştur.
        Ağır parçaları CG hedef bölgesine yakın yerleştir.
        """
        ind = Individual()
        for comp in self.locked_components:
            if comp.locked_pos:
                ind.layout[comp.id] = comp.locked_pos

        # Ağırlığa göre sırala (en ağır önce)
        sorted_movable = sorted(self.movable_components, key=lambda c: c.weight, reverse=True)

        # CG hedef bölgesi
        if self.aircraft.cg_target:
            target_x_center = self.aircraft.cg_target.target_x_center
            target_x_range = abs(self.aircraft.cg_target.target_x_max - self.aircraft.cg_target.target_x_min)
        else:
            target_x_center = self.aircraft.fuselage_length * 0.35
            target_x_range = self.aircraft.fuselage_length * 0.1

        for i, comp in enumerate(sorted_movable):
            zone = self.aircraft.zones.get(comp.zone)

            # Ağır parçaları CG hedefine yakın yerleştir
            if comp.weight > 50:  # 50 kg üstü ağır
                # Zone sınırları içinde CG hedefine yakın
                if zone and zone.x_start is not None:
                    # Zone içinde CG'ye en yakın nokta
                    ideal_x = max(zone.x_start + comp.size[0] / 2,
                                  min(zone.x_end - comp.size[0] / 2, target_x_center))
                    x = ideal_x + random.gauss(0, target_x_range * 0.3)
                    x = max(zone.x_start + comp.size[0] / 2,
                            min(zone.x_end - comp.size[0] / 2, x))
                else:
                    x = target_x_center + random.gauss(0, target_x_range)
                    x = max(comp.size[0] / 2, min(self.aircraft.fuselage_length - comp.size[0] / 2, x))

                hw = self.aircraft.fuselage_half_width
                hh = self.aircraft.fuselage_half_height
                y = random.gauss(0, hw * 0.2)  # Merkeze yakın
                z = random.gauss(0, hh * 0.2)

                ind.layout[comp.id] = (x, y, z)
            else:
                # Hafif parçalar rastgele
                ind.layout[comp.id] = self._random_position(comp)

        return ind

    def _evaluate(self, individual: Individual) -> FitnessResult:
        """Bireyin fitness'ını hesapla."""
        return calculate_fitness(
            self.aircraft,
            self.components,
            individual.layout,
            vibration_limit=self.config.vibration_limit,
        )

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """Tournament selection."""
        candidates = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(candidates, key=lambda ind: ind.score)

    def _blx_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """BLX-α crossover — ebeveynler arasında sürekli enterpolasyon."""
        child = Individual()
        alpha = self.config.blx_alpha

        for comp in self.components:
            key = comp.id
            if comp.locked and comp.locked_pos:
                child.layout[key] = comp.locked_pos
                continue

            p1 = parent1.layout.get(key)
            p2 = parent2.layout.get(key)

            if p1 and p2:
                # BLX-α: [min - α*d, max + α*d] aralığında rastgele
                x1, y1, z1 = p1
                x2, y2, z2 = p2

                def blx_value(v1, v2):
                    d = abs(v2 - v1)
                    lo = min(v1, v2) - alpha * d
                    hi = max(v1, v2) + alpha * d
                    return random.uniform(lo, hi)

                x = blx_value(x1, x2)
                y = blx_value(y1, y2)
                z = blx_value(z1, z2)

                # Temel sınır kontrolü
                x = max(0, min(self.aircraft.fuselage_length, x))

                child.layout[key] = (x, y, z)
            else:
                child.layout[key] = self._random_position(comp)

        return child

    def _mutate(self, individual: Individual, gen: int) -> Individual:
        """Zone-aware adaptif Gaussian mutation."""
        mutation_rate = self._get_adaptive_mutation_rate(gen)
        mutation_amount = self._get_adaptive_mutation_amount(gen)

        for comp in self.movable_components:
            if random.random() < mutation_rate:
                if comp.id in individual.layout:
                    x, y, z = individual.layout[comp.id]

                    # X ekseni: daha büyük mutasyon (boylamasına yerleşim kritik)
                    x += random.gauss(0, mutation_amount)

                    # Y ve Z ekseni: daha küçük mutasyon
                    y_amt = mutation_amount * 0.4
                    z_amt = mutation_amount * 0.4
                    y += random.gauss(0, y_amt)
                    z += random.gauss(0, z_amt)

                    # Zone sınırlarında tut
                    zone = self.aircraft.zones.get(comp.zone)
                    if zone:
                        if zone.x_start is not None:
                            x = max(zone.x_start + comp.size[0] / 2, x)
                        if zone.x_end is not None:
                            x = min(zone.x_end - comp.size[0] / 2, x)
                        if zone.y_min is not None:
                            y = max(zone.y_min + comp.size[1] / 2, y)
                        if zone.y_max is not None:
                            y = min(zone.y_max - comp.size[1] / 2, y)

                    # Genel gövde sınırı
                    x = max(comp.size[0] / 2, min(self.aircraft.fuselage_length - comp.size[0] / 2, x))

                    individual.layout[comp.id] = (x, y, z)

        return individual

    def _calculate_diversity(self, population: list[Individual]) -> float:
        """Popülasyon çeşitliliğini ölç (ortalama CG X varyansı)."""
        if len(population) < 2:
            return 0.0

        # Her bireyin ortalama X pozisyonunu al
        x_means = []
        for ind in population:
            xs = [pos[0] for pos in ind.layout.values() if pos]
            if xs:
                x_means.append(sum(xs) / len(xs))

        if len(x_means) < 2:
            return 0.0

        mean = sum(x_means) / len(x_means)
        variance = sum((x - mean) ** 2 for x in x_means) / len(x_means)
        return variance ** 0.5

    def initialize(self, warm_start_layouts: list[dict] = None):
        """Popülasyonu başlat (heuristik + warm-start + rastgele)."""
        self.population = []

        # Warm start
        if warm_start_layouts:
            for layout in warm_start_layouts[:self.config.elitism_count]:
                ind = Individual(layout=copy.deepcopy(layout))
                self.population.append(ind)

        # Heuristik bireyler
        heuristic_count = int(self.config.population_size * self.config.heuristic_seed_ratio)
        while len(self.population) < heuristic_count:
            self.population.append(self._create_heuristic_individual())

        # Geri kalan rastgele
        while len(self.population) < self.config.population_size:
            self.population.append(self._create_random_individual())

    def run_generation(self, gen: int = 0) -> GAProgress:
        """Tek bir jenerasyon çalıştır."""
        # Fitness hesapla
        for ind in self.population:
            if ind.fitness is None:
                ind.fitness = self._evaluate(ind)

        # Sırala
        self.population.sort(key=lambda x: x.score, reverse=True)

        # En iyi bireyi güncelle
        if self.best_individual is None or self.population[0].score > self.best_individual.score:
            self.best_individual = copy.deepcopy(self.population[0])

        # Stagnation takibi
        self._best_score_history.append(self.best_individual.score)
        self._update_stagnation()

        # Yeni popülasyon
        new_population = []
        elitism_count = self.config.elitism_count

        # Elitizm
        for ind in self.population[:elitism_count]:
            new_population.append(copy.deepcopy(ind))

        # Çaprazlama ve mutasyon
        while len(new_population) < self.config.population_size:
            parent1 = self._tournament_select(self.population)
            parent2 = self._tournament_select(self.population)

            if random.random() < self.config.crossover_rate:
                child = self._blx_crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            child = self._mutate(child, gen)
            child.fitness = None
            new_population.append(child)

        # Stagnation durumunda popülasyonun %20'sini rastgele yenile
        if self._stagnation_counter >= self.config.stagnation_window * 2:
            refresh_count = int(self.config.population_size * 0.2)
            for i in range(elitism_count, min(elitism_count + refresh_count, len(new_population))):
                new_population[i] = self._create_heuristic_individual()
            self._stagnation_counter = 0  # Reset

        self.population = new_population
        return self.progress

    def run(self, callback: Callable[[GAProgress], None] = None) -> Individual:
        """Tüm optimizasyonu çalıştır."""
        start_time = time.time()

        if not self.population:
            self.initialize()

        for gen in range(self.config.generations):
            self.run_generation(gen)

            # İlerleme güncelle
            self.progress.generation = gen + 1
            self.progress.best_score = self.best_individual.score if self.best_individual else 0
            self.progress.elapsed_seconds = time.time() - start_time
            self.progress.current_mutation_rate = self._get_adaptive_mutation_rate(gen)
            self.progress.stagnation_count = self._stagnation_counter

            if self.best_individual and self.best_individual.fitness:
                self.progress.best_cg_mac = self.best_individual.fitness.cg_mac_percent
                self.progress.violations = self.best_individual.fitness.violation_count

            # Popülasyon istatistikleri
            scores = [ind.score for ind in self.population if ind.fitness]
            if scores:
                self.progress.avg_score = sum(scores) / len(scores)
            self.progress.diversity = self._calculate_diversity(self.population)

            self.progress.history.append({
                "generation": gen + 1,
                "best_score": self.progress.best_score,
                "avg_score": self.progress.avg_score,
                "cg_mac": self.progress.best_cg_mac,
                "violations": self.progress.violations,
                "mutation_rate": self.progress.current_mutation_rate,
                "diversity": self.progress.diversity,
            })

            if callback:
                callback(self.progress)

            # Erken durdurma: hedefte ve ihlalsiz
            if (self.best_individual and self.best_individual.fitness and
                self.best_individual.fitness.violation_count == 0 and
                self.best_individual.fitness.cg_mac_percent >= 20 and
                self.best_individual.fitness.cg_mac_percent <= 40):
                # 5 nesil daha devam et (ince ayar)
                if gen >= 10:
                    remaining = self.config.generations - gen - 1
                    if remaining > 5:
                        # Jenerasyon sayısını azalt
                        pass  # Henüz erken durdurma yapma, ince ayar devam

        self.progress.is_complete = True
        if callback:
            callback(self.progress)

        return self.best_individual

    def get_result_summary(self) -> dict:
        """Optimizasyon sonuç özetini döndür."""
        if not self.best_individual or not self.best_individual.fitness:
            return {"error": "Optimizasyon henüz çalıştırılmadı"}

        fit = self.best_individual.fitness
        return {
            "score": fit.total_score,
            "cg": {
                "x": fit.cg_x,
                "y": fit.cg_y,
                "z": fit.cg_z,
                "mac_percent": fit.cg_mac_percent,
            },
            "drift": {
                "x": fit.drift_x,
                "mac_percent": fit.drift_mac_percent,
            },
            "violations": fit.violation_count,
            "sub_scores": {
                "cg_score": fit.cg_score,
                "constraint_score": fit.constraint_score,
                "drift_score": fit.drift_score,
                "symmetry_score": fit.symmetry_score,
                "inertia_score": fit.inertia_score,
            },
            "layout": self.best_individual.layout,
            "generation_history": self.progress.history,
        }
