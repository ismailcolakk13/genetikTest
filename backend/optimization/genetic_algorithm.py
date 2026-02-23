# -*- coding: utf-8 -*-
"""
Genetik Algoritma motoru.
Popülasyon yönetimi, seçilim, çaprazlama, mutasyon ve evrim döngüsü.
"""

import copy
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
    """Genetik algoritma konfigürasyonu."""
    population_size: int = 100
    generations: int = 50
    elitism_count: int = 10
    tournament_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_amount: float = 15.0  # Maksimum kaydırma miktarı (cm)
    vibration_limit: float = 100.0


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


class GeneticAlgorithm:
    """
    Genetik Algoritma optimizasyon motoru.

    Mevcut ucakYerlesim.py'deki GA mantığı korunmuş,
    modüler ve genişletilebilir hale getirilmiştir.
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

        # Kilitli olmayan komponentler (sadece bunlar optimize edilecek)
        self.movable_components = [c for c in components if not c.locked]
        self.locked_components = [c for c in components if c.locked]

        # Komponent sözlüğü
        self._comp_dict = {c.id: c for c in components}

        # Callback (ilerleme bildirim)
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[GAProgress], None]):
        """İlerleme callback fonksiyonu ayarla."""
        self._progress_callback = callback

    def _random_position(self, comp: Component) -> tuple:
        """Komponent için rastgele geçerli pozisyon üret."""
        zone_name = comp.zone
        zone = self.aircraft.zones.get(zone_name)

        if zone:
            # Bölge sınırları içinde rastgele X
            if zone.x_start is not None and zone.x_end is not None:
                padding_x = comp.size[0] / 2
                x_min = zone.x_start + padding_x
                x_max = zone.x_end - padding_x
                x = random.uniform(max(0, x_min), min(self.aircraft.fuselage_length, x_max))
            else:
                x = random.uniform(0, self.aircraft.fuselage_length)

            # Y ekseni
            if zone.y_min is not None and zone.y_max is not None:
                # Kanat bölgesi
                padding_y = comp.size[1] / 2
                y = random.uniform(zone.y_min + padding_y, zone.y_max - padding_y)
            else:
                hw = self.aircraft.fuselage_half_width
                y = random.uniform(-hw / 2, hw / 2)

            # Z ekseni
            hh = self.aircraft.fuselage_half_height
            z = random.uniform(-hh / 2, hh / 2)
        else:
            # Bölge tanımlı değilse genel gövde içi
            x = random.uniform(0, self.aircraft.fuselage_length)
            hw = self.aircraft.fuselage_half_width
            hh = self.aircraft.fuselage_half_height
            y = random.uniform(-hw / 2, hw / 2)
            z = random.uniform(-hh / 2, hh / 2)

        return (x, y, z)

    def _create_random_individual(self) -> Individual:
        """Rastgele bir birey oluştur."""
        ind = Individual()

        # Kilitli parçaları sabit pozisyona yerleştir
        for comp in self.locked_components:
            if comp.locked_pos:
                ind.layout[comp.id] = comp.locked_pos

        # Kilitli olmayan parçaları rastgele yerleştir
        for comp in self.movable_components:
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

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Uniform crossover."""
        child = Individual()

        for comp in self.components:
            key = comp.id
            if comp.locked and comp.locked_pos:
                child.layout[key] = comp.locked_pos
            elif random.random() < 0.5:
                child.layout[key] = parent1.layout.get(key, self._random_position(comp))
            else:
                child.layout[key] = parent2.layout.get(key, self._random_position(comp))

        return child

    def _mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation."""
        for comp in self.movable_components:
            if random.random() < self.config.mutation_rate:
                if comp.id in individual.layout:
                    x, y, z = individual.layout[comp.id]
                    amt = self.config.mutation_amount

                    x += random.gauss(0, amt)
                    y += random.gauss(0, amt / 3)
                    z += random.gauss(0, amt / 3)

                    # Temel sınır kontrolü
                    x = max(0, min(self.aircraft.fuselage_length, x))

                    individual.layout[comp.id] = (x, y, z)

        return individual

    def initialize(self, warm_start_layouts: list[dict] = None):
        """
        Popülasyonu başlat.

        Args:
            warm_start_layouts: Önceki iyi tasarımlar (warm start için)
        """
        self.population = []

        # Warm start: Geçmiş iyi tasarımları popülasyona ekle
        if warm_start_layouts:
            for layout in warm_start_layouts[:self.config.elitism_count]:
                ind = Individual(layout=copy.deepcopy(layout))
                self.population.append(ind)

        # Geri kalanını rastgele doldur
        while len(self.population) < self.config.population_size:
            self.population.append(self._create_random_individual())

    def run_generation(self) -> GAProgress:
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

        # Yeni popülasyon oluştur
        new_population = []

        # Elitizm
        for ind in self.population[:self.config.elitism_count]:
            new_population.append(copy.deepcopy(ind))

        # Çaprazlama ve mutasyon
        while len(new_population) < self.config.population_size:
            parent1 = self._tournament_select(self.population)
            parent2 = self._tournament_select(self.population)

            if random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            child = self._mutate(child)
            child.fitness = None  # Yeniden değerlendirilecek
            new_population.append(child)

        self.population = new_population

        return self.progress

    def run(self, callback: Callable[[GAProgress], None] = None) -> Individual:
        """
        Tüm optimizasyonu çalıştır.

        Args:
            callback: Her jenerasyon sonunda çağrılacak fonksiyon

        Returns:
            En iyi birey
        """
        start_time = time.time()

        # Başlangıç popülasyonu
        if not self.population:
            self.initialize()

        for gen in range(self.config.generations):
            self.run_generation()

            # İlerleme güncelle
            self.progress.generation = gen + 1
            self.progress.best_score = self.best_individual.score if self.best_individual else 0
            self.progress.elapsed_seconds = time.time() - start_time

            if self.best_individual and self.best_individual.fitness:
                self.progress.best_cg_mac = self.best_individual.fitness.cg_mac_percent
                self.progress.violations = self.best_individual.fitness.violation_count

            self.progress.history.append({
                "generation": gen + 1,
                "best_score": self.progress.best_score,
                "cg_mac": self.progress.best_cg_mac,
                "violations": self.progress.violations,
            })

            if callback:
                callback(self.progress)

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
