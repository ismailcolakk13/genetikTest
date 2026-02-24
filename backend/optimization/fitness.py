# -*- coding: utf-8 -*-
"""
Fitness fonksiyonu.
Çok amaçlı değerlendirme: CG sapması, kısıt ihlalleri, simetri, inertia.
"""

from dataclasses import dataclass
from ..models.aircraft import AircraftType, Component
from ..utils.cg_calculator import calculate_cg, analyze_fuel_drift, calculate_inertia
from .constraints import evaluate_all_constraints


@dataclass
class FitnessResult:
    """Fitness değerlendirme sonucu."""
    total_score: float = 0.0

    # Alt skorlar
    cg_score: float = 0.0
    constraint_score: float = 0.0
    symmetry_score: float = 0.0
    drift_score: float = 0.0
    inertia_score: float = 0.0

    # CG bilgileri
    cg_x: float = 0.0
    cg_y: float = 0.0
    cg_z: float = 0.0
    cg_mac_percent: float = 0.0

    # Drift bilgileri
    drift_x: float = 0.0
    drift_mac_percent: float = 0.0

    # Kısıt ihlal sayısı
    violation_count: int = 0


# Fitness ağırlıkları (v2 — dengelenmiş)
WEIGHTS = {
    "cg": 2000.0,           # CG sapması en yüksek öncelik
    "constraints": 0.5,     # Constraint cezaları zaten yüksek (5000/ihlal)
    "drift": 300.0,         # Yakıt drift ikincil
    "symmetry_y": 400.0,    # Lateral simetri önemli
    "inertia": 30.0,        # En düşük öncelik
    "feasibility_bonus": 5000.0,  # Sıfır ihlal bonusu
}


def calculate_fitness(
    aircraft: AircraftType,
    components: list[Component],
    layout: dict[str, tuple],
    weights: dict = None,
    vibration_limit: float = 100.0,
) -> FitnessResult:
    """
    Bir tasarım bireyi için fitness skoru hesapla.

    Args:
        aircraft: Uçak tipi
        components: Komponent listesi
        layout: Komponent yerleşimi {id: (x, y, z)}
        weights: Ağırlık katsayıları (opsiyonel)
        vibration_limit: Titreşim mesafe limiti

    Returns:
        FitnessResult nesnesi
    """
    w = weights or WEIGHTS
    result = FitnessResult()

    # 1. CG hesabı (tam depo)
    cg = calculate_cg(aircraft, components, layout, fuel_ratio=1.0)
    result.cg_x = cg.cg_x
    result.cg_y = cg.cg_y
    result.cg_z = cg.cg_z
    result.cg_mac_percent = cg.cg_mac_percent

    # CG skoru (hedefe yakınlık)
    # MAC yüzdesindeki sapma + Y/Z sapması
    cg_penalty = cg.mac_error * w.get("cg", WEIGHTS["cg"])

    # Y ve Z eksenlerinde de merkezde olmalı
    lateral_penalty = (abs(cg.cg_y) + abs(cg.cg_z)) * w.get("symmetry_y", WEIGHTS["symmetry_y"])

    result.cg_score = -(cg_penalty + lateral_penalty)

    # 2. Kısıt ihlalleri
    constraints = evaluate_all_constraints(
        aircraft, components, layout, vibration_limit=vibration_limit
    )
    result.constraint_score = -constraints.total_penalty * w.get("constraints", WEIGHTS["constraints"])
    result.violation_count = constraints.violation_count

    # 3. Yakıt drift analizi
    drift = analyze_fuel_drift(aircraft, components, layout)
    result.drift_x = drift.drift_x
    result.drift_mac_percent = drift.drift_mac_percent

    # Drift cezası
    result.drift_score = -drift.drift_mac_percent * w.get("drift", WEIGHTS["drift"])

    # 4. İnertia skoru
    inertia = calculate_inertia(
        components, layout, (cg.cg_x, cg.cg_y, cg.cg_z), fuel_ratio=1.0
    )
    # İnertia simetri oranına göre ödüllendirme
    result.inertia_score = -(1.0 - inertia.symmetry_ratio) * w.get("inertia", WEIGHTS["inertia"]) * 100

    # Feasibility bonus: sıfır ihlal ödülü
    feasibility_bonus = 0.0
    if result.violation_count == 0:
        feasibility_bonus = w.get("feasibility_bonus", 5000.0)

    # Toplam skor
    result.total_score = (
        result.cg_score +
        result.constraint_score +
        result.drift_score +
        result.inertia_score +
        feasibility_bonus
    )

    return result
