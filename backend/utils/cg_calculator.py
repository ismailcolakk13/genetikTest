# -*- coding: utf-8 -*-
"""
CG (Ağırlık Merkezi) hesaplama modülü.
MAC bazlı CG yüzdesi, yakıt drift analizi ve inertia tensör hesabı.
"""

from dataclasses import dataclass
from ..models.aircraft import AircraftType, Component


@dataclass
class CGResult:
    """CG hesaplama sonucu."""
    # Ağırlık merkezi koordinatları
    cg_x: float = 0.0
    cg_y: float = 0.0
    cg_z: float = 0.0

    # MAC bazlı CG yüzdesi
    cg_mac_percent: float = 0.0

    # Toplam kütle
    total_mass: float = 0.0

    # Hedeften sapma
    mac_error: float = 0.0  # MAC yüzdesindeki sapma
    distance_error: float = 0.0  # 3D mesafe hatası

    # Hedef aralık içinde mi
    in_target: bool = False


@dataclass
class DriftAnalysis:
    """Yakıt tüketimi CG drift analizi."""
    cg_full: CGResult = None  # Tam depo
    cg_empty: CGResult = None  # Boş depo
    drift_x: float = 0.0  # X eksenindeki kayma (cm)
    drift_mac_percent: float = 0.0  # MAC yüzdesindeki kayma
    max_drift: float = 0.0  # Tüm senaryolardaki maksimum drift

    # Tüm doluluk senaryoları
    scenarios: list = None

    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = []


@dataclass
class InertiaResult:
    """İnertia tensör sonucu."""
    ixx: float = 0.0  # Roll ekseni
    iyy: float = 0.0  # Pitch ekseni
    izz: float = 0.0  # Yaw ekseni

    @property
    def symmetry_ratio(self) -> float:
        """Lateral simetri oranı (1.0 = mükemmel simetri)."""
        if self.ixx == 0:
            return 0.0
        return min(self.ixx, self.izz) / max(self.ixx, self.izz)


def calculate_cg(
    aircraft: AircraftType,
    components: list[Component],
    layout: dict[str, tuple],
    fuel_ratio: float = 1.0
) -> CGResult:
    """
    Ağırlık merkezi hesapla.

    Args:
        aircraft: Uçak tipi
        components: Komponent listesi
        layout: Komponent yerleşimi {komponent_id: (x, y, z)}
        fuel_ratio: Yakıt doluluk oranı (0.0 - 1.0)

    Returns:
        CGResult nesnesi
    """
    total_mass = 0.0
    moment_x = 0.0
    moment_y = 0.0
    moment_z = 0.0

    for comp in components:
        if comp.id not in layout:
            continue

        pos = layout[comp.id]
        mass = comp.weight

        # Yakıt tankı ise doluluk oranına göre ağırlık ekle
        if comp.is_fuel_tank and comp.fuel_capacity > 0:
            mass += comp.fuel_capacity * fuel_ratio

        total_mass += mass
        moment_x += mass * pos[0]
        moment_y += mass * pos[1]
        moment_z += mass * pos[2]

    if total_mass == 0:
        return CGResult()

    cg_x = moment_x / total_mass
    cg_y = moment_y / total_mass
    cg_z = moment_z / total_mass

    # MAC bazlı CG yüzdesi
    cg_mac = 0.0
    mac_error = 0.0
    in_target = False
    distance_error = 0.0

    if aircraft.cg_target:
        cg_mac = aircraft.cg_target.x_to_mac(cg_x)

        # MAC yüzdesindeki sapma
        target_center = aircraft.cg_target.mac_percent_center
        mac_error = abs(cg_mac - target_center)

        # Hedef aralık kontrolü
        in_target = (
            aircraft.cg_target.mac_percent_min <= cg_mac <= aircraft.cg_target.mac_percent_max
        )

        # 3D mesafe hatası
        target_x = aircraft.cg_target.target_x_center
        distance_error = (
            (cg_x - target_x) ** 2 +
            cg_y ** 2 +  # Y hedefi 0
            cg_z ** 2    # Z hedefi 0
        ) ** 0.5

    return CGResult(
        cg_x=cg_x,
        cg_y=cg_y,
        cg_z=cg_z,
        cg_mac_percent=cg_mac,
        total_mass=total_mass,
        mac_error=mac_error,
        distance_error=distance_error,
        in_target=in_target,
    )


def analyze_fuel_drift(
    aircraft: AircraftType,
    components: list[Component],
    layout: dict[str, tuple],
    fuel_ratios: list[float] = None
) -> DriftAnalysis:
    """
    Farklı yakıt doluluk senaryolarında CG kaymasını analiz eder.

    Args:
        aircraft: Uçak tipi
        components: Komponent listesi
        layout: Komponent yerleşimi
        fuel_ratios: Test edilecek doluluk oranları

    Returns:
        DriftAnalysis nesnesi
    """
    if fuel_ratios is None:
        fuel_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    scenarios = []
    for ratio in fuel_ratios:
        cg = calculate_cg(aircraft, components, layout, fuel_ratio=ratio)
        scenarios.append({"fuel_ratio": ratio, "cg": cg})

    # Full ve empty CG
    cg_full = calculate_cg(aircraft, components, layout, fuel_ratio=1.0)
    cg_empty = calculate_cg(aircraft, components, layout, fuel_ratio=0.0)

    # Drift hesabı
    drift_x = abs(cg_full.cg_x - cg_empty.cg_x)
    drift_mac = abs(cg_full.cg_mac_percent - cg_empty.cg_mac_percent)

    # Maksimum drift (tüm senaryolar arasında)
    cg_xs = [s["cg"].cg_x for s in scenarios]
    max_drift = max(cg_xs) - min(cg_xs) if cg_xs else 0.0

    return DriftAnalysis(
        cg_full=cg_full,
        cg_empty=cg_empty,
        drift_x=drift_x,
        drift_mac_percent=drift_mac,
        max_drift=max_drift,
        scenarios=scenarios,
    )


def calculate_inertia(
    components: list[Component],
    layout: dict[str, tuple],
    cg: tuple[float, float, float],
    fuel_ratio: float = 1.0
) -> InertiaResult:
    """
    CG etrafındaki inertia tensörünü hesaplar (diyagonal elemanlar).
    Nokta kütle yaklaşımı kullanılır.

    Args:
        components: Komponent listesi
        layout: Komponent yerleşimi
        cg: Ağırlık merkezi (x, y, z)
        fuel_ratio: Yakıt doluluk oranı

    Returns:
        InertiaResult nesnesi
    """
    ixx = 0.0  # Roll (Y-Z düzleminde)
    iyy = 0.0  # Pitch (X-Z düzleminde)
    izz = 0.0  # Yaw (X-Y düzleminde)

    for comp in components:
        if comp.id not in layout:
            continue

        pos = layout[comp.id]
        mass = comp.weight

        if comp.is_fuel_tank and comp.fuel_capacity > 0:
            mass += comp.fuel_capacity * fuel_ratio

        # CG'den olan mesafe bileşenleri
        dx = pos[0] - cg[0]
        dy = pos[1] - cg[1]
        dz = pos[2] - cg[2]

        # Nokta kütle inertia tensörü
        ixx += mass * (dy ** 2 + dz ** 2)
        iyy += mass * (dx ** 2 + dz ** 2)
        izz += mass * (dx ** 2 + dy ** 2)

    return InertiaResult(ixx=ixx, iyy=iyy, izz=izz)
