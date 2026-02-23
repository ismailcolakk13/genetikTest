# -*- coding: utf-8 -*-
"""
Uçak ve komponent veri modelleri.
JSON konfigürasyon dosyalarından uçak geometrisi ve komponent bilgilerini yükler.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Zone:
    """Uçak üzerindeki bir bölge tanımı."""
    name: str
    x_start: float = 0.0
    x_end: float = 0.0
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    description: str = ""

    @property
    def x_center(self) -> float:
        return (self.x_start + self.x_end) / 2

    @property
    def x_length(self) -> float:
        return self.x_end - self.x_start


@dataclass
class WingGeometry:
    """Kanat geometri parametreleri."""
    sweep_angle_deg: float = 0.0
    span: float = 0.0
    chord_root: float = 0.0
    chord_tip: float = 0.0
    position_x: float = 0.0
    position_z: float = 0.0
    dihedral_deg: float = 0.0


@dataclass
class StabilizerGeometry:
    """Stabilizör geometri parametreleri."""
    span: float = 0.0
    chord_root: float = 0.0
    chord_tip: float = 0.0
    position_x: float = 0.0
    sweep_angle_deg: float = 0.0
    count: int = 1
    height: float = 0.0
    cant_angle_deg: float = 0.0


@dataclass
class CGTarget:
    """CG hedef tanımı (MAC bazlı)."""
    mac_percent_min: float = 25.0
    mac_percent_max: float = 35.0
    mac_leading_edge_x: float = 0.0
    mac_length: float = 0.0

    @property
    def mac_percent_center(self) -> float:
        return (self.mac_percent_min + self.mac_percent_max) / 2

    def mac_to_x(self, mac_percent: float) -> float:
        """MAC yüzdesini X koordinatına çevir."""
        return self.mac_leading_edge_x + (mac_percent / 100.0) * self.mac_length

    def x_to_mac(self, x: float) -> float:
        """X koordinatını MAC yüzdesine çevir."""
        if self.mac_length == 0:
            return 0.0
        return ((x - self.mac_leading_edge_x) / self.mac_length) * 100.0

    @property
    def target_x_min(self) -> float:
        return self.mac_to_x(self.mac_percent_min)

    @property
    def target_x_max(self) -> float:
        return self.mac_to_x(self.mac_percent_max)

    @property
    def target_x_center(self) -> float:
        return self.mac_to_x(self.mac_percent_center)


@dataclass
class AircraftType:
    """Bir uçak tipinin tam tanımı."""
    id: str
    name: str
    description: str = ""
    units: str = "cm"

    # Gövde boyutları
    fuselage_length: float = 0.0
    fuselage_width: float = 0.0
    fuselage_height: float = 0.0
    nose_length: float = 0.0
    nose_shape: str = "ogive"
    mid_start: float = 0.0
    mid_end: float = 0.0
    tail_start: float = 0.0
    tail_end: float = 0.0
    cross_section: str = "circular"

    # Alt geometriler
    wing: Optional[WingGeometry] = None
    horizontal_stabilizer: Optional[StabilizerGeometry] = None
    vertical_stabilizer: Optional[StabilizerGeometry] = None

    # Bölgeler
    zones: dict[str, Zone] = field(default_factory=dict)

    # CG hedefi
    cg_target: Optional[CGTarget] = None

    # Ağırlık bilgileri
    max_fuel_weight: float = 0.0
    empty_weight: float = 0.0

    @property
    def fuselage_half_width(self) -> float:
        return self.fuselage_width / 2

    @property
    def fuselage_half_height(self) -> float:
        return self.fuselage_height / 2


@dataclass
class Component:
    """Uçak içi bir komponent/ekipman."""
    id: str
    name: str
    weight: float  # kg
    size: tuple[float, float, float]  # (dx, dy, dz) cm
    zone: str  # Yerleştirilmesi gereken bölge
    locked: bool = False
    locked_pos: Optional[tuple[float, float, float]] = None
    vibration_sensitive: bool = False
    vibration_source: bool = False
    category: str = "general"
    description: str = ""
    fuel_capacity: float = 0.0
    proximity_to: Optional[str] = None

    @property
    def is_fuel_tank(self) -> bool:
        return self.category == "fuel"

    @property
    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]


def load_aircraft_type(json_path: str) -> AircraftType:
    """JSON dosyasından uçak tipi yükle."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    geom = data.get("geometry", {})

    # Kanat
    wing_data = geom.get("wing")
    wing = WingGeometry(**wing_data) if wing_data else None

    # Yatay stabilizör
    hs_data = geom.get("horizontal_stabilizer")
    h_stab = StabilizerGeometry(**hs_data) if hs_data else None

    # Dikey stabilizör
    vs_data = geom.get("vertical_stabilizer")
    v_stab = StabilizerGeometry(**vs_data) if vs_data else None

    # CG hedefi
    cg_data = data.get("cg_target")
    cg_target = CGTarget(**cg_data) if cg_data else None

    # Bölgeler
    zones = {}
    for zone_name, zone_data in data.get("zones", {}).items():
        zones[zone_name] = Zone(name=zone_name, **zone_data)

    return AircraftType(
        id=data["id"],
        name=data["name"],
        description=data.get("description", ""),
        units=data.get("units", "cm"),
        fuselage_length=geom.get("fuselage_length", 0),
        fuselage_width=geom.get("fuselage_width", 0),
        fuselage_height=geom.get("fuselage_height", 0),
        nose_length=geom.get("nose_length", 0),
        nose_shape=geom.get("nose_shape", "ogive"),
        mid_start=geom.get("mid_start", 0),
        mid_end=geom.get("mid_end", 0),
        tail_start=geom.get("tail_start", 0),
        tail_end=geom.get("tail_end", 0),
        cross_section=geom.get("cross_section", "circular"),
        wing=wing,
        horizontal_stabilizer=h_stab,
        vertical_stabilizer=v_stab,
        zones=zones,
        cg_target=cg_target,
        max_fuel_weight=data.get("max_fuel_weight", 0),
        empty_weight=data.get("empty_weight", 0),
    )


def load_components(json_path: str) -> list[Component]:
    """JSON dosyasından komponent listesi yükle."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    components = []
    for comp_data in data.get("components", []):
        comp = Component(
            id=comp_data["id"],
            name=comp_data.get("name", comp_data["id"]),
            weight=comp_data["weight"],
            size=tuple(comp_data["size"]),
            zone=comp_data["zone"],
            locked=comp_data.get("locked", False),
            locked_pos=tuple(comp_data["locked_pos"]) if comp_data.get("locked_pos") else None,
            vibration_sensitive=comp_data.get("vibration_sensitive", False),
            vibration_source=comp_data.get("vibration_source", False),
            category=comp_data.get("category", "general"),
            description=comp_data.get("description", ""),
            fuel_capacity=comp_data.get("fuel_capacity", 0),
            proximity_to=comp_data.get("proximity_to"),
        )
        components.append(comp)

    return components


def list_available_aircraft(data_dir: str) -> list[dict]:
    """Mevcut uçak tiplerini listele."""
    aircraft_dir = os.path.join(data_dir, "aircraft_types")
    result = []
    if not os.path.exists(aircraft_dir):
        return result
    for filename in os.listdir(aircraft_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(aircraft_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                result.append({
                    "id": data["id"],
                    "name": data["name"],
                    "description": data.get("description", ""),
                    "file": filename,
                })
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def get_component_library_path(aircraft_id: str, data_dir: str) -> Optional[str]:
    """Uçak tipine uygun komponent kütüphanesi dosya yolunu bul."""
    comp_dir = os.path.join(data_dir, "component_libraries")
    if not os.path.exists(comp_dir):
        return None

    # Önce direkt eşleşme dene
    for filename in os.listdir(comp_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(comp_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("aircraft_type") == aircraft_id:
                    return filepath
            except (json.JSONDecodeError, KeyError):
                continue

    return None
