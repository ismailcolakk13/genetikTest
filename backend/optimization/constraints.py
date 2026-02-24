# -*- coding: utf-8 -*-
"""
Kısıt (constraint) motoru — v2 kalibre edilmiş.
Komponent yerleşimi için çeşitli fiziksel ve tasarım kısıtlarını değerlendirir.
"""

from dataclasses import dataclass, field
from ..models.aircraft import AircraftType, Component
from ..utils.geometry import boxes_collide, is_inside_fuselage, is_inside_zone, distance_3d


@dataclass
class ConstraintViolation:
    """Tek bir kısıt ihlali."""
    constraint_type: str
    component_id: str
    description: str
    penalty: float
    related_component: str = ""


@dataclass
class ConstraintResult:
    """Kısıt değerlendirme sonucu."""
    violations: list[ConstraintViolation] = field(default_factory=list)
    total_penalty: float = 0.0
    violation_count: int = 0

    def add(self, violation: ConstraintViolation):
        self.violations.append(violation)
        self.total_penalty += violation.penalty
        self.violation_count += 1


# Ceza ağırlıkları (v2 — kalibre edilmiş)
PENALTY_COLLISION = 5000    # Düşürüldü: CG sinyalini bastırmaması için
PENALTY_BOUNDARY = 3000     # Orantılı ceza ile
PENALTY_ZONE = 2000         # Düşürüldü
PENALTY_VIBRATION = 1500    # Hafif düşürüldü
PENALTY_PROXIMITY = 800     # İkincil kısıt
PENALTY_SYMMETRY = 300      # Tercih, zorunlu değil


def _box_penetration_depth(pos1, size1, pos2, size2):
    """İki AABB kutusunun penetrasyon derinliğini hesapla."""
    overlap = 0.0
    for axis in range(3):
        half1 = size1[axis] / 2
        half2 = size2[axis] / 2
        min1 = pos1[axis] - half1
        max1 = pos1[axis] + half1
        min2 = pos2[axis] - half2
        max2 = pos2[axis] + half2

        axis_overlap = min(max1, max2) - max(min1, min2)
        if axis_overlap > 0:
            overlap += axis_overlap
    return overlap


def evaluate_all_constraints(
    aircraft: AircraftType,
    components: list[Component],
    layout: dict[str, tuple],
    vibration_limit: float = 100.0
) -> ConstraintResult:
    """
    Tüm kısıtları değerlendir.

    Args:
        aircraft: Uçak tipi
        components: Komponent listesi
        layout: Komponent yerleşimi
        vibration_limit: Titreşim hassas parçalar için minimum mesafe

    Returns:
        ConstraintResult nesnesi
    """
    result = ConstraintResult()
    comp_dict = {c.id: c for c in components}

    # 1. Çakışma kontrolü — orantılı penetrasyon cezası
    keys = [k for k in layout.keys()]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            id1, id2 = keys[i], keys[j]
            if id1 not in comp_dict or id2 not in comp_dict:
                continue

            c1, c2 = comp_dict[id1], comp_dict[id2]
            if boxes_collide(layout[id1], c1.size, layout[id2], c2.size):
                # Penetrasyon derinliğine orantılı ceza
                depth = _box_penetration_depth(layout[id1], c1.size, layout[id2], c2.size)
                avg_size = (sum(c1.size) + sum(c2.size)) / 6
                severity = min(depth / (avg_size + 1), 3.0)
                penalty = PENALTY_COLLISION * (0.3 + 0.7 * severity)

                result.add(ConstraintViolation(
                    constraint_type="COLLISION",
                    component_id=id1,
                    related_component=id2,
                    description=f"{c1.name} ile {c2.name} çakışıyor",
                    penalty=penalty,
                ))

    # 2. Gövde sınırı kontrolü
    for comp_id, pos in layout.items():
        if comp_id not in comp_dict:
            continue
        comp = comp_dict[comp_id]

        # Kanat bölgesi parçaları gövde içi kontrolünden muaf
        if comp.zone.startswith("WING_"):
            continue

        if not is_inside_fuselage(aircraft, pos, comp.size):
            result.add(ConstraintViolation(
                constraint_type="BOUNDARY",
                component_id=comp_id,
                description=f"{comp.name} gövde dışına taşıyor",
                penalty=PENALTY_BOUNDARY,
            ))

    # 3. Bölge kısıtları
    for comp_id, pos in layout.items():
        if comp_id not in comp_dict:
            continue
        comp = comp_dict[comp_id]

        if not is_inside_zone(aircraft, comp.zone, pos, comp.size):
            result.add(ConstraintViolation(
                constraint_type="ZONE",
                component_id=comp_id,
                description=f"{comp.name} tanımlı bölgesi ({comp.zone}) dışında",
                penalty=PENALTY_ZONE,
            ))

    # 4. Titreşim hassasiyeti kontrolü
    vib_sources = [
        comp_id for comp_id, comp in comp_dict.items()
        if comp.vibration_source and comp_id in layout
    ]
    vib_sensitive = [
        comp_id for comp_id, comp in comp_dict.items()
        if comp.vibration_sensitive and comp_id in layout
    ]

    for src_id in vib_sources:
        for sens_id in vib_sensitive:
            dist = distance_3d(layout[src_id], layout[sens_id])
            if dist < vibration_limit:
                deficit = vibration_limit - dist
                result.add(ConstraintViolation(
                    constraint_type="VIBRATION",
                    component_id=sens_id,
                    related_component=src_id,
                    description=f"{comp_dict[sens_id].name} titreşim kaynağı {comp_dict[src_id].name}'e çok yakın ({dist:.1f} cm < {vibration_limit} cm)",
                    penalty=PENALTY_VIBRATION * (deficit / vibration_limit),
                ))

    # 5. Yakınlık kısıtları (proximity_to)
    for comp_id, comp in comp_dict.items():
        if comp.proximity_to and comp_id in layout:
            target_id = comp.proximity_to
            if target_id in layout:
                dist = distance_3d(layout[comp_id], layout[target_id])
                proximity_limit = 150.0
                if dist > proximity_limit:
                    result.add(ConstraintViolation(
                        constraint_type="PROXIMITY",
                        component_id=comp_id,
                        related_component=target_id,
                        description=f"{comp.name}, {comp_dict[target_id].name}'e yakın olmalı (mesafe: {dist:.1f} cm)",
                        penalty=PENALTY_PROXIMITY * (dist / proximity_limit),
                    ))

    # 6. Lateral simetri kontrolü (sol-sağ çiftleri)
    _check_symmetry(comp_dict, layout, result)

    return result


def _check_symmetry(
    comp_dict: dict[str, Component],
    layout: dict[str, tuple],
    result: ConstraintResult
):
    """Sol-sağ komponent çiftlerinin simetrisini kontrol et."""
    checked = set()
    for comp_id in layout:
        if comp_id.endswith("_L") and comp_id not in checked:
            pair_id = comp_id[:-2] + "_R"
            if pair_id in layout:
                checked.add(comp_id)
                checked.add(pair_id)

                pos_l = layout[comp_id]
                pos_r = layout[pair_id]

                y_diff = abs(abs(pos_l[1]) - abs(pos_r[1]))
                x_diff = abs(pos_l[0] - pos_r[0])
                z_diff = abs(pos_l[2] - pos_r[2])

                total_asym = x_diff + y_diff + z_diff
                if total_asym > 20.0:
                    result.add(ConstraintViolation(
                        constraint_type="SYMMETRY",
                        component_id=comp_id,
                        related_component=pair_id,
                        description=f"{comp_id} ve {pair_id} simetrik değil (asimetri: {total_asym:.1f} cm)",
                        penalty=PENALTY_SYMMETRY * (total_asym / 20.0),
                    ))
