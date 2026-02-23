# -*- coding: utf-8 -*-
"""
Geometri yardımcı fonksiyonları.
Gövde kesit hesabı, çakışma kontrolü ve gövde içi sınır kontrolü.
"""

import math
from ..models.aircraft import AircraftType, Component


def get_fuselage_radius(aircraft: AircraftType, x: float, axis: str = "y") -> float:
    """
    Verilen X konumundaki gövde yarıçapını döndürür.
    Parametrik: Uçak tipine göre farklı kesit profili.

    Args:
        aircraft: Uçak tipi nesnesi
        x: Boylamasına konum (cm)
        axis: "y" (genişlik) veya "z" (yükseklik)

    Returns:
        O noktadaki yarıçap (cm)
    """
    if x < 0 or x > aircraft.fuselage_length:
        return 0.0

    max_radius = aircraft.fuselage_half_width if axis == "y" else aircraft.fuselage_half_height

    nose_end = aircraft.nose_length
    mid_start = aircraft.mid_start
    mid_end = aircraft.mid_end
    tail_start = aircraft.tail_start

    if x < nose_end:
        # Burun bölgesi
        ratio = x / nose_end if nose_end > 0 else 1.0

        if aircraft.nose_shape == "ogive":
            # Ogive profil (savaş uçağı)
            return ratio ** 0.6 * max_radius
        elif aircraft.nose_shape == "diamond":
            # Diamond profil (stealth UCAV)
            return ratio ** 0.4 * max_radius
        else:
            # Rounded (İHA)
            return ratio ** 0.5 * max_radius

    elif x < mid_end:
        # Orta gövde (maksimum kesit)
        return max_radius

    elif x < aircraft.fuselage_length:
        # Kuyruk (lineer daralma)
        tail_length = aircraft.fuselage_length - mid_end
        if tail_length <= 0:
            return max_radius
        ratio = (x - mid_end) / tail_length
        # Kuyrukta %15'e kadar daralma (savaş uçağı nozul)
        return max_radius * (1 - ratio * 0.85)

    return 0.0


def boxes_collide(pos1: tuple, dim1: tuple, pos2: tuple, dim2: tuple) -> bool:
    """
    İki kutu (AABB) arasında çakışma kontrolü.

    Args:
        pos1, pos2: Kutuların merkez noktaları (x, y, z)
        dim1, dim2: Kutuların boyutları (dx, dy, dz)

    Returns:
        True ise çakışma var
    """
    for i in range(3):
        min1 = pos1[i] - dim1[i] / 2
        max1 = pos1[i] + dim1[i] / 2
        min2 = pos2[i] - dim2[i] / 2
        max2 = pos2[i] + dim2[i] / 2

        if min1 >= max2 or max1 <= min2:
            return False

    return True


def is_inside_fuselage(aircraft: AircraftType, pos: tuple, dim: tuple) -> bool:
    """
    Bir komponent kutusunun gövde içinde olup olmadığını kontrol eder.
    Parametrik gövde kesitiyle çalışır.

    Args:
        aircraft: Uçak tipi
        pos: Komponent merkez noktası (x, y, z)
        dim: Komponent boyutları (dx, dy, dz)

    Returns:
        True ise gövde içinde
    """
    x, y, z = pos
    dx, dy, dz = dim

    # 1. Boylamasına (X) kontrol
    x_min = x - dx / 2
    x_max = x + dx / 2

    if x_min < 0 or x_max > aircraft.fuselage_length:
        return False

    # 2. Radyal (kesit) kontrolü
    # Parçanın en uzak köşesi
    max_y_dist = abs(y) + dy / 2
    max_z_dist = abs(z) + dz / 2

    # Kontrol noktaları: ön, orta, arka
    check_points = [x_min, x, x_max]

    for cx in check_points:
        r_y = get_fuselage_radius(aircraft, cx, "y")
        r_z = get_fuselage_radius(aircraft, cx, "z")

        if r_y <= 0 or r_z <= 0:
            return False

        # Eliptik kesit kontrolü: (y/ry)^2 + (z/rz)^2 <= 1
        if r_y > 0 and r_z > 0:
            check = (max_y_dist / r_y) ** 2 + (max_z_dist / r_z) ** 2
            if check > 1.0:
                return False

    return True


def is_inside_zone(aircraft: AircraftType, zone_name: str, pos: tuple, dim: tuple) -> bool:
    """
    Komponent belirtilen bölge içinde mi kontrol eder.

    Args:
        aircraft: Uçak tipi
        zone_name: Bölge adı ("NOSE", "CENTER_FUSELAGE", vb.)
        pos: Komponent merkez noktası
        dim: Komponent boyutları

    Returns:
        True ise bölge içinde
    """
    if zone_name not in aircraft.zones:
        return True  # Bölge tanımlı değilse kısıt yok

    zone = aircraft.zones[zone_name]
    x, y, z = pos
    dx, dy, dz = dim

    # X ekseni kontrolü
    if zone.x_start is not None and zone.x_end is not None:
        if (x - dx / 2) < zone.x_start or (x + dx / 2) > zone.x_end:
            return False

    # Y ekseni kontrolü (kanat bölgeleri için)
    if zone.y_min is not None and zone.y_max is not None:
        if (y - dy / 2) < zone.y_min or (y + dy / 2) > zone.y_max:
            return False

    return True


def distance_3d(pos1: tuple, pos2: tuple) -> float:
    """İki nokta arasındaki 3D mesafe."""
    return math.sqrt(
        (pos1[0] - pos2[0]) ** 2 +
        (pos1[1] - pos2[1]) ** 2 +
        (pos1[2] - pos2[2]) ** 2
    )
