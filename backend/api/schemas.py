# -*- coding: utf-8 -*-
"""
Pydantic şemaları — API request/response modelleri.
"""

from pydantic import BaseModel, Field
from typing import Optional


# --- Request Modelleri ---

class OptimizeRequest(BaseModel):
    """Optimizasyon isteği."""
    aircraft_type_id: str = Field(..., description="Uçak tipi ID'si")
    population_size: int = Field(100, ge=10, le=500)
    generations: int = Field(50, ge=5, le=200)
    mutation_rate: float = Field(0.15, ge=0.0, le=1.0)
    vibration_limit: float = Field(100.0, ge=0.0)


class EvaluateRequest(BaseModel):
    """Manuel yerleşim değerlendirme isteği."""
    aircraft_type_id: str
    layout: dict[str, list[float]]  # {component_id: [x, y, z]}


class ComponentPosition(BaseModel):
    """Tek bir komponent pozisyonu."""
    component_id: str
    x: float
    y: float
    z: float


# --- Response Modelleri ---

class AircraftTypeResponse(BaseModel):
    """Uçak tipi özet bilgisi."""
    id: str
    name: str
    description: str = ""


class AircraftDetailResponse(BaseModel):
    """Uçak tipi detaylı bilgisi."""
    id: str
    name: str
    description: str
    units: str
    geometry: dict
    zones: dict
    cg_target: dict
    components: list[dict]


class CGResponse(BaseModel):
    """CG hesaplama sonucu."""
    cg_x: float
    cg_y: float
    cg_z: float
    cg_mac_percent: float
    in_target: bool
    total_mass: float


class OptimizeStatusResponse(BaseModel):
    """Optimizasyon durumu."""
    job_id: str
    status: str  # "running", "completed", "failed"
    progress: int  # 0-100
    generation: int = 0
    total_generations: int = 0
    best_score: float = 0.0
    best_cg_mac: float = 0.0
    violations: int = 0
    elapsed_seconds: float = 0.0


class ViolationResponse(BaseModel):
    """Kısıt ihlali."""
    type: str
    component: str
    related_component: str = ""
    description: str
    penalty: float


class OptimizeResultResponse(BaseModel):
    """Optimizasyon sonucu."""
    job_id: str
    score: float
    cg: CGResponse
    drift_x: float
    drift_mac_percent: float
    violations: list[ViolationResponse]
    layout: dict[str, list[float]]
    generation_history: list[dict]
    sub_scores: dict


class EvaluateResponse(BaseModel):
    """Değerlendirme sonucu."""
    score: float
    cg: CGResponse
    drift_x: float
    drift_mac_percent: float
    violations: list[ViolationResponse]
    sub_scores: dict
