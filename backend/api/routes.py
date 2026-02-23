# -*- coding: utf-8 -*-
"""
API route tanımları.
FastAPI endpoint'leri: uçak listeleme, optimizasyon, değerlendirme.
"""

import os
import uuid
import threading
from fastapi import APIRouter, HTTPException

from .schemas import (
    AircraftTypeResponse,
    AircraftDetailResponse,
    OptimizeRequest,
    OptimizeStatusResponse,
    OptimizeResultResponse,
    EvaluateRequest,
    EvaluateResponse,
    CGResponse,
    ViolationResponse,
)
from ..models.aircraft import load_aircraft_type, load_components, list_available_aircraft, get_component_library_path
from ..optimization.genetic_algorithm import GeneticAlgorithm, GAConfig, Individual
from ..optimization.fitness import calculate_fitness
from ..optimization.constraints import evaluate_all_constraints
from ..utils.cg_calculator import calculate_cg, analyze_fuel_drift

router = APIRouter(prefix="/api")

# Data dizini (proje kökünden)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "aircraft_data")

# Çalışan optimizasyon job'ları
_jobs: dict[str, dict] = {}


def _load_aircraft_and_components(aircraft_type_id: str):
    """Uçak tipi ve komponentlerini yükle."""
    # Uçak tipi dosyasını bul
    aircraft_dir = os.path.join(DATA_DIR, "aircraft_types")
    aircraft_file = None
    for f in os.listdir(aircraft_dir):
        if f.endswith('.json'):
            filepath = os.path.join(aircraft_dir, f)
            import json
            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if data.get("id") == aircraft_type_id:
                aircraft_file = filepath
                break

    if not aircraft_file:
        raise HTTPException(status_code=404, detail=f"Uçak tipi bulunamadı: {aircraft_type_id}")

    aircraft = load_aircraft_type(aircraft_file)

    # Komponent kütüphanesi
    comp_path = get_component_library_path(aircraft_type_id, DATA_DIR)
    if not comp_path:
        raise HTTPException(status_code=404, detail=f"Komponent kütüphanesi bulunamadı: {aircraft_type_id}")

    components = load_components(comp_path)
    return aircraft, components


# --- ENDPOINT'LER ---

@router.get("/aircraft-types", response_model=list[AircraftTypeResponse])
def get_aircraft_types():
    """Mevcut uçak tiplerini listele."""
    types = list_available_aircraft(DATA_DIR)
    return [AircraftTypeResponse(**t) for t in types]


@router.get("/aircraft/{aircraft_type_id}", response_model=AircraftDetailResponse)
def get_aircraft_detail(aircraft_type_id: str):
    """Uçak tipi detaylı bilgisi + komponent listesi."""
    aircraft, components = _load_aircraft_and_components(aircraft_type_id)

    # Geometry dict
    geom = {
        "fuselage_length": aircraft.fuselage_length,
        "fuselage_width": aircraft.fuselage_width,
        "fuselage_height": aircraft.fuselage_height,
        "nose_length": aircraft.nose_length,
        "nose_shape": aircraft.nose_shape,
        "mid_start": aircraft.mid_start,
        "mid_end": aircraft.mid_end,
        "tail_start": aircraft.tail_start,
        "tail_end": aircraft.tail_end,
        "cross_section": aircraft.cross_section,
    }
    if aircraft.wing:
        geom["wing"] = {
            "sweep_angle_deg": aircraft.wing.sweep_angle_deg,
            "span": aircraft.wing.span,
            "chord_root": aircraft.wing.chord_root,
            "chord_tip": aircraft.wing.chord_tip,
            "position_x": aircraft.wing.position_x,
            "position_z": aircraft.wing.position_z,
            "dihedral_deg": aircraft.wing.dihedral_deg,
        }

    # Zones dict
    zones = {}
    for name, z in aircraft.zones.items():
        zones[name] = {
            "x_start": z.x_start,
            "x_end": z.x_end,
            "description": z.description,
        }
        if z.y_min is not None:
            zones[name]["y_min"] = z.y_min
        if z.y_max is not None:
            zones[name]["y_max"] = z.y_max

    # CG target dict
    cg_target = {}
    if aircraft.cg_target:
        cg_target = {
            "mac_percent_min": aircraft.cg_target.mac_percent_min,
            "mac_percent_max": aircraft.cg_target.mac_percent_max,
            "mac_leading_edge_x": aircraft.cg_target.mac_leading_edge_x,
            "mac_length": aircraft.cg_target.mac_length,
            "target_x_min": aircraft.cg_target.target_x_min,
            "target_x_max": aircraft.cg_target.target_x_max,
        }

    # Components list
    comp_list = []
    for c in components:
        comp_dict = {
            "id": c.id,
            "name": c.name,
            "weight": c.weight,
            "size": list(c.size),
            "zone": c.zone,
            "locked": c.locked,
            "vibration_sensitive": c.vibration_sensitive,
            "vibration_source": c.vibration_source,
            "category": c.category,
            "description": c.description,
        }
        if c.locked_pos:
            comp_dict["locked_pos"] = list(c.locked_pos)
        if c.fuel_capacity > 0:
            comp_dict["fuel_capacity"] = c.fuel_capacity
        comp_list.append(comp_dict)

    return AircraftDetailResponse(
        id=aircraft.id,
        name=aircraft.name,
        description=aircraft.description,
        units=aircraft.units,
        geometry=geom,
        zones=zones,
        cg_target=cg_target,
        components=comp_list,
    )


@router.post("/optimize", response_model=OptimizeStatusResponse)
def start_optimization(request: OptimizeRequest):
    """Optimizasyon başlat (arka planda)."""
    aircraft, components = _load_aircraft_and_components(request.aircraft_type_id)

    job_id = str(uuid.uuid4())[:8]

    config = GAConfig(
        population_size=request.population_size,
        generations=request.generations,
        mutation_rate=request.mutation_rate,
        vibration_limit=request.vibration_limit,
    )

    ga = GeneticAlgorithm(aircraft, components, config)
    ga.initialize()

    _jobs[job_id] = {
        "status": "running",
        "ga": ga,
        "aircraft": aircraft,
        "components": components,
        "result": None,
    }

    # Arka planda çalıştır
    def run_job():
        try:
            ga.run()
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = ga.get_result_summary()
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return OptimizeStatusResponse(
        job_id=job_id,
        status="running",
        progress=0,
        total_generations=request.generations,
    )


@router.get("/optimize/{job_id}/status", response_model=OptimizeStatusResponse)
def get_optimization_status(job_id: str):
    """Optimizasyon durumunu sorgula."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job bulunamadı")

    job = _jobs[job_id]
    ga: GeneticAlgorithm = job["ga"]
    progress = ga.progress

    pct = int((progress.generation / progress.total_generations) * 100) if progress.total_generations > 0 else 0

    return OptimizeStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=pct,
        generation=progress.generation,
        total_generations=progress.total_generations,
        best_score=progress.best_score,
        best_cg_mac=progress.best_cg_mac,
        violations=progress.violations,
        elapsed_seconds=progress.elapsed_seconds,
    )


@router.get("/optimize/{job_id}/result", response_model=OptimizeResultResponse)
def get_optimization_result(job_id: str):
    """Optimizasyon sonucunu al."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job bulunamadı")

    job = _jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job henüz tamamlanmadı: {job['status']}")

    result = job["result"]
    ga: GeneticAlgorithm = job["ga"]
    aircraft = job["aircraft"]
    components = job["components"]

    # Layout'u list formatına çevir
    layout_lists = {k: list(v) for k, v in result["layout"].items()}

    # Kısıt ihlallerini al
    constraints = evaluate_all_constraints(
        aircraft, components,
        result["layout"],
        vibration_limit=ga.config.vibration_limit,
    )

    violations = [
        ViolationResponse(
            type=v.constraint_type,
            component=v.component_id,
            related_component=v.related_component,
            description=v.description,
            penalty=v.penalty,
        )
        for v in constraints.violations
    ]

    # CG bilgisi
    cg = calculate_cg(aircraft, components, result["layout"], fuel_ratio=1.0)

    return OptimizeResultResponse(
        job_id=job_id,
        score=result["score"],
        cg=CGResponse(
            cg_x=cg.cg_x,
            cg_y=cg.cg_y,
            cg_z=cg.cg_z,
            cg_mac_percent=cg.cg_mac_percent,
            in_target=cg.in_target,
            total_mass=cg.total_mass,
        ),
        drift_x=result["drift"]["x"],
        drift_mac_percent=result["drift"]["mac_percent"],
        violations=violations,
        layout=layout_lists,
        generation_history=result["generation_history"],
        sub_scores=result["sub_scores"],
    )


@router.post("/design/evaluate", response_model=EvaluateResponse)
def evaluate_design(request: EvaluateRequest):
    """Manuel bir yerleşimi değerlendir."""
    aircraft, components = _load_aircraft_and_components(request.aircraft_type_id)

    # Layout'u tuple formatına çevir
    layout = {k: tuple(v) for k, v in request.layout.items()}

    # Fitness hesapla
    fit = calculate_fitness(aircraft, components, layout)

    # CG hesapla
    cg = calculate_cg(aircraft, components, layout, fuel_ratio=1.0)

    # Drift analizi
    drift = analyze_fuel_drift(aircraft, components, layout)

    # Kısıtları değerlendir
    constraints = evaluate_all_constraints(aircraft, components, layout)
    violations = [
        ViolationResponse(
            type=v.constraint_type,
            component=v.component_id,
            related_component=v.related_component,
            description=v.description,
            penalty=v.penalty,
        )
        for v in constraints.violations
    ]

    return EvaluateResponse(
        score=fit.total_score,
        cg=CGResponse(
            cg_x=cg.cg_x,
            cg_y=cg.cg_y,
            cg_z=cg.cg_z,
            cg_mac_percent=cg.cg_mac_percent,
            in_target=cg.in_target,
            total_mass=cg.total_mass,
        ),
        drift_x=drift.drift_x,
        drift_mac_percent=drift.drift_mac_percent,
        violations=violations,
        sub_scores={
            "cg_score": fit.cg_score,
            "constraint_score": fit.constraint_score,
            "drift_score": fit.drift_score,
            "symmetry_score": fit.symmetry_score,
            "inertia_score": fit.inertia_score,
        },
    )
