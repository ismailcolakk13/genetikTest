from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from fastapi.middleware.cors import CORSMiddleware
import os

from modeller.aircraft import Aircraft
from modeller.komponent import Komponent
from algoritmalar.ga import run_ga
from algoritmalar.pso import run_pso
from algoritmalar.nsga2 import run_nsga2
from algoritmalar.nsga2_pso_hybrid import run_nsga2_pso_hybrid
from yardimcilar.gorsellestirici import (
    _bolge_yuzey_olustur, ucak_govdesi_olustur, ozel_parca_ciz
)
import plotly.graph_objects as go
import numpy as np

app = FastAPI(title="Aircraft Component Layout Optimizer")

# Son simülasyon sonucunu sakla (3D görselleştirme için)
_last_result = {"tasarim": None, "aircraft": None, "best_score": None, "best_cg": None, "algoritma": None}

# CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class KomponentRequest(BaseModel):
    id: str
    agirlik: float
    boyut: List[float] # [x, y, z]
    sabit_bolge: str
    sabit_pos: Optional[List[float]] = None
    kilitli: bool
    titresim_hassasiyeti: bool
    sicaklik_hassasiyeti: bool

class SimulationRequest(BaseModel):
    govde_uzunluk: float
    govde_cap: float
    target_cg_x_min: float
    target_cg_x_max: float
    target_cg_y: float
    target_cg_z: float
    max_yakit_agirligi: float
    titresim_limiti: float
    sicaklik_limiti: float
    komponentler: List[KomponentRequest]
    algoritma: str
    pop_size: int
    generations: int

@app.post("/api/run-simulation")
async def run_simulation(req: SimulationRequest):
    try:
        # 1. Konfigürasyonu Parse Et ve Nesnelere Çevir
        db_komponents = []
        for c in req.komponentler:
            # sabit_bolge "GOVDE/TABAN" gibi "/" ile ayrılmış birden fazla bölge olabilir.
            # Boş string veya "SERBEST" ise izin_verilen_bolgeler boş liste olur.
            if c.sabit_bolge and c.sabit_bolge not in ("", "SERBEST"):
                izin_bolgeler = [b.strip() for b in c.sabit_bolge.split("/") if b.strip()]
            else:
                izin_bolgeler = []

            komp = Komponent(
                id=c.id,
                agirlik=c.agirlik,
                boyut=tuple(c.boyut),
                izin_verilen_bolgeler=izin_bolgeler,
                sabit_pos=tuple(c.sabit_pos) if c.sabit_pos else None,
                kilitli=c.kilitli,
                titresim_hassasiyeti=c.titresim_hassasiyeti,
                sicaklik_hassasiyeti=c.sicaklik_hassasiyeti
            )
            db_komponents.append(komp)

        aircraft = Aircraft(
            govde_uzunluk=req.govde_uzunluk,
            govde_cap=req.govde_cap,
            target_cg_x_min=req.target_cg_x_min,
            target_cg_x_max=req.target_cg_x_max,
            target_cg_y=req.target_cg_y,
            target_cg_z=req.target_cg_z,
            max_yakit_agirligi=req.max_yakit_agirligi,
            titresim_limiti=req.titresim_limiti,
            sicaklik_limiti=req.sicaklik_limiti,
            komponentler_db=db_komponents
        )

        # 2. Algoritmayı Çalıştır
        if req.algoritma == "PSO":
            en_iyi_tasarim, best_score, best_cg = run_pso(req.pop_size, req.generations, aircraft)
        elif req.algoritma == "NSGA2":
            en_iyi_tasarim, best_score, best_cg = run_nsga2(req.pop_size, req.generations, aircraft)
        elif req.algoritma == "HYBRID_NSGA2_PSO":
            en_iyi_tasarim, best_score, best_cg = run_nsga2_pso_hybrid(req.pop_size, req.generations, aircraft)
        elif req.algoritma == "GA":
            en_iyi_tasarim, best_score, best_cg = run_ga(req.pop_size, req.generations, aircraft)
        else:
            raise HTTPException(status_code=400, detail="Bilinmeyen algoritma!")

        # 3. Sonuçları JSON'a Döndür
        tasarim_json = {}
        for k_id, k_pos in en_iyi_tasarim.yerlesim.items():
            db_item = next((item for item in aircraft.komponentler_db if item.id == k_id), None)
            if db_item:
                 tasarim_json[k_id] = {
                     "id": k_id,
                     "pos_x": k_pos[0],
                     "pos_y": k_pos[1],
                     "pos_z": k_pos[2],
                     "boyut": db_item.boyut,
                     "agirlik": db_item.agirlik,
                     "sabit_bolge": "/".join(db_item.izin_verilen_bolgeler) if db_item.izin_verilen_bolgeler else "SERBEST"
                 }

        # Son sonucu sakla (3D görselleştirme endpoint'i için)
        _last_result["tasarim"] = en_iyi_tasarim
        _last_result["aircraft"] = aircraft
        _last_result["best_score"] = best_score
        _last_result["best_cg"] = best_cg
        _last_result["algoritma"] = req.algoritma

        response_data = {
            "success": True,
            "algoritma_ismi": req.algoritma,
            "en_iyi_skor": best_score,
            "en_iyi_cg": {
                "x": round(best_cg[0], 2),
                "y": round(best_cg[1], 2),
                "z": round(best_cg[2], 2)
            },
            "tasarim": tasarim_json
        }

        return response_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get-3d-view", response_class=HTMLResponse)
async def get_3d_view():
    """Son simülasyonun 3D Plotly görselleştirmesini HTML olarak döndürür."""
    if _last_result["tasarim"] is None:
        raise HTTPException(status_code=404, detail="Henüz simülasyon çalıştırılmadı!")

    en_iyi_tasarim = _last_result["tasarim"]
    aircraft = _last_result["aircraft"]
    best_score = _last_result["best_score"]
    best_cg = _last_result["best_cg"]
    ALGORITMA = _last_result["algoritma"]

    fig = go.Figure()

    # 1. Bölge yüzeyleri
    for bolge in ["BURUN", "GOVDE", "KUYRUK", "TAVAN", "TABAN"]:
        for trace in _bolge_yuzey_olustur(bolge, aircraft):
            fig.add_trace(trace)

    # 2. Uçak gövdesi
    for parca in ucak_govdesi_olustur(aircraft):
        fig.add_trace(parca)

    # 3. Komponentler — gerçekçi şekiller
    colors = ['red', 'blue', 'orange', 'purple', 'green', 'brown', 'cyan']
    for k_id, pos in en_iyi_tasarim.yerlesim.items():
        boyut = next(item for item in aircraft.komponentler_db if item.id == k_id).boyut
        idx = aircraft.komponentler_db.index(
            next(item for item in aircraft.komponentler_db if item.id == k_id))

        karma_parcalar = ozel_parca_ciz(pos, boyut, colors[idx % len(colors)], k_id, aircraft)
        for t in karma_parcalar:
            fig.add_trace(t)

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2] + boyut[2] / 1.5],
            mode='text', text=[k_id], textposition="top center",
            textfont=dict(size=10, color="black", family="Arial Bold"), showlegend=False
        ))

    # 4. Hedef CG aralığı
    box_r = aircraft.govde_yaricap + 5
    fig.add_trace(go.Mesh3d(
        x=[aircraft.target_cg_x_min, aircraft.target_cg_x_max,
           aircraft.target_cg_x_max, aircraft.target_cg_x_min,
           aircraft.target_cg_x_min, aircraft.target_cg_x_max,
           aircraft.target_cg_x_max, aircraft.target_cg_x_min],
        y=[-box_r, -box_r, box_r, box_r, -box_r, -box_r, box_r, box_r],
        z=[-box_r, -box_r, -box_r, -box_r, box_r, box_r, box_r, box_r],
        color='gold', opacity=0.3, name='HEDEF CG ARALIĞI', alphahull=0
    ))

    # 5. CG gösterimi
    viz_z = aircraft.govde_yaricap + 40
    fig.add_trace(go.Scatter3d(
        x=[best_cg[0]], y=[best_cg[1]], z=[viz_z],
        mode='markers+text', marker=dict(size=12, color='black', symbol='diamond'),
        name='HESAPLANAN CG', text=["HESAPLANAN CG"], textposition="top center",
        textfont=dict(color='black')
    ))
    fig.add_trace(go.Scatter3d(
        x=[best_cg[0], best_cg[0]], y=[best_cg[1], best_cg[1]], z=[best_cg[2], viz_z],
        mode='lines', line=dict(color='black', width=3),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=[best_cg[0]], y=[best_cg[1]], z=[best_cg[2]],
        mode='markers', marker=dict(size=5, color='black'),
        name='Gerçek CG Konumu'
    ))

    target_x_visual = np.clip(best_cg[0], aircraft.target_cg_x_min, aircraft.target_cg_x_max)
    fig.add_trace(go.Scatter3d(
        x=[target_x_visual, best_cg[0]],
        y=[aircraft.target_cg_y, best_cg[1]],
        z=[aircraft.target_cg_z, best_cg[2]],
        mode='lines', line=dict(color='red', width=4, dash='dot'), name='CG Hatası'
    ))

    # Açık tema layout (Reverted to original)
    camera = dict(eye=dict(x=2.0, y=-2.0, z=1.0))
    fig.update_layout(
        title=dict(
            text=f"Ön Tasarım: Uçak İçi Sistem Yerleşimi Optimizasyonu ({ALGORITMA}) | Skor: {best_score:.0f}",
            font=dict(color='black', size=16)
        ),
        scene=dict(
            xaxis=dict(title='Uzunluk (cm)', range=[0, aircraft.govde_uzunluk],
                       backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title='Genişlik (cm)', range=[-200, 200]),
            zaxis=dict(title='Yükseklik (cm)', range=[-100, 100]),
            aspectmode='data',
            camera=camera
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        legend=dict(font=dict(color='black')),
        margin=dict(r=0, l=0, b=0, t=50)
    )

    # Plotly HTML'i döndür (tam sayfa, iframe'de gösterilecek)
    html_content = fig.to_html(
        full_html=True,
        include_plotlyjs='cdn',
        config={'responsive': True, 'displayModeBar': True}
    )
    return HTMLResponse(content=html_content)


# Mount the static directory to serve index.html
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Çalıştırma komutu: uvicorn app:app --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
