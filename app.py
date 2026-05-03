from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from fastapi.middleware.cors import CORSMiddleware
import os

from modeller.aircraft import Aircraft
from modeller.komponent import Komponent
from algoritmalar.ga import run_ga
from algoritmalar.pso import run_pso
from algoritmalar.nsga2 import run_nsga2

app = FastAPI(title="Aircraft Component Layout Optimizer")

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
            komp = Komponent(
                id=c.id,
                agirlik=c.agirlik,
                boyut=tuple(c.boyut),
                izin_verilen_bolgeler=[c.sabit_bolge] if c.sabit_bolge else [],
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

        # Detaylı istatistikleri test etmek isterseniz (fitness vs.) ekleyebiliriz.
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

# Mount the static directory to serve index.html
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Çalıştırma komutu: uvicorn app:app --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
