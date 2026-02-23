# -*- coding: utf-8 -*-
"""
FastAPI ana uygulama.
Uçak Yerleşim Optimizasyon Aracı — Backend Sunucu
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router

app = FastAPI(
    title="Uçak Yerleşim Optimizasyon Aracı",
    description="TUSAŞ Destekli Tez Projesi — CG-Bilinçli Yapay Zeka Destekli Komponent Yerleşimi",
    version="1.0.0",
)

# CORS — Frontend'den erişim izni
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API rotalarını ekle
app.include_router(router)

# Frontend static dosyaları (build sonrası)
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "ucak-yerlesim-optimizasyon"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
