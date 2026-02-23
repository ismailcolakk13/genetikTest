import { useState, useEffect, useCallback, useRef } from "react";
import AircraftViewer3D from "./components/AircraftViewer3D";
import "./index.css";

const API_BASE = "http://localhost:8000/api";

/* ===== API Yardımcıları ===== */
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API Hatası: ${res.status}`);
  return res.json();
}

async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`API Hatası: ${res.status}`);
  return res.json();
}

/* ===== Kategori Renk Eşleşmesi ===== */
const CATEGORY_LABELS = {
  propulsion: { label: "İtki", badge: "propulsion" },
  avionics: { label: "Aviyonik", badge: "avionics" },
  fuel: { label: "Yakıt", badge: "fuel" },
  landing_gear: { label: "İniş T.", badge: "systems" },
  weapon: { label: "Silah", badge: "locked" },
  systems: { label: "Sistem", badge: "systems" },
  electrical: { label: "Elektrik", badge: "avionics" },
  structure: { label: "Yapı", badge: "systems" },
  payload: { label: "Yük", badge: "fuel" },
};

/* ===== Uçak İkonları ===== */
const AIRCRAFT_ICONS = {
  fighter_5gen: "✈️",
  uav_male: "🛩️",
  uav_combat: "🦅",
};

export default function App() {
  // State
  const [aircraftTypes, setAircraftTypes] = useState([]);
  const [selectedTypeId, setSelectedTypeId] = useState(null);
  const [aircraftData, setAircraftData] = useState(null);
  const [selectedComponent, setSelectedComponent] = useState(null);
  const [layout, setLayout] = useState(null);
  const [cgPosition, setCgPosition] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Optimizasyon state
  const [optimizing, setOptimizing] = useState(false);
  const [optimizeStatus, setOptimizeStatus] = useState(null);
  const [optimizeResult, setOptimizeResult] = useState(null);
  const [gaConfig, setGaConfig] = useState({
    population_size: 100,
    generations: 50,
    mutation_rate: 0.15,
    vibration_limit: 100,
  });

  const pollRef = useRef(null);

  // Uçak tiplerini yükle
  useEffect(() => {
    fetchJSON(`${API_BASE}/aircraft-types`)
      .then(setAircraftTypes)
      .catch((e) => setError(e.message));
  }, []);

  // Uçak tipi seçildiğinde detayları yükle
  useEffect(() => {
    if (!selectedTypeId) return;

    setLoading(true);
    setOptimizeResult(null);
    setOptimizeStatus(null);
    setLayout(null);
    setCgPosition(null);

    fetchJSON(`${API_BASE}/aircraft/${selectedTypeId}`)
      .then((data) => {
        setAircraftData(data);
        // Varsayılan yerleşim: kilitli parçalar sabit pozisyonlarında
        const defaultLayout = {};
        for (const comp of data.components) {
          if (comp.locked && comp.locked_pos) {
            defaultLayout[comp.id] = comp.locked_pos;
          }
        }
        setLayout(defaultLayout);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [selectedTypeId]);

  // Optimizasyon başlat
  const startOptimization = useCallback(async () => {
    if (!selectedTypeId) return;

    setOptimizing(true);
    setOptimizeResult(null);

    try {
      const status = await postJSON(`${API_BASE}/optimize`, {
        aircraft_type_id: selectedTypeId,
        ...gaConfig,
      });

      setOptimizeStatus(status);

      // Polling başlat
      pollRef.current = setInterval(async () => {
        try {
          const s = await fetchJSON(
            `${API_BASE}/optimize/${status.job_id}/status`,
          );
          setOptimizeStatus(s);

          if (s.status === "completed") {
            clearInterval(pollRef.current);

            const result = await fetchJSON(
              `${API_BASE}/optimize/${status.job_id}/result`,
            );
            setOptimizeResult(result);
            setOptimizing(false);

            // Sonuç layout'unu uygula
            setLayout(result.layout);
            setCgPosition([result.cg.cg_x, result.cg.cg_y, result.cg.cg_z]);
          } else if (s.status === "failed") {
            clearInterval(pollRef.current);
            setOptimizing(false);
            setError("Optimizasyon başarısız oldu");
          }
        } catch (e) {
          console.error("Polling hatası:", e);
        }
      }, 500);
    } catch (e) {
      setError(e.message);
      setOptimizing(false);
    }
  }, [selectedTypeId, gaConfig]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // CG durum sınıfı
  const getCGStatusClass = () => {
    if (!optimizeResult?.cg) return "";
    return optimizeResult.cg.in_target ? "good" : "danger";
  };

  // CG bar pozisyonu (yüzde)
  const getCGBarPosition = () => {
    if (!optimizeResult?.cg || !aircraftData?.cg_target) return 50;
    const { cg_mac_percent } = optimizeResult.cg;
    return Math.max(0, Math.min(100, cg_mac_percent));
  };

  const getCGTargetRange = () => {
    if (!aircraftData?.cg_target) return { left: 25, width: 10 };
    const { mac_percent_min, mac_percent_max } = aircraftData.cg_target;
    return { left: mac_percent_min, width: mac_percent_max - mac_percent_min };
  };

  // Jenerasyon historisi için chart data
  const getChartData = () => {
    if (!optimizeResult?.generation_history) return [];
    return optimizeResult.generation_history;
  };

  return (
    <div className="app-layout">
      {/* ===== Header ===== */}
      <header className="app-header">
        <div style={{ display: "flex", alignItems: "center" }}>
          <h1>🛩️ Uçak Yerleşim Optimizasyonu</h1>
          <span className="subtitle">
            TUSAŞ Destekli Tez Projesi — CG-Bilinçli AI Tasarım Aracı
          </span>
        </div>
        <div className="header-actions">
          {error && (
            <span style={{ color: "var(--accent-red)", fontSize: 12 }}>
              ⚠️ {error}
            </span>
          )}
        </div>
      </header>

      {/* ===== Ana İçerik ===== */}
      <div className="app-main">
        {/* === Sol Panel === */}
        <div className="left-panel">
          {/* Uçak Tipi Seçimi */}
          <div className="panel-section">
            <div className="panel-section-title">Uçak Tipi</div>
            <div className="aircraft-selector">
              {aircraftTypes.map((type) => (
                <div
                  key={type.id}
                  className={`aircraft-option ${selectedTypeId === type.id ? "selected" : ""}`}
                  onClick={() => setSelectedTypeId(type.id)}
                >
                  <span className="aircraft-option-icon">
                    {AIRCRAFT_ICONS[type.id] || "✈️"}
                  </span>
                  <div className="aircraft-option-info">
                    <h3>{type.name}</h3>
                    <p>{type.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Komponent Listesi */}
          {aircraftData && (
            <div className="panel-section">
              <div className="panel-section-title">
                Komponentler ({aircraftData.components.length})
              </div>
              <div className="component-list">
                {aircraftData.components.map((comp) => {
                  const cat = CATEGORY_LABELS[comp.category] || {
                    label: comp.category,
                    badge: "",
                  };
                  return (
                    <div
                      key={comp.id}
                      className={`component-item ${selectedComponent === comp.id ? "selected" : ""} ${comp.locked ? "locked" : ""}`}
                      onClick={() =>
                        setSelectedComponent(
                          comp.id === selectedComponent ? null : comp.id,
                        )
                      }
                    >
                      <span className="comp-name">
                        <span className={`comp-badge ${cat.badge}`}>
                          {cat.label}
                        </span>
                        {comp.name || comp.id}
                        {comp.locked && " 🔒"}
                      </span>
                      <span className="comp-weight">{comp.weight} kg</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* === 3D Görüntüleyici === */}
        <AircraftViewer3D
          aircraftData={aircraftData}
          layout={layout}
          cgPosition={cgPosition}
          selectedComponent={selectedComponent}
          onComponentClick={setSelectedComponent}
        />

        {/* === Sağ Panel === */}
        <div className="right-panel">
          {/* CG Göstergesi */}
          <div className="cg-indicator">
            <div className="panel-section-title">Ağırlık Merkezi (CG)</div>

            <div className="cg-bar-container">
              {aircraftData?.cg_target && (
                <div
                  className="cg-target-zone"
                  style={{
                    left: `${getCGTargetRange().left}%`,
                    width: `${getCGTargetRange().width}%`,
                  }}
                />
              )}
              {optimizeResult?.cg && (
                <div
                  className="cg-marker"
                  style={{ left: `${getCGBarPosition()}%` }}
                />
              )}
            </div>

            <div className="cg-stats">
              <div className="cg-stat">
                <div className="cg-stat-label">CG (MAC %)</div>
                <div className={`cg-stat-value ${getCGStatusClass()}`}>
                  {optimizeResult?.cg
                    ? `${optimizeResult.cg.cg_mac_percent.toFixed(1)}%`
                    : "—"}
                </div>
              </div>
              <div className="cg-stat">
                <div className="cg-stat-label">CG Hedef</div>
                <div
                  className="cg-stat-value"
                  style={{ color: "var(--accent-emerald)" }}
                >
                  {aircraftData?.cg_target
                    ? `${aircraftData.cg_target.mac_percent_min}-${aircraftData.cg_target.mac_percent_max}%`
                    : "—"}
                </div>
              </div>
              <div className="cg-stat">
                <div className="cg-stat-label">CG Kayması</div>
                <div
                  className={`cg-stat-value ${optimizeResult?.drift_mac_percent > 5 ? "danger" : optimizeResult?.drift_mac_percent > 2 ? "warning" : "good"}`}
                >
                  {optimizeResult
                    ? `${optimizeResult.drift_mac_percent.toFixed(2)}%`
                    : "—"}
                </div>
              </div>
              <div className="cg-stat">
                <div className="cg-stat-label">Toplam Kütle</div>
                <div className="cg-stat-value">
                  {optimizeResult?.cg
                    ? `${optimizeResult.cg.total_mass.toFixed(0)} kg`
                    : "—"}
                </div>
              </div>
            </div>
          </div>

          {/* Optimizasyon Kontrolleri */}
          <div className="optimize-controls">
            <div className="panel-section-title">Optimizasyon</div>

            <div className="config-grid">
              <div className="config-item">
                <label>Popülasyon</label>
                <input
                  type="number"
                  value={gaConfig.population_size}
                  onChange={(e) =>
                    setGaConfig((c) => ({
                      ...c,
                      population_size: parseInt(e.target.value) || 100,
                    }))
                  }
                  min={10}
                  max={500}
                />
              </div>
              <div className="config-item">
                <label>Nesil Sayısı</label>
                <input
                  type="number"
                  value={gaConfig.generations}
                  onChange={(e) =>
                    setGaConfig((c) => ({
                      ...c,
                      generations: parseInt(e.target.value) || 50,
                    }))
                  }
                  min={5}
                  max={200}
                />
              </div>
              <div className="config-item">
                <label>Mutasyon Oranı</label>
                <input
                  type="number"
                  value={gaConfig.mutation_rate}
                  onChange={(e) =>
                    setGaConfig((c) => ({
                      ...c,
                      mutation_rate: parseFloat(e.target.value) || 0.15,
                    }))
                  }
                  min={0}
                  max={1}
                  step={0.05}
                />
              </div>
              <div className="config-item">
                <label>Titreşim Limiti</label>
                <input
                  type="number"
                  value={gaConfig.vibration_limit}
                  onChange={(e) =>
                    setGaConfig((c) => ({
                      ...c,
                      vibration_limit: parseFloat(e.target.value) || 100,
                    }))
                  }
                  min={0}
                />
              </div>
            </div>

            <button
              className="btn btn-primary"
              onClick={startOptimization}
              disabled={!selectedTypeId || optimizing}
            >
              {optimizing
                ? "⏳ Optimizasyon Devam Ediyor..."
                : "🚀 Optimizasyonu Başlat"}
            </button>

            {/* İlerleme Çubuğu */}
            {optimizeStatus && optimizing && (
              <div className="progress-container">
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${optimizeStatus.progress}%` }}
                  />
                </div>
                <div className="progress-text">
                  <span>
                    Nesil {optimizeStatus.generation}/
                    {optimizeStatus.total_generations}
                  </span>
                  <span>{optimizeStatus.elapsed_seconds.toFixed(1)}s</span>
                </div>
              </div>
            )}
          </div>

          {/* Skor Özeti */}
          {optimizeResult && (
            <div className="panel-section">
              <div className="panel-section-title">Skor Dağılımı</div>
              <div className="cg-stats">
                <div className="cg-stat">
                  <div className="cg-stat-label">Toplam Skor</div>
                  <div className="cg-stat-value">
                    {optimizeResult.score.toFixed(0)}
                  </div>
                </div>
                <div className="cg-stat">
                  <div className="cg-stat-label">CG Skoru</div>
                  <div className="cg-stat-value">
                    {optimizeResult.sub_scores.cg_score.toFixed(0)}
                  </div>
                </div>
                <div className="cg-stat">
                  <div className="cg-stat-label">Kısıt Skoru</div>
                  <div
                    className={`cg-stat-value ${optimizeResult.sub_scores.constraint_score < -1000 ? "danger" : "good"}`}
                  >
                    {optimizeResult.sub_scores.constraint_score.toFixed(0)}
                  </div>
                </div>
                <div className="cg-stat">
                  <div className="cg-stat-label">Drift Skoru</div>
                  <div className="cg-stat-value">
                    {optimizeResult.sub_scores.drift_score.toFixed(0)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Jenerasyon Grafiği */}
          {optimizeResult?.generation_history && (
            <div className="history-chart">
              <div className="panel-section-title">Jenerasyon İlerlemesi</div>
              <div className="mini-chart">
                {(() => {
                  const data = getChartData();
                  const scores = data.map((d) => d.best_score);
                  const min = Math.min(...scores);
                  const max = Math.max(...scores);
                  const range = max - min || 1;

                  return data.map((d, i) => (
                    <div
                      key={i}
                      className="chart-bar"
                      style={{
                        height: `${Math.max(3, ((d.best_score - min) / range) * 100)}%`,
                      }}
                      title={`Nesil ${d.generation}: ${d.best_score.toFixed(0)}`}
                    />
                  ));
                })()}
              </div>
            </div>
          )}

          {/* Kısıt İhlalleri */}
          <div className="violations-panel">
            <div className="panel-section-title">
              Kısıt İhlalleri{" "}
              {optimizeResult ? `(${optimizeResult.violations.length})` : ""}
            </div>

            {optimizeResult?.violations.length === 0 && (
              <div className="no-violations">
                ✅ Hiçbir kısıt ihlali yok — Mükemmel tasarım!
              </div>
            )}

            {optimizeResult?.violations.map((v, i) => (
              <div
                key={i}
                className={`violation-item ${v.type === "SYMMETRY" || v.type === "PROXIMITY" ? "warning" : ""}`}
              >
                <span className="violation-icon">
                  {v.type === "COLLISION"
                    ? "💥"
                    : v.type === "BOUNDARY"
                      ? "🚧"
                      : v.type === "ZONE"
                        ? "📍"
                        : v.type === "VIBRATION"
                          ? "📳"
                          : v.type === "SYMMETRY"
                            ? "⚖️"
                            : v.type === "PROXIMITY"
                              ? "📏"
                              : "⚠️"}
                </span>
                <span>{v.description}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ===== Status Bar ===== */}
      <div className="status-bar">
        <div className="status-left">
          <span>
            <span className={`status-dot ${optimizing ? "warning" : ""}`} />
            {optimizing ? "Optimizasyon çalışıyor" : "Hazır"}
          </span>
          {selectedTypeId && (
            <span>Uçak: {aircraftData?.name || selectedTypeId}</span>
          )}
        </div>
        <div className="status-right">
          <span>v1.0.0</span>
          <span>TUSAŞ Tez Projesi</span>
        </div>
      </div>
    </div>
  );
}
