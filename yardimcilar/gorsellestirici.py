import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# BÖLGE RENKLERİ (RGBA)
# ---------------------------------------------------------------------------
BOLGE_RENKLERI = {
    "BURUN":  "rgba(255, 160,  80, 0.13)",
    "GOVDE":  "rgba( 80, 160, 255, 0.10)",
    "KUYRUK": "rgba(180,  80, 255, 0.13)",
    "TAVAN":  "rgba( 80, 220, 150, 0.10)",
    "TABAN":  "rgba(255,  80, 160, 0.10)",
}
BOLGE_SINIR_RENKLERI = {
    "BURUN":  "rgba(255, 140,  50, 0.35)",
    "GOVDE":  "rgba( 50, 130, 220, 0.25)",
    "KUYRUK": "rgba(160,  50, 230, 0.35)",
    "TAVAN":  "rgba( 50, 190, 120, 0.30)",
    "TABAN":  "rgba(220,  50, 140, 0.30)",
}


def _bolge_yuzey_olustur(bolge_adi, aircraft):
    """
    Her bölge için fuselage şekline uyan, o bölgenin X ve Z
    sınırlarına göre kesilmiş yarı saydam 3D yüzey döndürür.

    BURUN / GOVDE / KUYRUK → Silindirik segment (tam çevre)
    TAVAN                  → Üst yarı silindir (z ≥ 0)
    TABAN                  → Alt yarı silindir (z ≤ 0)
    """
    traces = []

    x_min, x_max = aircraft.get_bolge_x_siniri(bolge_adi)
    z_min, z_max = aircraft.get_bolge_z_siniri(bolge_adi)

    # X ekseni boyunca örnekleme
    n_x = 30
    xs = np.linspace(x_min, x_max, n_x)

    fill_color  = BOLGE_RENKLERI[bolge_adi]
    edge_color  = BOLGE_SINIR_RENKLERI[bolge_adi]

    if bolge_adi in ("BURUN", "GOVDE", "KUYRUK"):
        # TAM ÇEVRELİ segment
        n_u = 24
        us = np.linspace(0, 2 * np.pi, n_u)
        U, XV = np.meshgrid(us, xs)
        R = np.array([aircraft.get_fuselage_radius(x) for x in xs])
        R_mat = np.tile(R.reshape(-1, 1), (1, n_u))

        X_surf = XV
        Y_surf = R_mat * np.cos(U)
        Z_surf = R_mat * np.sin(U) * 1.2   # Gövde ile aynı eliptik oran

        traces.append(go.Surface(
            x=X_surf, y=Y_surf, z=Z_surf,
            colorscale=[[0, fill_color], [1, fill_color]],
            showscale=False,
            opacity=0.18,
            name=f"Bölge: {bolge_adi}",
            hovertemplate=f"<b>Bölge: {bolge_adi}</b><extra></extra>",
            surfacecolor=np.zeros_like(X_surf),
        ))

        # Ön ve arka kapak çemberleri (kenar çizgi)
        for x_cap in [x_min, x_max]:
            r_cap = aircraft.get_fuselage_radius(x_cap)
            theta = np.linspace(0, 2 * np.pi, 60)
            y_cap = r_cap * np.cos(theta)
            z_cap = r_cap * np.sin(theta) * 1.2
            traces.append(go.Scatter3d(
                x=np.full_like(theta, x_cap), y=y_cap, z=z_cap,
                mode='lines',
                line=dict(color=edge_color, width=2),
                showlegend=False, hoverinfo='skip'
            ))

    else:
        # YARI-SİLİNDİR (TAVAN z≥0 / TABAN z≤0)
        n_u = 24
        if bolge_adi == "TAVAN":
            us = np.linspace(0, np.pi, n_u)          # üst yarı
        else:
            us = np.linspace(np.pi, 2 * np.pi, n_u)  # alt yarı

        U, XV = np.meshgrid(us, xs)
        R = np.array([aircraft.get_fuselage_radius(x) for x in xs])
        R_mat = np.tile(R.reshape(-1, 1), (1, n_u))

        X_surf = XV
        Y_surf = R_mat * np.cos(U)
        Z_surf = R_mat * np.sin(U) * 1.2

        traces.append(go.Surface(
            x=X_surf, y=Y_surf, z=Z_surf,
            colorscale=[[0, fill_color], [1, fill_color]],
            showscale=False,
            opacity=0.20,
            name=f"Bölge: {bolge_adi}",
            hovertemplate=f"<b>Bölge: {bolge_adi}</b><extra></extra>",
            surfacecolor=np.zeros_like(X_surf),
        ))

        # Kenar (açık yay) çizgileri ön ve arka
        for x_cap in [x_min, x_max]:
            r_cap = aircraft.get_fuselage_radius(x_cap)
            theta = us  # Aynı yay
            y_cap = r_cap * np.cos(theta)
            z_cap = r_cap * np.sin(theta) * 1.2
            traces.append(go.Scatter3d(
                x=np.full_like(theta, x_cap), y=y_cap, z=z_cap,
                mode='lines',
                line=dict(color=edge_color, width=2),
                showlegend=False, hoverinfo='skip'
            ))

    # Bölge etiketi (ortada, gövde dışı)
    x_mid = (x_min + x_max) / 2
    r_mid = aircraft.get_fuselage_radius(x_mid)
    if bolge_adi == "TABAN":
        z_lbl = -r_mid * 1.35
    else:
        z_lbl = r_mid * 1.35
    traces.append(go.Scatter3d(
        x=[x_mid], y=[0], z=[z_lbl],
        mode='text',
        text=[bolge_adi],
        textfont=dict(size=11, color=edge_color, family="Arial Black"),
        showlegend=False, hoverinfo='skip'
    ))

    return traces


# ---------------------------------------------------------------------------
# UÇAK GÖVDESİ
# ---------------------------------------------------------------------------

def ucak_govdesi_olustur(aircraft):
    traces = []

    # 1. GÖVDE (Fuselage)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, aircraft.govde_uzunluk, 40)
    u, v = np.meshgrid(u, v)

    r_values = np.array([aircraft.get_fuselage_radius(x) for x in v.flatten()]).reshape(v.shape)

    x_govde = v
    y_govde = r_values * np.cos(u)
    z_govde = r_values * np.sin(u) * 1.2

    traces.append(go.Surface(
        x=x_govde, y=y_govde, z=z_govde,
        opacity=0.15, colorscale='Greys', showscale=False,
        name='Gövde', hoverinfo='skip'
    ))

    for i in range(0, 40, 4):
        traces.append(go.Scatter3d(
            x=x_govde[i], y=y_govde[i], z=z_govde[i],
            mode='lines', line=dict(color='black', width=1),
            showlegend=False, hoverinfo='skip'
        ))

    # 2. KANATLAR
    kanat_x_bas = 80
    kanat_genislik = 40
    kanat_uzunluk = 360
    z_kanat = aircraft.govde_yaricap * 1.1

    x_w = [kanat_x_bas, kanat_x_bas+kanat_genislik, kanat_x_bas+kanat_genislik, kanat_x_bas]
    y_w = [-kanat_uzunluk/2, -kanat_uzunluk/2, kanat_uzunluk/2, kanat_uzunluk/2]
    z_w = [z_kanat, z_kanat, z_kanat, z_kanat]

    traces.append(go.Mesh3d(
        x=x_w, y=y_w, z=z_w,
        color='lightblue', opacity=0.5, name='Kanat',
        i=[0, 0], j=[1, 2], k=[2, 3]
    ))

    # 3. KUYRUK TAKIMI
    tail_x = aircraft.govde_uzunluk - 40
    h_stab_span = 120
    x_h = [tail_x, aircraft.govde_uzunluk, aircraft.govde_uzunluk, tail_x]
    y_h = [-h_stab_span/2, -h_stab_span/2, h_stab_span/2, h_stab_span/2]
    z_h = [0, 0, 0, 0]

    traces.append(go.Mesh3d(
        x=x_h, y=y_h, z=z_h, color='lightblue', opacity=0.5, name='Yatay Kuyruk',
        i=[0, 0], j=[1, 2], k=[2, 3]
    ))

    x_v = [tail_x, aircraft.govde_uzunluk, aircraft.govde_uzunluk, tail_x+10]
    y_v = [0, 0, 0, 0]
    z_v = [0, 0, 50, 50]

    traces.append(go.Mesh3d(
        x=x_v, y=y_v, z=z_v, color='lightblue', opacity=0.5, name='Dikey Kuyruk',
        i=[0, 0], j=[1, 2], k=[2, 3]
    ))

    return traces


# ---------------------------------------------------------------------------
# PARÇA KUTUSU
# ---------------------------------------------------------------------------

def parca_kutusu_ciz(pos, dim, color, name):
    x, y, z = pos
    dx, dy, dz = dim

    x_k = [x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2]
    y_k = [y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2]
    z_k = [z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2]

    return go.Mesh3d(
        x=x_k, y=y_k, z=z_k,
        color=color, opacity=1.0, name=name,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        hoverinfo='name'
    )


# ---------------------------------------------------------------------------
# ANA GÖRSELLEŞTİRME
# ---------------------------------------------------------------------------

def gorsellestir_tasarim(en_iyi_tasarim, best_score, best_cg, aircraft, ALGORITMA):
    fig = go.Figure()

    # 1. BÖLGE YÜZEYLERINI ÇİZ (Arka plana)
    for bolge in ["BURUN", "GOVDE", "KUYRUK", "TAVAN", "TABAN"]:
        for trace in _bolge_yuzey_olustur(bolge, aircraft):
            fig.add_trace(trace)

    # 2. UÇAK GÖVDESİNİ ÇİZ
    for parca in ucak_govdesi_olustur(aircraft):
        fig.add_trace(parca)

    # 3. KOMPONENTLERİ ÇİZ
    colors = ['red', 'blue', 'orange', 'purple', 'green', 'brown', 'cyan']
    for k_id, pos in en_iyi_tasarim.yerlesim.items():
        boyut = next(item for item in aircraft.komponentler_db if item.id == k_id).boyut
        idx   = aircraft.komponentler_db.index(next(item for item in aircraft.komponentler_db if item.id == k_id))

        fig.add_trace(parca_kutusu_ciz(pos, boyut, colors[idx % len(colors)], k_id))

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2] + boyut[2]/1.5],
            mode='text', text=[k_id], textposition="top center",
            textfont=dict(size=10, color="black"), showlegend=False
        ))

    # 4. HEDEF CG ARALIĞI
    box_r = aircraft.govde_yaricap + 5
    fig.add_trace(go.Mesh3d(
        x=[aircraft.target_cg_x_min, aircraft.target_cg_x_max,
           aircraft.target_cg_x_max, aircraft.target_cg_x_min,
           aircraft.target_cg_x_min, aircraft.target_cg_x_max,
           aircraft.target_cg_x_max, aircraft.target_cg_x_min],
        y=[-box_r, -box_r, box_r, box_r, -box_r, -box_r, box_r, box_r],
        z=[-box_r, -box_r, -box_r, -box_r, box_r, box_r, box_r, box_r],
        color='gold', opacity=0.3, name='HEDEF CG ARALIĞI',
        alphahull=0
    ))

    # 5. HESAPLANAN CG
    viz_z = aircraft.govde_yaricap + 40

    fig.add_trace(go.Scatter3d(
        x=[best_cg[0]], y=[best_cg[1]], z=[viz_z],
        mode='markers+text', marker=dict(size=12, color='black', symbol='diamond'),
        name='HESAPLANAN CG', text=["HESAPLANAN CG"], textposition="top center"
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

    target_x_visual = best_cg[0]
    if best_cg[0] < aircraft.target_cg_x_min:
        target_x_visual = aircraft.target_cg_x_min
    elif best_cg[0] > aircraft.target_cg_x_max:
        target_x_visual = aircraft.target_cg_x_max

    fig.add_trace(go.Scatter3d(
        x=[target_x_visual, best_cg[0]],
        y=[aircraft.target_cg_y, best_cg[1]],
        z=[aircraft.target_cg_z, best_cg[2]],
        mode='lines', line=dict(color='red', width=4, dash='dot'), name='CG Hatası'
    ))

    # --- LAYOUT ---
    camera = dict(eye=dict(x=2.0, y=-2.0, z=1.0))

    fig.update_layout(
        title=f"Ön Tasarım: Uçak İçi Sistem Yerleşimi Optimizasyonu ({ALGORITMA})",
        scene=dict(
            xaxis=dict(title='Uzunluk (cm)', range=[0, aircraft.govde_uzunluk],
                       backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title='Genişlik (cm)', range=[-200, 200]),
            zaxis=dict(title='Yükseklik (cm)', range=[-100, 100]),
            aspectmode='data',
            camera=camera
        ),
        margin=dict(r=0, l=0, b=0, t=50)
    )

    fig.show()
