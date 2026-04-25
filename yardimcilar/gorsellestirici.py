import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# BÖLGE RENKLERİ (RGBA) – eski tanımlar korundu
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

# Komponent renkleri
KOMPONENT_RENKLERI = {
    "Motor":        "red",
    "Batarya_Ana":  "royalblue",
    "Aviyonik_1":   "green",
    "Aviyonik_2":   "darkorange",
    "Payload_Kam":  "deeppink",
    "Yakit_Tanki":  "sienna",
    "Servo_Kuyruk": "purple",
}
_FALLBACK_COLORS = ["red", "blue", "green", "orange", "pink", "brown", "purple", "cyan"]


# ---------------------------------------------------------------------------
# YARDIMCI: Silindir Yüzeyi (go.Surface)
# ---------------------------------------------------------------------------

def _silindir_traces(cx, cy, cz, r, uzunluk, yon='x', n_u=30,
                     color='gray', opacity=0.90, name='', cap=True):
    """
    Silindir yan yüzeyi + opsiyonel kapaklar.
    """
    traces = []
    theta = np.linspace(0, 2 * np.pi, n_u)
    half = uzunluk / 2
    ts = np.array([-half, half])
    TH, TV = np.meshgrid(theta, ts)

    if yon == 'x':
        X = cx + TV
        Y = cy + r * np.cos(TH)
        Z = cz + r * np.sin(TH)
    elif yon == 'y':
        X = cx + r * np.cos(TH)
        Y = cy + TV
        Z = cz + r * np.sin(TH)
    else:  # 'z'
        X = cx + r * np.cos(TH)
        Y = cy + r * np.sin(TH)
        Z = cz + TV

    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, color], [1, color]],
        showscale=False, opacity=opacity, name=name,
        hovertemplate=f"<b>{name}</b><extra></extra>" if name else None,
        surfacecolor=np.zeros_like(X),
    ))

    if cap:
        for sign in [-1, 1]:
            if yon == 'x':
                px = [cx + sign * half] + [cx + sign * half] * n_u
                py = [cy] + list(cy + r * np.cos(theta))
                pz = [cz] + list(cz + r * np.sin(theta))
            elif yon == 'y':
                px = [cx] + list(cx + r * np.cos(theta))
                py = [cy + sign * half] + [cy + sign * half] * n_u
                pz = [cz] + list(cz + r * np.sin(theta))
            else:
                px = [cx] + list(cx + r * np.cos(theta))
                py = [cy] + list(cy + r * np.sin(theta))
                pz = [cz + sign * half] + [cz + sign * half] * n_u

            i_idx = [0] * n_u
            j_idx = list(range(1, n_u + 1))
            k_idx = list(range(2, n_u + 1)) + [1]

            traces.append(go.Mesh3d(
                x=px, y=py, z=pz,
                i=i_idx, j=j_idx, k=k_idx,
                color=color, opacity=opacity,
                showlegend=False, hoverinfo='skip',
            ))

    return traces


def _konik_traces(cx_start, cy, cz, r_start, r_end, uzunluk, yon='x',
                  n_u=28, color='gray', opacity=0.90, name=''):
    """r_start'dan r_end'e daralan konik yüzey."""
    traces = []
    theta = np.linspace(0, 2 * np.pi, n_u)
    xs = np.linspace(0, uzunluk, 20)
    rs = np.linspace(r_start, r_end, 20)
    TH, XV = np.meshgrid(theta, xs)
    R_mat = np.tile(rs.reshape(-1, 1), (1, n_u))

    if yon == 'x':
        X = cx_start + XV
        Y = cy + R_mat * np.cos(TH)
        Z = cz + R_mat * np.sin(TH)
    elif yon == 'y':
        Y = cy + XV - uzunluk / 2
        X = cx_start + R_mat * np.cos(TH)
        Z = cz + R_mat * np.sin(TH)
    else:
        Z = cz + XV - uzunluk / 2
        X = cx_start + R_mat * np.cos(TH)
        Y = cy + R_mat * np.sin(TH)

    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, color], [1, color]],
        showscale=False, opacity=opacity, name=name,
        surfacecolor=np.zeros_like(X),
        hovertemplate=f"<b>{name}</b><extra></extra>" if name else None,
    ))
    return traces


def _kutu_mesh(cx, cy, cz, dx, dy, dz, color='blue', opacity=0.90, name=''):
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    xs = [cx - hx, cx - hx, cx + hx, cx + hx, cx - hx, cx - hx, cx + hx, cx + hx]
    ys = [cy - hy, cy + hy, cy + hy, cy - hy, cy - hy, cy + hy, cy + hy, cy - hy]
    zs = [cz - hz, cz - hz, cz - hz, cz - hz, cz + hz, cz + hz, cz + hz, cz + hz]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, color=color, opacity=opacity, name=name,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        hovertemplate=f"<b>{name}</b><extra></extra>" if name else None,
    )


# ---------------------------------------------------------------------------
# MOTOR — Türbofan Jet Motor
# (X ekseni boylamasına, cx merkez, gövde ön-arka)
# ---------------------------------------------------------------------------

def _motor_ciz(cx, cy, cz, dx, dy, dz, color, name):
    """
    Gerçekçi türbofan/jet motor geometrisi:
      - Fan girişi (geniş silindir)
      - İç çekirdek gövdesi (daha dar silindir)
      - Kompresör kasması (orta gövde)
      - Yanma odası (şişkin orta bölüm)
      - Egzoz nozulu (daralan konik)
      - Fan kanatçıkları (radyal levhalar)
      - Orta nacelle spinner (konik burun)
    """
    traces = []
    r_fan = min(dy, dz) / 2 * 0.88   # fan dış yarıçapı
    r_core = r_fan * 0.45              # çekirdek yarıçapı
    r_inner = r_core * 0.35            # iç egzoz çekirdeği

    # Bölüm uzunlukları
    inlet_len    = dx * 0.12
    core_len     = dx * 0.55
    combust_len  = dx * 0.15
    nozzle_len   = dx * 0.18

    x0 = cx - dx / 2   # motor burnu

    # --- 1. Fan Girişi (nacelle) ---
    fan_cx = x0 + inlet_len / 2
    traces += _silindir_traces(fan_cx, cy, cz, r_fan, inlet_len,
                               yon='x', n_u=36, color='#708090', opacity=0.75,
                               name=name, cap=True)

    # Fan iç halkası (giriş diski koyulaştırıcı - görsel)
    traces += _silindir_traces(fan_cx - inlet_len * 0.48, cy, cz,
                               r_fan * 0.50, 0.5, yon='x', n_u=28,
                               color='#222222', opacity=0.85, name='', cap=True)

    # --- Fan Kanatçıkları ---
    n_fan_blades = 12
    blade_len = inlet_len * 0.85
    blade_chord = r_fan - r_core * 1.05
    blade_cx = x0 + inlet_len / 2
    for i in range(n_fan_blades):
        ang = 2 * np.pi * i / n_fan_blades
        # Her kanattaki dış sınır noktaları (sweep açılı)
        r_in = r_core * 1.05
        r_out = r_fan * 0.95
        # Kanat 4 köşe
        sweep = 0.15 * blade_len   # geri-sweep
        pts_x = [blade_cx - blade_len / 2,
                 blade_cx + blade_len / 2 - sweep,
                 blade_cx + blade_len / 2,
                 blade_cx - blade_len / 2]
        pts_y_in  = cy + r_in * np.cos(ang)
        pts_z_in  = cz + r_in * np.sin(ang)
        pts_y_out = cy + r_out * np.cos(ang)
        pts_z_out = cz + r_out * np.sin(ang)
        traces.append(go.Mesh3d(
            x=[pts_x[0], pts_x[1], pts_x[2], pts_x[3]],
            y=[pts_y_in, pts_y_in, pts_y_out, pts_y_out],
            z=[pts_z_in, pts_z_in, pts_z_out, pts_z_out],
            i=[0], j=[1], k=[2],
            color='#B0C4DE', opacity=0.80,
            showlegend=False, hoverinfo='skip'
        ))
        traces.append(go.Mesh3d(
            x=[pts_x[0], pts_x[2], pts_x[3]],
            y=[pts_y_in, pts_y_out, pts_y_out],
            z=[pts_z_in, pts_z_out, pts_z_out],
            i=[0], j=[1], k=[2],
            color='#B0C4DE', opacity=0.80,
            showlegend=False, hoverinfo='skip'
        ))

    # --- Spinner (Konik burun) ---
    spinner_len = inlet_len * 0.55
    traces += _konik_traces(x0, cy, cz, 0.5, r_core * 0.95, spinner_len,
                            yon='x', n_u=24, color='#CCCCCC', opacity=0.90, name='')

    # --- 2. Kompresör/Çekirdek Gövdesi ---
    core_cx = x0 + inlet_len + core_len / 2
    traces += _silindir_traces(core_cx, cy, cz, r_core, core_len,
                               yon='x', n_u=32, color=color, opacity=0.88,
                               name='', cap=False)

    # Kompresör aşama halkaları (görsel)
    n_stages = 6
    for si in range(n_stages):
        ring_x = x0 + inlet_len + core_len * (si + 0.5) / n_stages
        ring_r = r_core * (1.0 + 0.06 * np.sin(si * np.pi / n_stages))
        theta_r = np.linspace(0, 2 * np.pi, 30)
        traces.append(go.Scatter3d(
            x=np.full(30, ring_x),
            y=cy + ring_r * np.cos(theta_r),
            z=cz + ring_r * np.sin(theta_r),
            mode='lines', line=dict(color='#555555', width=2),
            showlegend=False, hoverinfo='skip'
        ))

    # --- 3. Yanma Odası (şişkin bölge) ---
    combust_x = x0 + inlet_len + core_len
    combust_r = r_core * 1.18
    combust_cx = combust_x + combust_len / 2
    traces += _silindir_traces(combust_cx, cy, cz, combust_r, combust_len,
                               yon='x', n_u=28, color='#8B0000', opacity=0.82,
                               name='', cap=False)

    # Yakıt enjektörleri (küçük silindir bumps etrafında)
    n_inj = 8
    inj_ang_list = np.linspace(0, 2 * np.pi, n_inj, endpoint=False)
    for ang in inj_ang_list:
        iy = cy + combust_r * np.cos(ang)
        iz = cz + combust_r * np.sin(ang)
        traces += _silindir_traces(combust_cx, iy, iz, combust_r * 0.12, combust_len * 0.4,
                                   yon='x', n_u=8, color='#444444', opacity=0.90, name='', cap=True)

    # --- 4. Egzoz Nozulu (daralan konik) ---
    nozzle_x_start = x0 + inlet_len + core_len + combust_len
    traces += _konik_traces(nozzle_x_start, cy, cz, combust_r, r_inner,
                            nozzle_len, yon='x', n_u=28, color='#444444', opacity=0.85, name='')

    # İç egzoz koni (sıcak akış)
    traces += _konik_traces(nozzle_x_start + nozzle_len * 0.2, cy, cz,
                            r_inner * 0.85, 0.2, nozzle_len * 0.65,
                            yon='x', n_u=20, color='#FF4500', opacity=0.50, name='')

    # Egzoz alevleri (şeffaf kırmızı/sarı çizgiler)
    nozzle_end = nozzle_x_start + nozzle_len
    n_jets = 7
    for ji in range(n_jets):
        frac = ji / (n_jets - 1)
        jy = cy + r_inner * 0.65 * np.cos(2 * np.pi * frac)
        jz = cz + r_inner * 0.65 * np.sin(2 * np.pi * frac)
        jet_len = dx * 0.10 * (0.6 + 0.4 * np.random.default_rng(ji).random())
        traces.append(go.Scatter3d(
            x=[nozzle_end, nozzle_end + jet_len],
            y=[jy, jy * 0.7], z=[jz, jz * 0.7],
            mode='lines', line=dict(color='rgba(255,180,0,0.45)', width=3),
            showlegend=False, hoverinfo='skip'
        ))

    # --- Dış Fan Kılıfı (Bypass duct) ---
    bypass_cx = x0 + inlet_len + core_len * 0.35
    bypass_len = core_len * 0.70
    traces += _silindir_traces(bypass_cx, cy, cz, r_fan * 0.97, bypass_len,
                               yon='x', n_u=32, color='#778899', opacity=0.25,
                               name='', cap=False)

    return traces


# ---------------------------------------------------------------------------
# BATARYA — Lityum Paket
# ---------------------------------------------------------------------------

def _batarya_ciz(cx, cy, cz, dx, dy, dz, color, name):
    traces = []
    traces.append(_kutu_mesh(cx, cy, cz, dx, dy, dz, color=color, opacity=0.88, name=name))

    # Hücre silindirleri
    n_hucre = 3
    h_r = min(dx, dy) / (n_hucre * 2.8)
    h_height = dz * 0.18
    xs_offsets = np.linspace(-dx * 0.28, dx * 0.28, n_hucre)
    for x_off in xs_offsets:
        traces += _silindir_traces(cx + x_off, cy, cz + dz / 2 + h_height / 2,
                                   h_r, h_height, yon='z', n_u=20,
                                   color='#AAAAAA', opacity=0.80, name='', cap=True)

    # Terminaller
    for sign in [-1, 1]:
        traces.append(_kutu_mesh(cx + sign * (dx / 2 + dx * 0.04), cy, cz,
                                  dx * 0.08, dy * 0.20, dz * 0.12,
                                  color='#DDDDDD', opacity=0.95, name=''))
    return traces


# ---------------------------------------------------------------------------
# AVİYONİK 1 — AHRS (Attitude & Heading Reference System)
# Kompakt kutu, üstünde IMU port kapağı, ön yüzde konnektör satırı
# ---------------------------------------------------------------------------

def _aviyonik_ahrs_ciz(cx, cy, cz, dx, dy, dz, color, name):
    traces = []

    # Ana gövde — ince alüminyum kutu
    traces.append(_kutu_mesh(cx, cy, cz, dx, dy, dz, color=color, opacity=0.88, name=name))

    # Üst IMU kapağı (biraz dışarı çıkıntılı, merkezi yuvarlak)
    cap_r = min(dx, dy) * 0.18
    cap_h = dz * 0.12
    traces += _silindir_traces(cx, cy, cz + dz / 2 + cap_h / 2, cap_r, cap_h,
                               yon='z', n_u=24, color='#888888', opacity=0.90, name='', cap=True)
    # IMU damper çemberi
    theta_d = np.linspace(0, 2 * np.pi, 30)
    traces.append(go.Scatter3d(
        x=cx + cap_r * np.cos(theta_d),
        y=cy + cap_r * np.sin(theta_d),
        z=np.full(30, cz + dz / 2),
        mode='lines', line=dict(color='black', width=2),
        showlegend=False, hoverinfo='skip'
    ))

    # Ön yüz — DB9 benzeri konnektör
    n_pins = 9
    pin_r = dz * 0.04
    pin_xs = np.linspace(cx - dx * 0.32, cx + dx * 0.32, n_pins)
    for px in pin_xs:
        traces.append(go.Scatter3d(
            x=[px, px], y=[cy - dy / 2 - 0.5, cy - dy / 2 - dz * 0.10],
            z=[cz, cz],
            mode='lines', line=dict(color='#FFDD00', width=3),
            showlegend=False, hoverinfo='skip'
        ))

    # Yan yüz — kalibrasyon delikleri
    for zi_off in [-dz * 0.25, dz * 0.25]:
        traces.append(go.Scatter3d(
            x=[cx + dx / 2], y=[cy], z=[cz + zi_off],
            mode='markers', marker=dict(size=3, color='black', symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))

    return traces


# ---------------------------------------------------------------------------
# AVİYONİK 2 — Uçuş Bilgisayarı (Flight Computer)
# 19" rack modülü tarzı: uzun dikdörtgen + ön panel + LED sıra + fan agregate
# ---------------------------------------------------------------------------

def _aviyonik_fcu_ciz(cx, cy, cz, dx, dy, dz, color, name):
    traces = []

    # Ana rack kasası
    traces.append(_kutu_mesh(cx, cy, cz, dx, dy, dz, color=color, opacity=0.88, name=name))

    # Ön panel (farklı renk)
    panel_depth = dy * 0.06
    traces.append(_kutu_mesh(cx, cy - dy / 2 + panel_depth / 2, cz,
                              dx, panel_depth, dz,
                              color='#1A2230', opacity=0.92, name=''))

    # LED indikatörleri (yeşil küçük noktalar)
    n_leds = 5
    led_xs = np.linspace(cx - dx * 0.35, cx + dx * 0.35, n_leds)
    for lx in led_xs:
        traces.append(go.Scatter3d(
            x=[lx], y=[cy - dy / 2], z=[cz + dz * 0.35],
            mode='markers', marker=dict(size=5, color='lime', symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))

    # Soğutma Fanı grili (arka, kare ızgara çizgi)
    ux = np.linspace(cx - dx * 0.25, cx + dx * 0.25, 5)
    for lx in ux:
        traces.append(go.Scatter3d(
            x=[lx, lx], y=[cy + dy / 2, cy + dy / 2],
            z=[cz - dz * 0.35, cz + dz * 0.35],
            mode='lines', line=dict(color='#333333', width=1),
            showlegend=False, hoverinfo='skip'
        ))
    uz = np.linspace(cz - dz * 0.35, cz + dz * 0.35, 5)
    for lz in uz:
        traces.append(go.Scatter3d(
            x=[cx - dx * 0.25, cx + dx * 0.25],
            y=[cy + dy / 2, cy + dy / 2],
            z=[lz, lz],
            mode='lines', line=dict(color='#333333', width=1),
            showlegend=False, hoverinfo='skip'
        ))
    # Fan çemberi
    theta_f = np.linspace(0, 2 * np.pi, 32)
    fan_r = min(dx, dz) * 0.22
    traces.append(go.Scatter3d(
        x=cx + fan_r * np.cos(theta_f),
        y=np.full(32, cy + dy / 2),
        z=cz + fan_r * np.sin(theta_f),
        mode='lines', line=dict(color='#444444', width=2),
        showlegend=False, hoverinfo='skip'
    ))

    # Ethernet portları (üstte 2 adet dikdörtgen çıkıntı)
    for sign in [-1, 1]:
        traces.append(_kutu_mesh(cx + sign * dx * 0.20, cy - dy / 2 - dx * 0.04, cz - dz * 0.28,
                                  dx * 0.12, dx * 0.08, dz * 0.14,
                                  color='#888888', opacity=0.95, name=''))

    # PCIe kart çıkıntısı (üstte ince levha)
    traces.append(_kutu_mesh(cx + dx * 0.30, cy, cz + dz / 2 + dz * 0.06,
                              dx * 0.30, dy * 0.80, dz * 0.12,
                              color='#2E5E2E', opacity=0.85, name=''))  # yeşil PCB

    return traces


# ---------------------------------------------------------------------------
# PAYLOAD KAMERA
# ---------------------------------------------------------------------------

def _kamera_ciz(cx, cy, cz, dx, dy, dz, color, name):
    traces = []
    traces.append(_kutu_mesh(cx, cy, cz, dx, dy, dz * 0.70, color=color, opacity=0.88, name=name))

    dome_r = min(dy, dx) * 0.38
    n_u, n_v = 22, 14
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, -np.pi / 2, n_v)
    U, V = np.meshgrid(u, v)
    Xd = cx + dome_r * np.cos(V) * np.cos(U)
    Yd = cy + dome_r * np.cos(V) * np.sin(U)
    Zd = cz - dz * 0.35 + dome_r * np.sin(V)

    traces.append(go.Surface(
        x=Xd, y=Yd, z=Zd,
        colorscale=[[0, 'navy'], [1, 'cornflowerblue']],
        showscale=False, opacity=0.85, name='',
        surfacecolor=np.ones_like(Xd) * 0.5, hoverinfo='skip',
    ))

    theta_ring = np.linspace(0, 2 * np.pi, 40)
    traces.append(go.Scatter3d(
        x=cx + dome_r * 0.55 * np.cos(theta_ring),
        y=cy + dome_r * 0.55 * np.sin(theta_ring),
        z=np.full(40, cz - dz * 0.35 - dome_r * 0.78),
        mode='lines', line=dict(color='rgba(180,220,255,0.6)', width=2),
        showlegend=False, hoverinfo='skip'
    ))
    return traces


# ---------------------------------------------------------------------------
# YAKIT TANKI — Kapsül
# ---------------------------------------------------------------------------

def _yakit_tanki_ciz(cx, cy, cz, dx, dy, dz, color, name):
    traces = []
    r = min(dy, dz) / 2 * 0.82
    cyl_len = dx * 0.60
    dome_r = r

    traces += _silindir_traces(cx, cy, cz, r, cyl_len, yon='x',
                               n_u=36, color=color, opacity=0.82, name=name, cap=False)

    for sign_dir in [-1, 1]:
        n_u2, n_v2 = 24, 12
        u2 = np.linspace(0, 2 * np.pi, n_u2)
        v2 = np.linspace(0, np.pi / 2, n_v2)
        U2, V2 = np.meshgrid(u2, v2)
        Xd = cx + sign_dir * (cyl_len / 2 + dome_r * np.sin(V2))
        Yd = cy + dome_r * np.cos(V2) * np.cos(U2)
        Zd = cz + dome_r * np.cos(V2) * np.sin(U2)
        traces.append(go.Surface(
            x=Xd, y=Yd, z=Zd,
            colorscale=[[0, color], [1, color]],
            showscale=False, opacity=0.82, name='',
            surfacecolor=np.zeros_like(Xd), hoverinfo='skip',
        ))

    fill_ratio = 0.65
    fill_z = cz - r + 2 * r * fill_ratio
    traces.append(go.Scatter3d(
        x=[cx - cyl_len / 2, cx + cyl_len / 2],
        y=[cy, cy], z=[fill_z, fill_z],
        mode='lines', line=dict(color='rgba(0,180,255,0.7)', width=3),
        showlegend=False, hoverinfo='skip'
    ))
    return traces


# ---------------------------------------------------------------------------
# SERVO
# ---------------------------------------------------------------------------

def _servo_ciz(cx, cy, cz, dx, dy, dz, color, name):
    traces = []
    traces.append(_kutu_mesh(cx, cy, cz, dx, dy, dz, color=color, opacity=0.92, name=name))
    traces.append(_kutu_mesh(cx, cy, cz + dz * 0.52, dx * 0.70, dy * 0.70, dz * 0.08,
                              color='#CCCCCC', opacity=0.90, name=''))

    arm_len = dx * 1.1
    arm_w = dx * 0.12
    traces.append(_kutu_mesh(cx + arm_len / 2, cy, cz + dz * 0.56 + arm_w / 2,
                              arm_len, arm_w, arm_w, color='#888888', opacity=0.95, name=''))
    traces.append(go.Scatter3d(
        x=[cx, cx], y=[cy, cy], z=[cz + dz * 0.42, cz + dz * 0.65],
        mode='lines', line=dict(color='white', width=4),
        showlegend=False, hoverinfo='skip'
    ))
    traces.append(go.Scatter3d(
        x=[cx + arm_len], y=[cy], z=[cz + dz * 0.56 + arm_w / 2],
        mode='markers', marker=dict(size=5, color='white'),
        showlegend=False, hoverinfo='skip'
    ))
    return traces


# ---------------------------------------------------------------------------
# KOMPONENT YÖNLENDİRİCİ
# ---------------------------------------------------------------------------

_SHAPE_MAP = {
    "Motor":        _motor_ciz,
    "Batarya_Ana":  _batarya_ciz,
    "Aviyonik_1":   _aviyonik_ahrs_ciz,   # AHRS
    "Aviyonik_2":   _aviyonik_fcu_ciz,    # Flight Computer
    "Payload_Kam":  _kamera_ciz,
    "Yakit_Tanki":  _yakit_tanki_ciz,
    "Servo_Kuyruk": _servo_ciz,
}


def komponent_ciz(k_id, pos, boyut, color):
    cx, cy, cz = pos
    dx, dy, dz = boyut
    func = _SHAPE_MAP.get(k_id)
    if func:
        return func(cx, cy, cz, dx, dy, dz, color, k_id)
    return [_kutu_mesh(cx, cy, cz, dx, dy, dz, color=color, opacity=0.90, name=k_id)]


# ---------------------------------------------------------------------------
# BÖLGE YÜZEYLERİ  (değişmedi)
# ---------------------------------------------------------------------------

def _bolge_yuzey_olustur(bolge_adi, aircraft):
    traces = []
    x_min, x_max = aircraft.get_bolge_x_siniri(bolge_adi)
    z_min, z_max = aircraft.get_bolge_z_siniri(bolge_adi)

    n_x = 30
    xs = np.linspace(x_min, x_max, n_x)
    fill_color = BOLGE_RENKLERI[bolge_adi]
    edge_color = BOLGE_SINIR_RENKLERI[bolge_adi]

    if bolge_adi in ("BURUN", "GOVDE", "KUYRUK"):
        n_u = 24
        us = np.linspace(0, 2 * np.pi, n_u)
        U, XV = np.meshgrid(us, xs)
        R = np.array([aircraft.get_fuselage_radius(x) for x in xs])
        R_mat = np.tile(R.reshape(-1, 1), (1, n_u))
        X_surf = XV
        Y_surf = R_mat * np.cos(U)
        Z_surf = R_mat * np.sin(U) * 1.2
        traces.append(go.Surface(
            x=X_surf, y=Y_surf, z=Z_surf,
            colorscale=[[0, fill_color], [1, fill_color]],
            showscale=False, opacity=0.18, name=f"Bölge: {bolge_adi}",
            hovertemplate=f"<b>Bölge: {bolge_adi}</b><extra></extra>",
            surfacecolor=np.zeros_like(X_surf),
        ))
        for x_cap in [x_min, x_max]:
            r_cap = aircraft.get_fuselage_radius(x_cap)
            theta = np.linspace(0, 2 * np.pi, 60)
            traces.append(go.Scatter3d(
                x=np.full_like(theta, x_cap),
                y=r_cap * np.cos(theta),
                z=r_cap * np.sin(theta) * 1.2,
                mode='lines', line=dict(color=edge_color, width=2),
                showlegend=False, hoverinfo='skip'
            ))
    else:
        n_u = 24
        us = np.linspace(0, np.pi, n_u) if bolge_adi == "TAVAN" \
            else np.linspace(np.pi, 2 * np.pi, n_u)
        U, XV = np.meshgrid(us, xs)
        R = np.array([aircraft.get_fuselage_radius(x) for x in xs])
        R_mat = np.tile(R.reshape(-1, 1), (1, n_u))
        X_surf = XV
        Y_surf = R_mat * np.cos(U)
        Z_surf = R_mat * np.sin(U) * 1.2
        traces.append(go.Surface(
            x=X_surf, y=Y_surf, z=Z_surf,
            colorscale=[[0, fill_color], [1, fill_color]],
            showscale=False, opacity=0.20, name=f"Bölge: {bolge_adi}",
            hovertemplate=f"<b>Bölge: {bolge_adi}</b><extra></extra>",
            surfacecolor=np.zeros_like(X_surf),
        ))
        for x_cap in [x_min, x_max]:
            r_cap = aircraft.get_fuselage_radius(x_cap)
            traces.append(go.Scatter3d(
                x=np.full_like(us, x_cap),
                y=r_cap * np.cos(us),
                z=r_cap * np.sin(us) * 1.2,
                mode='lines', line=dict(color=edge_color, width=2),
                showlegend=False, hoverinfo='skip'
            ))

    x_mid = (x_min + x_max) / 2
    r_mid = aircraft.get_fuselage_radius(x_mid)
    z_lbl = -r_mid * 1.35 if bolge_adi == "TABAN" else r_mid * 1.35
    traces.append(go.Scatter3d(
        x=[x_mid], y=[0], z=[z_lbl],
        mode='text', text=[bolge_adi],
        textfont=dict(size=11, color=edge_color, family="Arial Black"),
        showlegend=False, hoverinfo='skip'
    ))

    return traces


# ---------------------------------------------------------------------------
# UÇAK GÖVDESİ  (değişmedi)
# ---------------------------------------------------------------------------

def ucak_govdesi_olustur(aircraft):
    traces = []

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, aircraft.govde_uzunluk, 40)
    u, v = np.meshgrid(u, v)
    r_values = np.array([aircraft.get_fuselage_radius(x) for x in v.flatten()]).reshape(v.shape)
    x_govde = v
    y_govde = r_values * np.cos(u)
    z_govde = r_values * np.sin(u) * 1.2
    
    traces.append(go.Surface(
        x=x_govde, y=y_govde, z=z_govde, opacity=0.15, colorscale='Greys', showscale=False, name='Gövde', hoverinfo='skip'
    ))
    for i in range(0, 40, 4): 
        traces.append(go.Scatter3d(x=x_govde[i], y=y_govde[i], z=z_govde[i], mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))

    # 2. KANATLAR (High Wing - Üstten Kanat)
    kanat_x_bas = 80
    kanat_genislik = 50 
    kanat_uzunluk = 360 
    z_kanat = aircraft.govde_yaricap * 1.1 
    
    x_w = [kanat_x_bas, kanat_x_bas+kanat_genislik, kanat_x_bas+kanat_genislik, kanat_x_bas]
    y_w = [-kanat_uzunluk/2, -kanat_uzunluk/2, kanat_uzunluk/2, kanat_uzunluk/2]
    z_w = [z_kanat, z_kanat, z_kanat, z_kanat]
    traces.append(go.Mesh3d(x=x_w, y=y_w, z=z_w, color='lightblue', opacity=0.4, name='Ana Kanat', i=[0, 0], j=[1, 2], k=[2, 3]))

    # CESSNA ÖZEL: Pervane Diski (Propeller)
    theta = np.linspace(0, 2*np.pi, 30)
    r_prop = 45
    traces.append(go.Mesh3d(x=[0]*len(theta), y=r_prop*np.cos(theta), z=r_prop*np.sin(theta), alphahull=0, color='silver', opacity=0.3, name='Pervane Diski', hoverinfo='skip'))

    # CESSNA ÖZEL: Kanat Destek Dikmeleri (Wing Struts)
    traces.append(go.Scatter3d(x=[kanat_x_bas+10, kanat_x_bas+10], y=[-aircraft.govde_yaricap, -kanat_uzunluk/3], z=[-aircraft.govde_yaricap*0.8, z_kanat], mode='lines', line=dict(color='gray', width=4), name='Sol Dikme'))
    traces.append(go.Scatter3d(x=[kanat_x_bas+10, kanat_x_bas+10], y=[aircraft.govde_yaricap, kanat_uzunluk/3], z=[-aircraft.govde_yaricap*0.8, z_kanat], mode='lines', line=dict(color='gray', width=4), name='Sağ Dikme'))

    # CESSNA ÖZEL: İniş Takımları (Tricycle Gear)
    z_yer = -aircraft.govde_yaricap - 40
    
    # 1. Burun İniş Takımı
    traces.append(go.Scatter3d(x=[20, 15], y=[0, 0], z=[-aircraft.govde_yaricap, z_yer + 5], 
                               mode='lines', line=dict(color='#666666', width=8), name='Burun Dikmesi', hoverinfo='skip'))
    traces.extend(_silindir_traces(15, 0, z_yer + 5, r=8, uzunluk=6, yon='y', color='#111111', name='Burun Tekeri', cap=True))
    traces.extend(_silindir_traces(15, 0, z_yer + 5, r=4, uzunluk=6.5, yon='y', color='#DDDDDD', name='', cap=True)) # Jant
    
    # 2. Ana İniş Takımları
    # Sol Dikme ve Tekerlek
    traces.append(go.Scatter3d(x=[kanat_x_bas+15, kanat_x_bas+30], y=[0, -45], z=[-aircraft.govde_yaricap*0.8, z_yer + 6], 
                               mode='lines', line=dict(color='#666666', width=8), name='Sol Ana Dikme', hoverinfo='skip'))
    traces.extend(_silindir_traces(kanat_x_bas+30, -45, z_yer + 6, r=10, uzunluk=8, yon='y', color='#111111', name='Sol Tekerlek', cap=True))
    traces.extend(_silindir_traces(kanat_x_bas+30, -45, z_yer + 6, r=5, uzunluk=8.5, yon='y', color='#DDDDDD', name='', cap=True)) # Jant
    
    # Sağ Dikme ve Tekerlek
    traces.append(go.Scatter3d(x=[kanat_x_bas+15, kanat_x_bas+30], y=[0, 45], z=[-aircraft.govde_yaricap*0.8, z_yer + 6], 
                               mode='lines', line=dict(color='#666666', width=8), name='Sağ Ana Dikme', hoverinfo='skip'))
    traces.extend(_silindir_traces(kanat_x_bas+30, 45, z_yer + 6, r=10, uzunluk=8, yon='y', color='#111111', name='Sağ Tekerlek', cap=True))
    traces.extend(_silindir_traces(kanat_x_bas+30, 45, z_yer + 6, r=5, uzunluk=8.5, yon='y', color='#DDDDDD', name='', cap=True)) # Jant

    # 3. KUYRUK TAKIMI (TAIL) - Cessna stili geriye yatık (swept back)
    tail_x = aircraft.govde_uzunluk - 50
    h_stab_span = 120
    traces.append(go.Mesh3d(x=[tail_x, tail_x+40, tail_x+40, tail_x], y=[-h_stab_span/2, -h_stab_span/2, h_stab_span/2, h_stab_span/2], z=[0, 0, 0, 0], color='lightblue', opacity=0.4, name='Yatay Kuyruk', i=[0, 0], j=[1, 2], k=[2, 3]))
    traces.append(go.Mesh3d(x=[tail_x, aircraft.govde_uzunluk, aircraft.govde_uzunluk+15, tail_x+40], y=[0, 0, 0, 0], z=[0, 0, 60, 60], color='lightblue', opacity=0.4, name='Dikey Kuyruk', i=[0, 0], j=[1, 2], k=[2, 3]))

    return traces

def dondur_3d(xs, ys, zs, cx, cy, cz, roll_deg, pitch_deg, yaw_deg):
    if roll_deg == 0 and pitch_deg == 0 and yaw_deg == 0:
        return xs, ys, zs
    r_x, r_y, r_z = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)], [0, np.sin(r_x), np.cos(r_x)]])
    Ry = np.array([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0], [-np.sin(r_y), 0, np.cos(r_y)]])
    Rz = np.array([[np.cos(r_z), -np.sin(r_z), 0], [np.sin(r_z), np.cos(r_z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    pts = np.vstack((np.array(xs)-cx, np.array(ys)-cy, np.array(zs)-cz))
    rot = R @ pts
    return (rot[0]+cx).tolist(), (rot[1]+cy).tolist(), (rot[2]+cz).tolist()

def kutu_trace(x, y, z, dx, dy, dz, color, name, cx, cy, cz, r=0, p=0, yw=0):
    x_k = [x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2]
    y_k = [y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2]
    z_k = [z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2]
    x_k, y_k, z_k = dondur_3d(x_k, y_k, z_k, cx, cy, cz, r, p, yw)
    return go.Mesh3d(
        x=x_k, y=y_k, z=z_k, color=color, opacity=1.0, name=name,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], hoverinfo='name'
    )

def silindir_trace(x, y, z, dx, dy, dz, color, name, cx, cy, cz, r=0, p=0, yw=0):
    theta = np.linspace(0, 2*np.pi, 15)
    xs, ys, zs = [], [], []
    r_y, r_z = dy / 2, dz / 2
    for t in theta:
        xs.extend([x - dx/2, x + dx/2])
        ys.extend([y + r_y*np.cos(t), y + r_y*np.cos(t)])
        zs.extend([z + r_z*np.sin(t), z + r_z*np.sin(t)])
    xs, ys, zs = dondur_3d(xs, ys, zs, cx, cy, cz, r, p, yw)
    return go.Mesh3d(x=xs, y=ys, z=zs, alphahull=0, color=color, name=name, hoverinfo='name')

def elipsoit_trace(x, y, z, dx, dy, dz, color, name, cx, cy, cz, r=0, p=0, yw=0):
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    xs, ys, zs = [], [], []
    for uu in u:
        for vv in v:
            xs.append(x + (dx/2) * np.cos(uu) * np.sin(vv))
            ys.append(y + (dy/2) * np.sin(uu) * np.sin(vv))
            zs.append(z + (dz/2) * np.cos(vv))
    xs, ys, zs = dondur_3d(xs, ys, zs, cx, cy, cz, r, p, yw)
    return go.Mesh3d(x=xs, y=ys, z=zs, alphahull=0, color=color, name=name, hoverinfo='name')

def ozel_parca_ciz(pos, dim, color, name):
    """Komponentleri isimlerine göre girintili çıkıntılı (karmaşık) geometriler olarak çizer"""
    x, y, z = pos
    dx, dy, dz = dim
    name_lower = name.lower()
    traces = []
    
    r_deg, p_deg, yw_deg = 0, 0, 0
    
    if "motor" in name_lower:
        p_deg = 5 # Motor egzozu çok hafif yukarı açılı dursun
        traces.append(silindir_trace(x + dx*0.1, y, z, dx*0.8, dy, dz, color, name, x, y, z, r_deg, p_deg, yw_deg))
        traces.append(elipsoit_trace(x - dx*0.4, y, z, dx*0.2, dy*0.8, dz*0.8, "silver", name + " Koni", x, y, z, r_deg, p_deg, yw_deg))
        traces.append(silindir_trace(x + dx*0.5, y, z, dx*0.2, dy*0.5, dz*0.5, "darkgray", name + " Egzoz", x, y, z, r_deg, p_deg, yw_deg))
    elif "yakit" in name_lower or "tank" in name_lower:
        yw_deg = 90 # Kanat hizasında uzanması için 90 derece (Y ekseni boyunca) döndürüyoruz
        y_loc_left = y + 90  # Sol kanat altı/içi
        y_loc_right = y - 90 # Sağ kanat altı/içi
        z_offset = z + 35    # Gövde merkezinden üstte olan kanat hizasına yükseltiyoruz
        
        tank_uzunluk = 100
        tank_kalinlik = 18
        
        # Sol Kanat Yakıt Tankı (Hap formu yakalamak için silindir+elipsoit kombinasyonu)
        traces.append(silindir_trace(x, y_loc_left, z_offset, tank_uzunluk, tank_kalinlik, tank_kalinlik, color, name+"_Sol", x, y_loc_left, z_offset, r_deg, p_deg, yw_deg))
        traces.append(elipsoit_trace(x, y_loc_left, z_offset, tank_uzunluk + 20, tank_kalinlik, tank_kalinlik, color, name+"_Sol", x, y_loc_left, z_offset, r_deg, p_deg, yw_deg))
        
        # Sağ Kanat Yakıt Tankı
        traces.append(silindir_trace(x, y_loc_right, z_offset, tank_uzunluk, tank_kalinlik, tank_kalinlik, color, name+"_Sag", x, y_loc_right, z_offset, r_deg, p_deg, yw_deg))
        traces.append(elipsoit_trace(x, y_loc_right, z_offset, tank_uzunluk + 20, tank_kalinlik, tank_kalinlik, color, name+"_Sag", x, y_loc_right, z_offset, r_deg, p_deg, yw_deg))
    elif "kam" in name_lower or "payload" in name_lower:
        p_deg = -35 # Kamera merceği yere doğru bakar (-35 pitch)
        traces.append(kutu_trace(x, y, z + dz*0.2, dx, dy, dz*0.6, color, name, x, y, z, r_deg, p_deg, yw_deg))
        traces.append(silindir_trace(x, y, z - dz*0.2, dx*0.5, dy*0.5, dz*0.4, "black", name + " Lens", x, y, z, r_deg, p_deg, yw_deg))
    elif "aviyonik" in name_lower:
        yw_deg = 15 # Kartlar daha dinamik bir hava vermesi için Z ekseninde (yaw) 15 derece çevrilsin
        traces.append(kutu_trace(x, y, z - dz*0.1, dx, dy, dz*0.8, color, name, x, y, z, r_deg, p_deg, yw_deg))
        num_fins = 5
        fin_dx = dx * 0.8 / num_fins
        for i in range(num_fins):
            fx = (x - dx*0.35) + i * (dx * 0.8 / num_fins)
            traces.append(kutu_trace(fx, y, z + dz*0.35, fin_dx*0.5, dy*0.8, dz*0.15, "silver", name + " Fin", x, y, z, r_deg, p_deg, yw_deg))
    elif "batarya" in name_lower:
        yw_deg = -20 # Çapraz pil montajı dizilimi
        h_dx, h_dy = dx*0.4, dy*0.4
        traces.append(kutu_trace(x - dx*0.25, y - dy*0.25, z, h_dx, h_dy, dz, color, name, x, y, z, r_deg, p_deg, yw_deg))
        traces.append(kutu_trace(x + dx*0.25, y - dy*0.25, z, h_dx, h_dy, dz, color, name, x, y, z, r_deg, p_deg, yw_deg))
        traces.append(kutu_trace(x - dx*0.25, y + dy*0.25, z, h_dx, h_dy, dz, color, name, x, y, z, r_deg, p_deg, yw_deg))
        traces.append(kutu_trace(x + dx*0.25, y + dy*0.25, z, h_dx, h_dy, dz, color, name, x, y, z, r_deg, p_deg, yw_deg))
    else:
        # L Şeklinde parçalar (örn Servo) 45 derece eğimli konulsun
        p_deg, yw_deg = 10, 45 
        traces.append(kutu_trace(x - dx*0.2, y, z, dx*0.6, dy, dz, color, name, x, y, z, r_deg, p_deg, yw_deg))
        traces.append(kutu_trace(x + dx*0.3, y, z - dz*0.2, dx*0.4, dy, dz*0.6, color, name, x, y, z, r_deg, p_deg, yw_deg))
        
    return traces

def gorsellestir_tasarim(en_iyi_tasarim, best_score, best_cg, aircraft, ALGORITMA):
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
        # Boyut bilgisini DB'den çek
        boyut = next(item for item in aircraft.komponentler_db if item.id == k_id).boyut
        idx = aircraft.komponentler_db.index(next(item for item in aircraft.komponentler_db if item.id == k_id))
        colors = ['red', 'blue', 'orange', 'purple', 'green', 'brown', 'cyan']
        
        # Kutuları girintili çıkıntılı çiz
        karma_parcalar = ozel_parca_ciz(pos, boyut, colors[idx % len(colors)], k_id)
        for t in karma_parcalar:
            fig.add_trace(t)
        
        # Etiket ekle (Havada asılı yazı)
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2] + boyut[2] / 1.5],
            mode='text', text=[k_id], textposition="top center",
            textfont=dict(size=10, color="black"), showlegend=False
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

    target_x_visual = np.clip(best_cg[0], aircraft.target_cg_x_min, aircraft.target_cg_x_max)
    fig.add_trace(go.Scatter3d(
        x=[target_x_visual, best_cg[0]],
        y=[aircraft.target_cg_y, best_cg[1]],
        z=[aircraft.target_cg_z, best_cg[2]],
        mode='lines', line=dict(color='red', width=4, dash='dot'), name='CG Hatası'
    ))

    # Layout — açık tema (eski)
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
