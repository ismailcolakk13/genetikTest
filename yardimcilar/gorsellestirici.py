import plotly.graph_objects as go
import numpy as np

def ucak_govdesi_olustur(aircraft):
    """
    Cessna 172 görünümüne sahip uçak geometrisi (Mesh) oluşturur.
    """
    traces = []
    
    # 1. GÖVDE (FUSELAGE)
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
    traces.append(go.Scatter3d(x=[20, 15], y=[0, 0], z=[-aircraft.govde_yaricap, z_yer], mode='lines+markers', line=dict(color='black', width=6), marker=dict(size=8, color='darkgray'), name='Burun Tekeri'))
    traces.append(go.Scatter3d(x=[kanat_x_bas+30, kanat_x_bas+15, kanat_x_bas+30], y=[-40, 0, 40], z=[z_yer, -aircraft.govde_yaricap*0.8, z_yer], mode='lines+markers', line=dict(color='black', width=6), marker=dict(size=10, color='darkgray'), name='Ana İniş Takımları'))

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

    # 1. Uçak Gövdesini Çiz
    ucak_parcalari = ucak_govdesi_olustur(aircraft)
    for parca in ucak_parcalari:
        fig.add_trace(parca)
    
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
            x=[pos[0]], y=[pos[1]], z=[pos[2] + boyut[2]/1.5], # Kutunun biraz üstüne
            mode='text', text=[k_id], textposition="top center",
            textfont=dict(size=10, color="black"), showlegend=False
        ))

    # 3. Ağırlık Merkezi (CG) Göstergeleri
    # Hedef CG Aralığı (Altın Sarısı Yarı Şeffaf Kutu - Yakıt tankıyla karışmasın diye)
    # Görünür olması için gövde çapından biraz daha geniş çiziyoruz.
    box_r = aircraft.govde_yaricap + 5 # Yarıçaptan 5cm daha geniş
    fig.add_trace(go.Mesh3d(
        x=[aircraft.target_cg_x_min, aircraft.target_cg_x_max, aircraft.target_cg_x_max, aircraft.target_cg_x_min, aircraft.target_cg_x_min, aircraft.target_cg_x_max, aircraft.target_cg_x_max, aircraft.target_cg_x_min],
        y=[-box_r, -box_r, box_r, box_r, -box_r, -box_r, box_r, box_r],
        z=[-box_r, -box_r, -box_r, -box_r, box_r, box_r, box_r, box_r],
        color='gold', opacity=0.3, name='HEDEF CG ARALIĞI',
        alphahull=0
    ))

    # Hesaplanan (Sonuç) CG - Görünürlük için yukarı taşıyoruz
    viz_z = aircraft.govde_yaricap + 40 # Gövdenin üstünde, her zaman görünür olması için

    fig.add_trace(go.Scatter3d(
        x=[best_cg[0]], y=[best_cg[1]], z=[viz_z],
        mode='markers+text', marker=dict(size=12, color='black', symbol='diamond'),
        name='HESAPLANAN CG', text=["HESAPLANAN CG"], textposition="top center"
    ))

    # Gerçek CG noktasına dikey çizgi (Drop line)
    fig.add_trace(go.Scatter3d(
        x=[best_cg[0], best_cg[0]], y=[best_cg[1], best_cg[1]], z=[best_cg[2], viz_z],
        mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'
    ))

    # Gerçek CG noktası (İçeride kalan küçük nokta)
    fig.add_trace(go.Scatter3d(
        x=[best_cg[0]], y=[best_cg[1]], z=[best_cg[2]],
        mode='markers', marker=dict(size=5, color='black'), 
        name='Gerçek CG Konumu'
    ))

    # Çizgi Çek (Hata payını görselleştirmek için - En yakın sınıra)
    target_x_visual = best_cg[0]
    if best_cg[0] < aircraft.target_cg_x_min: target_x_visual = aircraft.target_cg_x_min
    elif best_cg[0] > aircraft.target_cg_x_max: target_x_visual = aircraft.target_cg_x_max

    fig.add_trace(go.Scatter3d(
        x=[target_x_visual, best_cg[0]], y=[aircraft.target_cg_y, best_cg[1]], z=[aircraft.target_cg_z, best_cg[2]],
        mode='lines', line=dict(color='red', width=4, dash='dot'), name='CG Hatası'
    ))

    # --- AYARLAR VE SAHNE DÜZENİ ---
    camera = dict(
        eye=dict(x=2.0, y=-2.0, z=1.0) # Kamerayı çaprazdan baktır
    )

    fig.update_layout(
        title=f"Ön Tasarım: Uçak İçi Sistem Yerleşimi Optimizasyonu ({ALGORITMA})",
        scene=dict(
            xaxis=dict(title='Uzunluk (cm)', range=[0, aircraft.govde_uzunluk], backgroundcolor="rgb(240, 240, 240)"),
            yaxis=dict(title='Genişlik (cm)', range=[-200, 200]), # Kanatları kapsasın diye geniş
            zaxis=dict(title='Yükseklik (cm)', range=[-100, 100]),
            aspectmode='data', # Gerçek oranları koru (Uçak basık görünmesin)
            camera=camera
        ),
        margin=dict(r=0, l=0, b=0, t=50) # Kenar boşluklarını azalt
    )

    fig.show()
