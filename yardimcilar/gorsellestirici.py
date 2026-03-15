import plotly.graph_objects as go
import numpy as np

def ucak_govdesi_olustur(aircraft):
    """
    Cessna benzeri basit bir uçak geometrisi (Mesh) oluşturur.
    """
    traces = []
    
    # 1. GÖVDE (FUSELAGE) - Aerodinamik bir tüp
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, aircraft.govde_uzunluk, 40)
    u, v = np.meshgrid(u, v)
    
    # Gövde şeklini belirleyen yarıçap fonksiyonu (Tapering)
    r_values = np.array([aircraft.get_fuselage_radius(x) for x in v.flatten()]).reshape(v.shape)
    
    x_govde = v
    y_govde = r_values * np.cos(u)
    z_govde = r_values * np.sin(u) * 1.2 # Yükseklik biraz daha eliptik olsun
    
    # Gövdeyi Yarı Şeffaf Çiz
    traces.append(go.Surface(
        x=x_govde, y=y_govde, z=z_govde,
        opacity=0.15, colorscale='Greys', showscale=False, name='Gövde', hoverinfo='skip'
    ))
    
    # Gövde Tel Kafes (Wireframe) çizgileri
    for i in range(0, 40, 4): 
        traces.append(go.Scatter3d(
            x=x_govde[i], y=y_govde[i], z=z_govde[i],
            mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'
        ))

    # 2. KANATLAR (WINGS) - High Wing Cessna Tipi
    kanat_x_bas = 80
    kanat_genislik = 40 # Chord
    kanat_uzunluk = 360 # Span (Gövdeden taşan)
    z_kanat = aircraft.govde_yaricap * 1.1 # Gövdenin üstünde
    
    x_w = [kanat_x_bas, kanat_x_bas+kanat_genislik, kanat_x_bas+kanat_genislik, kanat_x_bas]
    y_w = [-kanat_uzunluk/2, -kanat_uzunluk/2, kanat_uzunluk/2, kanat_uzunluk/2]
    z_w = [z_kanat, z_kanat, z_kanat, z_kanat]
    
    traces.append(go.Mesh3d(
        x=x_w, y=y_w, z=z_w,
        color='lightblue', opacity=0.5, name='Kanat',
        i=[0, 0], j=[1, 2], k=[2, 3] # Yüzey örme indeksleri
    ))

    # 3. KUYRUK TAKIMI (TAIL)
    # Yatay Stabilize
    tail_x = aircraft.govde_uzunluk - 40
    h_stab_span = 120
    x_h = [tail_x, aircraft.govde_uzunluk, aircraft.govde_uzunluk, tail_x]
    y_h = [-h_stab_span/2, -h_stab_span/2, h_stab_span/2, h_stab_span/2]
    z_h = [0, 0, 0, 0]
    
    traces.append(go.Mesh3d(
        x=x_h, y=y_h, z=z_h, color='lightblue', opacity=0.5, name='Yatay Kuyruk',
        i=[0, 0], j=[1, 2], k=[2, 3]
    ))
    
    # Dikey Stabilize (Rudder)
    x_v = [tail_x, aircraft.govde_uzunluk, aircraft.govde_uzunluk, tail_x+10]
    y_v = [0, 0, 0, 0]
    z_v = [0, 0, 50, 50] # 50 birim yukarı
    
    traces.append(go.Mesh3d(
        x=x_v, y=y_v, z=z_v, color='lightblue', opacity=0.5, name='Dikey Kuyruk',
        i=[0, 0], j=[1, 2], k=[2, 3]
    ))

    return traces

def parca_kutusu_ciz(pos, dim, color, name):
    """Komponentleri katı kutular olarak çizer"""
    x, y, z = pos
    dx, dy, dz = dim
    
    # Küp Köşeleri
    x_k = [x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2]
    y_k = [y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2]
    z_k = [z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2]
    
    return go.Mesh3d(
        x=x_k, y=y_k, z=z_k,
        color=color, opacity=1.0, name=name,
        # Küp yüzey tanımları (index based)
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        hoverinfo='name'
    )

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
        
        # Kutuyu çiz
        fig.add_trace(parca_kutusu_ciz(pos, boyut, colors[idx % len(colors)], k_id))
        
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
