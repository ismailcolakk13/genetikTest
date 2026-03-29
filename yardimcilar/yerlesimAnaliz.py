def analiz_yap(en_iyi_tasarim, best_score, best_cg, aircraft, ALGORITMA):
    
    print(f"\n--- TASARIM ANALİZİ ({ALGORITMA}) ---")

    # 1. CG Hedefe Yakınlık Kontrolü
    cg_x, cg_y, cg_z = best_cg
    # X ekseninde hedef aralığa göre sapma hesabı
    if cg_x < aircraft.target_cg_x_min:
        dx = aircraft.target_cg_x_min - cg_x
    elif cg_x > aircraft.target_cg_x_max:
        dx = cg_x - aircraft.target_cg_x_max
    else:
        dx = 0.0

    # Toplam mesafe hatası (X aralığı, Y=0 ve Z=0 hedeflerine göre)
    dist_error = (dx**2 + (cg_y - aircraft.target_cg_y)**2 + (cg_z - aircraft.target_cg_z)**2)**0.5

    if dist_error < 2.0:
        print(f"✅ CG hedefe çok yakın (Sapma: {dist_error:.2f} cm)")
    elif dist_error < 15.0:
        print(f"⚠️ CG hedefe orta mesafede (Sapma: {dist_error:.2f} cm)")
    else:
        print(f"❌ CG hedeften uzak (Sapma: {dist_error:.2f} cm)")

    # 2. Yakıt Tankı Etkisi Kontrolü
    # Yakıt tankı ağırlık merkezinden (CG) ne kadar uzaksa, yakıt azaldıkça uçağın dengesi o kadar bozulur.
    yakit_pos = en_iyi_tasarim.yerlesim.get("Yakit_Tanki", (0, 0, 0))
    hedef_merkez_x = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2

    if abs(yakit_pos[0] - hedef_merkez_x) > 10.0:
        print(f"⛽ Yakıt tankının X konumu ({yakit_pos[0]:.1f}) ideal merkezden uzak. Yakıt tüketimi CG'yi ETKİLEYECEK.")
    else:
        print(f"⛽ Yakıt tankı ideal merkeze çok yakın. Yakıt tüketiminin dengeye etkisi MİNİMUM.")

    # 3. Genel Skor Yorumu
    # Ceza sistemi olduğu için skor 0'a ne kadar yakınsa (negatif değerler) o kadar iyidir.
    if best_score > -4000:
        print(f"🏆 Tasarım çok iyi (Skor: {best_score:.0f})")
    elif best_score > -6000:
        print(f"👍 Tasarım kabul edilebilir (Skor: {best_score:.0f})")
    else:
        print(f"🚫 Tasarım zayıf (Skor: {best_score:.0f})")

    # 4. SICAKLIK PROFİLİ ANALİZİ
    print("\n--- SICAKLIK PROFİLİ ANALİZİ ---")
    pos_motor = en_iyi_tasarim.yerlesim.get("Motor")
    if pos_motor:
        sicaklik_ihlali_var = False
        for k_id, pos in en_iyi_tasarim.yerlesim.items():
            parca_db = next(item for item in aircraft.komponentler_db if item.id == k_id)
            if parca_db.sicaklik_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                if mesafe < aircraft.sicaklik_limiti:
                    print(f"🔥 {k_id}: Motora çok yakın ({mesafe:.1f} cm) - SICAKLIK RİSKİ! (Limit: {aircraft.sicaklik_limiti} cm)")
                    sicaklik_ihlali_var = True
                elif mesafe < aircraft.sicaklik_limiti * 1.5:
                    print(f"⚠️ {k_id}: Motora mesafe sınırda ({mesafe:.1f} cm) - DİKKAT")
                else:
                    print(f"✅ {k_id}: Motordan güvenli mesafede ({mesafe:.1f} cm)")
        if not sicaklik_ihlali_var:
            print("✅ Tüm ısıya hassas parçalar güvenli mesafede.")
        
    print("\n--- DENGE ANALİZİ (CG DRIFT) ---")

    # Denge Analizi Hesaplamaları (Sadece X ekseni için)
    bos_agirlik = 0
    bos_moment_x = 0
    dolu_agirlik = 0
    dolu_moment_x = 0

    for k_id, pos in en_iyi_tasarim.yerlesim.items():
        db_item = next(item for item in aircraft.komponentler_db if item.id == k_id)
        mass = db_item.agirlik

        # Bos depo için moment (Yakıt = 0)
        bos_agirlik += mass
        bos_moment_x += mass * pos[0]

        # Dolu depo için moment (Yakıt = MAX)
        if k_id == "Yakit_Tanki":
            dolu_agirlik += (mass + aircraft.max_yakit_agirligi)
            dolu_moment_x += (mass + aircraft.max_yakit_agirligi) * pos[0]
        else:
            dolu_agirlik += mass
            dolu_moment_x += mass * pos[0]

    cg_bos_x = bos_moment_x / bos_agirlik
    cg_dolu_x = dolu_moment_x / dolu_agirlik
    cg_kaymasi = abs(cg_dolu_x - cg_bos_x)

    yakit_pos_x = yakit_pos[0]

    print(f"Yakit Tanki Konumu (X): {yakit_pos_x:.2f} cm")
    print(f"CG (Dolu Depo)        : {cg_dolu_x:.2f} cm")
    print(f"CG (Bos Depo)         : {cg_bos_x:.2f} cm")
    print(f"CG Kaymasi (Drift)    : {cg_kaymasi:.2f} cm")

    # Uyarı Mekanizması
    if cg_kaymasi > 5.0:
        print("❌ KRİTİK: Yakıt tüketimi CG'yi çok fazla kaydırıyor! Uçuş stabilitesi tehlikede.")
    elif cg_kaymasi > 2.0:
        print("⚠️ DİKKAT: Yakıt tüketimi dengeyi etkiliyor. Trim ayarı gerekecek.")
    else:
        print("✅ MÜKEMMEL: Yakıt tankı ideal konumda. Yakıt tüketiminin dengeye etkisi minimum.")
    print("-----------------------\n")
    print(f"\n--- YERLEŞİM DETAYLARI ({ALGORITMA}) ---")

    for k_id, pos in en_iyi_tasarim.yerlesim.items():
        print(f"📍 {k_id}: Gövde Başından {pos[0]:.1f} cm geride.")