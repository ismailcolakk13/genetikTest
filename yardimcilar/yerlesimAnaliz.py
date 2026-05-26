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
    # Sol ve Sağ tank konumlarının ortalama X'i kullanılır
    yakit_sol = en_iyi_tasarim.yerlesim.get("Yakit_Tanki_Sol")
    yakit_sag = en_iyi_tasarim.yerlesim.get("Yakit_Tanki_Sag")
    if yakit_sol and yakit_sag:
        yakit_pos = ((yakit_sol[0] + yakit_sag[0]) / 2,
                     (yakit_sol[1] + yakit_sag[1]) / 2,
                     (yakit_sol[2] + yakit_sag[2]) / 2)
    else:
        yakit_pos = en_iyi_tasarim.yerlesim.get("Yakit_Tanki", (0, 0, 0))
    hedef_merkez_x = (aircraft.target_cg_x_min + aircraft.target_cg_x_max) / 2

    if abs(yakit_pos[0] - hedef_merkez_x) > 10.0:
        print(f"⛽ Yakıt tanklarının X konumu ({yakit_pos[0]:.1f}) ideal merkezden uzak. Yakıt tüketimi CG'yi ETKİLEYECEK.")
    else:
        print(f"⛽ Yakıt tankları ideal merkeze çok yakın. Yakıt tüketiminin dengeye etkisi MİNİMUM.")

    # 3. Fiziksel İhlal ve Genel Skor Yorumu
    # Skor temelli eşik güvenilmez — CG hatası tek başına skoru -5000'in altına çekebilir.
    # Bunun yerine çakışma ve taşmayı doğrudan re-kontrol ediyoruz.
    from yardimcilar.yardimciFonksiyonlar import kutular_cakisiyor_mu
    kmap_check = aircraft.komponentler_map
    keys_check = list(en_iyi_tasarim.yerlesim.keys())

    has_collision = False
    for i in range(len(keys_check)):
        for j in range(i + 1, len(keys_check)):
            k1_id, k2_id = keys_check[i], keys_check[j]
            if k1_id.startswith("Yakit_Tanki") != k2_id.startswith("Yakit_Tanki"):
                continue
            k1, k2 = kmap_check[k1_id], kmap_check[k2_id]
            if k1.kilitli and k2.kilitli:
                continue
            if kutular_cakisiyor_mu(en_iyi_tasarim.yerlesim[k1_id], k1.boyut,
                                     en_iyi_tasarim.yerlesim[k2_id], k2.boyut):
                has_collision = True
                break
        if has_collision:
            break

    has_overflow = False
    for k_id, pos in en_iyi_tasarim.yerlesim.items():
        if k_id.startswith("Yakit_Tanki"):
            continue
        komp = kmap_check[k_id]
        if komp.kilitli:  # sabit konumlu parçalar tasarım gereği muaf
            continue
        if not aircraft.govde_icinde_mi(pos, komp.boyut):
            has_overflow = True
            break

    if has_collision:
        print(f"🚫 Tasarım ZAYIF (Parça çakışması var!) | Skor: {best_score:.0f}")
    elif has_overflow:
        print(f"🚫 Tasarım ZAYIF (Gövdeden taşma var!) | Skor: {best_score:.0f}")
    elif best_score >= -2500:
        print(f"🏆 Tasarım ÇOK İYİ (Fiziksel ihlal yok ve Denge harika) | Skor: {best_score:.0f}")
    else:
        print(f"👍 Tasarım KABUL EDİLEBİLİR (Fiziksel ihlal yok, ancak denge daha iyi olabilir) | Skor: {best_score:.0f}")

    # 4. SICAKLIK PROFİLİ ANALİZİ
    print("\n--- SICAKLIK PROFİLİ ANALİZİ ---")
    pos_motor = en_iyi_tasarim.yerlesim.get("Motor")
    if pos_motor:
        sicaklik_ihlali_var = False
        kmap = aircraft.komponentler_map
        for k_id, pos in en_iyi_tasarim.yerlesim.items():
            parca_db = kmap[k_id]
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
            
    # 5. TİTREŞİM PROFİLİ ANALİZİ
    print("\n--- TİTREŞİM PROFİLİ ANALİZİ ---")
    if pos_motor:
        titresim_ihlali_var = False
        for k_id, pos in en_iyi_tasarim.yerlesim.items():
            parca_db = kmap[k_id]
            if parca_db.titresim_hassasiyeti:
                mesafe = ((pos[0]-pos_motor[0])**2 + (pos[1]-pos_motor[1])**2 + (pos[2]-pos_motor[2])**2)**0.5
                if mesafe < aircraft.titresim_limiti:
                    print(f"📳 {k_id}: Motora çok yakın ({mesafe:.1f} cm) - TİTREŞİM RİSKİ! (Limit: {aircraft.titresim_limiti} cm)")
                    titresim_ihlali_var = True
                elif mesafe < aircraft.titresim_limiti * 1.5:
                    print(f"⚠️ {k_id}: Motora mesafe sınırda ({mesafe:.1f} cm) - DİKKAT")
                else:
                    print(f"✅ {k_id}: Motordan güvenli mesafede ({mesafe:.1f} cm)")
        if not titresim_ihlali_var:
            print("✅ Tüm titreşime hassas parçalar güvenli mesafede.")

        
    print("\n--- DENGE ANALİZİ (CG DRIFT) ---")

    # Denge Analizi Hesaplamaları (Sadece X ekseni için)
    bos_agirlik = 0
    bos_moment_x = 0
    dolu_agirlik = 0
    dolu_moment_x = 0

    kmap = aircraft.komponentler_map
    for k_id, pos in en_iyi_tasarim.yerlesim.items():
        db_item = kmap[k_id]
        mass = db_item.agirlik

        # Bos depo için moment (Yakıt = 0)
        bos_agirlik += mass
        bos_moment_x += mass * pos[0]

        # Dolu depo için moment (Yakıt = MAX, Sol+Sağ eşit pay)
        if k_id in ("Yakit_Tanki_Sol", "Yakit_Tanki_Sag"):
            dolu_agirlik += (mass + aircraft.max_yakit_agirligi * 0.5)
            dolu_moment_x += (mass + aircraft.max_yakit_agirligi * 0.5) * pos[0]
        elif k_id == "Yakit_Tanki":
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