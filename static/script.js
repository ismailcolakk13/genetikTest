const PRESETS = {};

PRESETS.default = {
    global: { govde_uzunluk: 300, govde_cap: 60, target_cg_x_min: 90, target_cg_x_max: 110, target_cg_y: 0, target_cg_z: 0, max_yakit_agirligi: 50, sicaklik_limiti: 30, titresim_limiti: 50 },
    components: [
    // Motor → BURUN'da sabit, kilitli
    { id: "Motor", w: 40.0, d: [60, 40, 40], reg: "BURUN", lock: true, vib: false, temp: false, sabit_pos: [30, 0, 0] },

    // Batarya → Ağır, denge için merkez/alt gövde
    { id: "Batarya_Ana", w: 15.0, d: [20, 15, 10], reg: "GOVDE/TABAN", lock: false, vib: false, temp: true, sabit_pos: null },

    // Aviyonikler → Titreşim ve sıcaklıktan uzak, gövde üst bölgesi
    { id: "Aviyonik_1", w: 5.0, d: [15, 15, 5], reg: "GOVDE/TAVAN", lock: false, vib: true, temp: true, sabit_pos: null },
    { id: "Aviyonik_2", w: 5.0, d: [15, 15, 5], reg: "GOVDE/TAVAN", lock: false, vib: true, temp: true, sabit_pos: null },

    // Payload Kamera → Burun altında
    { id: "Payload_Kam", w: 10.0, d: [20, 20, 20], reg: "BURUN/TABAN", lock: false, vib: true, temp: false, sabit_pos: null },

    // Yakıt Tankları → kanat içi
    { id: "Yakit_Tanki_Sol", w: 20.0, d: [110, 22, 6], reg: "GOVDE", lock: false, vib: false, temp: false, sabit_pos: null },
    { id: "Yakit_Tanki_Sag", w: 20.0, d: [110, 22, 6], reg: "GOVDE", lock: false, vib: false, temp: false, sabit_pos: null },

    // Servo → Kuyruk trim servosu
    { id: "Servo_Kuyruk", w: 2.0, d: [5, 5, 5], reg: "KUYRUK", lock: false, vib: false, temp: false, sabit_pos: null },

    // === KOLTUKLAR ===
    // Pilot koltuğu (sol ön)
    { id: "Koltuk_Pilot", w: 8.0, d: [30, 15, 40], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [80, -8, 0] },
    // Pilot (insan yükü)
    { id: "Pilot", w: 90.0, d: [25, 12, 35], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [80, -8, 0] },
    // Yardımcı pilot koltuğu (sağ ön)
    { id: "Koltuk_Yardimci", w: 8.0, d: [30, 15, 40], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [80, 8, 0] },
    // Arka sol yolcu koltuğu
    { id: "Koltuk_Arka_Sol", w: 7.0, d: [28, 15, 38], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [120, -8, 0] },
    // Arka sağ yolcu koltuğu
    { id: "Koltuk_Arka_Sag", w: 7.0, d: [28, 15, 38], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [120, 8, 0] },
    // Bagaj bölmesi
    { id: "Bagaj", w: 15.0, d: [35, 30, 22], reg: "GOVDE", lock: true, vib: false, temp: false, sabit_pos: [160, 0, -2] },
    ]
};

// Gerçekçi Cessna 172 preset'i.
// Koordinat: 3D görselleştirmenin doğal ölçeği (gövde ~300 cm, çap 60 cm) —
// kanat kabinin üstüne, iniş takımı kanat altına, kuyruk arkaya doğru oturur.
// Mutlak cm gerçek uçakla birebir değil; AĞIRLIKLAR (kg) ve GÖRELİ yerleşim
// (motor önde, koltuklar/yakıt kanat hizasında, bagaj arkada) gerçekçidir.
// "Yapi_Govde" = boş gövde/kanat/kuyruk yapı kütlesi (CG'yi kanat altına çeker).
PRESETS.cessna172 = {
    global: { govde_uzunluk: 300, govde_cap: 60, target_cg_x_min: 90, target_cg_x_max: 110, target_cg_y: 0, target_cg_z: 0, max_yakit_agirligi: 109, sicaklik_limiti: 30, titresim_limiti: 50 },
    components: [
        // Lycoming O-320 motor + pervane (burunda sabit)
        { id: "Motor", w: 135, d: [60, 40, 40], reg: "BURUN", lock: true, vib: false, temp: false, sabit_pos: [30, 0, 0] },
        // Boş gövde/kanat/kuyruk yapısı — CG'yi kanat altına çeken ana kütle (alçak, merkez)
        { id: "Yapi_Govde", w: 450, d: [26, 24, 16], reg: "GOVDE", lock: true, vib: false, temp: false, sabit_pos: [120, 0, -8] },
        // Akü (ısıya hassas) + aviyonikler (titreşim+ısıya hassas) — optimize edilecek serbest parçalar
        { id: "Batarya", w: 11, d: [20, 15, 10], reg: "GOVDE/TABAN", lock: false, vib: false, temp: true, sabit_pos: null },
        { id: "Aviyonik_1", w: 7, d: [15, 15, 5], reg: "GOVDE/TAVAN", lock: false, vib: true, temp: true, sabit_pos: null },
        { id: "Aviyonik_2", w: 7, d: [15, 15, 5], reg: "GOVDE/TAVAN", lock: false, vib: true, temp: true, sabit_pos: null },
        // Ön koltuklar + pilot/yardımcı pilot
        { id: "Koltuk_Pilot", w: 10, d: [30, 15, 38], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [80, -8, 0] },
        { id: "Pilot", w: 80, d: [25, 12, 34], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [80, -8, 0] },
        { id: "Koltuk_CoPilot", w: 10, d: [30, 15, 38], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [80, 8, 0] },
        { id: "CoPilot", w: 75, d: [25, 12, 34], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [80, 8, 0] },
        // Arka koltuklar + yolcular
        { id: "Koltuk_Arka_Sol", w: 9, d: [28, 15, 36], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [120, -8, 0] },
        { id: "Yolcu_Arka_Sol", w: 75, d: [24, 12, 32], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [120, -8, 0] },
        { id: "Koltuk_Arka_Sag", w: 9, d: [28, 15, 36], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [120, 8, 0] },
        { id: "Yolcu_Arka_Sag", w: 75, d: [24, 12, 32], reg: "GOVDE/TABAN", lock: true, vib: false, temp: false, sabit_pos: [120, 8, 0] },
        // Bagaj bölmesi (arka koltuk arkası)
        { id: "Bagaj", w: 25, d: [35, 30, 22], reg: "GOVDE", lock: true, vib: false, temp: false, sabit_pos: [160, 0, -2] },
        // Kanat içi yakıt tankları (yüksek kanat, kanat köküne konumlu — Sol/Sag çizimde otomatik konumlanır)
        { id: "Yakit_Tanki_Sol", w: 5, d: [50, 22, 8], reg: "GOVDE", lock: true, vib: false, temp: false, sabit_pos: [95, -30, 18] },
        { id: "Yakit_Tanki_Sag", w: 5, d: [50, 22, 8], reg: "GOVDE", lock: true, vib: false, temp: false, sabit_pos: [95, 30, 18] },
        // Kuyruk trim servosu
        { id: "Servo_Kuyruk", w: 1.5, d: [6, 6, 6], reg: "KUYRUK", lock: true, vib: false, temp: false, sabit_pos: [270, 0, 0] }
    ]
};

const tbody = document.getElementById('componentsBody');
const template = document.getElementById('compRowTemplate');

function addComponentRow(data) {
    const clone = template.content.cloneNode(true);
    const row = clone.querySelector('tr');

    row.querySelector('.c-id').value = data.id || 'New_Part';
    row.querySelector('.c-weight').value = data.w || 10;
    row.querySelector('.c-dx').value = data.d ? data.d[0] : 10;
    row.querySelector('.c-dy').value = data.d ? data.d[1] : 10;
    row.querySelector('.c-dz').value = data.d ? data.d[2] : 10;
    row.querySelector('.c-region').value = data.reg || 'GOVDE';
    row.querySelector('.c-locked').checked = data.lock || false;
    row.querySelector('.c-vib').checked = data.vib || false;
    row.querySelector('.c-temp').checked = data.temp || false;

    // Sabit pozisyon alanları
    const posX = row.querySelector('.c-pos-x');
    const posY = row.querySelector('.c-pos-y');
    const posZ = row.querySelector('.c-pos-z');
    if (data.sabit_pos) {
        posX.value = data.sabit_pos[0];
        posY.value = data.sabit_pos[1];
        posZ.value = data.sabit_pos[2];
    } else {
        posX.value = '';
        posY.value = '';
        posZ.value = '';
    }

    // Kilitli değilse sabit_pos alanlarını gizle
    const posContainer = row.querySelector('.pos-inputs');
    posContainer.style.display = data.lock ? 'flex' : 'none';

    // Kilitli durumu değişince sabit_pos alanlarını göster/gizle
    const lockedCheckbox = row.querySelector('.c-locked');
    lockedCheckbox.addEventListener('change', () => {
        posContainer.style.display = lockedCheckbox.checked ? 'flex' : 'none';
    });

    row.querySelector('.del-btn').addEventListener('click', () => {
        row.remove();
    });

    tbody.appendChild(row);
}

// Preset yükle: global parametreleri doldur + komponent tablosunu yeniden kur
function applyPreset(name) {
    const preset = PRESETS[name] || PRESETS.default;
    const g = preset.global;
    for (const key in g) {
        const el = document.getElementById(key);
        if (el) el.value = g[key];
    }
    tbody.innerHTML = '';
    preset.components.forEach(c => addComponentRow(c));
}

// İlk yükleme: varsayılan preset
applyPreset('default');

const presetSelect = document.getElementById('presetSelect');
if (presetSelect) {
    presetSelect.addEventListener('change', () => applyPreset(presetSelect.value));
}

document.getElementById('addCompBtn').addEventListener('click', () => {
    addComponentRow({});
});

document.getElementById('runBtn').addEventListener('click', async () => {
    const btn = document.getElementById('runBtn');
    const loading = document.getElementById('loading');
    const resultsArea = document.getElementById('resultsArea');

    // UI Loading state
    btn.disabled = true;
    btn.innerHTML = 'Solving...';
    loading.style.display = 'block';

    // Scrape data
    const reqData = {
        govde_uzunluk: parseFloat(document.getElementById('govde_uzunluk').value),
        govde_cap: parseFloat(document.getElementById('govde_cap').value),
        target_cg_x_min: parseFloat(document.getElementById('target_cg_x_min').value),
        target_cg_x_max: parseFloat(document.getElementById('target_cg_x_max').value),
        target_cg_y: parseFloat(document.getElementById('target_cg_y').value),
        target_cg_z: parseFloat(document.getElementById('target_cg_z').value),
        max_yakit_agirligi: parseFloat(document.getElementById('max_yakit_agirligi').value),
        titresim_limiti: parseFloat(document.getElementById('titresim_limiti').value),
        sicaklik_limiti: parseFloat(document.getElementById('sicaklik_limiti').value),
        pop_size: parseInt(document.getElementById('pop_size').value),
        generations: parseInt(document.getElementById('generations').value),
        algoritma: document.getElementById('algoritma').value,
        komponentler: []
    };

    const rows = tbody.querySelectorAll('tr');
    rows.forEach(r => {
        const id = r.querySelector('.c-id').value;
        if (id.trim() === '') return;

        let lock = r.querySelector('.c-locked').checked;

        // Sabit pozisyon: 3 alan doldurulmuşsa kullan, yoksa null
        const posX = r.querySelector('.c-pos-x').value;
        const posY = r.querySelector('.c-pos-y').value;
        const posZ = r.querySelector('.c-pos-z').value;
        let sabitPos = null;
        if (lock && posX !== '' && posY !== '' && posZ !== '') {
            sabitPos = [parseFloat(posX), parseFloat(posY), parseFloat(posZ)];
        }

        // Region: Birden fazla bölge "/" ile ayrılmış olabilir (ör: "GOVDE/TABAN")
        const regionValue = r.querySelector('.c-region').value;

        let cData = {
            id: id,
            agirlik: parseFloat(r.querySelector('.c-weight').value),
            boyut: [
                parseFloat(r.querySelector('.c-dx').value),
                parseFloat(r.querySelector('.c-dy').value),
                parseFloat(r.querySelector('.c-dz').value)
            ],
            sabit_bolge: regionValue,
            kilitli: lock,
            titresim_hassasiyeti: r.querySelector('.c-vib').checked,
            sicaklik_hassasiyeti: r.querySelector('.c-temp').checked,
            sabit_pos: sabitPos
        };
        reqData.komponentler.push(cData);
    });

    try {
        const res = await fetch('/api/run-simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqData)
        });

        const data = await res.json();

        if (!res.ok) {
            alert('Error: ' + JSON.stringify(data));
        } else {
            // Update UI Results
            document.getElementById('resFitness').textContent = data.en_iyi_skor.toFixed(2);
            document.getElementById('resCG').textContent = `[${data.en_iyi_cg.x}, ${data.en_iyi_cg.y}, ${data.en_iyi_cg.z}]`;
            document.getElementById('resAlgo').textContent = data.algoritma_ismi;

            // Mühendislik analizini render et (CG, denge, sıcaklık, titreşim, fiziksel)
            renderAnalysis(data.analiz);

            // Log components
            const logBox = document.getElementById('resultLog');
            logBox.innerHTML = '';
            Object.values(data.tasarim).forEach(k => {
                const line = document.createElement('div');
                line.textContent = `> ${k.id} placed at (${k.pos_x.toFixed(1)}, ${k.pos_y.toFixed(1)}, ${k.pos_z.toFixed(1)}) in ${k.sabit_bolge}`;
                logBox.appendChild(line);
            });

            resultsArea.style.display = 'flex';

            // 3D Viewer: iframe'i yükle
            const viewerArea = document.getElementById('viewerArea');
            const viewer3d = document.getElementById('viewer3d');
            // Cache-bust ile yeniden yükle
            viewer3d.src = '/api/get-3d-view?t=' + Date.now();
            viewerArea.style.display = 'flex';

            // 3D viewer'a smooth scroll
            setTimeout(() => {
                viewerArea.scrollIntoView({ behavior: 'smooth' });
            }, 300);
        }
    } catch (e) {
        console.error(e);
        alert('Network Error! Is the backend running?');
    }

    btn.disabled = false;
    btn.innerHTML = '<ion-icon name="play-outline"></ion-icon> Run Optimization Sequence';
    loading.style.display = 'none';
});

// Fullscreen toggle for 3D viewer
document.getElementById('openFullscreen').addEventListener('click', () => {
    const viewerPanel = document.getElementById('viewerArea');
    const btn = document.getElementById('openFullscreen');
    viewerPanel.classList.toggle('fullscreen');
    if (viewerPanel.classList.contains('fullscreen')) {
        btn.innerHTML = '<ion-icon name="contract-outline"></ion-icon> Exit Fullscreen';
    } else {
        btn.innerHTML = '<ion-icon name="expand-outline"></ion-icon> Fullscreen';
    }
});

// ESC key to exit fullscreen
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const viewerPanel = document.getElementById('viewerArea');
        if (viewerPanel.classList.contains('fullscreen')) {
            viewerPanel.classList.remove('fullscreen');
            document.getElementById('openFullscreen').innerHTML = '<ion-icon name="expand-outline"></ion-icon> Fullscreen';
        }
    }
});

// ============================================================
//  Mühendislik analizi render — backend'in döndürdüğü `analiz`
//  sözlüğünü (CG, denge, sıcaklık, titreşim, fiziksel) kartlara çevirir.
//  Eskiden bu analiz sadece terminale yazılıyordu.
// ============================================================
const STATUS_CLASS = { iyi: 'ok', guvenli: 'ok', orta: 'warn', sinir: 'warn', kotu: 'crit', risk: 'crit' };
const STATUS_LABEL = { iyi: 'İYİ', guvenli: 'GÜVENLİ', orta: 'DİKKAT', sinir: 'SINIRDA', kotu: 'KRİTİK', risk: 'RİSK' };

const sCls = d => STATUS_CLASS[d] || 'warn';
const sLbl = d => STATUS_LABEL[d] || '—';

function profileCard(title, p, sub) {
    let overall = 'guvenli';
    if (!p.motor_var) {
        overall = 'orta';
    } else if (p.ihlal) {
        overall = 'risk';
    } else if (p.parcalar.some(x => x.durum === 'sinir')) {
        overall = 'sinir';
    }

    let body;
    if (!p.motor_var) {
        body = `<div class="ac-empty">Motor bulunamadı — mesafe hesaplanamadı.</div>`;
    } else if (p.parcalar.length === 0) {
        body = `<div class="ac-empty">İlgili hassas parça yok.</div>`;
    } else {
        body = `<div class="ac-rows">` + p.parcalar.map(x =>
            `<div><span>${x.id}</span><span class="pill sm ${sCls(x.durum)}">${x.mesafe} cm</span></div>`
        ).join('') + `</div>`;
    }

    return `
        <div class="analysis-card">
            <div class="ac-head"><span class="ac-title">${title}</span><span class="pill ${sCls(overall)}">${sLbl(overall)}</span></div>
            <div class="ac-sub">${sub} · limit ${p.limit} cm</div>
            ${body}
        </div>`;
}

function renderAnalysis(a) {
    const area = document.getElementById('analysisArea');
    if (!area) return;
    if (!a) { area.innerHTML = '<div class="ac-empty">Analiz verisi yok.</div>'; return; }

    const cards = [];

    // Ağırlık merkezi
    cards.push(`
        <div class="analysis-card">
            <div class="ac-head"><span class="ac-title">Ağırlık Merkezi (CG)</span><span class="pill ${sCls(a.cg.durum)}">${sLbl(a.cg.durum)}</span></div>
            <div class="ac-main">${a.cg.sapma}<span class="ac-unit">cm sapma</span></div>
            <div class="ac-rows">
                <div><span>Konum X / Y / Z</span><span>${a.cg.x} / ${a.cg.y} / ${a.cg.z}</span></div>
                <div><span>Hedef X aralığı</span><span>${a.cg.hedef_x_min} – ${a.cg.hedef_x_max}</span></div>
            </div>
            <div class="ac-note">${a.cg.mesaj}</div>
        </div>`);

    // Denge / yakıt CG kayması
    cards.push(`
        <div class="analysis-card">
            <div class="ac-head"><span class="ac-title">Denge — Yakıt CG Kayması</span><span class="pill ${sCls(a.denge.durum)}">${sLbl(a.denge.durum)}</span></div>
            <div class="ac-main">${a.denge.kayma}<span class="ac-unit">cm drift</span></div>
            <div class="ac-rows">
                <div><span>CG (dolu depo)</span><span>${a.denge.cg_dolu_x} cm</span></div>
                <div><span>CG (boş depo)</span><span>${a.denge.cg_bos_x} cm</span></div>
                <div><span>Yakıt tankı X</span><span>${a.denge.yakit_x} cm</span></div>
            </div>
            <div class="ac-note">${a.denge.mesaj}</div>
        </div>`);

    // Sıcaklık ve titreşim profilleri
    cards.push(profileCard('Sıcaklık Profili', a.sicaklik, 'Motora mesafe'));
    cards.push(profileCard('Titreşim Profili', a.titresim, 'Motora mesafe'));

    // Fiziksel kontrol
    cards.push(`
        <div class="analysis-card">
            <div class="ac-head"><span class="ac-title">Fiziksel Kontrol</span><span class="pill ${sCls(a.fiziksel.durum)}">${sLbl(a.fiziksel.durum)}</span></div>
            <div class="ac-rows">
                <div><span>Parça çakışması</span><span>${a.fiziksel.cakisma ? 'VAR' : 'yok'}</span></div>
                <div><span>Gövdeden taşma</span><span>${a.fiziksel.tasma ? 'VAR' : 'yok'}</span></div>
                <div><span>Skor</span><span>${a.fiziksel.skor}</span></div>
            </div>
            <div class="ac-note">${a.fiziksel.mesaj}</div>
        </div>`);

    area.innerHTML = cards.join('');
}