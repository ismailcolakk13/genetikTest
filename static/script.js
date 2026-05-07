// ─── Initial Components ───
const initialComponents = [
    { id: "Motor",        w: 40.0, d: [60, 40, 40], reg: "BURUN",  lock: true,  vib: false, temp: false },
    { id: "Batarya_Ana",  w: 15.0, d: [20, 15, 10], reg: "GOVDE",  lock: false, vib: false, temp: true  },
    { id: "Aviyonik_1",   w: 5.0,  d: [15, 15, 5],  reg: "GOVDE",  lock: false, vib: true,  temp: true  },
    { id: "Aviyonik_2",   w: 5.0,  d: [15, 15, 5],  reg: "GOVDE",  lock: false, vib: true,  temp: true  },
    { id: "Payload_Kam",  w: 10.0, d: [20, 20, 20], reg: "TABAN",  lock: false, vib: true,  temp: false },
    { id: "Yakit_Tanki",  w: 40.0, d: [50, 40, 30], reg: "GOVDE",  lock: false, vib: false, temp: false },
    { id: "Servo_Kuyruk", w: 2.0,  d: [5, 5, 5],   reg: "KUYRUK", lock: false, vib: false, temp: false },
];

const tbody = document.getElementById('componentsBody');
const template = document.getElementById('compRowTemplate');

function updateCount() {
    const count = tbody.querySelectorAll('tr').length;
    document.getElementById('compCount').textContent = count;
}

function addComponentRow(data) {
    const clone = template.content.cloneNode(true);
    const row = clone.querySelector('tr');

    row.querySelector('.c-id').value     = data.id || 'New_Part';
    row.querySelector('.c-weight').value = data.w  || 10;
    row.querySelector('.c-dx').value     = data.d ? data.d[0] : 10;
    row.querySelector('.c-dy').value     = data.d ? data.d[1] : 10;
    row.querySelector('.c-dz').value     = data.d ? data.d[2] : 10;
    row.querySelector('.c-region').value = data.reg || 'GOVDE';
    row.querySelector('.c-locked').checked = data.lock || false;
    row.querySelector('.c-vib').checked    = data.vib  || false;
    row.querySelector('.c-temp').checked   = data.temp || false;

    row.querySelector('.del-btn').addEventListener('click', () => {
        row.style.animation = 'fadeIn 0.2s ease reverse';
        setTimeout(() => { row.remove(); updateCount(); }, 200);
    });

    tbody.appendChild(row);
    updateCount();
}

// Populate initial
initialComponents.forEach(c => addComponentRow(c));

// Add new
document.getElementById('addCompBtn').addEventListener('click', () => {
    addComponentRow({});
    // Scroll to bottom of table
    const tableWrap = document.querySelector('.table-wrap');
    if (tableWrap) tableWrap.scrollTop = tableWrap.scrollHeight;
});

// ─── Tab switching ───
document.querySelectorAll('.tab-item').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab-item').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
    });
});

// ─── Run Optimization ───
document.getElementById('runBtn').addEventListener('click', async () => {
    const btn       = document.getElementById('runBtn');
    const progress  = document.getElementById('progressWrap');
    const results   = document.getElementById('resultsArea');
    const resTab    = document.getElementById('resultsTab');
    const startTime = performance.now();

    // Loading state
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Optimizing…';
    progress.classList.add('active');
    results.classList.remove('visible');

    // Collect form data
    const reqData = {
        govde_uzunluk:     parseFloat(document.getElementById('govde_uzunluk').value),
        govde_cap:         parseFloat(document.getElementById('govde_cap').value),
        target_cg_x_min:   parseFloat(document.getElementById('target_cg_x_min').value),
        target_cg_x_max:   parseFloat(document.getElementById('target_cg_x_max').value),
        target_cg_y:       parseFloat(document.getElementById('target_cg_y').value),
        target_cg_z:       parseFloat(document.getElementById('target_cg_z').value),
        max_yakit_agirligi: parseFloat(document.getElementById('max_yakit_agirligi').value),
        titresim_limiti:   parseFloat(document.getElementById('titresim_limiti').value),
        sicaklik_limiti:   parseFloat(document.getElementById('sicaklik_limiti').value),
        pop_size:          parseInt(document.getElementById('pop_size').value),
        generations:       parseInt(document.getElementById('generations').value),
        algoritma:         document.getElementById('algoritma').value,
        komponentler:      []
    };

    const rows = tbody.querySelectorAll('tr');
    rows.forEach(r => {
        const id = r.querySelector('.c-id').value;
        if (id.trim() === '') return;

        let lock = r.querySelector('.c-locked').checked;
        let cData = {
            id,
            agirlik: parseFloat(r.querySelector('.c-weight').value),
            boyut: [
                parseFloat(r.querySelector('.c-dx').value),
                parseFloat(r.querySelector('.c-dy').value),
                parseFloat(r.querySelector('.c-dz').value)
            ],
            sabit_bolge: r.querySelector('.c-region').value,
            kilitli: lock,
            titresim_hassasiyeti: r.querySelector('.c-vib').checked,
            sicaklik_hassasiyeti: r.querySelector('.c-temp').checked,
            sabit_pos: null
        };

        if (lock && cData.sabit_bolge === 'BURUN') {
            cData.sabit_pos = [30.0, 0.0, 0.0];
        }
        reqData.komponentler.push(cData);
    });

    try {
        const res = await fetch('/api/run-simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqData)
        });

        const data = await res.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        if (!res.ok) {
            showError(JSON.stringify(data.detail || data));
        } else {
            // Update metrics
            document.getElementById('resFitness').textContent = data.en_iyi_skor.toFixed(4);
            document.getElementById('resCG').textContent =
                `(${data.en_iyi_cg.x}, ${data.en_iyi_cg.y}, ${data.en_iyi_cg.z})`;
            document.getElementById('resAlgo').textContent = data.algoritma_ismi;

            // Build log
            const logBox = document.getElementById('resultLog');
            logBox.innerHTML = '';

            // Summary line
            addLogLine(logBox, `✅ Optimization completed in ${elapsed}s`);
            addLogLine(logBox, `📊 Population: ${reqData.pop_size} | Generations: ${reqData.generations}`);
            addLogLine(logBox, `─`.repeat(50));

            Object.values(data.tasarim).forEach((k, i) => {
                setTimeout(() => {
                    addLogLine(logBox,
                        `📦 ${k.id} → (${k.pos_x.toFixed(1)}, ${k.pos_y.toFixed(1)}, ${k.pos_z.toFixed(1)}) ∈ ${k.sabit_bolge} [${k.agirlik}kg]`
                    );
                }, i * 80);
            });

            // Show results
            results.classList.add('visible');
            resTab.style.display = 'flex';

            // Smooth scroll
            setTimeout(() => {
                results.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 300);
        }
    } catch (e) {
        console.error(e);
        showError('Network Error — Is the backend running at port 8000?');
    }

    // Reset button
    btn.disabled = false;
    btn.innerHTML = '<ion-icon name="play-outline"></ion-icon> Run Optimization';
    progress.classList.remove('active');
});

function addLogLine(container, text) {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = `<span class="dot"></span> ${text}`;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
}

function showError(msg) {
    const results = document.getElementById('resultsArea');
    results.classList.add('visible');
    document.getElementById('statusBadge').className = 'badge badge-warning';
    document.getElementById('statusBadge').textContent = 'Error';
    document.getElementById('resFitness').textContent = '—';
    document.getElementById('resCG').textContent = '—';
    document.getElementById('resAlgo').textContent = '—';
    const logBox = document.getElementById('resultLog');
    logBox.innerHTML = '';
    addLogLine(logBox, `❌ ${msg}`);
}
