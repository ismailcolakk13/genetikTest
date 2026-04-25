const initialComponents = [
    { id: "Motor", w: 40.0, d: [60, 40, 40], reg: "BURUN", lock: true, vib: false },
    { id: "Batarya_Ana", w: 15.0, d: [20, 15, 10], reg: "GOVDE", lock: false, vib: false },
    { id: "Aviyonik_1", w: 5.0, d: [15, 15, 5], reg: "GOVDE", lock: false, vib: true },
    { id: "Aviyonik_2", w: 5.0, d: [15, 15, 5], reg: "GOVDE", lock: false, vib: true },
    { id: "Payload_Kam", w: 10.0, d: [20, 20, 20], reg: "ON_ALT", lock: false, vib: true },
    { id: "Yakit_Tanki", w: 40.0, d: [50, 40, 30], reg: "MERKEZ", lock: false, vib: false },
    { id: "Servo_Kuyruk", w: 2.0, d: [5, 5, 5], reg: "KUYRUK", lock: false, vib: false },
];

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
    
    row.querySelector('.del-btn').addEventListener('click', () => {
        row.remove();
    });
    
    tbody.appendChild(row);
}

// Initial populate
initialComponents.forEach(c => addComponentRow(c));

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
        titresim_limiti: 50.0, // Optional ui mapping
        pop_size: parseInt(document.getElementById('pop_size').value),
        generations: parseInt(document.getElementById('generations').value),
        algoritma: document.getElementById('algoritma').value,
        komponentler: []
    };
    
    const rows = tbody.querySelectorAll('tr');
    rows.forEach(r => {
        const id = r.querySelector('.c-id').value;
        if(id.trim() === '') return;
        
        let lock = r.querySelector('.c-locked').checked;
        let cData = {
            id: id,
            agirlik: parseFloat(r.querySelector('.c-weight').value),
            boyut: [
                parseFloat(r.querySelector('.c-dx').value),
                parseFloat(r.querySelector('.c-dy').value),
                parseFloat(r.querySelector('.c-dz').value)
            ],
            sabit_bolge: r.querySelector('.c-region').value,
            kilitli: lock,
            titresim_hassasiyeti: r.querySelector('.c-vib').checked,
            sabit_pos: null
        };
        // Mocking fixed pos for nose if locked, for demonstration
        if (lock && cData.sabit_bolge === 'BURUN') {
            cData.sabit_pos = [30.0, 0.0, 0.0];
        }
        reqData.komponentler.push(cData);
    });
    
    try {
        const res = await fetch('http://127.0.0.1:8000/api/run-simulation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(reqData)
        });
        
        const data = await res.json();
        
        if (!res.ok) {
            alert('Error: ' + JSON.stringify(data));
        } else {
            // Update UI Results
            document.getElementById('resFitness').textContent = data.en_iyi_skor.toFixed(4);
            document.getElementById('resCG').textContent = `[${data.en_iyi_cg.x}, ${data.en_iyi_cg.y}, ${data.en_iyi_cg.z}]`;
            document.getElementById('resAlgo').textContent = data.algoritma_ismi;
            
            // Log components
            const logBox = document.getElementById('resultLog');
            logBox.innerHTML = '';
            Object.values(data.tasarim).forEach(k => {
                const line = document.createElement('div');
                line.textContent = `> ${k.id} placed at (${k.pos_x.toFixed(1)}, ${k.pos_y.toFixed(1)}, ${k.pos_z.toFixed(1)}) in ${k.sabit_bolge}`;
                logBox.appendChild(line);
            });
            
            resultsArea.style.display = 'flex';
            // Scroll to bottom smoothly since component pane shrinks
            resultsArea.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (e) {
        console.error(e);
        alert('Network Error! Is the backend running?');
    }
    
    btn.disabled = false;
    btn.innerHTML = '<ion-icon name="play-outline"></ion-icon> Run Optimization Sequence';
    loading.style.display = 'none';
});
