import subprocess
import sys
import os
import re
import csv
from datetime import datetime

class Logger(object):
    def __init__(self, filename="test_results.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.log.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test Çalıştırması Başladı\n")
        self.log.write("-" * 50 + "\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

algorithms = {
    1: "GA",
    2: "PSO",
    3: "NSGA-II",
    4: "Hybrid"
}

results = {1: [], 2: [], 3: [], 4: []}

print("Test başlıyor...")
for algo_id, algo_name in algorithms.items():
    for i in range(10):
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['NO_VIZ'] = '1'
        process = subprocess.Popen(
            [sys.executable, 'main.py'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', env=env
        )
        stdout, stderr = process.communicate(input=f"{algo_id}\n")

        scores = re.findall(r'Skor: ([-]?\d+)', stdout)
        # Match all three quality levels printed by analiz_yap
        statuses = re.findall(r'(ÇOK İYİ|KABUL EDİLEBİLİR|ZAYIF)', stdout)

        if not scores:
            print(f"  [UYARI] {algo_name} Run {i+1}: Skor bilgisi bulunamadı. Stderr:\n{stderr[:500]}")

        score = int(scores[-1]) if scores else 0
        status = statuses[-1] if statuses else "BİLİNMİYOR"
        
        heat_status = "RISK" if "SICAKLIK RİSKİ!" in stdout else "OK"
        vibration_status = "RISK" if "TİTREŞİM RİSKİ!" in stdout else "OK"
        
        results[algo_id].append((score, status))
        print(f"{algo_name} Run {i+1}: Skor={score}, Durum={status}, Heat={heat_status}, Vib={vibration_status}")

        csv_filename = "test_results.csv"
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["Date", "Algorithm", "Run", "Score", "Status", "Heat", "Vibration"])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), algo_name, i+1, score, status, heat_status, vibration_status])

print("\n--- ÖZET ---")
for algo_id, algo_name in algorithms.items():
    run_data = results[algo_id]
    if not run_data:
        print(f"{algo_name}: Sonuç yok")
        continue
    scores = [s[0] for s in run_data]
    statuses = [s[1] for s in run_data]
    avg_score = sum(scores) / len(scores)
    # ÇOK İYİ and KABUL EDİLEBİLİR both count as physically valid designs
    iyi_count   = statuses.count('ÇOK İYİ')
    kabul_count = statuses.count('KABUL EDİLEBİLİR')
    zayif_count = statuses.count('ZAYIF')
    bilinmiyor  = statuses.count('BİLİNMİYOR')
    gecerli     = iyi_count + kabul_count
    print(f"{algo_name}: Ort Skor={avg_score:+.1f} | "
          f"ÇOK İYİ={iyi_count}/10 | KABUL={kabul_count}/10 | "
          f"ZAYIF={zayif_count}/10 | BİLİNMİYOR={bilinmiyor}/10 | "
          f"Geçerli={gecerli}/10 | Skorlar={scores}")
