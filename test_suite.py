import subprocess
import re

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
        process = subprocess.Popen(['python3', 'main.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=f"{algo_id}\n")
        
        scores = re.findall(r'Skor: ([-]?\d+)', stdout)
        statuses = re.findall(r'(KABUL EDİLEBİLİR|ZAYIF)', stdout)
        
        score = int(scores[-1]) if scores else 0
        status = statuses[-1] if statuses else "BİLİNMİYOR"
        results[algo_id].append((score, status))
        print(f"{algo_name} Run {i+1}: Skor={score}, Durum={status}")

print("\n--- ÖZET ---")
for algo_id, algo_name in algorithms.items():
    scores = [s[0] for s in results[algo_id]]
    statuses = [s[1] for s in results[algo_id]]
    avg_score = sum(scores) / len(scores)
    kabul_count = statuses.count('KABUL EDİLEBİLİR')
    zayif_count = statuses.count('ZAYIF')
    print(f"{algo_name}: Ortalama Skor={avg_score:.1f} | KABUL={kabul_count}/10 | ZAYIF={zayif_count}/10 | Skorlar={scores}")
