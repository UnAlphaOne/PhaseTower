import torch, time, os, psutil
from model import PhaseTower
from utils import get_mnist_loaders

def rapl_energy_mJ():
    """lit l'énergie CPU via RAPL (Intel) – retourne mJ"""
    try:
        # package-0
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            uj_before = int(f.read())
        return uj_before / 1_000        # µJ → mJ
    except Exception as e:
        print("RAPL non disponible", e)
        return None

def bench(n=100):
    _, test_loader = get_mnist_loaders(batch_size=1)
    model = PhaseTower(784, [(512, 0.08), (256, 0.05), (128, 0.03), (10, 1.0)])
    model.eval()

    data_iter = iter(test_loader)
    # warm-up
    for _ in range(10):
        x, _ = next(data_iter)
        with torch.no_grad():
            _ = model(x)

    # mesures
    times, energies = [], []
    for _ in range(n):
        x, _ = next(data_iter)
        t0 = time.perf_counter()
        e0 = rapl_energy_mJ()
        with torch.no_grad():
            _ = model(x)
        t1 = time.perf_counter()
        e1 = rapl_energy_mJ()
        times.append((t1 - t0) * 1e3)      # ms
        if e0 and e1:
            energies.append(e1 - e0)       # mJ

    print(f"PhaseTower – {n} inférences")
    print(f"Temps moyen  : {sum(times)/n:.3f} ms")
    print(f"Énergie moy. : {sum(energies)/n:.3f} mJ" if energies else "RAPL indisponible")
    print(f"CPU usage    : {psutil.cpu_percent()} %")


if __name__ == "__main__":
    bench(200)