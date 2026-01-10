import torch
from model import PhaseTower, PhaseMemory
from utils import get_mnist_loaders, mnist_gauss_noise
import matplotlib.pyplot as plt

def ood_demo(tau=0.2):
    ckpt = torch.load("checkpoints/phasetower_mnist.pt", map_location='cpu')
    model = PhaseTower(784, [(512, 0.08), (256, 0.05), (128, 0.03), (10, 1.0)])
    model.load_state_dict(ckpt["model"])
    memory = ckpt["memory"]

    # 1. clean
    _, test_loader = get_mnist_loaders(batch_size=1)
    # 2. bruité
    noisy_loader = torch.utils.data.DataLoader(
        mnist_gauss_noise(test_loader.dataset, sigma=0.5),
        batch_size=1, shuffle=False)

    def reject_or_predict(x):
        _, z = model(x)
        sim = torch.abs(memory.M @ z.conj())
        if sim.max() < tau:
            return "unknown"
        return str(memory.recall(z))

    # visualise 10 images bruitées
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i, (x, y) in enumerate(noisy_loader):
        if i >= 10:
            break
        pred = reject_or_predict(x)
        ax = axs[i // 5, i % 5]
        ax.imshow(x.view(28, 28), cmap='gray')
        ax.set_title(f"pred: {pred}")
        ax.axis('off')
    plt.suptitle("OOD rejection (tau=0.2) – red = unknown")
    plt.tight_layout()
    plt.savefig("ood_grid.png")
    print("Saved ood_grid.png")


if __name__ == "__main__":
    ood_demo()