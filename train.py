import torch, time, argparse
from model import PhaseTower, PhaseMemory
from utils import get_mnist_loaders, mnist_rotation, mnist_gauss_noise
from tqdm import tqdm

def eval_model(model, loader, desc="test"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            logits, _ = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def main(args):
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    layers = [(512, 0.08), (256, 0.05), (128, 0.03), (10, 1.0)]
    model = PhaseTower(784, layers, num_classes=10)
    memory = PhaseMemory(max_mem=10000, dim=model.codebook.shape[1], k=5)

    # ---------- 1 epoch ----------
    model.train()
    t0 = time.time()
    for x, y in tqdm(train_loader, desc="train"):
        for xi, yi in zip(x, y):
            logits, z = model(xi)
            model.forward_with_target(xi, yi, eta=0.1)
            memory.memorize(z, yi.item())
    train_time = time.time() - t0

    # ---------- tests ----------
    acc_clean = eval_model(model, test_loader, "test clean")
    acc_rot   = eval_model(model, mnist_rotation(test_loader.dataset), "test rotated")
    acc_noise = eval_model(model, mnist_gauss_noise(test_loader.dataset), "test noisy")

    print(f"Epoch 1 done in {train_time:.1f} s")
    print(f"Clean accuracy : {acc_clean:.2%}")
    print(f"Rotated        : {acc_rot:.2%}")
    print(f"Noisy          : {acc_noise:.2%}")

    # ---------- composition demo ----------
    from random import randint
    idx1 = randint(0, 100)          # exemple classe 1
    idx3 = randint(0, 100)          # exemple classe 3
    z13 = memory.compose(memory.ptr-100+idx1, memory.ptr-100+idx3, 13)
    if z13 is not None:
        pred = memory.recall(z13)
        print("Composition 1+3 -> label 13 , recall =", pred)

    # sauvegarde
    torch.save({"model": model.state_dict(), "memory": memory}, "checkpoints/phasetower_mnist.pt")
    print("Saved to checkpoints/phasetower_mnist.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args)