import torch
from model import PhaseTower, PhaseMemory
from utils import get_mnist_loaders
from torchvision.datasets import MNIST
import torchvision.transforms as T

# charge le modèle sauvegardé
ckpt = torch.load("checkpoints/phasetower_mnist.pt", map_location='cpu')
model = PhaseTower(784, [(512, 0.08), (256, 0.05), (128, 0.03), (10, 1.0)])
model.load_state_dict(ckpt["model"])
memory = ckpt["memory"]

# récupère deux images : 1 et 3
test_set = MNIST("./data", train=False, download=False,
                 transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))]))
idx1 = next(i for i, (_,y) in enumerate(test_set) if y == 1)
idx3 = next(i for i, (_,y) in enumerate(test_set) if y == 3)
x1, y1 = test_set[idx1]
x3, y3 = test_set[idx3]

# mémorise
_, z1 = model(x1)
_, z3 = model(x3)
memory.memorize(z1, 1)
memory.memorize(z3, 3)

# création classe 13
z13 = memory.compose(memory.ptr-2, memory.ptr-1, 13)
print("Composition 1+3 créée :", z13 is not None)

# test sur image « 13 » artificielle
x13 = (x1 + x3) / 2
_, z13_test = model(x13)
pred = memory.recall(z13_test)
print("Prédiction image 13 :", pred)