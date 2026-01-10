import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

# ------------------------- PhaseBlock -------------------------
class PhaseBlock:
    """
    Bloc résiduel complexe
      - K : nombre de neurones
      - sparsity : fraction active
      - d_in : dimension d'entrée (complexe ou réel)
    """
    def __init__(self, d_in, K, sparsity):
        self.K = K
        self.active = max(1, int(sparsity * K))
        self.P = torch.randn(K, d_in) / math.sqrt(d_in)   # directions
        self.P = F.normalize(self.P, dim=1)
        self.phi = torch.zeros(K)                         # phases
        self.lam = torch.complex(torch.tensor(1.0), torch.tensor(0.0))

    def forward(self, z_in, target=None, eta=0.1):
        """
        z_in : vecteur complexe (B, d_in)  ou  (d_in,)
        retourne z_out complexe (B, K_out) ou (K_out,)
        si target est fourni -> mise-à-jour locale
        """
        single = z_in.dim() == 1
        if single:
            z_in = z_in.unsqueeze(0)          # (1, d_in)

        B, d = z_in.shape
        scores = (z_in.real @ self.P.real.T + z_in.imag @ self.P.imag.T)  # (B, K)
        val, idx = scores.topk(self.active, dim=1)                         # (B, active)

        # construction du phasor actif
        z_active = val * torch.exp(1j * self.phi[idx])                     # (B, active)

        # saut résiduel
        K_out = min(self.K, z_in.size(1))
        z_residual = self.lam * z_in[:, :K_out]                            # (B, K_out)
        z_out = z_active[:, :K_out] + z_residual                           # (B, K_out)
        z_out = F.normalize(z_out, p=2, dim=1) * math.sqrt(K_out)

        # mise-à-jour locale si cible fournie
        if target is not None:
            delta = target.unsqueeze(0) - z_out          # (B, K_out)
            # rotation
            self.phi.scatter_add_(0, idx[:, :K_out].reshape(-1),
                                  eta * torch.atan2(delta.imag, delta.real).reshape(-1))
            # déplacement radial
            upd = eta * delta.real.norm(dim=0, keepdim=True).T * self.P[idx[:, :K_out]]
            self.P[idx[:, :K_out]] += upd

        return z_out.squeeze(0) if single else z_out


# ------------------------- PhaseTower -------------------------
class PhaseTower:
    def __init__(self, input_dim, layers_cfg, num_classes=10):
        # layers_cfg : liste de (K, sparsity)
        self.blocks = []
        d = input_dim
        for K, sparsity in layers_cfg:
            self.blocks.append(PhaseBlock(d, K, sparsity))
            d = K
        self.codebook = torch.complex(torch.randn(num_classes, d), torch.randn(num_classes, d))
        self.codebook = F.normalize(self.codebook, dim=1)

    def forward(self, x):
        z = torch.complex(x, torch.zeros_like(x))
        for blk in self.blocks:
            z = blk.forward(z)
        logits = (z.unsqueeze(0) @ self.codebook.conj().T).real.squeeze()
        return logits, z

    def forward_with_target(self, x, y, eta=0.1):
        """1 forward + 1 backward cible-complexe"""
        logits, z = self.forward(x)
        # cible complexe : one-hot dans la direction de la classe
        target = self.codebook[y]
        # rétro-projection cible-complexe
        z_in = torch.complex(x, torch.zeros_like(x))
        for blk in reversed(self.blocks):
            target = blk.forward(z_in, target=target, eta=eta)
            z_in = z_in[:blk.blocks[0].K if hasattr(blk,'blocks') else blk.K]
        return logits


# ------------------------- PhaseMemory -------------------------
class PhaseMemory:
    def __init__(self, max_mem=10000, dim=10, k=5, alpha=0.01):
        self.M = torch.zeros(max_mem, dim, dtype=torch.cfloat)
        self.labels = torch.zeros(max_mem, dtype=torch.long)
        self.v = torch.zeros(max_mem)
        self.ptr = 0
        self.max_mem = max_mem
        self.k = k
        self.alpha = alpha

    def memorize(self, z, y):
        idx = self.ptr % self.max_mem
        self.M[idx] = z.detach().clone()
        self.labels[idx] = y
        self.v[idx] = 1.0
        self.ptr += 1

    def recall(self, z):
        sim = torch.abs(self.M @ z.conj())
        val, idx = sim.topk(self.k, dim=0)
        self.v[idx] += val - self.alpha
        return self.labels[idx].mode().values.item()

    def compose(self, idx1, idx2, new_label, tau=0.7):
        z1 = self.M[idx1]
        z2 = self.M[idx2]
        z_plus = (z1 + z2) / (z1 + z2).norm()
        if torch.abs(self.M @ z_plus.conj()).max() < tau:
            self.memorize(z_plus, new_label)
            return z_plus
        return None