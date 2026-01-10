import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

# ------------------------- PhaseBlock -------------------------
class PhaseBlock(nn.Module):
    """
    Bloc résiduel complexe
      - K : nombre de neurones
      - sparsity : fraction active
      - d_in : dimension d'entrée (complexe ou réel)
    """
    def __init__(self, d_in, K, sparsity):
        super().__init__()
        self.K = K
        self.d_in = d_in
        self.active = max(1, int(sparsity * K))
        
        # Initialisation d'un tenseur COMPLEXE
        P_real = torch.randn(K, d_in) / math.sqrt(d_in)
        P_imag = torch.randn(K, d_in) / math.sqrt(d_in)
        P_init = torch.complex(P_real, P_imag)
        P_init = F.normalize(P_init, dim=1)
        self.P = nn.Parameter(P_init)   # directions (complexe)
        
        self.phi = nn.Parameter(torch.zeros(K))                         # phases
        self.lam = nn.Parameter(torch.complex(torch.tensor(1.0), torch.tensor(0.0)))
        
        # Buffers pour les mises à jour
        self.register_buffer('_phi_updates', torch.zeros(K))
        self.register_buffer('_P_updates', torch.zeros_like(P_init))

    def forward(self, z_in, target=None, eta=0.1):
        """
        z_in : vecteur complexe (B, d_in)  ou  (d_in,)
        retourne z_out complexe (B, K) ou (K,)
        si target est fourni -> mise-à-jour locale
        """
        single = z_in.dim() == 1
        if single:
            z_in = z_in.unsqueeze(0)          # (1, d_in)

        # Vérifier si z_in est complexe, sinon le convertir
        if not torch.is_complex(z_in):
            z_in = torch.complex(z_in, torch.zeros_like(z_in))

        B, d = z_in.shape
        
        # Normalisation des poids P à chaque forward
        P_norm = F.normalize(self.P, dim=1)
        
        # Calcul des scores avec produit scalaire complexe
        scores = torch.real(z_in @ P_norm.conj().T)  # (B, K)
        
        val, idx = scores.topk(self.active, dim=1)   # (B, active)

        # construction du phasor actif - créer un tenseur de taille (B, K)
        z_active_full = torch.zeros(B, self.K, dtype=torch.cfloat, device=z_in.device)
        
        # Remplir les neurones actifs
        for b in range(B):
            active_values = val[b] * torch.exp(1j * self.phi[idx[b]])
            z_active_full[b, idx[b]] = active_values
        
        # saut résiduel - adapter la dimension si nécessaire
        if d >= self.K:
            z_residual = self.lam * z_in[:, :self.K]
        else:
            # Si la dimension d'entrée est plus petite que K, on pad avec des zéros
            padding = torch.zeros(B, self.K - d, dtype=torch.cfloat, device=z_in.device)
            z_residual = self.lam * torch.cat([z_in, padding], dim=1)
        
        z_out = z_active_full + z_residual                           # (B, K)
        z_out = F.normalize(z_out, p=2, dim=1) * math.sqrt(self.K)

        # mise-à-jour locale si cible fournie
        if target is not None:
            # S'assurer que target a la bonne dimension
            if target.dim() == 1:
                target = target.unsqueeze(0)
            
            # Adapter la dimension de target si nécessaire
            target_B, target_dim = target.shape
            if target_dim != self.K:
                if target_dim > self.K:
                    target = target[:, :self.K]
                else:
                    padding = torch.zeros(target_B, self.K - target_dim, 
                                         dtype=torch.cfloat, device=target.device)
                    target = torch.cat([target, padding], dim=1)
            
            delta = target - z_out          # (B, K)
            
            # Réinitialiser les buffers de mise à jour
            self._phi_updates.zero_()
            self._P_updates.zero_()
            
            # Accumuler les mises à jour
            for b in range(B):
                # Mise à jour des phases
                phase_updates = eta * torch.atan2(
                    delta[b, idx[b]].imag, 
                    delta[b, idx[b]].real
                )
                self._phi_updates.index_add_(0, idx[b], phase_updates)
                
                # Mise à jour des poids P
                delta_norm = torch.norm(delta[b, idx[b]].real, dim=0, keepdim=True)
                upd = eta * delta_norm * self.P.data[idx[b]]
                self._P_updates.index_add_(0, idx[b], upd)
            
            # Appliquer les mises à jour (hors graphe de calcul)
            with torch.no_grad():
                self.phi.data += self._phi_updates
                self.P.data += self._P_updates

        return z_out.squeeze(0) if single else z_out


# ------------------------- PhaseTower -------------------------
class PhaseTower(nn.Module):
    def __init__(self, input_dim, layers_cfg, num_classes=10):
        super().__init__()
        # layers_cfg : liste de (K, sparsity)
        self.blocks = nn.ModuleList()
        d = input_dim
        for K, sparsity in layers_cfg:
            self.blocks.append(PhaseBlock(d, K, sparsity))
            d = K
        
        # Initialisation et normalisation du codebook
        codebook_init = torch.complex(
            torch.randn(num_classes, d), 
            torch.randn(num_classes, d)
        )
        codebook_init = F.normalize(codebook_init, dim=1)
        self.codebook = nn.Parameter(codebook_init)

    def forward(self, x):
        # Convertir l'entrée en tenseur complexe
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        
        z = x
        for blk in self.blocks:
            z = blk.forward(z)
        
        # Calcul des logits (produit scalaire complexe)
        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1, d)
        
        # Produit scalaire complexe : z @ codebook.conj().T
        logits = torch.real(z @ self.codebook.conj().T)
        
        if logits.size(0) == 1:
            logits = logits.squeeze(0)
        
        return logits, z

    def forward_with_target(self, x, y, eta=0.1):
        """1 forward + 1 backward cible-complexe"""
        logits, z = self.forward(x)
        
        # Mode évaluation pour éviter les opérations in-place problématiques
        self.eval()
        
        # cible complexe : one-hot dans la direction de la classe
        target = self.codebook[y]
        
        # Stocker les entrées intermédiaires pendant le forward
        intermediate_inputs = []
        current_input = x
        if not torch.is_complex(current_input):
            current_input = torch.complex(current_input, torch.zeros_like(current_input))
        
        for blk in self.blocks:
            intermediate_inputs.append(current_input)
            # Forward sans mise à jour
            with torch.no_grad():
                current_input = blk.forward(current_input)
        
        # Rétro-propagation cible-complexe
        current_target = target
        
        # Si target est 1D, le convertir en 2D
        if current_target.dim() == 1:
            current_target = current_target.unsqueeze(0)
        
        # Parcourir les blocs en sens inverse
        for i in range(len(self.blocks) - 1, -1, -1):
            blk = self.blocks[i]
            z_input = intermediate_inputs[i]
            
            # Vérifier et adapter les dimensions
            if current_target.dim() == 1:
                current_target = current_target.unsqueeze(0)
            
            target_dim = current_target.shape[1] if current_target.dim() > 1 else current_target.shape[0]
            
            if target_dim != blk.K:
                if target_dim > blk.K:
                    # Si trop grand, prendre seulement les premières dimensions
                    current_target = current_target[:, :blk.K]
                else:
                    # Si trop petit, pad avec des zéros
                    if current_target.dim() == 1:
                        current_target = current_target.unsqueeze(0)
                    
                    padding_size = blk.K - current_target.shape[1]
                    padding = torch.zeros(current_target.shape[0], padding_size,
                                         dtype=torch.cfloat, device=current_target.device)
                    current_target = torch.cat([current_target, padding], dim=1)
            
            # Mode entraînement pour ce bloc seulement
            blk.train()
            current_target = blk.forward(z_input, target=current_target, eta=eta)
        
        # Retour en mode entraînement
        self.train()
        
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
        # Si z est 1D, le convertir en 2D pour le produit matriciel
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Vérifier si M n'est pas vide
        if self.ptr == 0:
            return -1
        
        sim = torch.abs(self.M @ z.conj().T).squeeze()  # (max_mem,)
        k_val = min(self.k, len(sim))
        val, idx = sim.topk(k_val, dim=0)
        
        if val.numel() > 0:
            self.v[idx] += val - self.alpha
        
        if len(idx) > 0:
            return self.labels[idx].mode().values.item()
        else:
            return -1  # Valeur par défaut si pas de rappel

    def compose(self, idx1, idx2, new_label, tau=0.7):
        if self.ptr == 0:
            return None
            
        z1 = self.M[idx1]
        z2 = self.M[idx2]
        z_plus = (z1 + z2) / (torch.norm(z1 + z2) + 1e-8)
        
        if self.M.shape[0] > 0:
            sim = torch.abs(self.M @ z_plus.conj())
            if sim.max() < tau:
                self.memorize(z_plus, new_label)
                return z_plus
        else:
            self.memorize(z_plus, new_label)
            return z_plus
        
        return None