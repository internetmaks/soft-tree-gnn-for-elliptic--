"""Modele: różniczkowalne miękkie drzewo decyzyjne oraz hybryda End-to-End GNN + Soft Tree."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

EMBED_DIM = 100


class HeadChunkedGATConv(GATConv):
    """GATConv liczący głowice uwagi w paczkach (z checkpointingiem każdej paczki).

    Matematycznie identyczny z `GATConv` (te same parametry), ale nie materializuje naraz
    tensora wiadomości [E, heads, C] – przy 8 głowicach i ~440 tys. krawędzi to >1,4 GB
    na tensor, co przekracza pamięć GPU (np. 16 GB w Radeon RX 6950 XT) przy treningu
    full-batch. Obsługuje wejście `x` (Tensor) bez cech krawędzi.
    """

    def __init__(self, *args, head_chunk: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_chunk = max(1, head_chunk)

    @staticmethod
    def _attend(xp, a_src, a_dst, src, dst, num_nodes, negative_slope, dropout, training):
        # xp: [N, h, C]; a_src/a_dst: [N, h]
        alpha = F.leaky_relu(a_src[src] + a_dst[dst], negative_slope)
        alpha = softmax(alpha, dst, num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=dropout, training=training)
        msg = xp[src] * alpha.unsqueeze(-1)
        return torch.zeros(num_nodes, xp.size(1), xp.size(2), device=xp.device,
                           dtype=xp.dtype).index_add_(0, dst, msg)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        H, C, N = self.heads, self.out_channels, x.size(0)
        xp = self.lin(x).view(N, H, C)
        a_src = (xp * self.att_src).sum(dim=-1)
        a_dst = (xp * self.att_dst).sum(dim=-1)
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst = edge_index[0], edge_index[1]

        outs = []
        use_ckpt = self.training and torch.is_grad_enabled()
        for s in range(0, H, self.head_chunk):
            e = min(s + self.head_chunk, H)
            args = (xp[:, s:e], a_src[:, s:e], a_dst[:, s:e], src, dst, N,
                    self.negative_slope, self.dropout, self.training)
            outs.append(checkpoint(self._attend, *args, use_reentrant=False)
                        if use_ckpt else self._attend(*args))
        out = torch.cat(outs, dim=1)
        out = out.reshape(N, H * C) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out


class DifferentiableSoftDecisionTree(nn.Module):
    """Miękkie drzewo decyzyjne (Frosst & Hinton) z regularyzacją przeciw zapaści węzłów.

    Węzły wewnętrzne (2^d - 1) są indeksowane poziomami (BFS): węzeł o pozycji j na
    głębokości k ma indeks 2^k - 1 + j. Liście (2^d) są indeksowane tak, że kolejne bity
    indeksu (od najstarszego) oznaczają wybory lewo (0) / prawo (1) na kolejnych poziomach.
    """

    def __init__(self, in_features: int, num_classes: int = 2, depth: int = 4,
                 penalty_lambda: float = 0.05):
        super().__init__()
        self.depth = depth
        self.num_internal = 2**depth - 1
        self.num_leaves = 2**depth
        self.penalty_lambda = penalty_lambda

        # Filtry w_i i obciążenia b_i oraz wyuczalna odwrotna temperatura beta
        self.inner_nodes = nn.Linear(in_features, self.num_internal)
        self.beta = nn.Parameter(torch.ones(self.num_internal))
        # Logity rozkładów klas w liściach (Q^l = softmax)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.1)

        depth_weights, ema_decays = [], []
        for d in range(depth):
            num_nodes_at_d = 2**d
            depth_weights.extend([2.0**(-d)] * num_nodes_at_d)
            # Okno czasowe EMA rośnie wykładniczo z głębokością
            decay = 1.0 - (0.1 * (0.5**d))
            ema_decays.extend([decay] * num_nodes_at_d)

        self.register_buffer('depth_weights', torch.tensor(depth_weights, dtype=torch.float32))
        self.register_buffer('ema_decays', torch.tensor(ema_decays, dtype=torch.float32))
        self.register_buffer('ema_alpha', torch.full((self.num_internal,), 0.5, dtype=torch.float32))

    def forward(self, z: torch.Tensor):
        """Zwraca (leaf_path_probs [B,2^d], leaf_q [2^d,C], internal_path_probs [B,2^d-1], p_right [B,2^d-1])."""
        batch_size = z.size(0)
        logits = self.inner_nodes(z) * self.beta
        p_right = torch.sigmoid(logits)
        p_left = 1.0 - p_right

        path_probs = torch.ones(batch_size, 1, device=z.device)
        internal_path_probs_list = []

        start_idx = 0
        for d in range(self.depth):
            num_nodes_at_d = 2**d
            internal_path_probs_list.append(path_probs)
            p_r = p_right[:, start_idx: start_idx + num_nodes_at_d].unsqueeze(2)
            p_l = p_left[:, start_idx: start_idx + num_nodes_at_d].unsqueeze(2)
            path_probs = path_probs.unsqueeze(2) * torch.cat([p_l, p_r], dim=2)
            path_probs = path_probs.view(batch_size, -1)
            start_idx += num_nodes_at_d

        internal_path_probs = torch.cat(internal_path_probs_list, dim=1)
        leaf_q = F.softmax(self.leaf_logits, dim=-1)
        return path_probs, leaf_q, internal_path_probs, p_right

    def compute_loss(self, leaf_path_probs: torch.Tensor, leaf_q: torch.Tensor,
                     internal_path_probs: torch.Tensor, p_right: torch.Tensor,
                     targets: torch.Tensor, class_weights: torch.Tensor,
                     reg_internal_path_probs: torch.Tensor = None,
                     reg_p_right: torch.Tensor = None) -> torch.Tensor:
        """Ważona oczekiwana entropia krzyżowa na liściach + kara przeciw zapaści węzłów.

        Jeżeli podano `reg_*` (np. wszystkie węzły okresu treningowego, w tym `unknown`),
        służą one do wyznaczenia alpha_i (regularyzacja semi-supervised); w przeciwnym
        razie używane są tensory węzłów oznaczonych.
        """
        eps = 1e-8
        # Porównanie zamiast F.one_hot (scatter nieobsługiwany przez DirectML)
        classes = torch.arange(leaf_q.size(-1), device=targets.device)
        target_one_hot = (targets.unsqueeze(1) == classes.unsqueeze(0)).float()
        weighted_target = target_one_hot * class_weights.unsqueeze(0)

        log_q = torch.log(leaf_q + eps)
        leaf_ce = -torch.matmul(weighted_target, log_q.t())
        cls_loss = torch.sum(leaf_path_probs * leaf_ce, dim=1).mean()

        P_in = reg_internal_path_probs if reg_internal_path_probs is not None else internal_path_probs
        p_r = reg_p_right if reg_p_right is not None else p_right

        denom = torch.sum(P_in, dim=0) + eps
        current_alpha = torch.sum(P_in * p_r, dim=0) / denom

        if self.training:
            with torch.no_grad():
                self.ema_alpha.mul_(self.ema_decays).add_(current_alpha * (1.0 - self.ema_decays))

        alpha = 0.5 * current_alpha + 0.5 * self.ema_alpha.detach()
        alpha = torch.clamp(alpha, min=eps, max=1.0 - eps)

        penalty_per_node = -0.5 * torch.log(alpha) - 0.5 * torch.log(1.0 - alpha)
        penalty = torch.sum(self.depth_weights * penalty_per_node)
        return cls_loss + self.penalty_lambda * penalty

    def predict_probs(self, z: torch.Tensor, mode: str = 'max_path_leaf') -> torch.Tensor:
        """Rozkład klas: 'max_path_leaf' (jeden, najbardziej prawdopodobny liść) lub 'soft_average'."""
        leaf_path_probs, leaf_q, _, _ = self.forward(z)
        if mode == 'soft_average':
            return torch.matmul(leaf_path_probs, leaf_q)
        elif mode == 'max_path_leaf':
            max_leaf_idx = torch.argmax(leaf_path_probs, dim=1)
            return leaf_q[max_leaf_idx]
        raise ValueError(f"Nieznany tryb predykcji: {mode}")


class EndToEndGNNSoftTree(nn.Module):
    """Enkoder GNN + (opcjonalne połączenie pomijające) + różniczkowalne miękkie drzewo.

    Z = [X || H] gdy `use_skip=True` (H – 100-wymiarowe osadzenie GNN), w p.p. Z = H.
    Cały model jest trenowany łącznie (end-to-end).
    """

    BACKBONES = ('gcn', 'gat', 'gat_resnet')

    def __init__(self, in_channels: int, backbone: str = 'gcn', tree_depth: int = 4,
                 use_skip: bool = True, penalty_lambda: float = 0.05,
                 num_classes: int = 2, dropout: float = 0.5, grad_checkpoint: bool = False,
                 gat_head_chunk: int = 0):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        if backbone not in self.BACKBONES:
            raise ValueError(f"Nieznany backbone: {backbone}. Dostępne: {self.BACKBONES}")
        self.in_channels = in_channels
        self.backbone = backbone
        self.use_skip = use_skip
        self.dropout = dropout
        hid = EMBED_DIM

        def gat(*a, **k):
            # gat_head_chunk > 0: liczenie głowic uwagi paczkami (mniejsze zużycie VRAM)
            if gat_head_chunk > 0:
                return HeadChunkedGATConv(*a, head_chunk=gat_head_chunk, **k)
            return GATConv(*a, **k)

        if backbone == 'gcn':
            self.conv1 = GCNConv(in_channels, hid)
            self.conv2 = GCNConv(hid, hid)
        elif backbone == 'gat':
            self.conv1 = gat(in_channels, hid, heads=8, concat=True)       # -> 800
            self.conv2 = gat(hid * 8, hid, heads=8, concat=False)          # -> 100 (średnia głów)
        else:  # gat_resnet
            self.conv1 = gat(in_channels, hid, heads=4, concat=True)       # -> 400
            self.conv2 = gat(hid * 4, hid, heads=4, concat=True)           # -> 400
            self.conv3 = gat(hid * 4, hid, heads=1, concat=False)          # -> 100

        tree_in = hid + in_channels if use_skip else hid
        self.tree_in_features = tree_in
        self.tree = DifferentiableSoftDecisionTree(
            in_features=tree_in, num_classes=num_classes, depth=tree_depth,
            penalty_lambda=penalty_lambda)

    def _conv(self, conv: nn.Module, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Warstwa konwolucyjna; opcjonalnie z checkpointingiem gradientów (oszczędność pamięci GPU)."""
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(conv, h, edge_index, use_reentrant=False)
        return conv(h, edge_index)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """100-wymiarowe osadzenie GNN H (ostatnia warstwa bez aktywacji)."""
        p = self.dropout
        if self.backbone == 'gcn':
            h = F.dropout(x, p=p, training=self.training)
            h = F.relu(self._conv(self.conv1, h, edge_index))
            h = F.dropout(h, p=p, training=self.training)
            return self._conv(self.conv2, h, edge_index)
        if self.backbone == 'gat':
            h = F.dropout(x, p=p, training=self.training)
            h = F.elu(self._conv(self.conv1, h, edge_index))
            h = F.dropout(h, p=p, training=self.training)
            return self._conv(self.conv2, h, edge_index)
        # gat_resnet
        h0 = F.dropout(x, p=p, training=self.training)
        h1 = F.elu(self._conv(self.conv1, h0, edge_index))
        h1_d = F.dropout(h1, p=p, training=self.training)
        h2 = F.elu(self._conv(self.conv2, h1_d, edge_index)) + h1          # połączenie rezydualne h2 + h1
        h2 = F.dropout(h2, p=p, training=self.training)
        return self._conv(self.conv3, h2, edge_index)

    def get_tree_input(self, x: torch.Tensor, edge_index: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zwraca (Z, H), gdzie Z = [X || H] (skip) lub H."""
        h = self.encode(x, edge_index)
        z = torch.cat([x, h], dim=1) if self.use_skip else h
        return z, h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Zwraca (leaf_path_probs, leaf_q, internal_path_probs, p_right, z)."""
        z, _ = self.get_tree_input(x, edge_index)
        leaf_path_probs, leaf_q, internal_path_probs, p_right = self.tree(z)
        return leaf_path_probs, leaf_q, internal_path_probs, p_right, z

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor, edge_index: torch.Tensor,
                      mode: str = 'max_path_leaf') -> torch.Tensor:
        """Rozkłady klas [N, C] dla wszystkich węzłów grafu."""
        z, _ = self.get_tree_input(x, edge_index)
        return self.tree.predict_probs(z, mode=mode)
