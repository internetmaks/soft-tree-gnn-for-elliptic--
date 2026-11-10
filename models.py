"""Modele: różniczkowalne miękkie drzewo decyzyjne oraz hybryda End-to-End GNN + Soft Tree."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

EMBED_DIM = 100
NUM_LOCAL_FEATURES = 93


class HeadChunkedGATConv(GATConv):
    """GATConv liczący głowice uwagi w paczkach (czysto na GPU, niskie zużycie VRAM).

    Matematycznie identyczny z `GATConv` (te same parametry `lin`, `att_src`, `att_dst`,
    `bias`), ale:
    * wykonuje projekcję liniową `F.linear` osobno dla każdej paczki głowic zamiast
      materializować pełny tensor `[N, heads, C]` (652 MB dla 8 głowic na Elliptic)
      oraz pełny tensor wiadomości `[E, heads, C]` (1,4 GB),
    * dla `concat=False` akumuluje sumę po głowicach w locie do `[N, C]`,
    * na DirectML używa czysto-GPU softmaxu opartego o `index_add_` (bez fallbacku na CPU),
    * buforuje `edge_index` z dodanymi pętlami własnymi.
    """

    def __init__(self, *args, head_chunk: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_chunk = max(1, head_chunk)
        self._cached_loop_key = None
        self._cached_loop_ei = None

    @staticmethod
    def _edge_softmax(alpha: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if alpha.device.type != 'privateuseone':
            return softmax(alpha, dst, num_nodes=num_nodes)
        # Czysty GPU softmax dla DirectML (unika nieobsługiwanego scatter_reduce_)
        alpha_max = alpha.detach().amax(dim=0, keepdim=True)
        exp = (alpha - alpha_max).exp()
        denom = torch.zeros(num_nodes, alpha.size(1), device=alpha.device,
                            dtype=alpha.dtype).index_add_(0, dst, exp)
        return exp / (denom.index_select(0, dst) + 1e-16)

    @classmethod
    def _attend_chunk(cls, x: torch.Tensor, w_chunk: torch.Tensor,
                      a_src: torch.Tensor, a_dst: torch.Tensor,
                      src: torch.Tensor, dst: torch.Tensor, num_nodes: int,
                      negative_slope: float, dropout: float, training: bool) -> torch.Tensor:
        h, c = a_src.size(1), a_src.size(2)
        xp = F.linear(x, w_chunk).view(num_nodes, h, c)
        s_logit = (xp * a_src).sum(dim=-1)
        d_logit = (xp * a_dst).sum(dim=-1)
        alpha = F.leaky_relu(s_logit.index_select(0, src) + d_logit.index_select(0, dst),
                             negative_slope)
        alpha = cls._edge_softmax(alpha, dst, num_nodes)
        alpha = F.dropout(alpha, p=dropout, training=training)
        msg = xp.index_select(0, src) * alpha.unsqueeze(-1)
        return torch.zeros(num_nodes, h, c, device=x.device,
                           dtype=x.dtype).index_add_(0, dst, msg)

    def _get_loop_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if not self.add_self_loops:
            return edge_index
        key = (id(edge_index), int(edge_index.size(1)), num_nodes, edge_index.device)
        if self._cached_loop_key != key:
            ei, _ = remove_self_loops(edge_index)
            ei, _ = add_self_loops(ei, num_nodes=num_nodes)
            self._cached_loop_key = key
            self._cached_loop_ei = ei
        return self._cached_loop_ei

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                use_ckpt: bool = False) -> torch.Tensor:
        H, C, N = self.heads, self.out_channels, x.size(0)
        edge_index = self._get_loop_edge_index(edge_index, N)
        src, dst = edge_index[0], edge_index[1]
        w = self.lin.weight.view(H, C, -1)

        outs = []
        acc = None
        ckpt = use_ckpt and self.training and torch.is_grad_enabled()
        for s in range(0, H, self.head_chunk):
            e = min(s + self.head_chunk, H)
            wc = w[s:e].reshape((e - s) * C, -1)
            args = (x, wc, self.att_src[:, s:e, :], self.att_dst[:, s:e, :],
                    src, dst, N, self.negative_slope, self.dropout, self.training)
            o = (checkpoint(self._attend_chunk, *args, use_reentrant=False)
                 if ckpt else self._attend_chunk(*args))
            if self.concat:
                outs.append(o.reshape(N, (e - s) * C))
            else:
                acc = o.sum(dim=1) if acc is None else acc + o.sum(dim=1)

        out = torch.cat(outs, dim=1) if self.concat else (acc / H)
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
                     reg_p_right: torch.Tensor = None,
                     soft_leaf_path_probs: torch.Tensor = None,
                     soft_targets: torch.Tensor = None,
                     distill_alpha: float = 0.5) -> torch.Tensor:
        """Ważona oczekiwana entropia krzyżowa na liściach + kara przeciw zapaści węzłów.

        Jeżeli podano `reg_*` (np. wszystkie węzły okresu treningowego, w tym `unknown`),
        służą one do wyznaczenia alpha_i (regularyzacja semi-supervised); w przeciwnym
        razie używane są tensory węzłów oznaczonych.
        Jeżeli podano `soft_targets` (np. predykcje nauczyciela dla wszystkich węzłów
        treningowych z kroków 1-29, w tym 77% `unknown`), funkcja łączy stratę na
        prawdziwych etykietach z destylacją wiedzy (Frosst & Hinton).
        """
        eps = 1e-8
        log_q = torch.log(leaf_q + eps)

        if targets.dim() == 2:
            target_one_hot = targets.float()
        else:
            # Porównanie zamiast F.one_hot (scatter nieobsługiwany przez DirectML)
            classes = torch.arange(leaf_q.size(-1), device=targets.device)
            target_one_hot = (targets.unsqueeze(1) == classes.unsqueeze(0)).float()
        weighted_target = target_one_hot * class_weights.unsqueeze(0)

        leaf_ce = -torch.matmul(weighted_target, log_q.t())
        cls_loss = torch.sum(leaf_path_probs * leaf_ce, dim=1).mean()

        if soft_targets is not None:
            s_lpp = soft_leaf_path_probs if soft_leaf_path_probs is not None else leaf_path_probs
            weighted_soft = soft_targets.float() * class_weights.unsqueeze(0)
            soft_leaf_ce = -torch.matmul(weighted_soft, log_q.t())
            soft_cls_loss = torch.sum(s_lpp * soft_leaf_ce, dim=1).mean()
            cls_loss = (1.0 - distill_alpha) * cls_loss + distill_alpha * soft_cls_loss

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


class DifferentiableSoftDecisionForest(nn.Module):
    """Komitet (las) K niezależnych różniczkowalnych miękkich drzew decyzyjnych.

    Każde drzewo otrzymuje losowo zamaskowaną podprzestrzeń cech wejściowych
    (`nn.Dropout1d`), co wymusza różnorodność filtrów w węzłach wewnętrznych i
    redukuje wariancję w dolnych partiach drzewa.
    """

    def __init__(self, in_features: int, num_classes: int = 2, depth: int = 4,
                 num_trees: int = 5, penalty_lambda: float = 0.05, feature_dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.depth = depth
        self.num_trees = num_trees
        self.penalty_lambda = penalty_lambda
        self.feature_dropout = feature_dropout
        self.feature_dropouts = nn.ModuleList([
            nn.Dropout1d(p=feature_dropout) if feature_dropout > 0 else nn.Identity()
            for _ in range(num_trees)
        ])
        self.trees = nn.ModuleList([
            DifferentiableSoftDecisionTree(
                in_features=in_features,
                num_classes=num_classes,
                depth=depth,
                penalty_lambda=penalty_lambda
            ) for _ in range(num_trees)
        ])

    def forward(self, z: torch.Tensor):
        outputs = []
        for drop, tree in zip(self.feature_dropouts, self.trees):
            z_dropped = drop(z.unsqueeze(-1)).squeeze(-1) if self.training else z
            outputs.append(tree(z_dropped))
        return outputs

    def compute_loss(self, forest_outputs, targets: torch.Tensor, class_weights: torch.Tensor,
                     train_labeled_mask: torch.Tensor = None, reg_mask: torch.Tensor = None,
                     soft_targets: torch.Tensor = None, distill_mask: torch.Tensor = None,
                     distill_alpha: float = 0.5) -> torch.Tensor:
        total_loss = 0.0
        for tree, (leaf_path_probs, leaf_q, internal_path_probs, p_right) in zip(self.trees, forest_outputs):
            if train_labeled_mask is not None:
                l_pp = leaf_path_probs[train_labeled_mask]
                i_pp = internal_path_probs[train_labeled_mask]
                p_r = p_right[train_labeled_mask]
                t_y = targets[train_labeled_mask] if targets.size(0) == leaf_path_probs.size(0) else targets
            else:
                l_pp, i_pp, p_r, t_y = leaf_path_probs, internal_path_probs, p_right, targets

            reg_i_pp = internal_path_probs[reg_mask] if reg_mask is not None else None
            reg_p_r = p_right[reg_mask] if reg_mask is not None else None

            s_lpp, s_t = None, None
            if soft_targets is not None:
                if distill_mask is not None:
                    s_lpp = leaf_path_probs[distill_mask]
                    s_t = soft_targets[distill_mask] if soft_targets.size(0) == leaf_path_probs.size(0) else soft_targets
                else:
                    s_lpp = leaf_path_probs
                    s_t = soft_targets

            total_loss = total_loss + tree.compute_loss(
                l_pp, leaf_q, i_pp, p_r, t_y, class_weights,
                reg_internal_path_probs=reg_i_pp, reg_p_right=reg_p_r,
                soft_leaf_path_probs=s_lpp, soft_targets=s_t, distill_alpha=distill_alpha
            )
        return total_loss / self.num_trees

    def predict_probs(self, z: torch.Tensor, mode: str = 'max_path_leaf') -> torch.Tensor:
        probs_list = [tree.predict_probs(z, mode=mode) for tree in self.trees]
        return torch.stack(probs_list, dim=0).mean(dim=0)


class EndToEndGNNSoftTree(nn.Module):
    """Enkoder GNN + (opcjonalne połączenie pomijające) + różniczkowalne miękkie drzewo / las.

    Z = [X || LayerNorm(H)] gdy `use_skip=True` i `normalize_emb=True` (H – 100-wymiarowe
    osadzenie GNN), w p.p. bez normalizacji lub Z = H. Gdy `asymmetric_features=True`,
    warstwy GNN otrzymują wyłącznie pierwsze 93 cechy lokalne `X[:, :93]`, natomiast
    gałąź Skip Connection przekazuje do drzewa wszystkie cechy tabelaryczne `X`.
    """

    BACKBONES = ('gcn', 'gat', 'gat_resnet')

    def __init__(self, in_channels: int, backbone: str = 'gcn', tree_depth: int = 4,
                 use_skip: bool = True, penalty_lambda: float = 0.05,
                 num_classes: int = 2, dropout: float = 0.5, grad_checkpoint: bool = False,
                 gat_head_chunk: int = 0, normalize_emb: bool = True,
                 asymmetric_features: bool = False, num_local_features: int = NUM_LOCAL_FEATURES,
                 num_trees: int = 1, feature_dropout: float = 0.2):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        if backbone not in self.BACKBONES:
            raise ValueError(f"Nieznany backbone: {backbone}. Dostępne: {self.BACKBONES}")
        self.in_channels = in_channels
        self.backbone = backbone
        self.use_skip = use_skip
        self.dropout = dropout
        self.normalize_emb = normalize_emb
        self.asymmetric_features = asymmetric_features
        self.num_local_features = num_local_features
        self.num_trees = num_trees
        self.feature_dropout = feature_dropout
        hid = EMBED_DIM

        gnn_in = min(num_local_features, in_channels) if asymmetric_features else in_channels
        self.gnn_in_channels = gnn_in

        def gat(*a, **k):
            # gat_head_chunk > 0: liczenie głowic uwagi paczkami (mniejsze zużycie VRAM)
            if gat_head_chunk > 0:
                return HeadChunkedGATConv(*a, head_chunk=gat_head_chunk, **k)
            return GATConv(*a, **k)

        if backbone == 'gcn':
            self.conv1 = GCNConv(gnn_in, hid)
            self.conv2 = GCNConv(hid, hid)
        elif backbone == 'gat':
            self.conv1 = gat(gnn_in, hid, heads=8, concat=True)            # -> 800
            self.conv2 = gat(hid * 8, hid, heads=8, concat=False)          # -> 100 (średnia głów)
        else:  # gat_resnet
            self.conv1 = gat(gnn_in, hid, heads=4, concat=True)            # -> 400
            self.conv2 = gat(hid * 4, hid, heads=4, concat=True)           # -> 400
            self.conv3 = gat(hid * 4, hid, heads=1, concat=False)          # -> 100

        self.emb_norm = nn.LayerNorm(hid) if normalize_emb else nn.Identity()

        tree_in = hid + in_channels if use_skip else hid
        self.tree_in_features = tree_in
        if num_trees > 1:
            self.tree = DifferentiableSoftDecisionForest(
                in_features=tree_in, num_classes=num_classes, depth=tree_depth,
                num_trees=num_trees, penalty_lambda=penalty_lambda,
                feature_dropout=feature_dropout)
        else:
            self.tree = DifferentiableSoftDecisionTree(
                in_features=tree_in, num_classes=num_classes, depth=tree_depth,
                penalty_lambda=penalty_lambda)

    def freeze_gnn_encoder(self, freeze: bool = True) -> None:
        """Zamraża (`requires_grad=False`) lub odmraża parametry enkodera GNN (poza drzewem)."""
        req = not freeze
        for name, param in self.named_parameters():
            if not name.startswith('tree.'):
                param.requires_grad = req

    def _conv(self, conv: nn.Module, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Warstwa konwolucyjna; opcjonalnie z checkpointingiem gradientów (oszczędność pamięci GPU)."""
        if isinstance(conv, HeadChunkedGATConv):
            return conv(h, edge_index, use_ckpt=self.grad_checkpoint)
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(conv, h, edge_index, use_reentrant=False)
        return conv(h, edge_index)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """100-wymiarowe osadzenie GNN H (ostatnia warstwa bez aktywacji)."""
        if self.asymmetric_features:
            x = x[:, :self.gnn_in_channels]
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
        """Zwraca (Z, H), gdzie Z = [X || H] (skip) lub H, a H jest opcjonalnie znormalizowane LayerNorm."""
        h = self.encode(x, edge_index)
        h = self.emb_norm(h)
        z = torch.cat([x, h], dim=1) if self.use_skip else h
        return z, h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Zwraca (leaf_path_probs, leaf_q, internal_path_probs, p_right, z) dla pojedynczego drzewa
        lub (forest_outputs, z) dla lasu miękkich drzew."""
        z, _ = self.get_tree_input(x, edge_index)
        if isinstance(self.tree, DifferentiableSoftDecisionForest):
            return self.tree(z), z
        leaf_path_probs, leaf_q, internal_path_probs, p_right = self.tree(z)
        return leaf_path_probs, leaf_q, internal_path_probs, p_right, z

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor, edge_index: torch.Tensor,
                      mode: str = 'max_path_leaf') -> torch.Tensor:
        """Rozkłady klas [N, C] dla wszystkich węzłów grafu."""
        z, _ = self.get_tree_input(x, edge_index)
        return self.tree.predict_probs(z, mode=mode)
