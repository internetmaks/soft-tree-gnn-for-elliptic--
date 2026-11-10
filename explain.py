"""Moduł XAI: ścieżki w miękkim drzewie, wpływ cech oraz projekcja UMAP osadzeń."""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from models import EMBED_DIM, DifferentiableSoftDecisionForest, EndToEndGNNSoftTree


def _tree_feature_names(model: EndToEndGNNSoftTree, feature_names: List[str]) -> List[str]:
    """Nazwy wymiarów wejścia drzewa: [cechy tabelaryczne (jeśli skip) || GNN_emb_0..99]."""
    names = list(feature_names) if model.use_skip else []
    return names + [f"GNN_emb_{k}" for k in range(EMBED_DIM)]


def _path_to_leaf(leaf_idx: int, depth: int) -> List[Dict]:
    """Rozkłada indeks liścia na ścieżkę (indeks węzła wewnętrznego, kierunek) od korzenia.

    Węzeł o pozycji j na głębokości k ma indeks 2^k - 1 + j; kolejne bity indeksu liścia
    (od najstarszego) to wybory lewo (0) / prawo (1).
    """
    path = []
    for k in range(depth):
        prefix = leaf_idx >> (depth - k)
        direction = (leaf_idx >> (depth - k - 1)) & 1
        path.append({'depth': k, 'node_idx': (2**k - 1) + prefix,
                     'direction': 'right' if direction else 'left'})
    return path


@torch.no_grad()
def select_illicit_nodes(model: EndToEndGNNSoftTree, data, mask: Optional[torch.Tensor] = None,
                         inference_mode: str = 'max_path_leaf', top_n: int = 3) -> List[int]:
    """Wybiera `top_n` prawdziwych węzłów illicit (domyślnie z testu) o najwyższym P(illicit)."""
    model.eval()
    mask = data.test_mask if mask is None else mask
    probs = model.predict_probs(data.x, data.edge_index, mode=inference_mode)[:, 1]
    cand = torch.nonzero(mask & (data.y == 1), as_tuple=False).view(-1)
    if cand.numel() == 0:
        return []
    order = torch.argsort(probs[cand], descending=True)[:top_n]
    return cand[order].cpu().tolist()


@torch.no_grad()
def explain_node(model: EndToEndGNNSoftTree, data, node_idx: int, top_k: int = 10,
                 verbose: bool = True) -> Dict:
    """Ścieżka o największym prawdopodobieństwie dla węzła + Top-K cech na każdym węźle drzewa.

    Wpływ cechy j w węźle i: c_ij = z_j * w_ij (znak: dodatni -> ku prawej gałęzi). Na
    ścieżce raportujemy też udział cech tabelarycznych vs 100 cech osadzenia GNN
    w sumie |c_ij| (oraz w normie L1 filtru |w_ij|) oraz diagnostykę skali aktywacji
    (weryfikacja efektu `normalize_emb` / `nn.LayerNorm`).
    Dla lasu miękkich drzew (`DifferentiableSoftDecisionForest`) wybierane jest drzewo
    reprezentatywne o najwyższej pewności dla klasy dominującej, a prawdopodobieństwo
    końcowe pochodzi z uśrednienia komitetu drzew.
    """
    model.eval()
    z_all, _ = model.get_tree_input(data.x, data.edge_index)
    z = z_all[node_idx].unsqueeze(0)

    is_forest = isinstance(model.tree, DifferentiableSoftDecisionForest)
    if is_forest:
        forest = model.tree
        ensemble_probs = forest.predict_probs(z, mode='max_path_leaf')[0]
        best_tree_idx = 0
        best_score = -1.0
        for idx_t, t_mod in enumerate(forest.trees):
            l_pp_t, l_q_t, _, _ = t_mod(z)
            l_idx_t = int(torch.argmax(l_pp_t, dim=1).item())
            score_t = float(l_q_t[l_idx_t, 1].item() * l_pp_t[0, l_idx_t].item())
            if score_t > best_score:
                best_score = score_t
                best_tree_idx = idx_t
        tree = forest.trees[best_tree_idx]
    else:
        tree = model.tree
        best_tree_idx = 0
        ensemble_probs = None

    leaf_pp, leaf_q, _, p_right = tree(z)
    leaf_idx = int(torch.argmax(leaf_pp, dim=1).item())
    names = _tree_feature_names(model, data.feature_names)
    n_tab = z.size(1) - EMBED_DIM  # 0 gdy brak połączenia pomijającego
    z_vec = z[0]

    steps = []
    tot_abs_tab, tot_abs_all, tot_w_tab, tot_w_all = 0.0, 0.0, 0.0, 0.0
    for step in _path_to_leaf(leaf_idx, tree.depth):
        i = step['node_idx']
        w_i = tree.inner_nodes.weight[i]                       # [w_tab || w_graph]
        beta_i = tree.beta[i]
        contrib = (z_vec * w_i) * beta_i                       # wkład do logitu węzła (z beta)
        abs_c = contrib.abs()
        abs_w = w_i.abs()

        # Top-K najbardziej wpływowych cech (wśród WSZYSTKICH wymiarów wejścia drzewa)
        k = min(top_k, abs_c.numel())
        top_vals, top_idx = torch.topk(abs_c, k)
        top_features = [{
            'feature': names[j],
            'is_tabular': bool(j < n_tab),
            'x_value': float(z_vec[j].item()),
            'weight': float(w_i[j].item()),
            'contribution': float(contrib[j].item()),
        } for j in top_idx.tolist()]
        # Top-K wśród cech tabelarycznych (interpretowalne cechy transakcji)
        top_tabular = []
        if n_tab > 0:
            kt = min(top_k, n_tab)
            _, tidx = torch.topk(abs_c[:n_tab], kt)
            top_tabular = [{
                'feature': names[j], 'x_value': float(z_vec[j].item()),
                'weight': float(w_i[j].item()), 'contribution': float(contrib[j].item()),
            } for j in tidx.tolist()]

        share_tab = float(abs_c[:n_tab].sum() / (abs_c.sum() + 1e-12)) if n_tab > 0 else 0.0
        w_share_tab = float(abs_w[:n_tab].sum() / (abs_w.sum() + 1e-12)) if n_tab > 0 else 0.0
        tot_abs_tab += float(abs_c[:n_tab].sum())
        tot_abs_all += float(abs_c.sum())
        tot_w_tab += float(abs_w[:n_tab].sum())
        tot_w_all += float(abs_w.sum())

        steps.append({
            **step,
            'p_right': float(p_right[0, i].item()),
            'bias': float(tree.inner_nodes.bias[i].item()),
            'beta': float(beta_i.item()),
            'top_features': top_features,
            'top_tabular_features': top_tabular,
            'tabular_share_contrib': share_tab,
            'tabular_share_weights': w_share_tab,
            'graph_share_contrib': 1.0 - share_tab if n_tab > 0 else 1.0,
        })

    probs = ensemble_probs if ensemble_probs is not None else leaf_q[leaf_idx]
    tab_mean_abs = float(z_vec[:n_tab].abs().mean().item()) if n_tab > 0 else 0.0
    gnn_mean_abs = float(z_vec[n_tab:].abs().mean().item())
    gnn_max_abs = float(z_vec[n_tab:].abs().max().item())

    result = {
        'node_idx': node_idx,
        'tx_id': data.tx_ids[node_idx] if hasattr(data, 'tx_ids') else None,
        'time_step': int(data.time_step[node_idx].item()),
        'true_label': int(data.y[node_idx].item()),
        'leaf_idx': leaf_idx,
        'leaf_path_prob': float(leaf_pp[0, leaf_idx].item()),
        'leaf_class_probs': probs.cpu().tolist(),
        'p_illicit': float(probs[1].item()),
        'is_forest': is_forest,
        'tree_idx': best_tree_idx,
        'normalize_emb': bool(getattr(model, 'normalize_emb', False)),
        'tabular_mean_abs': tab_mean_abs,
        'gnn_mean_abs': gnn_mean_abs,
        'gnn_max_abs': gnn_max_abs,
        'path': steps,
        'tabular_share_contrib_total': tot_abs_tab / (tot_abs_all + 1e-12) if n_tab > 0 else 0.0,
        'tabular_share_weights_total': tot_w_tab / (tot_w_all + 1e-12) if n_tab > 0 else 0.0,
        'n_tabular_features': n_tab,
        'n_gnn_features': EMBED_DIM,
    }
    if verbose:
        print_explanation(result)
    return result


def print_explanation(res: Dict) -> None:
    """Czytelny wydruk wyjaśnienia jednego węzła."""
    print("=" * 78)
    print(f"Węzeł {res['node_idx']} (txId={res['tx_id']}, krok {res['time_step']}, "
          f"etykieta={res['true_label']})")
    forest_info = f" [Las: drzewo #{res['tree_idx']}]" if res.get('is_forest') else ""
    print(f"Liść {res['leaf_idx']}{forest_info} | P(ścieżki)={res['leaf_path_prob']:.3f} | "
          f"P(illicit)={res['p_illicit']:.3f} | LayerNorm={res.get('normalize_emb', False)}")
    print(f"  Skala wejścia drzewa: |X_tab|_mean={res.get('tabular_mean_abs', 0.0):.3f}, "
          f"|H_gnn|_mean={res.get('gnn_mean_abs', 0.0):.3f}, |H_gnn|_max={res.get('gnn_max_abs', 0.0):.3f}")
    for s in res['path']:
        print(f"  [głębokość {s['depth']}] węzeł {s['node_idx']:2d} -> {s['direction']:5s} "
              f"(p_right={s['p_right']:.3f}, beta={s['beta']:.2f})")
        for f in s['top_features'][:5]:
            kind = 'TAB' if f['is_tabular'] else 'GNN'
            print(f"      {kind} {f['feature']:<22s} x={f['x_value']:+.3f} "
                  f"w={f['weight']:+.3f} wkład={f['contribution']:+.3f}")
        print(f"      udział cech tabelarycznych w |wkładzie|: {s['tabular_share_contrib']:.1%}")
    n_tab, n_g = res['n_tabular_features'], res['n_gnn_features']
    print(f"  Sumarycznie (ścieżka): cechy tabelaryczne ({n_tab}) = "
          f"{res['tabular_share_contrib_total']:.1%} wkładu, "
          f"osadzenie GNN ({n_g}) = {1 - res['tabular_share_contrib_total'] if n_tab else 1.0:.1%}")
    print(f"  Udział tabelaryczny w normie L1 filtrów na ścieżce: "
          f"{res['tabular_share_weights_total']:.1%}")


def explanation_to_dataframe(res: Dict) -> pd.DataFrame:
    """Płaska tabela (węzeł drzewa x cecha) z Top-K cechami dla zapisu do CSV."""
    rows = []
    for s in res['path']:
        for rank, f in enumerate(s['top_features'], 1):
            rows.append({'node_idx': res['node_idx'], 'tree_node': s['node_idx'],
                         'depth': s['depth'], 'direction': s['direction'], 'rank': rank,
                         **f})
    return pd.DataFrame(rows)


@torch.no_grad()
def visualize_embeddings_umap(model: EndToEndGNNSoftTree, data, mask: Optional[torch.Tensor] = None,
                              save_path: str = 'umap_embeddings.png', max_points: int = 20000,
                              representation: str = 'gnn', seed: int = 42,
                              n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Rzutowanie UMAP ukrytych reprezentacji węzłów (kolor = klasa).

    representation: 'gnn' – osadzenie H (100-d), 'tree_input' – wejście drzewa Z.
    Domyślnie węzły testowe; przy nadmiarze losowa podpróbka (zachowane wszystkie illicit).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import umap

    model.eval()
    z, h = model.get_tree_input(data.x, data.edge_index)
    emb = h if representation == 'gnn' else z
    mask = data.test_mask if mask is None else mask
    idx = torch.nonzero(mask, as_tuple=False).view(-1).cpu().numpy()
    y = data.y.cpu().numpy()

    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        illicit = idx[y[idx] == 1]
        licit = idx[y[idx] == 0]
        keep = max(max_points - len(illicit), 0)
        licit = rng.choice(licit, size=min(keep, len(licit)), replace=False)
        idx = np.concatenate([illicit, licit])

    coords = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                       random_state=seed).fit_transform(emb[idx].cpu().numpy())

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for label, color, name in [(0, 'tab:blue', 'licit'), (1, 'tab:red', 'illicit')]:
        sel = y[idx] == label
        ax.scatter(coords[sel, 0], coords[sel, 1], s=4 if label == 0 else 8, c=color,
                   alpha=0.4 if label == 0 else 0.8, label=f"{name} (n={int(sel.sum())})")
    ax.set_title(f"UMAP: {'osadzenie GNN H' if representation == 'gnn' else 'wejście drzewa Z'}")
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.legend(markerscale=3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return coords
