"""Moduł XAI: ścieżki w miękkim drzewie / lesie, wpływ cech oraz projekcja UMAP osadzeń."""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from models import EMBED_DIM, DifferentiableSoftDecisionForest, EndToEndGNNSoftTree

PRE_DRIFT_LABEL = 'Pre-Drift (35-42)'
POST_DRIFT_LABEL = 'Post-Drift (43-49)'


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


def _regime_of(time_step: int) -> str:
    if 35 <= time_step <= 42:
        return PRE_DRIFT_LABEL
    if time_step >= 43:
        return POST_DRIFT_LABEL
    return 'Train/Val (1-34)'


@torch.no_grad()
def select_illicit_nodes(model: EndToEndGNNSoftTree, data, mask: Optional[torch.Tensor] = None,
                         inference_mode: str = 'max_path_leaf', top_n: int = 3,
                         probs: Optional[torch.Tensor] = None, diverse_steps: bool = True) -> List[int]:
    """Wybiera `top_n` prawdziwych węzłów illicit (domyślnie z testu) o najwyższym P(illicit).

    `probs` – P(illicit) faktycznie użyte do oceny (np. z ewaluacji continual); `diverse_steps`
    – najpierw po jednym węźle z każdego kroku czasowego (malejąco po P(illicit)), potem reszta.
    """
    model.eval()
    mask = data.test_mask if mask is None else mask
    if probs is None:
        probs = model.predict_probs(data.x, data.edge_index, mode=inference_mode)[:, 1]
    cand = torch.nonzero(mask & (data.y == 1), as_tuple=False).view(-1)
    if cand.numel() == 0 or top_n <= 0:
        return []
    order = cand[torch.argsort(probs[cand], descending=True)].cpu().tolist()
    if not diverse_steps:
        return order[:top_n]
    steps = data.time_step.cpu()
    chosen, seen_steps = [], set()
    for n in order:
        s = int(steps[n])
        if s not in seen_steps:
            chosen.append(n)
            seen_steps.add(s)
        if len(chosen) >= top_n:
            return chosen
    for n in order:
        if n not in chosen:
            chosen.append(n)
        if len(chosen) >= top_n:
            break
    return chosen


@torch.no_grad()
def select_explanation_nodes(model: EndToEndGNNSoftTree, data, top_n: int = 3,
                             inference_mode: str = 'max_path_leaf',
                             probs_pre: Optional[torch.Tensor] = None,
                             probs_post: Optional[torch.Tensor] = None) -> List[Tuple[str, int]]:
    """`top_n` węzłów illicit z Pre-Drift (35-42) ORAZ `top_n` z Post-Drift (43-49)."""
    pre_mask = data.test_pre_drift_mask
    post_mask = data.test_drift_step43_mask | data.test_post_drift_mask
    pre = select_illicit_nodes(model, data, pre_mask, inference_mode, top_n, probs=probs_pre)
    post = select_illicit_nodes(model, data, post_mask, inference_mode, top_n,
                                probs=probs_post if probs_post is not None else probs_pre)
    return [(PRE_DRIFT_LABEL, n) for n in pre] + [(POST_DRIFT_LABEL, n) for n in post]


def _tree_path_attribution(tree, z: torch.Tensor, inference_mode: str):
    """Ścieżka o maks. prawdopodobieństwie w jednym drzewie + wkłady |beta * x * w| wzdłuż niej."""
    leaf_pp, leaf_q, _, p_right = tree(z)
    leaf_idx = int(torch.argmax(leaf_pp, dim=1).item())
    if inference_mode == 'soft_average':
        p_ill = float(torch.matmul(leaf_pp, leaf_q)[0, 1].item())
    else:
        p_ill = float(leaf_q[leaf_idx, 1].item())
    z_vec = z[0]
    abs_sum = torch.zeros_like(z_vec)
    support_sum = torch.zeros_like(z_vec)
    path = _path_to_leaf(leaf_idx, tree.depth)
    for step in path:
        i = step['node_idx']
        contrib = z_vec * tree.inner_nodes.weight[i] * tree.beta[i]
        abs_sum += contrib.abs()
        sign = 1.0 if step['direction'] == 'right' else -1.0
        support_sum += sign * contrib                  # >0: cecha wspiera wybraną gałąź
    return {
        'leaf_idx': leaf_idx, 'leaf_pp': leaf_pp, 'leaf_q': leaf_q, 'p_right': p_right,
        'leaf_path_prob': float(leaf_pp[0, leaf_idx].item()), 'p_illicit': p_ill,
        'abs_sum': abs_sum, 'support_sum': support_sum, 'path': path,
    }


@torch.no_grad()
def explain_node(model: EndToEndGNNSoftTree, data, node_idx: int, top_k: int = 10,
                 verbose: bool = True, inference_mode: str = 'max_path_leaf',
                 z_all: Optional[torch.Tensor] = None, p_illicit_used: Optional[float] = None,
                 threshold_used: Optional[float] = None, explained_with: str = 'base') -> Dict:
    """Wyjaśnienie węzła: szczegółowa ścieżka najpewniejszego drzewa + atrybucja całego lasu.

    * Wpływ cechy j w węźle i: c_ij = beta_i * z_j * w_ij (znak dodatni -> ku prawej gałęzi).
    * `per_tree_p_illicit`, `per_tree_leaf_idx` – głosy wszystkich K drzew.
    * `ensemble_top_features` – Top-K cech wg (1/K) sum_k sum_d |beta_kd * x_j * w_kdj| po
      aktywnych ścieżkach wszystkich drzew (dla pojedynczego drzewa K = 1).
    * Udział cech tabelarycznych vs 100 cech GNN w |wkładzie| oraz w normie L1 filtrów,
      a także diagnostyka skali wejścia (efekt `normalize_emb` / `nn.LayerNorm`).
    """
    model.eval()
    if z_all is None:
        z_all, _ = model.get_tree_input(data.x, data.edge_index)
    z = z_all[node_idx].unsqueeze(0)
    names = _tree_feature_names(model, data.feature_names)
    n_tab = z.size(1) - EMBED_DIM  # 0 gdy brak połączenia pomijającego
    z_vec = z[0]

    is_forest = isinstance(model.tree, DifferentiableSoftDecisionForest)
    trees = model.tree.member_trees()
    attrs = [_tree_path_attribution(t, z, inference_mode) for t in trees]
    per_tree_p = [a['p_illicit'] for a in attrs]
    ensemble_p = float(np.mean(per_tree_p))
    vote_thr = threshold_used if threshold_used is not None else 0.5
    pred_illicit = (p_illicit_used if p_illicit_used is not None else ensemble_p) >= vote_thr

    # Najpewniejsze drzewo dla przewidywanej klasy
    def confidence(a):
        q = a['p_illicit'] if pred_illicit else 1.0 - a['p_illicit']
        return q * a['leaf_path_prob']
    best_tree_idx = int(np.argmax([confidence(a) for a in attrs]))
    tree, best = trees[best_tree_idx], attrs[best_tree_idx]

    # Atrybucja całego lasu
    ens_abs = torch.stack([a['abs_sum'] for a in attrs]).mean(dim=0)
    ens_support = torch.stack([a['support_sum'] for a in attrs]).mean(dim=0)
    k_ens = min(top_k, ens_abs.numel())
    _, ens_idx = torch.topk(ens_abs, k_ens)
    ensemble_top_features = [{
        'feature': names[j], 'is_tabular': bool(j < n_tab), 'x_value': float(z_vec[j].item()),
        'mean_abs_contribution': float(ens_abs[j].item()),
        'mean_path_support': float(ens_support[j].item()),
    } for j in ens_idx.tolist()]
    ens_tab_share = float(ens_abs[:n_tab].sum() / (ens_abs.sum() + 1e-12)) if n_tab > 0 else 0.0

    # Szczegółowa ścieżka najpewniejszego drzewa
    p_right = best['p_right']
    steps = []
    tot_abs_tab, tot_abs_all, tot_w_tab, tot_w_all = 0.0, 0.0, 0.0, 0.0
    for step in best['path']:
        i = step['node_idx']
        w_i = tree.inner_nodes.weight[i]                       # [w_tab || w_graph]
        beta_i = tree.beta[i]
        contrib = (z_vec * w_i) * beta_i                       # wkład do logitu węzła (z beta)
        abs_c = contrib.abs()
        abs_w = w_i.abs()

        k = min(top_k, abs_c.numel())
        _, top_idx = torch.topk(abs_c, k)
        top_features = [{
            'feature': names[j],
            'is_tabular': bool(j < n_tab),
            'x_value': float(z_vec[j].item()),
            'weight': float(w_i[j].item()),
            'contribution': float(contrib[j].item()),
        } for j in top_idx.tolist()]
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

    leaf_q_best = best['leaf_q'][best['leaf_idx']]
    t_step = int(data.time_step[node_idx].item())
    tab_mean_abs = float(z_vec[:n_tab].abs().mean().item()) if n_tab > 0 else 0.0
    result = {
        'node_idx': int(node_idx),
        'tx_id': data.tx_ids[node_idx] if hasattr(data, 'tx_ids') else None,
        'time_step': t_step,
        'regime': _regime_of(t_step),
        'true_label': int(data.y[node_idx].item()),
        'explained_with': explained_with,
        'inference_mode': inference_mode,
        'p_illicit': ensemble_p,
        'p_illicit_used': float(p_illicit_used) if p_illicit_used is not None else ensemble_p,
        'threshold_used': float(threshold_used) if threshold_used is not None else None,
        'predicted_illicit': bool(pred_illicit),
        'is_forest': is_forest,
        'num_trees': len(trees),
        'per_tree_p_illicit': per_tree_p,
        'per_tree_leaf_idx': [a['leaf_idx'] for a in attrs],
        'per_tree_leaf_path_prob': [a['leaf_path_prob'] for a in attrs],
        'trees_voting_illicit': int(sum(p >= 0.5 for p in per_tree_p)),
        'ensemble_top_features': ensemble_top_features,
        'ensemble_tabular_share_contrib': ens_tab_share,
        'tree_idx': best_tree_idx,
        'leaf_idx': best['leaf_idx'],
        'leaf_path_prob': best['leaf_path_prob'],
        'leaf_class_probs': leaf_q_best.cpu().tolist(),
        'normalize_emb': bool(getattr(model, 'normalize_emb', False)),
        'tabular_mean_abs': tab_mean_abs,
        'gnn_mean_abs': float(z_vec[n_tab:].abs().mean().item()),
        'gnn_max_abs': float(z_vec[n_tab:].abs().max().item()),
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
          f"{res.get('regime', '')}, etykieta={res['true_label']}, model={res.get('explained_with', 'base')})")
    thr = res.get('threshold_used')
    thr_txt = f" | próg={thr:.3f}" if thr is not None else ""
    print(f"P(illicit) użyte={res.get('p_illicit_used', res['p_illicit']):.3f}{thr_txt} | "
          f"LayerNorm={res.get('normalize_emb', False)}")
    if res.get('num_trees', 1) > 1:
        votes = ", ".join(f"{p:.2f}" for p in res['per_tree_p_illicit'])
        print(f"  Głosy drzew P(illicit): [{votes}] | liście: {res['per_tree_leaf_idx']} | "
              f"drzewa za illicit: {res['trees_voting_illicit']}/{res['num_trees']}")
    print(f"  Skala wejścia drzewa: |X_tab|_mean={res.get('tabular_mean_abs', 0.0):.3f}, "
          f"|H_gnn|_mean={res.get('gnn_mean_abs', 0.0):.3f}, |H_gnn|_max={res.get('gnn_max_abs', 0.0):.3f}")
    print(f"  Top cechy lasu (śr. |β·x·w| po ścieżkach {res.get('num_trees', 1)} drzew, "
          f"udział tabelaryczny {res.get('ensemble_tabular_share_contrib', 0.0):.1%}):")
    for f in res.get('ensemble_top_features', [])[:5]:
        kind = 'TAB' if f['is_tabular'] else 'GNN'
        print(f"      {kind} {f['feature']:<22s} x={f['x_value']:+.3f} "
              f"|wkład|={f['mean_abs_contribution']:.3f} wsparcie ścieżki={f['mean_path_support']:+.3f}")
    forest_info = f" [drzewo #{res['tree_idx']}]" if res.get('is_forest') else ""
    print(f"  Ścieżka{forest_info}: liść {res['leaf_idx']} | P(ścieżki)={res['leaf_path_prob']:.3f} | "
          f"Q(illicit)={res['leaf_class_probs'][1]:.3f}")
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
    """Płaska tabela do CSV: Top-K cech na ścieżce najpewniejszego drzewa (`section='path'`)
    oraz Top-K cech całego lasu (`section='ensemble'`)."""
    base = {'node_idx': res['node_idx'], 'tx_id': res.get('tx_id'), 'time_step': res['time_step'],
            'regime': res.get('regime'), 'true_label': res['true_label'],
            'explained_with': res.get('explained_with'), 'p_illicit_used': res.get('p_illicit_used')}
    rows = []
    for s in res['path']:
        for rank, f in enumerate(s['top_features'], 1):
            rows.append({**base, 'section': 'path', 'tree_idx': res.get('tree_idx', 0),
                         'tree_node': s['node_idx'], 'depth': s['depth'],
                         'direction': s['direction'], 'rank': rank, **f})
    for rank, f in enumerate(res.get('ensemble_top_features', []), 1):
        rows.append({**base, 'section': 'ensemble', 'tree_idx': -1, 'tree_node': -1, 'depth': -1,
                     'direction': '', 'rank': rank, **f})
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
    with open(save_path, 'wb') as f:
        fig.savefig(f, format='png', dpi=150)
    plt.close(fig)
    return coords
