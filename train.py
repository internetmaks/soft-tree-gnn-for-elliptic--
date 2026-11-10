"""Pętla treningowa, early stopping, metryki AML oraz ewaluacja Concept Drift."""
import copy
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from models import EndToEndGNNSoftTree

DEFAULT_CLASS_WEIGHTS = (0.3, 0.7)   # [licit, illicit]
METRIC_NAMES = ['Precision_Illicit', 'Recall_Illicit', 'F1_Illicit', 'Micro_F1', 'MCC', 'ROC_AUC']


def compute_metrics(y_true: np.ndarray, p_illicit: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Metryki dla niezbalansowanego AML; klasa pozytywna = illicit (1)."""
    y_pred = (p_illicit >= threshold).astype(np.int64)
    if len(y_true) == 0:
        return {k: float('nan') for k in METRIC_NAMES}
    try:
        auc = roc_auc_score(y_true, p_illicit) if len(np.unique(y_true)) == 2 else float('nan')
    except ValueError:
        auc = float('nan')
    return {
        'Precision_Illicit': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall_Illicit': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1_Illicit': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Micro_F1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'ROC_AUC': auc,
    }


@torch.no_grad()
def predict_all_probs(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf') -> torch.Tensor:
    """Prawdopodobieństwo klasy illicit dla WSZYSTKICH węzłów grafu (jedna propagacja)."""
    model.eval()
    return model.predict_probs(data.x, data.edge_index, mode=inference_mode)[:, 1]


@torch.no_grad()
def evaluate_mask(model: EndToEndGNNSoftTree, data, mask: torch.Tensor,
                  inference_mode: str = 'max_path_leaf',
                  probs: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Metryki na podanej masce. `probs` (P(illicit) dla wszystkich węzłów) pozwala uniknąć ponownej propagacji."""
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    y_true = data.y[mask].cpu().numpy()
    p = probs[mask].detach().cpu().numpy()
    return compute_metrics(y_true, p)


def _train_period_mask(data) -> torch.Tensor:
    """Wszystkie węzły (także `unknown`) z kroków 1-29 – do semi-supervised regularyzacji."""
    return (data.time_step >= 1) & (data.time_step <= 29)


def train_model(model: EndToEndGNNSoftTree, data, epochs: int = 500, lr: float = 1e-3,
                weight_decay: float = 1e-4, patience: int = 50,
                class_weights=DEFAULT_CLASS_WEIGHTS, inference_mode: str = 'max_path_leaf',
                semi_supervised_reg: bool = False, verbose: bool = True, log_every: int = 10):
    """Trening full-batch; strata tylko na `train_mask`, early stopping po F1_Illicit na `val_mask`.

    Zwraca (model z wczytanym najlepszym stanem, historia treningu).
    """
    device = next(model.parameters()).device
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    reg_mask = _train_period_mask(data) if semi_supervised_reg else None
    history = {'epoch': [], 'loss': [], 'val_F1_Illicit': []}
    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        # Propagacja na całym grafie (z węzłami unknown), strata tylko na train_mask
        leaf_pp, leaf_q, int_pp, p_right, _ = model(data.x, data.edge_index)
        m = data.train_mask
        loss = model.tree.compute_loss(
            leaf_pp[m], leaf_q, int_pp[m], p_right[m], data.y[m], cw,
            reg_internal_path_probs=int_pp[reg_mask] if reg_mask is not None else None,
            reg_p_right=p_right[reg_mask] if reg_mask is not None else None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        val_f1 = evaluate_mask(model, data, data.val_mask, inference_mode)['F1_Illicit']
        history['epoch'].append(epoch)
        history['loss'].append(float(loss.item()))
        history['val_F1_Illicit'].append(float(val_f1))

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())   # best_model_state
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % log_every == 0):
            print(f"Epoka {epoch:4d} | loss {loss.item():.4f} | val F1_Illicit {val_f1:.4f} "
                  f"| najlepszy {best_f1:.4f} (ep. {best_epoch})")
        if bad_epochs >= patience:
            if verbose:
                print(f"Early stopping w epoce {epoch} (najlepsza epoka: {best_epoch}, val F1={best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    history['best_val_F1_Illicit'] = best_f1
    return model, history


def evaluate_regimes(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf') -> pd.DataFrame:
    """Metryki w 4 reżimach czasowych: Całość 35-49, Pre-Drift 35-42, Szok 43, Post-Drift 44-49."""
    probs = predict_all_probs(model, data, inference_mode)
    regimes = [
        ('Całość (35-49)', data.test_mask),
        ('Pre-Drift (35-42)', data.test_pre_drift_mask),
        ('Szok (krok 43)', data.test_drift_step43_mask),
        ('Post-Drift (44-49)', data.test_post_drift_mask),
    ]
    rows = []
    for name, mask in regimes:
        res = evaluate_mask(model, data, mask, inference_mode, probs=probs)
        res = {'Reżim': name, 'N': int(mask.sum()), 'N_illicit': int((data.y[mask] == 1).sum()), **res}
        rows.append(res)
    return pd.DataFrame(rows).set_index('Reżim')


def evaluate_per_timestep(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf',
                          save_csv: Optional[str] = None, save_plot: Optional[str] = None) -> pd.DataFrame:
    """Ewaluacja krok po kroku t = 35..49; opcjonalny zapis tabeli (CSV) i wykresu (PNG)."""
    probs = predict_all_probs(model, data, inference_mode)
    labeled = data.y != -1
    rows = []
    for t in range(35, 50):
        mask = (data.time_step == t) & labeled
        res = evaluate_mask(model, data, mask, inference_mode, probs=probs)
        rows.append({'time_step': t, 'N': int(mask.sum()),
                     'N_illicit': int((data.y[mask] == 1).sum()), **res})
    df = pd.DataFrame(rows).set_index('time_step')

    if save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        df.to_csv(save_csv)
    if save_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(os.path.abspath(save_plot)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for col, style in [('F1_Illicit', 'o-'), ('Precision_Illicit', 's--'),
                           ('Recall_Illicit', '^--'), ('MCC', 'd:')]:
            ax.plot(df.index, df[col], style, label=col)
        ax.axvline(43, color='red', linestyle=':', label='Zamknięcie darknetu (t=43)')
        ax.set_xlabel('Krok czasowy')
        ax.set_ylabel('Wartość metryki')
        ax.set_title('Metryki klasy illicit w kolejnych krokach testowych')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        fig.tight_layout()
        fig.savefig(save_plot, dpi=150)
        plt.close(fig)
    return df
