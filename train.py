"""Pętla treningowa, early stopping, strojenie progu, destylacja wiedzy oraz ciągłe douczanie po dryfcie."""
import copy
import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from models import DifferentiableSoftDecisionForest, EndToEndGNNSoftTree

DEFAULT_CLASS_WEIGHTS = (0.3, 0.7)   # [licit, illicit]
ADAPT_CLASS_WEIGHTS = (0.5, 0.5)     # zbalansowany bufor odtwarzania -> wagi równe
THRESHOLD_RANGE = (0.20, 0.85)
POST_DRIFT_THRESHOLD_RANGE = (0.50, 0.75)
TRAIN_LAST_STEP = 29
DRIFT_STEP = 43
LAST_STEP = 49
METRIC_NAMES = ['Precision_Illicit', 'Recall_Illicit', 'F1_Illicit', 'Micro_F1', 'MCC', 'ROC_AUC',
                'TP', 'FP', 'FN', 'TN']

ThresholdLike = Union[float, torch.Tensor]


def compute_metrics(y_true: np.ndarray, p_illicit: np.ndarray,
                    threshold: Union[float, np.ndarray] = 0.5) -> Dict[str, float]:
    """Metryki dla niezbalansowanego AML; klasa pozytywna = illicit (1).

    `threshold` może być skalarem lub wektorem progów (po jednym na węzeł), np. przy
    progach zależnych od kroku czasowego w ewaluacji Post-Drift.
    """
    if len(y_true) == 0:
        return {k: float('nan') for k in METRIC_NAMES}
    y_pred = (p_illicit >= threshold).astype(np.int64)
    try:
        auc = roc_auc_score(y_true, p_illicit) if len(np.unique(y_true)) == 2 else float('nan')
    except ValueError:
        auc = float('nan')
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {
        'Precision_Illicit': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall_Illicit': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1_Illicit': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Micro_F1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'ROC_AUC': auc,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
    }


def find_optimal_threshold(y_true: np.ndarray, p_illicit: np.ndarray,
                           low: float = THRESHOLD_RANGE[0], high: float = THRESHOLD_RANGE[1],
                           num_steps: int = 66) -> Tuple[float, float]:
    """Wyszukuje optymalny próg klasyfikacji w przedziale [low, high] maksymalizujący F1_Illicit."""
    if len(y_true) == 0:
        return 0.5, 0.0
    thresholds = np.linspace(low, high, num_steps)
    preds = p_illicit[:, None] >= thresholds[None, :]
    y_pos = (y_true == 1)[:, None]
    tp = np.sum(preds & y_pos, axis=0).astype(np.float64)
    fp = np.sum(preds & (~y_pos), axis=0).astype(np.float64)
    fn = np.sum((~preds) & y_pos, axis=0).astype(np.float64)
    denom = 2.0 * tp + fp + fn
    f1_scores = np.where(denom > 0, (2.0 * tp) / np.maximum(denom, 1e-12), 0.0)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


@torch.no_grad()
def predict_all_probs(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf') -> torch.Tensor:
    """Prawdopodobieństwo klasy illicit dla WSZYSTKICH węzłów grafu (jedna propagacja)."""
    model.eval()
    return model.predict_probs(data.x, data.edge_index, mode=inference_mode)[:, 1]


@torch.no_grad()
def tune_threshold_on_val(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf',
                          low: float = THRESHOLD_RANGE[0], high: float = THRESHOLD_RANGE[1],
                          probs: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """Wyznacza optymalny próg decyzyjny wyłącznie na masce walidacyjnej (kroki 30-34)."""
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    y_val = data.y[data.val_mask].cpu().numpy()
    p_val = probs[data.val_mask].detach().cpu().numpy()
    return find_optimal_threshold(y_val, p_val, low=low, high=high)


def _threshold_for_mask(threshold: ThresholdLike, mask: torch.Tensor):
    """Próg (skalar lub wektor numpy) dla węzłów z maski."""
    if isinstance(threshold, torch.Tensor):
        return threshold[mask].detach().cpu().numpy()
    return float(threshold)


def _mean_threshold(threshold: ThresholdLike, mask: torch.Tensor) -> float:
    if isinstance(threshold, torch.Tensor):
        sel = threshold[mask]
        return round(float(sel.mean().item()), 4) if sel.numel() > 0 else float('nan')
    return round(float(threshold), 4)


@torch.no_grad()
def evaluate_mask(model: EndToEndGNNSoftTree, data, mask: torch.Tensor,
                  inference_mode: str = 'max_path_leaf',
                  probs: Optional[torch.Tensor] = None,
                  threshold: ThresholdLike = 0.5) -> Dict[str, float]:
    """Metryki na podanej masce przy zadanym progu (skalar lub wektor progów per węzeł)."""
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    y_true = data.y[mask].cpu().numpy()
    p = probs[mask].detach().cpu().numpy()
    return compute_metrics(y_true, p, threshold=_threshold_for_mask(threshold, mask))


def _train_period_mask(data) -> torch.Tensor:
    """Wszystkie węzły (oznaczone ORAZ `unknown`, y == -1) z kroków 1-29 –
    do regularyzacji semi-supervised (EMA) i destylacji."""
    return data.time_step <= TRAIN_LAST_STEP


def _train_unknown_mask(data) -> torch.Tensor:
    """Nieoznaczone węzły (`unknown`, y == -1) wyłącznie z okresu treningowego (kroki 1-29)."""
    return _train_period_mask(data) & (data.y == -1)


@torch.no_grad()
def generate_soft_targets(teacher_model: EndToEndGNNSoftTree, data,
                          inference_mode: str = 'soft_average') -> torch.Tensor:
    """Generuje rozkłady prawdopodobieństwa [N, 2] modelu nauczyciela (Frosst & Hinton).

    Podczas destylacji wykorzystywane są wyłącznie węzły z kroków 1-29 (w tym 77% `unknown`),
    co gwarantuje całkowity brak wycieku informacji z walidacji (30-34) i testu (35-49).
    Nauczyciel musi być modelem SPRZED adaptacji post-drift (adaptacja widzi etykiety 30-48).
    """
    teacher_model.eval()
    return teacher_model.predict_probs(data.x, data.edge_index, mode=inference_mode).detach()


def _build_optimizer(model: torch.nn.Module, lr: float = 1e-3,
                     weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """Dzieli parametry na grupy z i bez L2 weight_decay (zgodnie z Frosst & Hinton 2017).

    Odwrotna temperatura `beta`, logity rozkładów liści `leaf_logits`, parametry `emb_norm`
    oraz wektory obciążeń `bias` mają `weight_decay=0.0`, aby nie spłaszczać rozkładów
    w liściach ku [0.5, 0.5] ani nie rozmywać ostrych podziałów sigmoidalnych.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ("beta", "leaf_logits", "emb_norm", "bias")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr)


def train_model(model: EndToEndGNNSoftTree, data, epochs: int = 500, lr: float = 1e-3,
                weight_decay: float = 1e-4, patience: int = 50,
                class_weights=DEFAULT_CLASS_WEIGHTS, inference_mode: str = 'max_path_leaf',
                semi_supervised_reg: Optional[bool] = True, tune_threshold: bool = True,
                soft_targets: Optional[torch.Tensor] = None, distill_alpha: float = 0.5,
                verbose: bool = True, log_every: int = 10):
    """Trening full-batch; strata na `train_mask` (oraz opcjonalnie destylacja na krokach 1-29),
    early stopping i strojenie progu (0.20-0.85) po F1_Illicit na `val_mask` (kroki 30-34).

    Gdy `semi_supervised_reg=True` (domyślnie), `reg_mask = (time_step <= 29)` – wszystkie węzły
    okresu treningowego niezależnie od etykiety (także `unknown`) – trafia do `compute_loss`
    jako źródło alpha_i i aktualizacji bufora EMA.
    Zwraca (model z wczytanym najlepszym stanem, historia treningu ze znalezionym `opt_threshold`).
    """
    if semi_supervised_reg is None:
        semi_supervised_reg = True
    semi_supervised_reg = bool(semi_supervised_reg)

    device = next(model.parameters()).device
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = _build_optimizer(model, lr=lr, weight_decay=weight_decay)

    gnn_trainable = any(p.requires_grad for n, p in model.named_parameters() if not n.startswith('tree.'))
    z_frozen = None
    if not gnn_trainable:
        model.eval()
        with torch.no_grad():
            z_frozen, _ = model.get_tree_input(data.x, data.edge_index)
            z_frozen = z_frozen.detach()

    train_period_mask = _train_period_mask(data)
    reg_mask = train_period_mask if semi_supervised_reg else None
    distill_mask = train_period_mask if soft_targets is not None else None
    if soft_targets is not None:
        soft_targets = soft_targets.to(device)
    is_forest = isinstance(model.tree, DifferentiableSoftDecisionForest)

    reg_n = int(reg_mask.sum().item()) if reg_mask is not None else 0
    reg_unknown_n = int((reg_mask & (data.y == -1)).sum().item()) if reg_mask is not None else 0
    if verbose:
        if reg_mask is not None:
            print(f"[semi-supervised EMA] AKTYWNA: reg_mask = time_step <= {TRAIN_LAST_STEP} "
                  f"({reg_n} węzłów, w tym {reg_unknown_n} unknown)")
        else:
            print("[semi-supervised EMA] WYŁĄCZONA (alpha_i liczone tylko na węzłach oznaczonych)")

    history = {'epoch': [], 'loss': [], 'val_F1_Illicit': [], 'val_threshold': []}
    best_f1, best_epoch, best_thr, best_state, bad_epochs = -1.0, 0, 0.5, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        m = data.train_mask

        if is_forest:
            forest_outputs = model.tree(z_frozen) if z_frozen is not None else model(data.x, data.edge_index)[0]
            loss = model.tree.compute_loss(
                forest_outputs, data.y, cw,
                train_labeled_mask=m, reg_mask=reg_mask,
                soft_targets=soft_targets, distill_mask=distill_mask,
                distill_alpha=distill_alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            loss_val = float(loss.item())
            del forest_outputs, loss
        else:
            if z_frozen is not None:
                leaf_pp, leaf_q, int_pp, p_right = model.tree(z_frozen)
            else:
                leaf_pp, leaf_q, int_pp, p_right, _ = model(data.x, data.edge_index)
            loss = model.tree.compute_loss(
                leaf_pp[m], leaf_q, int_pp[m], p_right[m], data.y[m], cw,
                reg_internal_path_probs=int_pp[reg_mask] if reg_mask is not None else None,
                reg_p_right=p_right[reg_mask] if reg_mask is not None else None,
                soft_leaf_path_probs=leaf_pp[distill_mask] if distill_mask is not None else None,
                soft_targets=soft_targets[distill_mask] if distill_mask is not None else None,
                distill_alpha=distill_alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            loss_val = float(loss.item())
            del leaf_pp, leaf_q, int_pp, p_right, loss

        if z_frozen is not None:
            model.eval()
            with torch.no_grad():
                probs = model.tree.predict_probs(z_frozen, mode=inference_mode)[:, 1]
        else:
            probs = predict_all_probs(model, data, inference_mode)
        if tune_threshold:
            epoch_thr, val_f1 = tune_threshold_on_val(model, data, inference_mode, probs=probs)
        else:
            epoch_thr = 0.5
            val_f1 = evaluate_mask(model, data, data.val_mask, inference_mode,
                                   probs=probs, threshold=0.5)['F1_Illicit']
        del probs

        history['epoch'].append(epoch)
        history['loss'].append(loss_val)
        history['val_F1_Illicit'].append(float(val_f1))
        history['val_threshold'].append(float(epoch_thr))

        if val_f1 > best_f1:
            best_f1, best_epoch, best_thr, bad_epochs = val_f1, epoch, epoch_thr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % log_every == 0):
            print(f"Epoka {epoch:4d} | loss {loss_val:.4f} | val F1_Illicit {val_f1:.4f} "
                  f"(thr={epoch_thr:.2f}) | najlepszy {best_f1:.4f} (ep. {best_epoch}, thr={best_thr:.2f})")
        if bad_epochs >= patience:
            if verbose:
                print(f"Early stopping w epoce {epoch} (najlepsza epoka: {best_epoch}, "
                      f"val F1={best_f1:.4f}, opt_thr={best_thr:.2f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if tune_threshold:
        best_thr, best_f1 = tune_threshold_on_val(model, data, inference_mode)
    history['best_epoch'] = best_epoch
    history['best_val_F1_Illicit'] = best_f1
    history['opt_threshold'] = best_thr
    history['semi_supervised_reg'] = semi_supervised_reg
    history['reg_mask_nodes'] = reg_n
    history['reg_mask_unknown_nodes'] = reg_unknown_n
    return model, history


def train_distilled_tree(student_model: EndToEndGNNSoftTree, data,
                         teacher_model: Optional[EndToEndGNNSoftTree] = None,
                         soft_targets: Optional[torch.Tensor] = None,
                         teacher_state_dict: Optional[dict] = None,
                         teacher_num_trees: int = 5, distill_alpha: float = 0.5,
                         epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-4,
                         patience: int = 50, class_weights=DEFAULT_CLASS_WEIGHTS,
                         inference_mode: str = 'max_path_leaf', tune_threshold: bool = True,
                         verbose: bool = True, log_every: int = 10):
    """Trenuje pojedyncze miękkie drzewo decyzyjne z destylacją wiedzy na 77% węzłów `unknown` (kroki 1-29).

    Zgodnie z Frosst & Hinton (2017):
    1. Jeśli nie przekazano `soft_targets` ani `teacher_model`, najpierw trenowany jest nauczyciel
       (Soft Forest K drzew) na krokach 1-29.
    2. Wytrenowany enkoder GNN nauczyciela (`conv*`, `emb_norm`) jest kopiowany do modelu ucznia
       i zamrażany (`freeze_gnn_encoder(True)`), dzięki czemu pojedyncze drzewo ucznia uczy się
       na tej samej wyuczonej reprezentacji grafowej bez ponownego przeuczania GNN.
    3. Miękkie cele `soft_targets` na węzłach z kroków 1-29 (w tym 77% `unknown`) są destylowane
       bez podwójnego mnożenia przez `class_weights`.
    """
    device = next(student_model.parameters()).device
    if soft_targets is None or (teacher_state_dict is None and teacher_model is None):
        if teacher_model is None:
            if verbose:
                print(f"[Destylacja] Trening modelu nauczyciela (Soft Forest, K={teacher_num_trees} drzew)...")
            gat_chunk = 0
            if hasattr(student_model, 'conv1') and hasattr(student_model.conv1, 'head_chunk'):
                gat_chunk = student_model.conv1.head_chunk
            teacher_model = EndToEndGNNSoftTree(
                in_channels=student_model.in_channels,
                backbone=student_model.backbone,
                tree_depth=student_model.tree.depth,
                use_skip=student_model.use_skip,
                penalty_lambda=student_model.tree.penalty_lambda,
                dropout=student_model.dropout,
                grad_checkpoint=student_model.grad_checkpoint,
                gat_head_chunk=gat_chunk,
                normalize_emb=student_model.normalize_emb,
                asymmetric_features=student_model.asymmetric_features,
                num_trees=teacher_num_trees,
            ).to(device)
            teacher_model, _ = train_model(
                teacher_model, data, epochs=epochs, lr=lr, weight_decay=weight_decay,
                patience=patience, class_weights=class_weights, inference_mode=inference_mode,
                semi_supervised_reg=True, tune_threshold=tune_threshold,
                verbose=verbose, log_every=log_every)
        if teacher_state_dict is None and teacher_model is not None:
            teacher_state_dict = teacher_model.state_dict()
        if soft_targets is None:
            soft_targets = generate_soft_targets(teacher_model, data, inference_mode='soft_average')
            if verbose:
                unk_n = int(_train_unknown_mask(data).sum().item())
                tot_n = int(_train_period_mask(data).sum().item())
                print(f"[Destylacja] Wygenerowano miękkie cele dla {tot_n} węzłów z kroków 1-29 "
                      f"(w tym {unk_n} węzłów unknown, {unk_n / max(tot_n, 1):.1%}).")

    # Kopiowanie i zamrożenie wytrenowanego enkodera GNN nauczyciela (conv*, emb_norm)
    if teacher_state_dict is not None:
        student_sd = student_model.state_dict()
        copied = 0
        for k, v in teacher_state_dict.items():
            if not k.startswith('tree.') and k in student_sd and student_sd[k].shape == v.shape:
                student_sd[k].copy_(v.to(student_sd[k].device))
                copied += 1
        if copied > 0:
            student_model.freeze_gnn_encoder(True)
            if verbose:
                print(f"[Destylacja] Przeniesiono i zamrożono enkoder GNN nauczyciela ({copied} tensorów).")

    try:
        student_model, history = train_model(
            student_model, data, epochs=epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, class_weights=class_weights, inference_mode=inference_mode,
            semi_supervised_reg=True, tune_threshold=tune_threshold,
            soft_targets=soft_targets, distill_alpha=distill_alpha,
            verbose=verbose, log_every=log_every)
    finally:
        student_model.freeze_gnn_encoder(False)
    return student_model, history


REGIMES = (
    ('Całość (35-49)', 'test_mask'),
    ('Pre-Drift (35-42)', 'test_pre_drift_mask'),
    ('Szok (krok 43)', 'test_drift_step43_mask'),
    ('Post-Drift (44-49)', 'test_post_drift_mask'),
)


def evaluate_regimes(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf',
                     threshold: ThresholdLike = 0.5, probs: Optional[torch.Tensor] = None) -> pd.DataFrame:
    """Metryki w 4 reżimach czasowych: Całość 35-49, Pre-Drift 35-42, Szok 43, Post-Drift 44-49.

    `threshold` może być wektorem progów per węzeł (progi zależne od kroku); wówczas kolumna
    `Opt_Threshold` raportuje średni próg zastosowany w danym reżimie.
    """
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    rows = []
    for name, attr in REGIMES:
        mask = getattr(data, attr)
        res = evaluate_mask(model, data, mask, inference_mode, probs=probs, threshold=threshold)
        rows.append({'Reżim': name, 'N': int(mask.sum()), 'N_illicit': int((data.y[mask] == 1).sum()),
                     'Opt_Threshold': _mean_threshold(threshold, mask), **res})
    return pd.DataFrame(rows).set_index('Reżim')


def evaluate_per_timestep(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf',
                          save_csv: Optional[str] = None, save_plot: Optional[str] = None,
                          threshold: ThresholdLike = 0.5, probs: Optional[torch.Tensor] = None) -> pd.DataFrame:
    """Ewaluacja krok po kroku t = 35..49 (z faktycznym progiem użytym w każdym kroku);
    opcjonalny zapis tabeli (CSV, UTF-8) i wykresu (PNG)."""
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    labeled = data.y != -1
    rows = []
    for t in range(35, LAST_STEP + 1):
        step_all = data.time_step == t
        mask = step_all & labeled
        res = evaluate_mask(model, data, mask, inference_mode, probs=probs, threshold=threshold)
        rows.append({'time_step': t, 'N': int(mask.sum()),
                     'N_illicit': int((data.y[mask] == 1).sum()),
                     'Opt_Threshold': _mean_threshold(threshold, step_all), **res})
    df = pd.DataFrame(rows).set_index('time_step')

    if save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        df.to_csv(save_csv, encoding='utf-8')
    if save_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(os.path.abspath(save_plot)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for col, style in [('F1_Illicit', 'o-'), ('Precision_Illicit', 's--'),
                           ('Recall_Illicit', '^--'), ('MCC', 'd:')]:
            ax.plot(df.index, df[col], style, label=col)
        ax.step(df.index, df['Opt_Threshold'], where='mid', color='gray', alpha=0.7,
                linestyle='-.', label='Próg decyzyjny (Opt_Threshold)')
        ax.axvline(DRIFT_STEP, color='red', linestyle=':', label='Zamknięcie darknetu (t=43)')
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


@torch.no_grad()
def get_adaptive_post_drift_threshold(model: EndToEndGNNSoftTree, z_all: torch.Tensor,
                                      val_mask: torch.Tensor, y_all: torch.Tensor,
                                      base_threshold: float, current_step_mask: torch.Tensor,
                                      min_pos_rate: float = 0.015,
                                      inference_mode: str = 'max_path_leaf',
                                      low: float = POST_DRIFT_THRESHOLD_RANGE[0],
                                      high: float = POST_DRIFT_THRESHOLD_RANGE[1],
                                      majority_frac: float = 0.6, majority_margin: float = 0.0,
                                      quantile_floor: float = 0.52,
                                      return_info: bool = False):
    """Próg tau_t dla kroku t >= 43 (wyznaczany PRZED predykcją kroku t, bez jego etykiet).

    1. Dolny próg głosowania większościowego (las): tau_maj = f*mean_max + (1-f)*mean_min - margin,
       gdzie f = ceil(0.6 K)/K (dla K=5: co najmniej 3/5 drzew musi skierować próbkę do liścia illicit).
       Zapobiega to spadkowi progu do ~0.35 i zalewowi fałszywych alarmów po zamknięciu darknetu.
    2. Ponowne strojenie F1 w [max(low, tau_maj), high] na zbiorze kalibracyjnym `val_mask`
       (walidacja 30-34 oraz ujawnione kroki post-drift < t), ograniczone z góry przez `base_threshold`
       oraz przez `tau_maj`, gdy na małej próbce post-drift próg walidacyjny jest zbyt restrykcyjny.
    """
    model.eval()
    tree = model.tree
    info = {}

    if hasattr(tree, 'trees'):
        k = len(tree.trees)
        votes = math.ceil(majority_frac * k)
        frac = votes / k
        max_illicit_per_tree = []
        min_illicit_per_tree = []
        for t_mod in tree.trees:
            q = torch.softmax(t_mod.leaf_logits, dim=-1)[:, 1]
            max_illicit_per_tree.append(q.max().item())
            min_illicit_per_tree.append(q.min().item())
        mean_max = sum(max_illicit_per_tree) / len(max_illicit_per_tree)
        mean_min = sum(min_illicit_per_tree) / len(min_illicit_per_tree)
        majority_th = frac * mean_max + (1.0 - frac) * mean_min - majority_margin
        info.update({'majority_votes': f"{votes}/{k}", 'mean_max_leaf': mean_max,
                     'mean_min_leaf': mean_min, 'majority_threshold': majority_th})
        floor_th = max(low, majority_th)
    else:
        q = torch.softmax(tree.leaf_logits, dim=-1)[:, 1]
        mean_max, mean_min = float(q.max().item()), float(q.min().item())
        majority_th = majority_frac * mean_max + (1.0 - majority_frac) * mean_min - majority_margin
        info.update({'majority_votes': '1/1', 'mean_max_leaf': mean_max,
                     'mean_min_leaf': mean_min, 'majority_threshold': majority_th})
        floor_th = max(low, quantile_floor, majority_th)

    probs_cal = model.predict_probs(z_all[val_mask], mode=inference_mode)[:, 1]
    y_cal = y_all[val_mask]
    lab = y_cal != -1
    p_np = probs_cal[lab].detach().cpu().numpy()
    y_np = y_cal[lab].detach().cpu().numpy()
    eff_high = max(floor_th, min(high, float(base_threshold)))
    best_th, best_f1 = find_optimal_threshold(
        y_np, p_np, low=floor_th, high=eff_high,
        num_steps=max(2, int(round((eff_high - floor_th) / 0.01)) + 1))
    best_th = min(best_th, float(base_threshold))
    info['retuned_threshold'] = best_th
    info['calibration_f1'] = best_f1

    # Jeśli w kroku t odsetek pozytywnych przy best_th jest poniżej min_pos_rate,
    # pozwalamy obniżyć próg maksymalnie do progu większościowego 3/5 drzew (floor_th), nigdy do 0.35
    probs_step = model.predict_probs(z_all[current_step_mask], mode=inference_mode)[:, 1]
    info['quantile_relaxed'] = False
    if probs_step.numel() > 0:
        pos_rate = float((probs_step >= best_th).float().mean().item())
        info['pos_rate_before_floor'] = pos_rate
        if pos_rate < min_pos_rate:
            quantile_th = float(torch.quantile(probs_step.detach().cpu().float(), 1.0 - min_pos_rate).item())
            relaxed = min(best_th, max(floor_th, quantile_th))
            info['quantile_relaxed'] = relaxed < best_th
            best_th = relaxed
    best_th = max(best_th, floor_th)

    final_th = round(float(best_th), 4)
    info['final_threshold'] = final_th
    if return_info:
        return final_th, info
    return final_th


def _build_balanced_replay(data, risk_scores: torch.Tensor, t: int, replay_size: int,
                           rng: np.random.Generator, recent_start: int = 30,
                           hard_frac: float = 0.5) -> torch.Tensor:
    """Bufor odtwarzania zbalansowany klasowo (50% illicit / 50% licit) z ujawnionych kroków <= t.

    * połowa illicit: wszystkie węzły illicit z kroku t + losowanie (z powtórzeniami, jeśli
      potrzeba) z węzłów illicit z kroków [recent_start, t],
    * połowa licit: `hard_frac` „trudnych” węzłów licit z [recent_start, t] o najwyższym
      bieżącym ryzyku p_illicit (potencjalne FP) + losowe węzły licit z całej historii 1..t.
    """
    y, ts = data.y, data.time_step
    dev = y.device

    def idx(mask):
        return torch.nonzero(mask, as_tuple=False).view(-1).cpu().numpy()

    recent = (ts >= recent_start) & (ts <= t)
    cur_ill = idx((ts == t) & (y == 1))
    ill_recent = idx(recent & (y == 1))
    lic_recent = idx(recent & (y == 0))
    lic_hist = idx((ts <= t) & (y == 0))

    n_half = max(replay_size // 2, 1)
    ill = cur_ill[:n_half]
    need = n_half - len(ill)
    if need > 0 and len(ill_recent) > 0:
        extra = rng.choice(ill_recent, size=need, replace=len(ill_recent) < need)
        ill = np.concatenate([ill, extra])

    n_hard = min(int(round(n_half * hard_frac)), len(lic_recent))
    if n_hard > 0:
        risk = risk_scores[torch.as_tensor(lic_recent, device=risk_scores.device)]
        top = torch.topk(risk, n_hard).indices.cpu().numpy()
        hard = lic_recent[top]
    else:
        hard = np.empty(0, dtype=np.int64)
    n_rand = min(n_half - len(hard), len(lic_hist))
    rand_lic = rng.choice(lic_hist, size=n_rand, replace=False) if n_rand > 0 else np.empty(0, dtype=np.int64)

    out = np.concatenate([ill, hard, rand_lic]).astype(np.int64)
    return torch.as_tensor(out, dtype=torch.long, device=dev)


def evaluate_continual_post_drift(
    model: EndToEndGNNSoftTree,
    data,
    inference_mode: str = 'max_path_leaf',
    threshold: float = 0.5,
    adapt_steps: int = 10,
    adapt_lr: float = 1e-3,
    replay_size: int = 2000,
    ewc_lambda: float = 1e-2,
    min_pos_rate: float = 0.015,
    hard_frac: float = 0.5,
    adapt_class_weights=ADAPT_CLASS_WEIGHTS,
    class_weights=None,
    seed: int = 42,
    verbose: bool = True,
    save_csv: Optional[str] = None,
    save_plot: Optional[str] = None,
):
    """Ewaluacja prekwencyjna z ciągłym douczaniem po szoku w kroku 43 (kroki 43-49).

    Protokół (bez wycieku danych z przyszłości):
    * kroki 35-42: model bazowy i statyczny próg walidacyjny `threshold`,
    * dla t = 43..49: (a) próg tau_t z `get_adaptive_post_drift_threshold` przy bieżącym stanie
      drzewa (zaadaptowanym na krokach < t), (b) predykcja kroku t, (c) ujawnienie etykiet kroku t
      i adaptacja przed krokiem t+1.
    * Adaptacja: zamrożony enkoder GNN (conv*, emb_norm), zamrożone `leaf_logits` i `beta`;
      aktualizowane są wyłącznie filtry routingu `inner_nodes` (w_i, b_i) na zbalansowanym
      buforze Replay (50/50), z kotwicą L2 (EWC-lite) `ewc_lambda * sum((p - p_pre)^2)`
      do wag sprzed dryfu.

    Zwraca (regimes_df, per_timestep_df, adapted_model, extras), gdzie extras zawiera
    `probs` (P(illicit) faktycznie użyte do oceny), `thresholds` (próg per węzeł),
    `step_thresholds` oraz `log` (diagnostyka każdego kroku).
    """
    if class_weights is not None and adapt_class_weights is None:
        adapt_class_weights = class_weights
    device = next(model.parameters()).device
    cw = torch.tensor(adapt_class_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)

    adapted = copy.deepcopy(model)
    adapted.freeze_gnn_encoder(True)
    tree = adapted.tree
    tree.set_routing_only_trainable(True)
    routing_params = tree.routing_parameters()
    anchor = [p.detach().clone() for p in routing_params]
    optimizer = torch.optim.AdamW(routing_params, lr=adapt_lr, weight_decay=0.0)
    is_forest = isinstance(tree, DifferentiableSoftDecisionForest)

    adapted.eval()
    with torch.no_grad():
        z_frozen, _ = adapted.get_tree_input(data.x, data.edge_index)
        z_frozen = z_frozen.detach()
        probs = tree.predict_probs(z_frozen, mode=inference_mode)[:, 1].clone()
    thresholds = torch.full_like(probs, float(threshold))

    val_idx = torch.nonzero(data.val_mask, as_tuple=False).view(-1)
    last_replay = None
    step_thresholds = {t: float(threshold) for t in range(35, DRIFT_STEP)}
    log = []

    for t in range(DRIFT_STEP, LAST_STEP + 1):
        step_all = data.time_step == t
        if not bool(step_all.any()):
            continue
        step_eval = step_all & (data.y != -1)
        step_probe = step_eval if bool(step_eval.any()) else step_all

        # (a) próg dla kroku t – wyłącznie z wiedzy dostępnej przed ujawnieniem etykiet kroku t
        post_revealed = torch.nonzero(
            (data.time_step >= DRIFT_STEP) & (data.time_step < t) & (data.y != -1),
            as_tuple=False).view(-1)
        calib_idx = post_revealed if post_revealed.numel() > 0 else val_idx
        tau_t, info = get_adaptive_post_drift_threshold(
            adapted, z_frozen, calib_idx, data.y, float(threshold), step_probe,
            min_pos_rate=min_pos_rate, inference_mode=inference_mode, return_info=True)

        # (b) predykcja kroku t bieżącym stanem drzewa
        tree.eval()
        with torch.no_grad():
            probs[step_all] = tree.predict_probs(z_frozen[step_all], mode=inference_mode)[:, 1]
        thresholds[step_all] = tau_t
        step_thresholds[t] = tau_t

        entry = {'time_step': t, 'threshold': tau_t, **info}
        if t >= LAST_STEP:
            log.append(entry)
            break

        # (c) ujawnienie etykiet kroku t -> adaptacja filtrów routingu przed krokiem t+1
        with torch.no_grad():
            risk = tree.predict_probs(z_frozen, mode='soft_average')[:, 1]
        replay_idx = _build_balanced_replay(data, risk, t, replay_size, rng, hard_frac=hard_frac)
        if replay_idx.numel() > 0:
            z_b, y_b = z_frozen[replay_idx], data.y[replay_idx]
            tree.train()
            loss_val = ewc_val = 0.0
            for _ in range(adapt_steps):
                optimizer.zero_grad()
                if is_forest:
                    loss = tree.compute_loss(tree(z_b), y_b, cw)
                else:
                    l_pp, l_q, i_pp, p_r = tree(z_b)
                    loss = tree.compute_loss(l_pp, l_q, i_pp, p_r, y_b, cw)
                ewc = sum(((p - a) ** 2).sum() for p, a in zip(routing_params, anchor))
                total = loss + ewc_lambda * ewc
                total.backward()
                torch.nn.utils.clip_grad_norm_(routing_params, max_norm=2.0)
                optimizer.step()
                loss_val, ewc_val = float(loss.item()), float(ewc.item())
            tree.eval()
            last_replay = replay_idx
            n_ill = int((y_b == 1).sum().item())
            entry.update({'replay_size': int(replay_idx.numel()), 'replay_illicit': n_ill,
                          'adapt_loss': loss_val, 'ewc_drift': ewc_val})
        log.append(entry)
        if verbose:
            print(f"[Continual t={t}] tau_t={tau_t:.4f} | replay={entry.get('replay_size', 0)} "
                  f"(illicit {entry.get('replay_illicit', 0)}) | loss={entry.get('adapt_loss', float('nan')):.4f} "
                  f"| ||θ-θ_pre||²={entry.get('ewc_drift', float('nan')):.4e}")

    regimes_df = evaluate_regimes(adapted, data, inference_mode, threshold=thresholds, probs=probs)
    per_step_df = evaluate_per_timestep(
        adapted, data, inference_mode, save_csv=save_csv, save_plot=save_plot,
        threshold=thresholds, probs=probs)
    if verbose:
        for name in ('Szok (krok 43)', 'Post-Drift (44-49)'):
            r = regimes_df.loc[name]
            print(f"[Continual] {name}: F1={r['F1_Illicit']:.4f} P={r['Precision_Illicit']:.4f} "
                  f"R={r['Recall_Illicit']:.4f} TP={int(r['TP'])} (śr. próg={r['Opt_Threshold']:.3f})")
    extras = {'probs': probs, 'thresholds': thresholds, 'step_thresholds': step_thresholds, 'log': log}
    return regimes_df, per_step_df, adapted, extras
