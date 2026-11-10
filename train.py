"""Pętla treningowa, early stopping, strojenie progu, destylacja wiedzy oraz ciągłe douczanie po dryfcie."""
import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from models import DifferentiableSoftDecisionForest, EndToEndGNNSoftTree

DEFAULT_CLASS_WEIGHTS = (0.3, 0.7)   # [licit, illicit]
THRESHOLD_RANGE = (0.20, 0.85)
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
    f1_scores = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
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


@torch.no_grad()
def evaluate_mask(model: EndToEndGNNSoftTree, data, mask: torch.Tensor,
                  inference_mode: str = 'max_path_leaf',
                  probs: Optional[torch.Tensor] = None,
                  threshold: float = 0.5) -> Dict[str, float]:
    """Metryki na podanej masce przy zadanym progu decyzyjnym (np. wyznaczonym na walidacji)."""
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    y_true = data.y[mask].cpu().numpy()
    p = probs[mask].detach().cpu().numpy()
    return compute_metrics(y_true, p, threshold=threshold)


def _train_period_mask(data) -> torch.Tensor:
    """Wszystkie węzły (także `unknown`) z kroków 1-29 – do semi-supervised regularyzacji i destylacji."""
    return (data.time_step >= 1) & (data.time_step <= 29)


def _train_unknown_mask(data) -> torch.Tensor:
    """Nieoznaczone węzły (`unknown`, y == -1) wyłącznie z okresu treningowego (kroki 1-29)."""
    return _train_period_mask(data) & (data.y == -1)


@torch.no_grad()
def generate_soft_targets(teacher_model: EndToEndGNNSoftTree, data,
                          inference_mode: str = 'soft_average') -> torch.Tensor:
    """Generuje rozkłady prawdopodobieństwa [N, 2] modelu nauczyciela (Frosst & Hinton).

    Podczas destylacji wykorzystywane są wyłącznie węzły z kroków 1-29 (w tym 77% `unknown`),
    co gwarantuje całkowity brak wycieku informacji z walidacji (30-34) i testu (35-49).
    """
    teacher_model.eval()
    return teacher_model.predict_probs(data.x, data.edge_index, mode=inference_mode).detach()


def train_model(model: EndToEndGNNSoftTree, data, epochs: int = 500, lr: float = 1e-3,
                weight_decay: float = 1e-4, patience: int = 50,
                class_weights=DEFAULT_CLASS_WEIGHTS, inference_mode: str = 'max_path_leaf',
                semi_supervised_reg: Optional[bool] = None, tune_threshold: bool = True,
                soft_targets: Optional[torch.Tensor] = None, distill_alpha: float = 0.5,
                verbose: bool = True, log_every: int = 10):
    """Trening full-batch; strata na `train_mask` (oraz opcjonalnie destylacja na krokach 1-29),
    early stopping i strojenie progu (0.20-0.85) po F1_Illicit na `val_mask` (kroki 30-34).

    Domyślnie dla enkodera GAT aktywowana jest synergia z regularyzacją półnadzorowaną EMA
    na wszystkich węzłach z kroków 1-29.
    Zwraca (model z wczytanym najlepszym stanem, historia treningu ze znalezionym `opt_threshold`).
    """
    device = next(model.parameters()).device
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if semi_supervised_reg is None:
        semi_supervised_reg = (model.backbone == 'gat') or (soft_targets is not None)

    train_period_mask = _train_period_mask(data)
    reg_mask = train_period_mask if semi_supervised_reg else None
    distill_mask = train_period_mask if soft_targets is not None else None
    if soft_targets is not None:
        soft_targets = soft_targets.to(device)

    history = {'epoch': [], 'loss': [], 'val_F1_Illicit': [], 'val_threshold': []}
    best_f1, best_epoch, best_thr, best_state, bad_epochs = -1.0, 0, 0.5, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        m = data.train_mask

        if isinstance(model.tree, DifferentiableSoftDecisionForest):
            forest_outputs, _ = model(data.x, data.edge_index)
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
    return model, history


def train_distilled_tree(student_model: EndToEndGNNSoftTree, data,
                         teacher_model: Optional[EndToEndGNNSoftTree] = None,
                         soft_targets: Optional[torch.Tensor] = None,
                         teacher_num_trees: int = 5, distill_alpha: float = 0.5,
                         epochs: int = 500, lr: float = 1e-3, weight_decay: float = 1e-4,
                         patience: int = 50, class_weights=DEFAULT_CLASS_WEIGHTS,
                         inference_mode: str = 'max_path_leaf', tune_threshold: bool = True,
                         verbose: bool = True, log_every: int = 10):
    """Trenuje pojedyncze miękkie drzewo decyzyjne z destylacją wiedzy na 77% węzłów `unknown` (kroki 1-29).

    Jeśli nie przekazano gotowych `soft_targets` ani `teacher_model`, funkcja najpierw trenuje
    model nauczyciela (las miękkich drzew z tym samym enkoderem GNN) na krokach 1-29, generuje
    rozkłady prawdopodobieństwa dla wszystkich węzłów z kroków 1-29 (w tym 77% nieoznaczonych
    węzłów `unknown`) i destyluje wiedzę do pojedynczego drzewa `student_model`.
    """
    device = next(student_model.parameters()).device
    if soft_targets is None:
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
        soft_targets = generate_soft_targets(teacher_model, data, inference_mode='soft_average')
        if verbose:
            unk_n = int(_train_unknown_mask(data).sum().item())
            tot_n = int(_train_period_mask(data).sum().item())
            print(f"[Destylacja] Wygenerowano miękkie cele dla {tot_n} węzłów z kroków 1-29 "
                  f"(w tym {unk_n} węzłów unknown, {unk_n / max(tot_n, 1):.1%}).")

    return train_model(
        student_model, data, epochs=epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, class_weights=class_weights, inference_mode=inference_mode,
        semi_supervised_reg=True, tune_threshold=tune_threshold,
        soft_targets=soft_targets, distill_alpha=distill_alpha,
        verbose=verbose, log_every=log_every)


def evaluate_regimes(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf',
                     threshold: float = 0.5, probs: Optional[torch.Tensor] = None) -> pd.DataFrame:
    """Metryki w 4 reżimach czasowych: Całość 35-49, Pre-Drift 35-42, Szok 43, Post-Drift 44-49."""
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    regimes = [
        ('Całość (35-49)', data.test_mask),
        ('Pre-Drift (35-42)', data.test_pre_drift_mask),
        ('Szok (krok 43)', data.test_drift_step43_mask),
        ('Post-Drift (44-49)', data.test_post_drift_mask),
    ]
    rows = []
    for name, mask in regimes:
        res = evaluate_mask(model, data, mask, inference_mode, probs=probs, threshold=threshold)
        res = {'Reżim': name, 'N': int(mask.sum()), 'N_illicit': int((data.y[mask] == 1).sum()),
               'Opt_Threshold': float(threshold), **res}
        rows.append(res)
    return pd.DataFrame(rows).set_index('Reżim')


def evaluate_per_timestep(model: EndToEndGNNSoftTree, data, inference_mode: str = 'max_path_leaf',
                          save_csv: Optional[str] = None, save_plot: Optional[str] = None,
                          threshold: float = 0.5, probs: Optional[torch.Tensor] = None) -> pd.DataFrame:
    """Ewaluacja krok po kroku t = 35..49; opcjonalny zapis tabeli (CSV) i wykresu (PNG)."""
    if probs is None:
        probs = predict_all_probs(model, data, inference_mode)
    labeled = data.y != -1
    rows = []
    for t in range(35, 50):
        mask = (data.time_step == t) & labeled
        res = evaluate_mask(model, data, mask, inference_mode, probs=probs, threshold=threshold)
        rows.append({'time_step': t, 'N': int(mask.sum()),
                     'N_illicit': int((data.y[mask] == 1).sum()),
                     'Opt_Threshold': float(threshold), **res})
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
        ax.set_title(f'Metryki klasy illicit w kolejnych krokach testowych (próg={threshold:.2f})')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        fig.tight_layout()
        fig.savefig(save_plot, dpi=150)
        plt.close(fig)
    return df


def _sample_replay_buffer(data, max_step: int, replay_size: int, rng: np.random.Generator) -> torch.Tensor:
    """Losuje zbalansowany bufor odtwarzania (Replay Buffer) z oznaczonych węzłów z kroków <= max_step."""
    hist_mask = (data.time_step <= max_step) & (data.y != -1)
    illicit_idx = torch.nonzero(hist_mask & (data.y == 1), as_tuple=False).view(-1).cpu().numpy()
    licit_idx = torch.nonzero(hist_mask & (data.y == 0), as_tuple=False).view(-1).cpu().numpy()

    n_illicit = min(len(illicit_idx), replay_size // 2)
    n_licit = min(len(licit_idx), replay_size - n_illicit)

    sel_illicit = rng.choice(illicit_idx, size=n_illicit, replace=False) if n_illicit > 0 else np.empty(0, dtype=np.int64)
    sel_licit = rng.choice(licit_idx, size=n_licit, replace=False) if n_licit > 0 else np.empty(0, dtype=np.int64)
    replay_idx = np.concatenate([sel_illicit, sel_licit])
    return torch.tensor(replay_idx, dtype=torch.long, device=data.x.device)


def evaluate_continual_post_drift(
    model: EndToEndGNNSoftTree,
    data,
    inference_mode: str = 'max_path_leaf',
    threshold: float = 0.5,
    adapt_steps: int = 15,
    adapt_lr: float = 5e-3,
    replay_size: int = 2000,
    class_weights=DEFAULT_CLASS_WEIGHTS,
    seed: int = 42,
    verbose: bool = True,
    save_csv: Optional[str] = None,
    save_plot: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, EndToEndGNNSoftTree]:
    """Ewaluacja z ciągłym douczaniem po szoku w kroku 43 (kroki 43-49) z buforem odtwarzania.

    Rygor braku wycieku danych z przyszłości:
    1. Kroki 35-43 są oceniane modelem sprzed adaptacji (przed ujawnieniem etykiet kroku 43).
    2. W każdym kroku t in {43, ..., 48}, po dokonaniu predykcji na kroku t, wagi enkodera GNN
       są zamrażane (`requires_grad=False`), a parametry liści i filtrów drzewa są aktualizowane
       na ujawnionych etykietach z kroku t połączonych z buforem odtwarzania (Replay Buffer)
       z wcześniejszych kroków czasowych (< t).
    3. Zaktualizowane drzewo wykonuje predykcję dla kolejnego kroku t + 1.

    Zwraca (regimes_df, per_timestep_df, adapted_model).
    """
    device = next(model.parameters()).device
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)

    adapted_model = copy.deepcopy(model)
    adapted_model.freeze_gnn_encoder(True)

    # 1. Predykcja bazowa dla kroków <= 43 oraz zamrożone wejście drzewa Z (GNN zamrożony)
    adapted_model.eval()
    with torch.no_grad():
        probs = predict_all_probs(adapted_model, data, inference_mode).clone()
        z_frozen, _ = adapted_model.get_tree_input(data.x, data.edge_index)
        z_frozen = z_frozen.detach()

    tree_params = [p for p in adapted_model.tree.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(tree_params, lr=adapt_lr, weight_decay=1e-4)
    labeled = data.y != -1

    # 2. Pętla adaptacji po każdym kroku t = 43..48 i predykcja na kroku t + 1
    for t in range(43, 49):
        cur_mask = (data.time_step == t) & labeled
        cur_idx = torch.nonzero(cur_mask, as_tuple=False).view(-1)
        if cur_idx.numel() > 0:
            replay_idx = _sample_replay_buffer(data, max_step=t - 1, replay_size=replay_size, rng=rng)
            # Przeważenie nowego reżimu względem bufora odtwarzania
            repeat_factor = max(1, replay_idx.numel() // max(cur_idx.numel(), 1))
            batch_idx = torch.cat([cur_idx.repeat(repeat_factor), replay_idx], dim=0)
            z_batch = z_frozen[batch_idx]
            y_batch = data.y[batch_idx]

            adapted_model.tree.train()
            for _ in range(adapt_steps):
                optimizer.zero_grad()
                if isinstance(adapted_model.tree, DifferentiableSoftDecisionForest):
                    forest_out = adapted_model.tree(z_batch)
                    loss = adapted_model.tree.compute_loss(forest_out, y_batch, cw)
                else:
                    l_pp, l_q, i_pp, p_r = adapted_model.tree(z_batch)
                    loss = adapted_model.tree.compute_loss(l_pp, l_q, i_pp, p_r, y_batch, cw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tree_params, max_norm=2.0)
                optimizer.step()

        # Predykcja na kroku t + 1 bez zaglądania do jego etykiet
        next_mask = data.time_step == (t + 1)
        adapted_model.eval()
        with torch.no_grad():
            probs[next_mask] = adapted_model.tree.predict_probs(
                z_frozen[next_mask], mode=inference_mode)[:, 1]

    regimes_df = evaluate_regimes(adapted_model, data, inference_mode, threshold=threshold, probs=probs)
    per_step_df = evaluate_per_timestep(
        adapted_model, data, inference_mode,
        save_csv=save_csv, save_plot=save_plot,
        threshold=threshold, probs=probs)
    if verbose:
        post_f1 = float(regimes_df.loc['Post-Drift (44-49)', 'F1_Illicit'])
        print(f"[Continual Post-Drift 43->49] Post-Drift F1_Illicit={post_f1:.4f} (próg={threshold:.2f})")
    return regimes_df, per_step_df, adapted_model
