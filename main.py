"""Skrypt uruchomieniowy: eksperymenty GNN + Soft Decision Tree (End-to-End) i badania ablacyjne.

Przykłady:
    python main.py --backbone gat_resnet --tree_depth 4 --use_skip True
    python main.py --backbone gcn --local_features_only True --semi_supervised_reg True
    python main.py --ablation          # predefiniowana siatka eksperymentów ablacyjnych
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from dataset import load_elliptic_pyg
from explain import (explain_node, explanation_to_dataframe, select_illicit_nodes,
                     visualize_embeddings_umap)
from models import EndToEndGNNSoftTree
from train import (DEFAULT_CLASS_WEIGHTS, evaluate_continual_post_drift,
                   evaluate_per_timestep, evaluate_regimes, generate_soft_targets,
                   train_distilled_tree, train_model)


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Oczekiwano wartości logicznej, otrzymano: {v}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-End Skip-GNN + Differentiable Soft Decision Tree (Elliptic++)")
    # Dane
    p.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    p.add_argument('--features_file', type=str, default='txs_features.csv')
    p.add_argument('--classes_file', type=str, default='txs_classes.csv')
    p.add_argument('--edges_file', type=str, default='txs_edgelist.csv')
    p.add_argument('--use_undirected', type=str2bool, default=False)
    # Model / ablacje
    p.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'gat_resnet'])
    p.add_argument('--tree_depth', type=int, default=4)
    p.add_argument('--use_skip', type=str2bool, default=True)
    p.add_argument('--local_features_only', type=str2bool, default=False)
    p.add_argument('--asymmetric_features', type=str2bool, default=False,
                   help='przekaż tylko pierwsze 93 cechy LF do warstw GNN, a pełne 182 cechy przez Skip Connection')
    p.add_argument('--normalize_emb', type=str2bool, default=True,
                   help='normalizacja nn.LayerNorm(100) osadzenia GNN przed konkatenacją Skip Connection')
    p.add_argument('--semi_supervised_reg', type=str2bool, default=None,
                   help='regularyzacja EMA na wszystkich węzłach kroków 1-29 (domyślnie True dla GAT)')
    p.add_argument('--tune_threshold', type=str2bool, default=True,
                   help='strojenie progu klasyfikacji (0.20-0.85) pod F1 na masce walidacyjnej (kroki 30-34)')
    p.add_argument('--num_trees', type=int, default=1,
                   help='liczba drzew w komitecie DifferentiableSoftDecisionForest (1 = pojedyncze drzewo)')
    p.add_argument('--feature_dropout', type=float, default=0.2,
                   help='prawdopodobieństwo Dropout1d cech w lesie miękkich drzew')
    p.add_argument('--distill', type=str2bool, default=False,
                   help='destylacja wiedzy z nauczyciela na 77%% nieoznaczonych węzłów unknown (kroki 1-29)')
    p.add_argument('--distill_alpha', type=float, default=0.5,
                   help='waga miękkich celów nauczyciela w funkcji straty podczas destylacji')
    p.add_argument('--teacher_trees', type=int, default=5,
                   help='liczba drzew modelu nauczyciela podczas destylacji')
    p.add_argument('--continual_post_drift', type=str2bool, default=False,
                   help='ciągłe douczanie drzewa po szoku w kroku 43 (kroki 43-49) z buforem Replay i zamrożonym GNN')
    p.add_argument('--adapt_steps', type=int, default=15)
    p.add_argument('--adapt_lr', type=float, default=5e-3)
    p.add_argument('--replay_size', type=int, default=2000)
    p.add_argument('--inference_mode', type=str, default='max_path_leaf', choices=['max_path_leaf', 'soft_average'])
    p.add_argument('--penalty_lambda', type=float, default=0.05)
    # Trening
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto',
                   help="'auto', 'rocm'/'cuda' (ROCm/CUDA), 'dml' (DirectML) lub 'cpu'")
    p.add_argument('--grad_checkpoint', type=str2bool, default=None,
                   help='checkpointing gradientów warstw GNN (domyślnie True na GPU dla GAT, oszczędza >10 GB VRAM)')
    p.add_argument('--gat_head_chunk', type=int, default=None,
                   help='liczba głowic GAT liczonych naraz (0 = standardowy GATConv); '
                        'domyślnie 2 na GPU (DirectML/ROCm/CUDA, oszczędza VRAM), 0 na CPU')
    # Wyjścia
    p.add_argument('--output_dir', type=str, default='results')
    p.add_argument('--explain_nodes', type=int, default=3, help='liczba węzłów illicit do wyjaśnienia (0 = pomiń)')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--umap', type=str2bool, default=False, help='wygeneruj wykres UMAP osadzeń')
    p.add_argument('--ablation', action='store_true', help='uruchom siatkę eksperymentów ablacyjnych')
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _directml_device():
    import torch_directml
    from dml_compat import install
    install()   # łatki scatter_add_/scatter_reduce_ potrzebne dla PyG na DirectML
    return torch_directml.device()


def resolve_device(name: str) -> torch.device:
    """'auto': ROCm/CUDA -> DirectML (AMD Radeon na Windows) -> CPU; 'rocm' jest aliasem 'cuda'."""
    if name == 'dml':
        return _directml_device()
    if name == 'rocm':
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch ROCm/HIP nie wykrył GPU (torch.cuda.is_available() == False). "
                               "Dla RX 6950 XT (gfx1030) upewnij się, że ustawiono HSA_OVERRIDE_GFX_VERSION=10.3.0.")
        return torch.device('cuda')
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        try:
            return _directml_device()
        except Exception as e:  # brak torch-directml / brak GPU
            print(f"[device] DirectML niedostępny ({e}); używam CPU")
            return torch.device('cpu')
    return torch.device(name)


def describe_device(device: torch.device) -> str:
    if device.type == 'cuda':
        idx = device.index or 0
        gpu_name = torch.cuda.get_device_name(idx)
        backend = f"ROCm/HIP {torch.version.hip}" if getattr(torch.version, 'hip', None) else f"CUDA {torch.version.cuda}"
        return f"{device} ({gpu_name}, {backend})"
    if device.type == 'privateuseone':
        try:
            import torch_directml
            gpu_name = torch_directml.device_name(device.index or 0).rstrip('\x00')
            return f"{device} ({gpu_name}, DirectML)"
        except Exception:
            return f"{device} (DirectML)"
    return str(device)


def run_experiment(args: argparse.Namespace, data=None, tag: str = None,
                   verbose: bool = True, soft_targets: torch.Tensor = None) -> dict:
    """Trenuje i ewaluuje jeden model; zapisuje tabele (CSV), wykresy i wyjaśnienia."""
    set_seed(args.seed)
    device = resolve_device(args.device)
    semi_reg = args.semi_supervised_reg
    if semi_reg is None:
        semi_reg = (args.backbone == 'gat') or bool(args.distill)

    tag = tag or (f"{args.backbone}_d{args.tree_depth}_skip{int(args.use_skip)}_"
                  f"norm{int(args.normalize_emb)}_asym{int(args.asymmetric_features)}_"
                  f"lf{int(args.local_features_only)}_semi{int(semi_reg)}_"
                  f"trees{args.num_trees}_distill{int(args.distill)}_{args.inference_mode}")
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    if data is None:
        data = load_elliptic_pyg(
            os.path.join(args.data_dir, args.features_file),
            os.path.join(args.data_dir, args.classes_file),
            os.path.join(args.data_dir, args.edges_file),
            use_undirected=args.use_undirected,
            local_features_only=args.local_features_only)
    data = data.to(device)
    if verbose:
        print(f"[{tag}] węzły={data.num_nodes}, krawędzie={data.edge_index.size(1)}, "
              f"cechy={data.x.size(1)}, urządzenie={describe_device(device)}")

    is_gpu = device.type in ('privateuseone', 'cuda')
    gat_head_chunk = args.gat_head_chunk
    if gat_head_chunk is None:
        gat_head_chunk = 2 if is_gpu else 0
    grad_ckpt = args.grad_checkpoint
    if grad_ckpt is None:
        grad_ckpt = is_gpu and (args.backbone in ('gat', 'gat_resnet'))
    model = EndToEndGNNSoftTree(
        in_channels=data.x.size(1), backbone=args.backbone, tree_depth=args.tree_depth,
        use_skip=args.use_skip, penalty_lambda=args.penalty_lambda,
        grad_checkpoint=grad_ckpt, gat_head_chunk=gat_head_chunk,
        normalize_emb=args.normalize_emb, asymmetric_features=args.asymmetric_features,
        num_trees=args.num_trees, feature_dropout=args.feature_dropout).to(device)

    if args.distill:
        model, history = train_distilled_tree(
            model, data, soft_targets=soft_targets,
            teacher_num_trees=args.teacher_trees, distill_alpha=args.distill_alpha,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, class_weights=DEFAULT_CLASS_WEIGHTS,
            inference_mode=args.inference_mode, tune_threshold=args.tune_threshold,
            verbose=verbose)
    else:
        model, history = train_model(
            model, data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, class_weights=DEFAULT_CLASS_WEIGHTS,
            inference_mode=args.inference_mode, semi_supervised_reg=semi_reg,
            tune_threshold=args.tune_threshold, verbose=verbose)

    opt_threshold = float(history.get('opt_threshold', 0.5))

    if args.continual_post_drift:
        regimes, per_step, model = evaluate_continual_post_drift(
            model, data, inference_mode=args.inference_mode, threshold=opt_threshold,
            adapt_steps=args.adapt_steps, adapt_lr=args.adapt_lr, replay_size=args.replay_size,
            class_weights=DEFAULT_CLASS_WEIGHTS, seed=args.seed, verbose=verbose,
            save_csv=os.path.join(out_dir, 'per_timestep_metrics.csv'),
            save_plot=os.path.join(out_dir, 'per_timestep_metrics.png'))
    else:
        regimes = evaluate_regimes(model, data, args.inference_mode, threshold=opt_threshold)
        per_step = evaluate_per_timestep(
            model, data, args.inference_mode,
            save_csv=os.path.join(out_dir, 'per_timestep_metrics.csv'),
            save_plot=os.path.join(out_dir, 'per_timestep_metrics.png'),
            threshold=opt_threshold)

    regimes.to_csv(os.path.join(out_dir, 'regime_metrics.csv'))
    hist_cols = [k for k in ('epoch', 'loss', 'val_F1_Illicit', 'val_threshold') if k in history]
    pd.DataFrame({k: history[k] for k in hist_cols}).to_csv(
        os.path.join(out_dir, 'training_history.csv'), index=False)
    torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'opt_threshold': opt_threshold},
               os.path.join(out_dir, 'model.pt'))

    if verbose:
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', 25)
        print(f"\n=== Wyniki testowe ({args.inference_mode}, Opt_Threshold={opt_threshold:.2f}) ===")
        print(regimes.round(4).to_string())
        print("\n=== Krok po kroku (35-49) ===")
        print(per_step.round(4).to_string())

    # XAI
    xai_tab_contrib, xai_tab_weights = float('nan'), float('nan')
    if args.explain_nodes > 0:
        nodes = select_illicit_nodes(model, data, inference_mode=args.inference_mode,
                                     top_n=args.explain_nodes)
        explanations, frames = [], []
        for n in nodes:
            res = explain_node(model, data, n, top_k=args.top_k, verbose=verbose)
            explanations.append(res)
            frames.append(explanation_to_dataframe(res))
        if explanations:
            xai_tab_contrib = float(np.mean([e['tabular_share_contrib_total'] for e in explanations]))
            xai_tab_weights = float(np.mean([e['tabular_share_weights_total'] for e in explanations]))
        if frames:
            pd.concat(frames).to_csv(os.path.join(out_dir, 'explanations_top_features.csv'), index=False)
            with open(os.path.join(out_dir, 'explanations.json'), 'w', encoding='utf-8') as f:
                json.dump(explanations, f, indent=2, ensure_ascii=False, default=str)
    if args.umap:
        visualize_embeddings_umap(model, data, save_path=os.path.join(out_dir, 'umap_embeddings.png'))

    # Zachowujemy miękkie cele z lasu drzew (jeśli to komitet), aby przyspieszyć destylację w ablacjach
    teacher_soft = None
    if args.num_trees > 1:
        teacher_soft = generate_soft_targets(model, data, inference_mode='soft_average').cpu()

    summary = {
        'tag': tag,
        'backbone': args.backbone,
        'tree_depth': args.tree_depth,
        'use_skip': args.use_skip,
        'normalize_emb': args.normalize_emb,
        'asymmetric_features': args.asymmetric_features,
        'local_features_only': args.local_features_only,
        'semi_supervised_reg': semi_reg,
        'num_trees': args.num_trees,
        'distill': args.distill,
        'continual_post_drift': args.continual_post_drift,
        'inference_mode': args.inference_mode,
        'best_epoch': history['best_epoch'],
        'best_val_F1_Illicit': history['best_val_F1_Illicit'],
        'Opt_Threshold': opt_threshold,
        'XAI_Tabular_Share_Contrib': xai_tab_contrib,
        'XAI_Tabular_Share_Weights': xai_tab_weights,
        '_teacher_soft_targets': teacher_soft,
    }
    for regime, row in regimes.iterrows():
        for m in ('Precision_Illicit', 'Recall_Illicit', 'F1_Illicit', 'MCC', 'ROC_AUC'):
            summary[f"{regime} | {m}"] = float(row[m])
    return summary


def ablation_grid(args: argparse.Namespace):
    """Siatka ablacji: warianty bazowe oraz nowe architektury GAT (LayerNorm, LF, Soft Forest, Destylacja)."""
    base = dict(vars(args))
    legacy_defaults = {'normalize_emb': False, 'semi_supervised_reg': False, 'tune_threshold': False,
                       'asymmetric_features': False, 'num_trees': 1, 'distill': False}
    configs = [
        ('baseline_skip', {**legacy_defaults}),
        ('no_skip', {**legacy_defaults, 'use_skip': False}),
        ('local_features_only', {**legacy_defaults, 'local_features_only': True}),
        ('semi_supervised_reg', {**legacy_defaults, 'semi_supervised_reg': True}),
        ('soft_average_inference', {**legacy_defaults, 'inference_mode': 'soft_average'}),
        ('gat', {**legacy_defaults, 'backbone': 'gat'}),
        ('gat_resnet', {**legacy_defaults, 'backbone': 'gat_resnet'}),
        ('gat_norm_semisup', {
            'backbone': 'gat', 'normalize_emb': True,
            'semi_supervised_reg': True, 'tune_threshold': True,
            'asymmetric_features': False, 'num_trees': 1, 'distill': False,
        }),
        ('gat_asymmetric_lf', {
            'backbone': 'gat', 'normalize_emb': True,
            'semi_supervised_reg': True, 'tune_threshold': True,
            'asymmetric_features': True, 'num_trees': 1, 'distill': False,
        }),
        ('gat_soft_forest', {
            'backbone': 'gat', 'normalize_emb': True,
            'semi_supervised_reg': True, 'tune_threshold': True,
            'asymmetric_features': True, 'num_trees': 5, 'feature_dropout': 0.2, 'distill': False,
        }),
        ('gat_distilled_tree', {
            'backbone': 'gat', 'normalize_emb': True,
            'semi_supervised_reg': True, 'tune_threshold': True,
            'asymmetric_features': True, 'num_trees': 1, 'distill': True,
        }),
    ]
    for name, override in configs:
        cfg = {**base, **override}
        yield name, argparse.Namespace(**cfg)


def run_with_fallback(args: argparse.Namespace, make_data=lambda: None, tag: str = None,
                      soft_targets: torch.Tensor = None) -> dict:
    """Uruchamia eksperyment; przy braku pamięci GPU najpierw ponawia na GPU z checkpointingiem
    i gat_head_chunk=1, a w ostateczności powtarza na CPU.

    Pełny graf Elliptic (204 tys. węzłów) z 8-głowicowym GAT potrafi przekroczyć 16 GB VRAM
    (np. Radeon RX 6950 XT) w trybie full-batch bez checkpointingu.
    """
    try:
        return run_experiment(args, data=make_data(), tag=tag, soft_targets=soft_targets)
    except RuntimeError as e:
        msg = str(e).lower()
        if args.device == 'cpu' or not ('video memory' in msg or 'out of memory' in msg):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.grad_checkpoint or (args.gat_head_chunk != 1):
            print(f"\n[device] Brak pamięci VRAM ({str(e).strip()[:80]}...). "
                  f"Ponawiam na GPU z --grad_checkpoint True --gat_head_chunk 1.")
            low_vram_args = argparse.Namespace(
                **{**vars(args), 'grad_checkpoint': True, 'gat_head_chunk': 1})
            try:
                return run_experiment(low_vram_args, data=make_data(), tag=tag, soft_targets=soft_targets)
            except RuntimeError as e2:
                msg2 = str(e2).lower()
                if not ('video memory' in msg2 or 'out of memory' in msg2):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print(f"\n[device] Brak pamięci GPU ({str(e).strip()[:90]}...). "
              f"Powtarzam eksperyment na CPU (wymuś bez fallbacku: --device cpu).")
        cpu_args = argparse.Namespace(**{**vars(args), 'device': 'cpu', 'gat_head_chunk': 0})
        return run_experiment(cpu_args, data=make_data(), tag=tag, soft_targets=soft_targets)


def main(argv=None):
    args = parse_args(argv)
    if not args.ablation:
        run_with_fallback(args)
        return

    # Ablacje: ładujemy dane osobno dla wariantu z/bez LF, reszta współdzielona
    cache = {}
    rows = []
    cached_teacher_soft = None
    for name, cfg in ablation_grid(args):
        key = (cfg.local_features_only, cfg.use_undirected)
        if key not in cache:
            cache[key] = load_elliptic_pyg(
                os.path.join(cfg.data_dir, cfg.features_file),
                os.path.join(cfg.data_dir, cfg.classes_file),
                os.path.join(cfg.data_dir, cfg.edges_file),
                use_undirected=cfg.use_undirected, local_features_only=cfg.local_features_only)
        print(f"\n########## ABLACJA: {name} ##########")
        soft_t = cached_teacher_soft if cfg.distill else None
        summary = run_with_fallback(cfg, make_data=cache[key].clone, tag=f"ablation_{name}",
                                    soft_targets=soft_t)
        if summary.get('_teacher_soft_targets') is not None:
            cached_teacher_soft = summary['_teacher_soft_targets']
        summary.pop('_teacher_soft_targets', None)
        summary['experiment'] = name
        rows.append(summary)
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'ablation_summary.csv'), index=False)

    print("\n=== Podsumowanie ablacji ===")
    df = pd.DataFrame(rows).set_index('experiment')
    summary_cols = ['best_epoch', 'best_val_F1_Illicit', 'Opt_Threshold']
    for regime in ('Całość (35-49)', 'Pre-Drift (35-42)', 'Post-Drift (44-49)'):
        for m in ('Precision_Illicit', 'Recall_Illicit', 'F1_Illicit', 'MCC'):
            col = f"{regime} | {m}"
            if col in df.columns:
                summary_cols.append(col)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 30)
    print(df[summary_cols].round(4).to_string())


if __name__ == '__main__':
    main()
