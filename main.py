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
from train import (DEFAULT_CLASS_WEIGHTS, evaluate_per_timestep, evaluate_regimes,
                   train_model)


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
    p.add_argument('--semi_supervised_reg', type=str2bool, default=False)
    p.add_argument('--inference_mode', type=str, default='max_path_leaf', choices=['max_path_leaf', 'soft_average'])
    p.add_argument('--penalty_lambda', type=float, default=0.05)
    # Trening
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto', help="'auto', 'dml' (AMD Radeon / DirectML), 'cuda' lub 'cpu'")
    p.add_argument('--grad_checkpoint', type=str2bool, default=False,
                   help='checkpointing gradientów warstw GNN (mniejsze zużycie VRAM, np. dla --backbone gat)')
    p.add_argument('--gat_head_chunk', type=int, default=None,
                   help='liczba głowic GAT liczonych naraz (0 = standardowy GATConv); '
                        'domyślnie 2 na DirectML (oszczędza VRAM), 0 na CPU/CUDA')
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
    """'auto': CUDA/ROCm -> DirectML (AMD Radeon na Windows) -> CPU; 'dml' wymusza DirectML."""
    if name == 'dml':
        return _directml_device()
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        try:
            return _directml_device()
        except Exception as e:  # brak torch-directml / brak GPU
            print(f"[device] DirectML niedostępny ({e}); używam CPU")
            return torch.device('cpu')
    return torch.device(name)


def run_experiment(args: argparse.Namespace, data=None, tag: str = None, verbose: bool = True) -> dict:
    """Trenuje i ewaluuje jeden model; zapisuje tabele (CSV), wykresy i wyjaśnienia."""
    set_seed(args.seed)
    device = resolve_device(args.device)
    tag = tag or (f"{args.backbone}_d{args.tree_depth}_skip{int(args.use_skip)}_"
                  f"lf{int(args.local_features_only)}_semi{int(args.semi_supervised_reg)}_{args.inference_mode}")
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
              f"cechy={data.x.size(1)}, urządzenie={device}")

    gat_head_chunk = args.gat_head_chunk
    if gat_head_chunk is None:
        gat_head_chunk = 2 if device.type == 'privateuseone' else 0
    model = EndToEndGNNSoftTree(
        in_channels=data.x.size(1), backbone=args.backbone, tree_depth=args.tree_depth,
        use_skip=args.use_skip, penalty_lambda=args.penalty_lambda,
        grad_checkpoint=args.grad_checkpoint, gat_head_chunk=gat_head_chunk).to(device)

    model, history = train_model(
        model, data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, class_weights=DEFAULT_CLASS_WEIGHTS,
        inference_mode=args.inference_mode, semi_supervised_reg=args.semi_supervised_reg,
        verbose=verbose)

    regimes = evaluate_regimes(model, data, args.inference_mode)
    per_step = evaluate_per_timestep(
        model, data, args.inference_mode,
        save_csv=os.path.join(out_dir, 'per_timestep_metrics.csv'),
        save_plot=os.path.join(out_dir, 'per_timestep_metrics.png'))
    regimes.to_csv(os.path.join(out_dir, 'regime_metrics.csv'))
    pd.DataFrame({k: history[k] for k in ('epoch', 'loss', 'val_F1_Illicit')}).to_csv(
        os.path.join(out_dir, 'training_history.csv'), index=False)
    torch.save({'state_dict': model.state_dict(), 'args': vars(args)}, os.path.join(out_dir, 'model.pt'))

    if verbose:
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', 20)
        print(f"\n=== Wyniki testowe ({args.inference_mode}) ===")
        print(regimes.round(4).to_string())
        print("\n=== Krok po kroku (35-49) ===")
        print(per_step.round(4).to_string())

    # XAI
    if args.explain_nodes > 0:
        nodes = select_illicit_nodes(model, data, inference_mode=args.inference_mode,
                                     top_n=args.explain_nodes)
        explanations, frames = [], []
        for n in nodes:
            res = explain_node(model, data, n, top_k=args.top_k, verbose=verbose)
            explanations.append(res)
            frames.append(explanation_to_dataframe(res))
        if frames:
            pd.concat(frames).to_csv(os.path.join(out_dir, 'explanations_top_features.csv'), index=False)
            with open(os.path.join(out_dir, 'explanations.json'), 'w', encoding='utf-8') as f:
                json.dump(explanations, f, indent=2, ensure_ascii=False, default=str)
    if args.umap:
        visualize_embeddings_umap(model, data, save_path=os.path.join(out_dir, 'umap_embeddings.png'))

    summary = {'tag': tag, 'backbone': args.backbone, 'tree_depth': args.tree_depth,
               'use_skip': args.use_skip, 'local_features_only': args.local_features_only,
               'semi_supervised_reg': args.semi_supervised_reg,
               'inference_mode': args.inference_mode,
               'best_epoch': history['best_epoch'],
               'best_val_F1_Illicit': history['best_val_F1_Illicit']}
    for regime, row in regimes.iterrows():
        for m in ('F1_Illicit', 'Precision_Illicit', 'Recall_Illicit', 'MCC', 'ROC_AUC'):
            summary[f"{regime} | {m}"] = float(row[m])
    return summary


def ablation_grid(args: argparse.Namespace):
    """Siatka ablacji: skip, cechy lokalne, semi-supervised reg., tryb inferencji, backbone."""
    base = dict(vars(args))
    configs = [
        ('baseline_skip', {}),
        ('no_skip', {'use_skip': False}),
        ('local_features_only', {'local_features_only': True}),
        ('semi_supervised_reg', {'semi_supervised_reg': True}),
        ('soft_average_inference', {'inference_mode': 'soft_average'}),
        ('gat', {'backbone': 'gat'}),
        ('gat_resnet', {'backbone': 'gat_resnet'}),
    ]
    for name, override in configs:
        cfg = {**base, **override}
        yield name, argparse.Namespace(**cfg)


def main(argv=None):
    args = parse_args(argv)
    if not args.ablation:
        run_experiment(args)
        return

    # Ablacje: ładujemy dane osobno dla wariantu z/bez LF, reszta współdzielona
    cache = {}
    rows = []
    for name, cfg in ablation_grid(args):
        key = (cfg.local_features_only, cfg.use_undirected)
        if key not in cache:
            cache[key] = load_elliptic_pyg(
                os.path.join(cfg.data_dir, cfg.features_file),
                os.path.join(cfg.data_dir, cfg.classes_file),
                os.path.join(cfg.data_dir, cfg.edges_file),
                use_undirected=cfg.use_undirected, local_features_only=cfg.local_features_only)
        print(f"\n########## ABLACJA: {name} ##########")
        summary = run_experiment(cfg, data=cache[key].clone(), tag=f"ablation_{name}")
        summary['experiment'] = name
        rows.append(summary)
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'ablation_summary.csv'), index=False)
    print("\n=== Podsumowanie ablacji ===")
    df = pd.DataFrame(rows).set_index('experiment')
    print(df[['best_epoch', 'best_val_F1_Illicit', 'Całość (35-49) | F1_Illicit',
              'Pre-Drift (35-42) | F1_Illicit', 'Post-Drift (44-49) | F1_Illicit']].round(4).to_string())


if __name__ == '__main__':
    main()
