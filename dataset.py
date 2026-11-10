"""Potok danych: wczytanie Elliptic / Elliptic++ i konstrukcja grafu PyTorch Geometric.

Kluczowe założenia metodologiczne:
* wszystkie węzły (również `unknown`) pozostają w grafie – filtrowanie odbywa się
  wyłącznie przez maski logiczne;
* imputacja (mediana) i standaryzacja są dopasowywane (`fit`) wyłącznie na węzłach
  z okresu treningowego (kroki 1-29), a następnie stosowane (`transform`) do całości;
* podział chronologiczny: train 1-29, val 30-34, test 35-49.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

NUM_LOCAL_FEATURES = 93
TRAIN_LAST_STEP = 29
VAL_RANGE = (30, 34)
TEST_RANGE = (35, 49)
PRE_DRIFT_RANGE = (35, 42)
DRIFT_STEP = 43
POST_DRIFT_RANGE = (44, 49)


def load_elliptic_pyg(
    features_path: str,
    classes_path: str,
    edges_path: str,
    use_undirected: bool = False,
    local_features_only: bool = False,
) -> Data:
    """Wczytuje zbiór Elliptic / Elliptic++ i buduje obiekt `torch_geometric.data.Data`.

    Args:
        features_path: ścieżka do `txs_features.csv` (txId, time_step, cechy...).
        classes_path: ścieżka do `txs_classes.csv` (txId, class).
        edges_path: ścieżka do `txs_edgelist.csv` (txId1, txId2).
        use_undirected: jeśli True, `edge_index` zostaje zamieniony na nieskierowany.
        local_features_only: jeśli True, zostaje tylko 93 cech lokalnych (ablacja LF).

    Returns:
        Data z polami: x, edge_index, y, time_step, 6 masek, feature_names, tx_ids.
    """
    # 1. Wczytanie surowych plików CSV
    df_feat = pd.read_csv(features_path)
    df_class = pd.read_csv(classes_path)
    df_edges = pd.read_csv(edges_path)

    # Ujednolicenie nazw pierwszych kolumn (Elliptic i Elliptic++ różnią się nazewnictwem)
    df_feat = df_feat.rename(columns={df_feat.columns[0]: "txId", df_feat.columns[1]: "time_step"})
    df_class = df_class.rename(columns={df_class.columns[0]: "txId", df_class.columns[1]: "class"})
    df_edges = df_edges.rename(columns={df_edges.columns[0]: "txId1", df_edges.columns[1]: "txId2"})

    # 2. Połączenie cech i klas po txId (kolejność wierszy = kolejność w pliku cech)
    df_class = df_class[["txId", "class"]].drop_duplicates(subset="txId")
    df = pd.merge(df_feat, df_class, on="txId", how="left")
    if len(df) != len(df_feat):
        raise RuntimeError("Połączenie cech z klasami zmieniło liczbę węzłów (zduplikowane txId?).")

    tx_ids = df["txId"].values
    num_nodes = len(df)

    # Mapowanie txId -> ciągły indeks [0, N-1]
    id_to_idx = pd.Series(np.arange(num_nodes, dtype=np.int64), index=tx_ids)

    # 3. Krawędzie (odrzucamy ewentualne krawędzie wskazujące na nieznane txId)
    src = df_edges["txId1"].map(id_to_idx)
    dst = df_edges["txId2"].map(id_to_idx)
    valid = src.notna() & dst.notna()
    edge_index = torch.tensor(
        np.stack([src[valid].values.astype(np.int64), dst[valid].values.astype(np.int64)]),
        dtype=torch.long,
    )
    if use_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    # 4. Etykiety: '1' (illicit) -> 1, '2' (licit) -> 0, '3' / 'unknown' / brak -> -1
    raw_classes = df["class"].astype(str).str.strip().values
    y_np = np.full(num_nodes, -1, dtype=np.int64)
    y_np[raw_classes == "1"] = 1
    y_np[raw_classes == "2"] = 0
    y = torch.tensor(y_np, dtype=torch.long)

    # 5. Kroki czasowe i maski (tylko węzły oznaczone y != -1)
    time_step = torch.tensor(df["time_step"].values.astype(np.int64), dtype=torch.long)
    labeled = y != -1

    def _range_mask(lo: int, hi: int) -> torch.Tensor:
        return (time_step >= lo) & (time_step <= hi) & labeled

    train_mask = _range_mask(1, TRAIN_LAST_STEP)
    val_mask = _range_mask(*VAL_RANGE)
    test_mask = _range_mask(*TEST_RANGE)
    test_pre_drift_mask = _range_mask(*PRE_DRIFT_RANGE)
    test_drift_step43_mask = _range_mask(DRIFT_STEP, DRIFT_STEP)
    test_post_drift_mask = _range_mask(*POST_DRIFT_RANGE)

    # 6. Macierz cech X: bezwzględne usunięcie txId, time_step, class oraz kolumn nieliczbowych
    candidate_cols = [c for c in df.columns if c not in ("txId", "time_step", "class")]
    feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    dropped = sorted(set(candidate_cols) - set(feature_cols))
    if dropped:
        print(f"[dataset] Usunięto kolumny nieliczbowe: {dropped}")
    if local_features_only:
        feature_cols = feature_cols[:NUM_LOCAL_FEATURES]  # LF = pierwsze 93 kolumny

    X_raw = df[feature_cols].values.astype(np.float32)
    X_raw[~np.isfinite(X_raw)] = np.nan  # +-inf traktujemy jak braki

    # 7. Imputacja + standaryzacja: fit TYLKO na krokach 1-29, transform na całości
    train_period = ((time_step >= 1) & (time_step <= TRAIN_LAST_STEP)).numpy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(X_raw[train_period])
    X_imputed = imputer.transform(X_raw)
    scaler.fit(X_imputed[train_period])
    X_scaled = scaler.transform(X_imputed).astype(np.float32)
    # Cechy stałe w okresie treningowym mają std=0 -> StandardScaler zwraca 0 (brak NaN)
    x = torch.tensor(X_scaled, dtype=torch.float32)

    # 8. Złożenie obiektu Data
    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_step
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.test_pre_drift_mask = test_pre_drift_mask
    data.test_drift_step43_mask = test_drift_step43_mask
    data.test_post_drift_mask = test_post_drift_mask
    data.feature_names = list(feature_cols)
    data.tx_ids = tx_ids
    return data


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    graph = load_elliptic_pyg(
        os.path.join(base, "txs_features.csv"),
        os.path.join(base, "txs_classes.csv"),
        os.path.join(base, "txs_edgelist.csv"),
    )
    print(f"Załadowano graf: {graph}")
    print(f"Cechy: {graph.x.size(1)} | węzły: {graph.num_nodes} | krawędzie: {graph.edge_index.size(1)}")
    for name in ("train_mask", "val_mask", "test_mask", "test_pre_drift_mask",
                 "test_drift_step43_mask", "test_post_drift_mask"):
        m = getattr(graph, name)
        print(f"{name:24s} n={int(m.sum()):6d} illicit={int((graph.y[m] == 1).sum()):5d}")