"""Warstwa zgodności DirectML (AMD Radeon, np. RX 6950 XT) dla PyTorch Geometric.

Backend `torch-directml` nie implementuje `Tensor.scatter_add_` (RuntimeError:
"Parametr jest niepoprawny" / "The parameter is incorrect") ani `scatter_reduce_`,
a z nich korzysta PyG (`torch_geometric.utils.scatter`) w GCNConv, GATConv i softmaxie
uwagi. `index_add_` działa poprawnie, więc:

* `scatter_add_` (indeks rozgłoszony wzdłuż `dim`, jak w PyG) -> `index_add_`,
* `scatter_reduce_` (max/min w softmaxie GAT) -> obliczenie na CPU i zwrot na GPU
  (przesyłane są tylko tensory rozmiaru [liczba_krawędzi, głowice]).

Wywołania na tensorach innych urządzeń (CPU/CUDA) pozostają bez zmian.
"""
import torch

_DML_TYPE = 'privateuseone'
_installed = False


def _is_dml(t: torch.Tensor) -> bool:
    return t.device.type == _DML_TYPE


def _broadcast_vector_index(index: torch.Tensor, dim: int):
    """Jeśli `index` jest rozgłoszonym wektorem wzdłuż `dim` (stride 0 poza `dim`), zwróć 1-D indeks."""
    if index.dim() == 1:
        return index
    if all(index.stride(d) == 0 or index.size(d) == 1 for d in range(index.dim()) if d != dim):
        sl = [0] * index.dim()
        sl[dim] = slice(None)
        return index[tuple(sl)]
    return None


def install() -> None:
    """Instaluje (idempotentnie) łatki na `Tensor.scatter_add_` i `Tensor.scatter_reduce_`."""
    global _installed
    if _installed:
        return
    orig_scatter_add_ = torch.Tensor.scatter_add_
    orig_scatter_reduce_ = torch.Tensor.scatter_reduce_

    def scatter_add_(self, dim, index, src):
        if not _is_dml(self):
            return orig_scatter_add_(self, dim, index, src)
        dim = dim % self.dim() if self.dim() > 0 else 0
        vec = _broadcast_vector_index(index, dim)
        if vec is not None and src.shape == index.shape:
            return self.index_add_(dim, vec, src)
        # Ogólny (wolniejszy) przypadek: obliczenia na CPU
        out = orig_scatter_add_(self.detach().cpu(), dim, index.cpu(), src.detach().cpu())
        return self.copy_(out.to(self.device))

    def scatter_reduce_(self, dim, index, src, reduce, *, include_self=True):
        if not _is_dml(self):
            return orig_scatter_reduce_(self, dim, index, src, reduce, include_self=include_self)
        out = orig_scatter_reduce_(self.detach().cpu(), dim, index.cpu(), src.detach().cpu(),
                                   reduce, include_self=include_self)
        return self.copy_(out.to(self.device))

    torch.Tensor.scatter_add_ = scatter_add_
    torch.Tensor.scatter_reduce_ = scatter_reduce_
    _installed = True
