from typing import Literal
from nptyping import NDArray
import numpy as np
from xarray import DataArray
from tensorflow import keras
from keras.models import Model
from jetstream_hugo.plots import Clusterplot
from jetstream_hugo.definitions import get_region


def half(x: int):
    if x % 2 == 0:
        return x // 2, x // 2
    return x // 2 + 1, x // 2


def rescale(X: NDArray | DataArray):
    Xmin, Xmax = X.min(), X.max()
    X = (X - Xmin) / (Xmax - Xmin)
    return X, Xmin, Xmax
    

def descale(X: NDArray, Xmin: float, Xmax: float):
    return Xmin + (Xmax - Xmin) * X


def preprocess(
    X: NDArray | DataArray,
    strides: list,
    strategy: Literal["crop"] | Literal["pad"] | list | tuple = "crop",
) -> NDArray | DataArray:
    if isinstance(strategy, str):
        strategy = (strategy, strategy)
    divisors = [1, 1]
    try:
        dims = list(X.dims)[1:]
    except AttributeError:
        dims = []
    for strides_ in strides:
        if isinstance(strides_, tuple):
            for i in range(2):
                divisors[i] *= strides_[i]
        else:
            for i in range(2):
                divisors[i] *= strides_
    for i, n in enumerate(X.shape[1:3]):
        factor = n / divisors[i]
        if strategy[i] == "crop":
            newn = int(np.floor(factor) * divisors[i])
            x1, x2 = half(n - newn)
            idxs = np.arange(x1, x1 + newn)
            if isinstance(X, NDArray):
                X = np.take(X, idxs, axis=i + 1)
            elif isinstance(X, DataArray):
                X = X.isel({dims[i]: idxs})
        elif strategy[i] == "pad":
            newn = int(np.ceil(factor) * divisors[i])
            x1, x2 = half(newn - n)
            if isinstance(X, NDArray):
                padding = np.zeros((X.ndim, 2), dtype=int)
                padding[i + 1] = [x1, x2]
                X = np.pad(X, padding, mode="edge")
            elif isinstance(X, DataArray):
                X = X.pad({dims[i]: (x1, x2)}, mode="edge")
    return X


def showcase_ae(
    da: DataArray, model: Model, n: int, symmetrize: bool = False, **kwargs
) -> Clusterplot:
    nrow = int(np.ceil(n / 3)) * 2
    ncol = 3
    tsteps = np.random.randint(da.shape[0], size=n)
    da = rescale(da[tsteps])
    to_plot1 = [da[t] for t, _ in enumerate(tsteps)]
    representations = [
        model.decoder(model.encoder(x.values[None, ..., None])) for x in to_plot1
    ]
    to_plot2 = [da[0].copy(data=y[0, ..., 0]) for y in representations]
    to_plot = [to_plot1, to_plot2]
    to_plot = [
        to_plot[i // 3 % 2][i % 3 + (i // 6) * 3] for i in range(len(tsteps) * 2)
    ]
    if symmetrize:
        to_plot = [2 * (tplt - 0.5) for tplt in to_plot]
    clu = Clusterplot(nrow, ncol, region=get_region(da))
    clu.add_contourf(to_plot, **kwargs)
    return clu


def showcase_desom(
    da: DataArray, model: Model, symmetrize: bool = False, **kwargs
) -> Clusterplot:
    if "time" in da.dims:
        da = da[0]
    decoded_centers = model.decoder(model.som_layer.prototypes)
    to_plot = [
        da.copy(data=decoded_center[..., 0]) for decoded_center in decoded_centers
    ]
    clu = Clusterplot(*model.som_layer.map_size, region=get_region(da), honeycomb=True)
    clu.add_contourf(to_plot, **kwargs)
    return clu


def phase_space(
    da: DataArray,
    decoder: Model,
    symmetrize: bool = True,
    ncol: int = 5,
    factor: int = -100,
    **kwargs,
) -> Clusterplot:
    if "time" in da.dims:
        da = da[0]
    latent_dim = decoder.layers[0].input.shape[1]
    nrow = int(np.ceil(latent_dim / ncol))
    embeddings = factor * np.eye(latent_dim)
    to_plot = [da.copy(data=decoder(embed[None, :])[0, ..., 0]) for embed in embeddings]
    if symmetrize:
        to_plot = [2 * (tplt - 0.5) for tplt in to_plot]
    clu = Clusterplot(nrow, ncol, region=get_region(da))
    clu.add_contourf(to_plot, **kwargs)
    return clu
