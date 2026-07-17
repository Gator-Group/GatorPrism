"""Data, graph, clustering, reproducibility, and output utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
from torch import Tensor


@dataclass(frozen=True)
class MultiOmicsData:
    """Aligned processed modality matrices and spatial coordinates."""

    features: Dict[str, np.ndarray]
    spatial: np.ndarray
    obs_names: np.ndarray

    @property
    def n_spots(self) -> int:
        return int(self.spatial.shape[0])

    @property
    def modality_dims(self) -> Dict[str, int]:
        return {name: int(matrix.shape[1]) for name, matrix in self.features.items()}


@dataclass(frozen=True)
class GraphFamily:
    """Graph family defined in the GatorPrism method."""

    spatial_edges: np.ndarray
    feature_edges: Dict[str, np.ndarray]
    consensus_feature_edges: np.ndarray
    modality_mixed_edges: Dict[str, np.ndarray]
    consensus_mixed_edges: np.ndarray


@dataclass(frozen=True)
class TorchGraphFamily:
    """Torch representation of :class:`GraphFamily`."""

    spatial_edges: Tensor
    feature_edges: Dict[str, Tensor]
    consensus_feature_edges: Tensor
    modality_mixed_edges: Dict[str, Tensor]
    consensus_mixed_edges: Tensor


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda``, or an explicit CUDA device."""

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested, but CUDA is unavailable.")
    return device


def parse_modality_specs(specifications: Sequence[str]) -> Dict[str, Path]:
    """Parse repeated ``NAME=/path/to/modality.h5ad`` arguments."""

    parsed: Dict[str, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Invalid modality specification {specification!r}; expected NAME=FILE."
            )
        name, path_text = specification.split("=", 1)
        name = name.strip()
        path = Path(path_text).expanduser().resolve()
        if not name:
            raise ValueError("A modality name must not be empty.")
        if name in parsed:
            raise ValueError(f"Duplicate modality name: {name}.")
        if not path.is_file():
            raise FileNotFoundError(f"Modality file does not exist: {path}")
        if path.suffix.lower() != ".h5ad":
            raise ValueError(f"GatorPrism expects .h5ad files, received: {path}")
        parsed[name] = path
    if len(parsed) < 2:
        raise ValueError("At least two --modality arguments are required.")
    return parsed


def _dense_float32(matrix: object, name: str) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} has no features.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np.ascontiguousarray(array)


def load_h5ad_modalities(
    modality_paths: Mapping[str, Path],
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
) -> MultiOmicsData:
    """Load and align co-registered processed modalities.

    The first modality defines spot order and coordinates.  Other modalities
    are reordered only when they contain exactly the same unique spot names.
    The function performs no normalization, PCA, or feature selection because
    the method takes preprocessed matrices ``X^m`` as input.
    """

    features: Dict[str, np.ndarray] = {}
    reference_names: Optional[np.ndarray] = None
    reference_spatial: Optional[np.ndarray] = None

    for index, (name, path) in enumerate(modality_paths.items()):
        adata = ad.read_h5ad(path)
        if not adata.obs_names.is_unique:
            raise ValueError(f"{name}: obs_names must be unique.")

        names = adata.obs_names.astype(str).to_numpy()
        if index == 0:
            reference_names = names
        elif np.array_equal(names, reference_names):
            pass
        elif set(names) == set(reference_names):
            adata = adata[reference_names].copy()
            names = adata.obs_names.astype(str).to_numpy()
        else:
            missing = len(set(reference_names) - set(names))
            extra = len(set(names) - set(reference_names))
            raise ValueError(
                f"{name}: spots are not co-registered with the first modality "
                f"({missing} missing, {extra} extra)."
            )

        if layer is None:
            matrix = adata.X
        else:
            if layer not in adata.layers:
                raise KeyError(f"{name}: layer {layer!r} is absent.")
            matrix = adata.layers[layer]
        features[name] = _dense_float32(matrix, f"{name}.X")

        if spatial_key in adata.obsm:
            coordinates = _dense_float32(
                adata.obsm[spatial_key], f"{name}.obsm[{spatial_key!r}]"
            )
            if coordinates.shape[1] != 2:
                raise ValueError(
                    f"{name}: spatial coordinates must have exactly two columns."
                )
            if reference_spatial is None:
                reference_spatial = coordinates
            elif not np.allclose(coordinates, reference_spatial, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"{name}: spatial coordinates disagree with the first modality."
                )

    if reference_spatial is None:
        raise KeyError(
            f"No input AnnData contains obsm[{spatial_key!r}] coordinates."
        )
    assert reference_names is not None
    expected_spots = reference_spatial.shape[0]
    if any(matrix.shape[0] != expected_spots for matrix in features.values()):
        raise ValueError("All modalities must contain the same number of spots.")

    return MultiOmicsData(
        features=features,
        spatial=np.ascontiguousarray(reference_spatial, dtype=np.float32),
        obs_names=np.asarray(reference_names, dtype=str),
    )


def symmetric_knn_edges(matrix: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Construct the symmetrized Euclidean kNN edge set from the method."""

    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("kNN input must be a matrix.")
    n_nodes = matrix.shape[0]
    if n_nodes < 2:
        raise ValueError("At least two spots are required to construct a graph.")
    if n_neighbors < 1 or n_neighbors >= n_nodes:
        raise ValueError(
            f"n_neighbors must lie in [1, {n_nodes - 1}], got {n_neighbors}."
        )

    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="euclidean",
        algorithm="auto",
    ).fit(matrix)
    queried = model.kneighbors(matrix, return_distance=False)
    # Remove the query spot itself explicitly.  This remains correct when
    # duplicated feature vectors make several zero-distance neighbors tie.
    indices = np.stack(
        [row[row != index][:n_neighbors] for index, row in enumerate(queried)],
        axis=0,
    )

    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), n_neighbors)
    columns = indices.reshape(-1).astype(np.int64, copy=False)
    directed = np.concatenate(
        (rows * n_nodes + columns, columns * n_nodes + rows)
    )
    directed = np.unique(directed)
    source = directed // n_nodes
    target = directed % n_nodes
    keep = source != target
    return np.ascontiguousarray(
        np.stack((source[keep], target[keep]), axis=0), dtype=np.int64
    )


def _edge_ids(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, n_edges].")
    return np.unique(edge_index[0] * n_nodes + edge_index[1])


def _ids_to_edges(edge_ids: np.ndarray, n_nodes: int) -> np.ndarray:
    edge_ids = np.asarray(edge_ids, dtype=np.int64)
    if edge_ids.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.ascontiguousarray(
        np.stack((edge_ids // n_nodes, edge_ids % n_nodes), axis=0),
        dtype=np.int64,
    )


def build_graph_family(
    features: Mapping[str, np.ndarray],
    spatial: np.ndarray,
    spatial_neighbors: int,
    feature_neighbors: int,
) -> GraphFamily:
    """Build spatial, modality feature/mixed, and consensus graphs."""

    names = tuple(features.keys())
    if len(names) < 2:
        raise ValueError("At least two modalities are required.")
    n_nodes = spatial.shape[0]

    spatial_edges = symmetric_knn_edges(spatial, spatial_neighbors)
    feature_edges = {
        name: symmetric_knn_edges(features[name], feature_neighbors)
        for name in names
    }

    spatial_ids = _edge_ids(spatial_edges, n_nodes)
    feature_ids = {
        name: _edge_ids(feature_edges[name], n_nodes) for name in names
    }
    consensus_ids = feature_ids[names[0]]
    for name in names[1:]:
        consensus_ids = np.intersect1d(
            consensus_ids, feature_ids[name], assume_unique=True
        )

    modality_mixed_edges = {
        name: _ids_to_edges(
            np.union1d(spatial_ids, feature_ids[name]), n_nodes
        )
        for name in names
    }
    consensus_mixed_edges = _ids_to_edges(
        np.union1d(spatial_ids, consensus_ids), n_nodes
    )

    return GraphFamily(
        spatial_edges=spatial_edges,
        feature_edges=feature_edges,
        consensus_feature_edges=_ids_to_edges(consensus_ids, n_nodes),
        modality_mixed_edges=modality_mixed_edges,
        consensus_mixed_edges=consensus_mixed_edges,
    )


def features_to_torch(
    features: Mapping[str, np.ndarray], device: torch.device
) -> Dict[str, Tensor]:
    return {
        name: torch.as_tensor(matrix, dtype=torch.float32, device=device)
        for name, matrix in features.items()
    }


def graphs_to_torch(
    graphs: GraphFamily, device: torch.device
) -> TorchGraphFamily:
    convert = lambda edge: torch.as_tensor(
        edge, dtype=torch.long, device=device
    )
    return TorchGraphFamily(
        spatial_edges=convert(graphs.spatial_edges),
        feature_edges={
            name: convert(edge) for name, edge in graphs.feature_edges.items()
        },
        consensus_feature_edges=convert(graphs.consensus_feature_edges),
        modality_mixed_edges={
            name: convert(edge)
            for name, edge in graphs.modality_mixed_edges.items()
        },
        consensus_mixed_edges=convert(graphs.consensus_mixed_edges),
    )


def cluster_with_mclust(
    embedding: np.ndarray,
    n_domains: int,
    seed: int,
) -> np.ndarray:
    """Cluster ``Z`` with the R ``mclust`` implementation."""

    rscript = shutil.which("Rscript")
    if rscript is None:
        environment_rscript = Path(sys.executable).with_name("Rscript")
        if environment_rscript.is_file():
            rscript = str(environment_rscript)
    if rscript is None:
        raise RuntimeError(
            "Rscript was not found. Install R and the R package 'mclust', or "
            "use --cluster-method kmeans only for debugging."
        )

    r_code = """
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(mclust))
x <- as.matrix(read.csv(args[1], header=FALSE, check.names=FALSE))
set.seed(as.integer(args[4]))
fit <- Mclust(x, G=as.integer(args[3]), verbose=FALSE)
if (is.null(fit$classification)) stop('mclust returned no classification')
write.table(fit$classification, file=args[2], row.names=FALSE,
            col.names=FALSE, quote=FALSE)
"""
    with tempfile.TemporaryDirectory(prefix="gatorprism_mclust_") as directory:
        directory = Path(directory)
        input_path = directory / "embedding.csv"
        output_path = directory / "labels.txt"
        np.savetxt(input_path, embedding, delimiter=",")
        command = [
            rscript,
            "-e",
            r_code,
            str(input_path),
            str(output_path),
            str(n_domains),
            str(seed),
        ]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"mclust failed: {message}")
        labels = np.loadtxt(output_path, dtype=np.int64, ndmin=1)

    if labels.shape != (embedding.shape[0],):
        raise RuntimeError("mclust returned an unexpected number of labels.")
    if np.unique(labels).size != n_domains:
        raise RuntimeError(
            f"mclust produced {np.unique(labels).size} non-empty domains; "
            f"expected {n_domains}."
        )
    return labels


def cluster_embedding(
    embedding: np.ndarray,
    n_domains: int,
    seed: int,
    method: str = "mclust",
) -> np.ndarray:
    """Cluster the fused representation ``Z`` into ``K`` domains."""

    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError("embedding must be a matrix.")
    if n_domains < 2 or n_domains >= embedding.shape[0]:
        raise ValueError("n_domains must be between 2 and n_spots - 1.")
    if method == "mclust":
        return cluster_with_mclust(embedding, n_domains, seed)
    if method == "kmeans":
        return KMeans(
            n_clusters=n_domains, n_init=20, random_state=seed
        ).fit_predict(embedding) + 1
    raise ValueError(f"Unknown clustering method: {method}")


def domain_coalition_profiles(
    labels: np.ndarray,
    gates: np.ndarray,
    coalition_names: Sequence[str],
) -> pd.DataFrame:
    """Compute mean coalition allocation for every inferred domain."""

    labels = np.asarray(labels)
    gates = np.asarray(gates)
    if gates.ndim != 2 or gates.shape[0] != labels.shape[0]:
        raise ValueError("gates and labels have incompatible shapes.")
    if gates.shape[1] != len(coalition_names):
        raise ValueError("coalition_names does not match the gate dimension.")

    rows = []
    for label in sorted(np.unique(labels)):
        mean_gate = gates[labels == label].mean(axis=0)
        row = {"domain": int(label), "n_spots": int((labels == label).sum())}
        row.update(
            {
                coalition_names[index]: float(mean_gate[index])
                for index in range(len(coalition_names))
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _json_ready(value: object) -> object:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_run_outputs(
    output_directory: Path,
    obs_names: np.ndarray,
    spatial: np.ndarray,
    labels: np.ndarray,
    embedding: np.ndarray,
    gates: np.ndarray,
    prototype_association: np.ndarray,
    coalition_names: Sequence[str],
    training_history: Sequence[Mapping[str, float]],
) -> None:
    """Save all interpretable inference artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    np.save(output_directory / "embedding.npy", embedding)
    np.save(output_directory / "coalition_weights.npy", gates)
    np.save(
        output_directory / "prototype_association.npy", prototype_association
    )

    pd.DataFrame(
        {
            "barcode": obs_names,
            "GatorPrism_domain": labels,
            "spatial_x": spatial[:, 0],
            "spatial_y": spatial[:, 1],
        }
    ).to_csv(output_directory / "predicted_domains.csv", index=False)

    gate_frame = pd.DataFrame(gates, columns=coalition_names)
    gate_frame.insert(0, "barcode", obs_names)
    gate_frame.to_csv(output_directory / "spot_coalition_weights.csv", index=False)

    domain_coalition_profiles(labels, gates, coalition_names).to_csv(
        output_directory / "domain_coalition_profiles.csv", index=False
    )
    pd.DataFrame(training_history).to_csv(
        output_directory / "training_history.csv", index=False
    )


__all__ = [
    "GraphFamily",
    "MultiOmicsData",
    "TorchGraphFamily",
    "build_graph_family",
    "cluster_embedding",
    "cluster_with_mclust",
    "domain_coalition_profiles",
    "features_to_torch",
    "graphs_to_torch",
    "load_h5ad_modalities",
    "parse_modality_specs",
    "resolve_device",
    "save_run_outputs",
    "set_seed",
    "symmetric_knn_edges",
    "write_json",
]
