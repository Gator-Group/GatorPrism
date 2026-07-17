from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _two_layer_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _add_self_loops(edge_index: Tensor, n_nodes: int) -> Tensor:
    loops = torch.arange(n_nodes, device=edge_index.device, dtype=torch.long)
    loops = torch.stack((loops, loops), dim=0)
    return torch.cat((edge_index, loops), dim=1)


class MultiHeadGraphAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or heads < 1:
            raise ValueError("hidden_dim and heads must be positive.")

        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = (self.hidden_dim + self.heads - 1) // self.heads
        self.inner_dim = self.heads * self.head_dim

        self.projection = nn.Linear(self.hidden_dim, self.inner_dim, bias=False)
        self.attention_source = nn.Parameter(
            torch.empty(self.heads, self.head_dim)
        )
        self.attention_target = nn.Parameter(
            torch.empty(self.heads, self.head_dim)
        )
        self.output_projection = nn.Linear(self.inner_dim, self.hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges].")

        n_nodes = x.shape[0]
        edges = _add_self_loops(edge_index, n_nodes)
        source, target = edges

        projected = self.projection(x).view(
            n_nodes, self.heads, self.head_dim
        )
        score = (
            (projected[source] * self.attention_source).sum(dim=-1)
            + (projected[target] * self.attention_target).sum(dim=-1)
        )
        score = F.leaky_relu(score, negative_slope=0.2)

        target_index = target[:, None].expand(-1, self.heads)
        maximum = torch.full(
            (n_nodes, self.heads),
            -torch.inf,
            dtype=score.dtype,
            device=score.device,
        )
        maximum.scatter_reduce_(
            0, target_index, score, reduce="amax", include_self=True
        )
        exponent = torch.exp(score - maximum[target])
        denominator = torch.zeros(
            (n_nodes, self.heads), dtype=score.dtype, device=score.device
        )
        denominator.scatter_add_(0, target_index, exponent)
        attention = exponent / denominator[target].clamp_min(1e-12)
        attention = self.attention_dropout(attention)

        message = projected[source] * attention.unsqueeze(-1)
        aggregated = torch.zeros(
            (n_nodes, self.heads, self.head_dim),
            dtype=message.dtype,
            device=message.device,
        )
        aggregate_index = target[:, None, None].expand_as(message)
        aggregated.scatter_add_(0, aggregate_index, message)
        aggregated = aggregated.reshape(n_nodes, self.inner_dim)
        update = self.output_projection(aggregated)
        return self.norm(x + self.feature_dropout(F.elu(update)))


class CoalitionGraphExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least one.")

        self.input_projection = _two_layer_mlp(
            input_dim, hidden_dim, hidden_dim, dropout
        )
        self.layers = nn.ModuleList(
            MultiHeadGraphAttention(hidden_dim, heads, dropout)
            for _ in range(n_layers)
        )
        self.output_projection = _two_layer_mlp(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        initial = self.input_projection(x)
        hidden = initial
        for layer in self.layers:
            hidden = layer(hidden, edge_index)
        return self.norm(initial + self.output_projection(hidden))


class GatorPrism(nn.Module):
    def __init__(
        self,
        modality_dims: Mapping[str, int],
        n_prototypes: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        encoder_layers: int = 3,
        alpha: float = 1.0,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if len(modality_dims) < 2:
            raise ValueError("GatorPrism requires at least two modalities.")
        if n_prototypes < 1:
            raise ValueError("n_prototypes must be positive.")
        if alpha <= 0 or gate_temperature <= 0:
            raise ValueError("alpha and gate_temperature must be positive.")
        if any(int(dimension) < 1 for dimension in modality_dims.values()):
            raise ValueError("Every modality dimension must be positive.")

        self.modality_names = tuple(modality_dims.keys())
        self.modality_dims = {
            name: int(modality_dims[name]) for name in self.modality_names
        }
        self.n_modalities = len(self.modality_names)
        self.n_coalitions = self.n_modalities + 1
        self.n_prototypes = int(n_prototypes)
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.gate_temperature = float(gate_temperature)
        self.coalition_names = ("joint",) + self.modality_names

        joint_input_dim = sum(self.modality_dims.values())
        self.joint_expert = CoalitionGraphExpert(
            joint_input_dim,
            hidden_dim,
            heads,
            dropout,
            encoder_layers,
        )
        self.modality_experts = nn.ModuleDict(
            {
                name: CoalitionGraphExpert(
                    self.modality_dims[name],
                    hidden_dim,
                    heads,
                    dropout,
                    encoder_layers,
                )
                for name in self.modality_names
            }
        )

        self.routing_prototypes = nn.Parameter(
            torch.empty(self.n_prototypes, self.hidden_dim)
        )
        nn.init.normal_(self.routing_prototypes, mean=0.0, std=0.1)
        self.router = _two_layer_mlp(
            2 * hidden_dim,
            hidden_dim,
            self.n_coalitions,
            dropout,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.decoders = nn.ModuleDict(
            {
                name: _two_layer_mlp(
                    2 * hidden_dim,
                    hidden_dim,
                    self.modality_dims[name],
                    dropout,
                )
                for name in self.modality_names
            }
        )

    def prototype_association(self, base: Tensor) -> Tensor:
        squared_distance = torch.cdist(
            base, self.routing_prototypes, p=2
        ).square()
        unnormalized = (1.0 + squared_distance / self.alpha).pow(
            -(self.alpha + 1.0) / 2.0
        )
        return unnormalized / unnormalized.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)

    def _validate_inputs(
        self,
        features: Mapping[str, Tensor],
        consensus_graph: Tensor,
        modality_graphs: Mapping[str, Tensor],
    ) -> int:
        if tuple(features.keys()) != self.modality_names:
            raise ValueError(
                "features must use the same modality names and order supplied "
                "when the model was initialized."
            )
        if set(modality_graphs) != set(self.modality_names):
            raise ValueError("modality_graphs must contain one graph per modality.")

        n_nodes = next(iter(features.values())).shape[0]
        for name in self.modality_names:
            matrix = features[name]
            if matrix.ndim != 2:
                raise ValueError(f"{name} features must be a matrix.")
            if matrix.shape != (n_nodes, self.modality_dims[name]):
                raise ValueError(
                    f"Unexpected shape for {name}: {tuple(matrix.shape)}."
                )
        for graph in (consensus_graph, *modality_graphs.values()):
            if graph.ndim != 2 or graph.shape[0] != 2:
                raise ValueError("Every graph must have shape [2, n_edges].")
        return n_nodes

    def forward(
        self,
        features: Mapping[str, Tensor],
        consensus_graph: Tensor,
        modality_graphs: Mapping[str, Tensor],
    ) -> Dict[str, object]:
        self._validate_inputs(features, consensus_graph, modality_graphs)

        joint_input = torch.cat(
            [features[name] for name in self.modality_names], dim=1
        )
        coalition_embeddings: Dict[str, Tensor] = {
            "joint": self.joint_expert(joint_input, consensus_graph)
        }
        for name in self.modality_names:
            coalition_embeddings[name] = self.modality_experts[name](
                features[name], modality_graphs[name]
            )

        coalition_stack = torch.stack(
            [coalition_embeddings[name] for name in self.coalition_names], dim=1
        )
        base = coalition_stack.mean(dim=1)
        association = self.prototype_association(base)
        prototype_context = association @ self.routing_prototypes
        gate_logits = self.router(
            torch.cat((base, prototype_context), dim=1)
        )
        gates = torch.softmax(
            gate_logits / self.gate_temperature, dim=1
        )
        routed = (coalition_stack * gates.unsqueeze(-1)).sum(dim=1)
        fused = self.fusion_norm(base + routed)

        reconstructions = {
            name: self.decoders[name](
                torch.cat((fused, coalition_embeddings[name]), dim=1)
            )
            for name in self.modality_names
        }

        return {
            "z": fused,
            "base": base,
            "prototype_association": association,
            "prototype_context": prototype_context,
            "gates": gates,
            "coalition_embeddings": coalition_embeddings,
            "coalition_stack": coalition_stack,
            "reconstructions": reconstructions,
            "coalition_names": self.coalition_names,
        }


@dataclass(frozen=True)
class LossWeights:
    consensus: float = 0.3
    neighborhood: float = 0.2
    cross_modal: float = 0.5
    spatial: float = 0.05
    balance: float = 0.01

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("Loss coefficients must be non-negative.")


def sample_distinct_non_edges(
    n_nodes: int,
    positive_edges: Tensor,
    n_samples: Optional[int] = None,
) -> Tensor:
    if n_nodes < 2:
        raise ValueError("At least two nodes are required for negative edges.")
    if positive_edges.ndim != 2 or positive_edges.shape[0] != 2:
        raise ValueError("positive_edges must have shape [2, n_edges].")

    device = positive_edges.device
    source, target = positive_edges.long()
    non_self = source != target
    positive_ids = torch.unique(source[non_self] * n_nodes + target[non_self])
    if n_samples is None:
        n_samples = int(positive_ids.numel())
    n_samples = int(n_samples)
    available = n_nodes * (n_nodes - 1) - int(positive_ids.numel())
    if n_samples > available:
        raise ValueError(
            f"Requested {n_samples} non-edges, but only {available} exist."
        )
    if n_samples == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    selected = torch.empty(0, dtype=torch.long, device=device)
    while selected.numel() < n_samples:
        remaining = n_samples - selected.numel()
        draw_count = max(remaining * 3, 128)
        candidates = torch.randint(
            n_nodes * n_nodes,
            (draw_count,),
            dtype=torch.long,
            device=device,
        )
        candidate_source = torch.div(candidates, n_nodes, rounding_mode="floor")
        candidate_target = candidates.remainder(n_nodes)
        candidates = candidates[candidate_source != candidate_target]
        if positive_ids.numel():
            candidates = candidates[~torch.isin(candidates, positive_ids)]
        if selected.numel():
            candidates = candidates[~torch.isin(candidates, selected)]
        candidates = torch.unique(candidates)
        if candidates.numel():
            candidates = candidates[
                torch.randperm(candidates.numel(), device=device)
            ]
            selected = torch.cat((selected, candidates[:remaining]))

    return torch.stack(
        (
            torch.div(selected, n_nodes, rounding_mode="floor"),
            selected.remainder(n_nodes),
        ),
        dim=0,
    )


def reconstruction_loss(
    reconstructions: Mapping[str, Tensor],
    features: Mapping[str, Tensor],
) -> Tensor:
    return torch.stack(
        [F.mse_loss(reconstructions[name], features[name]) for name in features]
    ).sum()


def edge_contrastive_loss(
    embedding: Tensor,
    positive_edges: Tensor,
    temperature: float,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if positive_edges.shape[1] == 0:
        raise ValueError("The positive edge set must not be empty.")

    source, target = positive_edges
    valid = source != target
    positives = torch.unique(source[valid] * embedding.shape[0] + target[valid])
    positive_edges = torch.stack(
        (
            torch.div(positives, embedding.shape[0], rounding_mode="floor"),
            positives.remainder(embedding.shape[0]),
        ),
        dim=0,
    )
    negative_edges = sample_distinct_non_edges(
        embedding.shape[0], positive_edges, positive_edges.shape[1]
    )

    normalized = F.normalize(embedding, dim=1)
    positive_score = (
        normalized[positive_edges[0]] * normalized[positive_edges[1]]
    ).sum(dim=1) / temperature
    negative_score = (
        normalized[negative_edges[0]] * normalized[negative_edges[1]]
    ).sum(dim=1) / temperature
    return -F.logsigmoid(positive_score).mean() - F.logsigmoid(
        -negative_score
    ).mean()


def _directional_nt_xent(
    anchors: Tensor,
    targets: Tensor,
    temperature: float,
    chunk_size: Optional[int],
) -> Tensor:
    n_nodes = anchors.shape[0]
    if chunk_size is None or chunk_size <= 0:
        chunk_size = n_nodes

    weighted_loss = anchors.new_tensor(0.0)
    for start in range(0, n_nodes, chunk_size):
        stop = min(start + chunk_size, n_nodes)
        logits = anchors[start:stop] @ targets.T / temperature
        labels = torch.arange(start, stop, device=anchors.device)
        weighted_loss = weighted_loss + F.cross_entropy(
            logits, labels, reduction="sum"
        )
    return weighted_loss / n_nodes


def symmetric_nt_xent(
    left: Tensor,
    right: Tensor,
    temperature: float,
    chunk_size: Optional[int] = None,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if left.shape != right.shape:
        raise ValueError("Cross-modal embeddings must have identical shapes.")

    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    return 0.5 * (
        _directional_nt_xent(left, right, temperature, chunk_size)
        + _directional_nt_xent(right, left, temperature, chunk_size)
    )


def cross_modal_alignment_loss(
    modality_embeddings: Mapping[str, Tensor],
    temperature: float,
    chunk_size: Optional[int] = None,
) -> Tensor:
    losses = [
        symmetric_nt_xent(
            modality_embeddings[left],
            modality_embeddings[right],
            temperature,
            chunk_size,
        )
        for left, right in combinations(modality_embeddings, 2)
    ]
    if not losses:
        raise ValueError("At least two modality embeddings are required.")
    return torch.stack(losses).mean()


def spatial_smoothness_loss(z: Tensor, spatial_edges: Tensor) -> Tensor:
    if spatial_edges.shape[1] == 0:
        raise ValueError("The spatial graph must contain at least one edge.")
    difference = z[spatial_edges[0]] - z[spatial_edges[1]]
    return difference.square().sum(dim=1).mean()


def load_balance_loss(gates: Tensor) -> Tensor:
    usage = gates.mean(dim=0)
    return (usage - 1.0 / gates.shape[1]).square().sum()


def compute_gatorprism_loss(
    outputs: Mapping[str, object],
    features: Mapping[str, Tensor],
    spatial_edges: Tensor,
    consensus_feature_edges: Tensor,
    modality_feature_edges: Mapping[str, Tensor],
    weights: LossWeights,
    edge_temperature: float = 0.2,
    modality_temperature: float = 0.2,
    nt_xent_chunk_size: Optional[int] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    z = outputs["z"]
    modality_embeddings = {
        name: outputs["coalition_embeddings"][name] for name in features
    }
    consensus_positives = (
        consensus_feature_edges
        if consensus_feature_edges.shape[1] > 0
        else spatial_edges
    )

    reconstruction = reconstruction_loss(outputs["reconstructions"], features)
    consensus = edge_contrastive_loss(
        z, consensus_positives, edge_temperature
    )
    neighborhood = torch.stack(
        [
            edge_contrastive_loss(
                modality_embeddings[name],
                modality_feature_edges[name],
                edge_temperature,
            )
            for name in features
        ]
    ).mean()
    cross_modal = cross_modal_alignment_loss(
        modality_embeddings,
        modality_temperature,
        nt_xent_chunk_size,
    )
    spatial = spatial_smoothness_loss(z, spatial_edges)
    balance = load_balance_loss(outputs["gates"])

    total = (
        reconstruction
        + weights.consensus * consensus
        + weights.neighborhood * neighborhood
        + weights.cross_modal * cross_modal
        + weights.spatial * spatial
        + weights.balance * balance
    )
    terms = {
        "total": total,
        "reconstruction": reconstruction,
        "consensus": consensus,
        "neighborhood": neighborhood,
        "cross_modal": cross_modal,
        "spatial": spatial,
        "balance": balance,
    }
    return total, terms
