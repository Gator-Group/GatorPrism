#!/usr/bin/env python3
"""Train GatorPrism and infer spatial domains from paired multi-omics data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
from typing import Dict, List

import torch

from GatorPrism import GatorPrism, LossWeights, compute_gatorprism_loss
from utils import (
    build_graph_family,
    cluster_embedding,
    features_to_torch,
    graphs_to_torch,
    load_h5ad_modalities,
    parse_modality_specs,
    resolve_device,
    save_run_outputs,
    set_seed,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype-conditioned coalition graph mixture-of-experts for "
            "unsupervised spatial multi-omics domain identification."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modality",
        action="append",
        required=True,
        metavar="NAME=FILE",
        help=(
            "Processed modality AnnData. Repeat once per modality, for example "
            "--modality RNA=rna.h5ad --modality ATAC=atac.h5ad."
        ),
    )
    parser.add_argument(
        "--n-domains",
        type=int,
        required=True,
        help="Number K of spatial domains used only by final clustering.",
    )
    parser.add_argument(
        "--n-prototypes",
        type=int,
        default=10,
        help="Number R of trainable routing prototypes (independent of K).",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="AnnData obsm key containing two-dimensional coordinates.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional AnnData layer used instead of X for every modality.",
    )
    parser.add_argument("--spatial-neighbors", type=int, default=6)
    parser.add_argument("--feature-neighbors", type=int, default=10)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--edge-temperature", type=float, default=0.2)
    parser.add_argument("--modality-temperature", type=float, default=0.2)
    parser.add_argument(
        "--nt-xent-chunk-size",
        type=int,
        default=1024,
        help="Anchor chunk size; all spots remain in-batch negatives.",
    )

    parser.add_argument("--lambda-consensus", type=float, default=0.3)
    parser.add_argument("--lambda-neighborhood", type=float, default=0.2)
    parser.add_argument("--lambda-cross-modal", type=float, default=0.5)
    parser.add_argument("--lambda-spatial", type=float, default=0.05)
    parser.add_argument("--lambda-balance", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=("mclust", "kmeans"),
        default="mclust",
        help=(
            "Final clustering of Z. mclust is the method-defined default; "
            "kmeans is provided only for debugging environments without R."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("saved_results/GatorPrism_run"),
    )
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.n_domains < 2:
        raise ValueError("--n-domains must be at least two.")
    if args.n_prototypes < 1:
        raise ValueError("--n-prototypes must be positive.")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer parameters.")
    if args.gradient_clip < 0:
        raise ValueError("--gradient-clip must be non-negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must lie in [0, 1).")
    for name in (
        "lambda_consensus",
        "lambda_neighborhood",
        "lambda_cross_modal",
        "lambda_spatial",
        "lambda_balance",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def train(
    model: GatorPrism,
    features: Dict[str, torch.Tensor],
    graphs,
    weights: LossWeights,
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: List[Dict[str, float]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            features,
            graphs.consensus_mixed_edges,
            graphs.modality_mixed_edges,
        )
        loss, terms = compute_gatorprism_loss(
            outputs=outputs,
            features=features,
            spatial_edges=graphs.spatial_edges,
            consensus_feature_edges=graphs.consensus_feature_edges,
            modality_feature_edges=graphs.feature_edges,
            weights=weights,
            edge_temperature=args.edge_temperature,
            modality_temperature=args.modality_temperature,
            nt_xent_chunk_size=args.nt_xent_chunk_size,
        )
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
        optimizer.step()

        row = {"epoch": epoch}
        row.update({name: float(value.detach().cpu()) for name, value in terms.items()})
        history.append(row)
        if (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.log_every == 0
        ):
            elapsed = time.time() - started
            print(
                f"epoch={epoch:04d}/{args.epochs} "
                f"loss={row['total']:.6f} "
                f"rec={row['reconstruction']:.6f} "
                f"cons={row['consensus']:.6f} "
                f"nbr={row['neighborhood']:.6f} "
                f"mod={row['cross_modal']:.6f} "
                f"sm={row['spatial']:.6f} "
                f"bal={row['balance']:.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return history


def run(args: argparse.Namespace) -> Path:
    validate_arguments(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    modality_paths = parse_modality_specs(args.modality)

    print("Loading co-registered processed modalities...", flush=True)
    data = load_h5ad_modalities(
        modality_paths,
        spatial_key=args.spatial_key,
        layer=args.layer,
    )
    if args.n_domains >= data.n_spots:
        raise ValueError("--n-domains must be smaller than the number of spots.")

    print("Constructing modality-specific and consensus graphs...", flush=True)
    graph_arrays = build_graph_family(
        data.features,
        data.spatial,
        spatial_neighbors=args.spatial_neighbors,
        feature_neighbors=args.feature_neighbors,
    )
    features = features_to_torch(data.features, device)
    graphs = graphs_to_torch(graph_arrays, device)

    model = GatorPrism(
        modality_dims=data.modality_dims,
        n_prototypes=args.n_prototypes,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        encoder_layers=args.encoder_layers,
        alpha=args.alpha,
        gate_temperature=args.gate_temperature,
    ).to(device)
    weights = LossWeights(
        consensus=args.lambda_consensus,
        neighborhood=args.lambda_neighborhood,
        cross_modal=args.lambda_cross_modal,
        spatial=args.lambda_spatial,
        balance=args.lambda_balance,
    )

    print(
        f"Training on {device}: spots={data.n_spots}, "
        f"modalities={data.modality_dims}, coalitions={model.coalition_names}, "
        f"consensus_feature_edges={graph_arrays.consensus_feature_edges.shape[1]}",
        flush=True,
    )
    history = train(model, features, graphs, weights, args)

    model.eval()
    with torch.inference_mode():
        outputs = model(
            features,
            graphs.consensus_mixed_edges,
            graphs.modality_mixed_edges,
        )
    embedding = outputs["z"].cpu().numpy()
    gates = outputs["gates"].cpu().numpy()
    association = outputs["prototype_association"].cpu().numpy()

    print(
        f"Clustering fused representation Z with {args.cluster_method} "
        f"(K={args.n_domains})...",
        flush=True,
    )
    labels = cluster_embedding(
        embedding,
        n_domains=args.n_domains,
        seed=args.seed,
        method=args.cluster_method,
    )

    output_directory = args.output.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    save_run_outputs(
        output_directory=output_directory,
        obs_names=data.obs_names,
        spatial=data.spatial,
        labels=labels,
        embedding=embedding,
        gates=gates,
        prototype_association=association,
        coalition_names=model.coalition_names,
        training_history=history,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "modality_dims": data.modality_dims,
        "modality_names": model.modality_names,
        "coalition_names": model.coalition_names,
        "n_prototypes": args.n_prototypes,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "encoder_layers": args.encoder_layers,
        "alpha": args.alpha,
        "gate_temperature": args.gate_temperature,
    }
    torch.save(checkpoint, output_directory / "GatorPrism_model.pt")

    configuration = vars(args).copy()
    configuration["output"] = str(output_directory)
    configuration["modality"] = {
        name: str(path) for name, path in modality_paths.items()
    }
    configuration["device_resolved"] = str(device)
    configuration["modality_dims"] = data.modality_dims
    configuration["coalition_names"] = list(model.coalition_names)
    configuration["loss_weights"] = asdict(weights)
    configuration["graph_edges"] = {
        "spatial": int(graph_arrays.spatial_edges.shape[1]),
        "consensus_feature": int(
            graph_arrays.consensus_feature_edges.shape[1]
        ),
        "consensus_mixed": int(
            graph_arrays.consensus_mixed_edges.shape[1]
        ),
        "feature": {
            name: int(edge.shape[1])
            for name, edge in graph_arrays.feature_edges.items()
        },
        "modality_mixed": {
            name: int(edge.shape[1])
            for name, edge in graph_arrays.modality_mixed_edges.items()
        },
    }
    write_json(output_directory / "config.json", configuration)

    print(f"Finished. Results saved to {output_directory}", flush=True)
    return output_directory


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
