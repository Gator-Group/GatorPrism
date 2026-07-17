# GatorPrism

GatorPrism is an unsupervised graph mixture-of-experts model for spatial domain identification in spatially co-registered multi-omics tissue sections. It learns one fused representation from two or more processed molecular modalities and uses `mclust` to obtain the final spatial domains.

This repository is a clean, standalone implementation of the method specified in `GatorPrism_Methods.md`. Its top-level organization follows the public [GatorTrio](https://github.com/zhangzh1328/GatorTrio) project style, while the model and objectives are specific to GatorPrism.

## Method overview

Given processed feature matrices \(X^1,\ldots,X^M\) and common spatial coordinates \(S\), GatorPrism performs the following steps:

1. Construct a symmetric spatial k-nearest-neighbor graph.
2. Construct one symmetric feature k-nearest-neighbor graph for every modality.
3. Form each modality-specific mixed graph by taking the union of its feature edges and the spatial edges.
4. Form a consensus graph from all spatial edges and the intersection of feature edges shared by every modality.
5. Encode the grand coalition with a joint graph-attention expert and each singleton modality with a separate graph-attention expert. Exactly \(M+1\) experts are instantiated.
6. Average the expert representations to obtain a reference representation, associate it with \(R\) trainable prototypes through a Student-t kernel, and use the resulting prototype context to generate spot-specific coalition weights.
7. Fuse the routed expert representations with a residual reference term and LayerNorm.
8. Optimize modality reconstruction, consensus-neighborhood contrast, modality-specific neighborhood contrast, symmetric cross-modal NT-Xent, spatial smoothness, and aggregate expert load balancing.
9. Apply an exact \(K\)-component `mclust` model to the fused representation \(Z\), then average spot-level routing weights within each inferred domain.

Routing prototypes are internal context variables. They are not spatial-domain assignments, and their number \(R\) is independent of the final number of domains \(K\). The implementation does not use target sharpening, a DEC clustering KL loss, annotations during training, intermediate modality-subset experts, or concatenated private embeddings for final clustering.

## Project structure

```text
.
├── main.py             # Full-batch training, inference, clustering, and export
├── GatorPrism.py       # GAT experts, prototype router, fusion, and all losses
├── utils.py            # AnnData loading, graph construction, mclust, and I/O
├── requirements.txt    # Python dependencies
├── saved_models/       # Optional user-managed checkpoint directory
└── saved_results/      # Default run outputs
```

The output directories are created automatically when a run is launched.

## Requirements

- Python 3.9 or later
- A CPU or CUDA installation of PyTorch 2.1 or later
- R with the `mclust` package for method-defined final clustering

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install `mclust` once from R:

```bash
Rscript -e "install.packages('mclust', repos='https://cloud.r-project.org')"
```

The model uses native PyTorch scatter operations and does not require `torch-geometric` or `torch-scatter`.

## Input data

Supply one `.h5ad` file per modality using repeated `--modality NAME=FILE` arguments. The input contract is:

- every modality contains the same uniquely named spots;
- modalities may have different feature dimensions;
- `adata.X`, or a shared layer selected by `--layer`, contains the processed feature matrix \(X^m\);
- at least one file contains two-dimensional coordinates in `adata.obsm["spatial"]`;
- if multiple files contain spatial coordinates, they must agree after spot alignment;
- annotations are not required and are never read for model training.

The loader reorders a modality only if its spot-name set exactly matches that of the first modality. It does not normalize, log-transform, scale, select features, or run PCA, because the mathematical method takes preprocessed matrices as input.

Do not pass a large raw peak-count matrix directly. For RNA–ATAC data, store appropriately processed low-dimensional RNA and ATAC features in the input `.h5ad` objects before running GatorPrism.

Example layout:

```text
data/
├── sample_RNA.h5ad
└── sample_ATAC.h5ad
```

## Run GatorPrism

From the project directory:

```bash
python main.py \
  --modality RNA=data/sample_RNA.h5ad \
  --modality ATAC=data/sample_ATAC.h5ad \
  --n-domains 14 \
  --n-prototypes 10 \
  --output saved_results/sample
```

For RNA–ADT data, only the modality name and file change:

```bash
python main.py \
  --modality RNA=data/sample_RNA.h5ad \
  --modality ADT=data/sample_ADT.h5ad \
  --n-domains 8 \
  --output saved_results/sample_rna_adt
```

Important options:

```text
--n-domains K                 Final number of mclust domains
--n-prototypes R              Trainable routing prototypes, independent of K
--spatial-neighbors           Spatial kNN size
--feature-neighbors           Per-modality molecular kNN size
--hidden-dim                  Common expert latent dimension
--heads                       GAT attention heads
--encoder-layers              GAT layers per coalition expert
--gate-temperature            Router softmax temperature
--edge-temperature            Edge-contrastive temperature
--modality-temperature        Cross-modal NT-Xent temperature
--nt-xent-chunk-size          Memory chunking; all spots remain negatives
--lambda-consensus            Consensus contrast coefficient
--lambda-neighborhood         Modality-neighborhood contrast coefficient
--lambda-cross-modal          Cross-modal alignment coefficient
--lambda-spatial              Spatial smoothness coefficient
--lambda-balance              Aggregate load-balancing coefficient
--cluster-method mclust       Method-defined final clustering
```

Run `python main.py --help` for all arguments and defaults.

`--cluster-method kmeans` is available only for smoke tests or environments without R. Results intended to match the manuscript method must use the default `mclust` option. There is no silent fallback from `mclust` to another clustering algorithm.

## Outputs

Each run creates:

```text
saved_results/sample/
├── GatorPrism_model.pt             # Final model state and architecture metadata
├── config.json                     # Complete run configuration and graph sizes
├── embedding.npy                   # Fused representation Z used by mclust
├── coalition_weights.npy           # Spot-by-coalition routing matrix
├── prototype_association.npy       # Internal Student-t routing associations
├── predicted_domains.csv           # Barcode, domain, and spatial coordinates
├── spot_coalition_weights.csv      # Named spot-level routing weights
├── domain_coalition_profiles.csv   # Mean routing allocation within each domain
└── training_history.csv            # Total and component losses by epoch
```

For \(M\) modalities, the coalition columns are ordered as `joint` followed by the modality names in the order given on the command line.

## Python API

The model can also be imported directly:

```python
from GatorPrism import GatorPrism

model = GatorPrism(
    modality_dims={"RNA": 30, "ATAC": 30},
    n_prototypes=10,
    hidden_dim=64,
    heads=4,
    encoder_layers=3,
)

outputs = model(
    features,               # ordered dict: modality -> [N, d_m] tensor
    consensus_mixed_edges,  # [2, E_cons] tensor
    modality_mixed_edges,   # dict: modality -> [2, E_m] tensor
)

z = outputs["z"]
gates = outputs["gates"]
```

## Method-to-code correspondence

| Method component | Implementation |
|---|---|
| Symmetric spatial and feature kNN graphs | `utils.symmetric_knn_edges` |
| Modality mixed and all-modality consensus graphs | `utils.build_graph_family` |
| Separate graph-attention coalition encoders | `CoalitionGraphExpert` |
| Coalition-averaged reference representation | `GatorPrism.forward` |
| Student-t prototype associations | `GatorPrism.prototype_association` |
| Two-layer prototype-conditioned router | `GatorPrism.router` |
| Residual weighted fusion and LayerNorm | `GatorPrism.forward` |
| Modality-aware reconstruction | `reconstruction_loss` |
| Consensus and private-neighborhood contrast | `edge_contrastive_loss` |
| Symmetric cross-modal NT-Xent | `symmetric_nt_xent` |
| Graph Dirichlet spatial regularization | `spatial_smoothness_loss` |
| Aggregate coalition load balancing | `load_balance_loss` |
| Joint objective | `compute_gatorprism_loss` |
| Final clustering of fused `Z` only | `utils.cluster_with_mclust` |
| Domain-level coalition profile | `utils.domain_coalition_profiles` |

## Reproducibility and scaling notes

- GatorPrism is trained full-batch because graph edges and cross-modal co-registration are defined over the whole section.
- NT-Xent uses every other spot as an in-batch negative. `--nt-xent-chunk-size` reduces peak memory without changing the denominator.
- Each graph-contrastive objective uses all distinct positive ordered edges and an equal number of uniformly sampled distinct non-edges without replacement.
- The random seed controls Python, NumPy, PyTorch, negative-edge sampling, and final clustering.
- Gate values are internal convex-combination coefficients, not externally calibrated biological probabilities.
