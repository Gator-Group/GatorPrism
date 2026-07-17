# GatorPrism: Prototype-Conditioned Routing across Coalition Graph Experts for Spatial Multi-Omics Integration

![model](https://github.com/Gator-Group/GatorPrism/blob/main/GatorPrism.png)

## Requirements

- python: 3.9
- numpy: 1.24 
- pandas: 2.0 
- scikit-learn: 1.3 
- scipy: 1.10 
- torch: 2.4 

## Project Structure

```bash
.
├── main.py             # Main training, clustering, and result-export workflow
├── GatorPrism.py       # Model architecture and self-supervised loss functions
├── utils.py            # Data loading, graph construction, mclust, and I/O utilities
├── requirements.txt    # Python dependencies
├── data/               # Folder for paired multi-omics .h5ad input files
└── saved_results/      # Model checkpoints and inference outputs
```

The `data/` directory is prepared by the user. The output directory is created automatically when training starts.

## Usage

### **1. Prepare your input data**

Place the two processed omics `.h5ad` datasets in the `./data/` directory.

Example dataset folder:

```bash
data/
 ├── human_lymph_node_A1_RNA.h5ad
 ├── human_lymph_node_A1_ADT.h5ad
 ├── mouse_embryonic_E11_RNA.h5ad
 ├── mouse_embryonic_E11_ATAC.h5ad
 ...
```

Each pair of `.h5ad` files must satisfy the following requirements:

- both modalities contain the same uniquely named spots;
- `adata.X` contains the processed feature matrix for that modality;
- modalities may contain different numbers of features;
- at least one file contains two-dimensional coordinates in `adata.obsm["spatial"]`;
- if both files contain spatial coordinates, the coordinates must agree after barcode alignment;
- spatial-domain annotations are not required and are not used during training.

If the same spots are stored in different orders, GatorPrism automatically reorders the second modality using `obs_names`. It raises an error when barcodes are missing or unmatched.

The loader does not perform normalization, log transformation, feature selection, scaling, or PCA. The `.h5ad` files must therefore contain preprocessed model inputs. In particular, do not pass a large raw ATAC peak-count matrix directly; use a suitable processed low-dimensional ATAC representation.

---

### **2. Run training and evaluation**

Run GatorPrism on an RNA–ADT dataset:

```bash
python main.py \
  --modality RNA=data/human_lymph_node_A1_RNA.h5ad \
  --modality ADT=data/human_lymph_node_A1_ADT.h5ad \
  --n-domains 10 \
  --output saved_results/human_lymph_node_A1
```

#### Optional configuration

- `--epochs`: number of full-batch training epochs;
- `--learning-rate`: AdamW learning rate;
- `--weight-decay`: AdamW weight decay;
- `--hidden-dim`: common coalition-expert latent dimension;
- `--heads`: number of graph-attention heads;
- `--encoder-layers`: graph-attention layers in each expert;
- `--spatial-neighbors`: spatial kNN neighborhood size;
- `--feature-neighbors`: molecular kNN neighborhood size;
- `--gate-temperature`: routing softmax temperature;
- `--edge-temperature`: graph-neighborhood contrastive temperature;
- `--modality-temperature`: cross-modal NT-Xent temperature;
- `--lambda-consensus`: consensus-neighborhood loss coefficient;
- `--lambda-neighborhood`: modality-specific neighborhood loss coefficient;
- `--lambda-cross-modal`: cross-modal alignment loss coefficient;
- `--lambda-spatial`: spatial smoothness coefficient;
- `--lambda-balance`: expert load-balancing coefficient;
- `--seed`: random seed;
- `--device`: `auto`, `cpu`, `cuda`, or a specific device such as `cuda:0`.

Display all available parameters:

```bash
python main.py --help
```
---

### **3. Output files**

After training completes, the selected output directory contains the trained model checkpoint and all inference results:

```bash
saved_results/
 ├── human_lymph_node_A1/
 │   ├── GatorPrism_model.pt
 │   ├── config.json
 │   ├── embedding.npy
 │   ├── coalition_weights.npy
 │   ├── prototype_association.npy
 │   ├── predicted_domains.csv
 │   ├── spot_coalition_weights.csv
 │   ├── domain_coalition_profiles.csv
 │   └── training_history.csv
 └── mouse_embryonic_E11/
     ├── GatorPrism_model.pt
     ├── config.json
     ├── embedding.npy
     ├── coalition_weights.npy
     ├── prototype_association.npy
     ├── predicted_domains.csv
     ├── spot_coalition_weights.csv
     ├── domain_coalition_profiles.csv
     └── training_history.csv
```

The files contain:

- `GatorPrism_model.pt`: trained model parameters and architecture metadata;
- `config.json`: input paths, hyperparameters, modality dimensions, and graph sizes;
- `embedding.npy`: fused spot representation `Z` used for final clustering;
- `coalition_weights.npy`: spot-by-coalition routing-weight matrix;
- `prototype_association.npy`: internal Student-t spot-to-prototype associations;
- `predicted_domains.csv`: barcodes, GatorPrism domains, and spatial coordinates;
- `spot_coalition_weights.csv`: named spot-level joint and modality-specific routing weights;
- `domain_coalition_profiles.csv`: mean coalition weights within each inferred domain;
- `training_history.csv`: total and component losses for every epoch.

For two modalities, the routing columns are ordered as `joint`, the first modality, and the second modality. For example, an RNA–ATAC run produces `joint`, `RNA`, and `ATAC` coalition weights.

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
    features,
    consensus_mixed_edges,
    modality_mixed_edges,
)

z = outputs["z"]
coalition_weights = outputs["gates"]
```

