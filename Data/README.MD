# Datasets

GatorPrism was evaluated on publicly available spatial multi-omics datasets and two paired simulated datasets. This page documents the source of each public dataset, the simulation model and parameters, and the expected data organization. Public datasets are not redistributed by this repository; users should obtain them from the original providers and comply with the corresponding terms of use.

## Public datasets

All experimental datasets used in this study are publicly accessible.

| Dataset | Modalities | Tissue or stage | Public source |
|---|---|---|---|
| Human Lymph Node | RNA + protein (ADT) | Human lymph-node sections A1 and D1 | [NCBI Gene Expression Omnibus: GSE263617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617) |
| Human Tonsil | RNA + protein (ADT) | Human FFPE tonsil | [10x Genomics Visium CytAssist Gene and Protein Expression Library of Human Tonsil](https://www.10xgenomics.com/datasets/gene-protein-expression-library-of-human-tonsil-cytassist-ffpe-2-standard) |
| ATAC–RNA-seq Mouse Brain | RNA + ATAC | Postnatal mouse-brain spatial multi-omics | [UCSC Cell Browser](https://brain-spatial-omics.cells.ucsc.edu/) |
| Mouse Embryonic MISAR-seq | RNA + ATAC | Mouse embryonic sections | [BioSino NODE: OEP003285](https://www.biosino.org/node/project/detail/OEP003285) |

### Human Lymph Node

The Human Lymph Node data are available from the NCBI Gene Expression Omnibus under accession [GSE263617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617). The GEO series contains paired Visium CytAssist transcriptomic and protein measurements for lymph-node sections A1 and D1. The processed RNA and protein data are linked to the corresponding transcriptomic sample records.

### Human Tonsil

The Human Tonsil dataset co-profiles transcript abundance and antibody-derived protein abundance using the Visium CytAssist Spatial Gene and Protein Expression assay. The dataset and associated Space Ranger outputs are available from the [10x Genomics public dataset portal](https://www.10xgenomics.com/datasets/gene-protein-expression-library-of-human-tonsil-cytassist-ffpe-2-standard). The provider reports that this dataset is distributed under the CC BY 4.0 licence.

### ATAC–RNA-seq Mouse Brain

The spatial ATAC–RNA-seq mouse-brain dataset jointly profiles chromatin accessibility and gene expression. Raw data and browser-based exploration are available through the [UCSC Cell Browser](https://brain-spatial-omics.cells.ucsc.edu/).

### Mouse Embryonic MISAR-seq

The mouse embryonic MISAR-seq dataset jointly profiles spatial transcriptomic and chromatin-accessibility signals across embryonic sections. The project is available through the BioSino National Omics Data Encyclopedia under accession [OEP003285](https://www.biosino.org/node/project/detail/OEP003285).

## Simulated datasets

We generated two paired spatial multi-omics datasets following the nonnegative spatial factorization framework described by [Townes and Engelhardt](https://doi.org/10.1038/s41592-022-01687-w). Both simulations use a shared latent spatial-factor model but differ in geometry, feature dimensions, and domain number:

- **Simulated RNA + ADT:** four geometric shape factors plus background, giving five ground-truth domains;
- **Simulated RNA + ATAC:** six nested square-ring factors, giving six ground-truth domains without a background class.

### Generative model

For spot $i$ and feature $j$, the expected count is

$$
\mu_{ij} = b + (FW^{\mathsf T})_{ij} + (UV^{\mathsf T})_{ij},
$$

where:

- $F \in \mathbb{R}^{N \times L_{\mathrm{sp}}}$ contains nonnegative spatial factors derived from a preset geometric pattern. The same $F$ is shared by both modalities within a paired dataset and defines the ground-truth spatial domains;
- $W \in \mathbb{R}^{J \times L_{\mathrm{sp}}}$ contains the feature-specific spatial loadings;
- $U \in \{0,1\}^{N \times L_{\mathrm{ns}}}$ contains non-spatial spot-level factors, with each entry independently sampled as $U_{il}\sim\operatorname{Bernoulli}(p_{\mathrm{ns}})$;
- $V \in \mathbb{R}^{J \times L_{\mathrm{ns}}}$ contains the corresponding non-spatial loadings;
- $b$ is the background mean.

Before the mean is calculated, the spatial and non-spatial loading blocks are scaled by $s_{\mathrm{sp}}$ and $1-s_{\mathrm{sp}}$, respectively. Each feature is therefore controlled by a mixture of domain-level spatial structure and spot-level non-spatial variation.

Counts are sampled from a negative-binomial likelihood parameterized by shape $r$ and mean $\mu_{ij}$:

$$
Y_{ij}\sim\operatorname{NegativeBinomial}
\left(\operatorname{shape}=r,\operatorname{mean}=\mu_{ij}\right).
$$

Under this parameterization, the variance is $\mu_{ij}+\mu_{ij}^{2}/r$. We use the same likelihood parameters for all modalities:

- background mean: $b=0.2$;
- overall mean expression level: $\bar{\mu}=20$;
- negative-binomial dispersion: $r=10$.

Raw counts $Y$ are stored in `adata.layers["counts"]`, whereas `adata.X` contains the transformed values $\log(1+Y)$ used by downstream methods.

### Spatial structure and ground truth

Spots are placed on a $36\times36$ grid, giving

$$
N=36^{2}=1{,}296
$$

spots. Coordinates are rescaled to a square spatial box. The ground-truth domain label for spot $i$ is

$$
y_i=\underset{l}{\arg\max}\;F_{il}.
$$

When all spatial factors for a spot are zero, that spot is assigned to the `Background` class.

#### Simulated RNA + ADT

The RNA + ADT dataset contains four spatial factors:

1. circle;
2. triangle;
3. square;
4. diamond.

Spots outside all four shapes form a background region. The resulting dataset therefore contains $K=5$ ground-truth domains.

#### Simulated RNA + ATAC

The RNA + ATAC dataset contains six nested concentric square rings defined by Chebyshev-distance bins anchored at the top-left corner. Every spot belongs to one of the six rings, giving $K=6$ ground-truth domains with no background region.

The two datasets intentionally use different spatial geometries and different domain numbers so that they remain visually and statistically distinct.

### Modality pairing

Within each simulated dataset, the two modalities share the same spatial-factor matrix $F$, spot coordinates, barcode order, and ground-truth domain labels. The modalities differ in their feature loadings, feature dimensions, and random seeds. This design produces complementary feature spaces while preserving a common spatial organization.

The feature dimensions are:

- RNA: 1,000 genes;
- ADT: 100 protein features;
- ATAC: 1,000 peaks.

The base random seed is 101. Modality $k$ uses seed $101+1000k$, ensuring deterministic but modality-specific sampling.

### Simulation parameters

| Parameter | Symbol | Value | Description |
|---|---:|---:|---|
| Grid side | — | 36 | $N=36^2=1{,}296$ spots |
| Spatial factors | $L_{\mathrm{sp}}$ | 4 for RNA + ADT; 6 for RNA + ATAC | Shape or ring factors |
| Non-spatial factors | $L_{\mathrm{ns}}$ | 3 | Dimension of non-spatial variation |
| Non-spatial activation probability | $p_{\mathrm{ns}}$ | 0.2 | $P(U_{il}=1)$ |
| Spatial share | $s_{\mathrm{sp}}$ | 0.55 | Relative spatial loading weight |
| Non-spatial share | $1-s_{\mathrm{sp}}$ | 0.45 | Relative non-spatial loading weight |
| Background mean | $b$ | 0.2 | Negative-binomial background mean |
| Mean expression | $\bar{\mu}$ | 20 | Overall feature mean |
| Negative-binomial dispersion | $r$ | 10 | Lower values produce greater overdispersion |
| Base seed | — | 101 | Modality $k$ uses seed $101+1000k$ |

### Per-dataset configuration

| Dataset | Geometry | Modality dimensions | Ground-truth domains |
|---|---|---|---:|
| Simulated RNA + ADT | Circle, triangle, square, diamond, and background | RNA: 1,000 genes; ADT: 100 proteins | 5 |
| Simulated RNA + ATAC | Six nested square rings | RNA: 1,000 genes; ATAC: 1,000 peaks | 6 |

## AnnData organization

Each modality is stored as a separate `.h5ad` object. Paired modalities must have identical `obs_names` and spatial coordinates.

| AnnData field | Contents |
|---|---|
| `adata.X` | Processed feature matrix, $\log(1+Y)$ for simulated data |
| `adata.layers["counts"]` | Raw simulated negative-binomial counts $Y$ |
| `adata.obsm["spatial"]` | Two-dimensional spot coordinates |
| `adata.obs["ground_truth"]` | Simulated ground-truth domain label |
| `adata.obs_names` | Shared spot identifiers used to align modalities |

An expected local layout is:

```text
data/
├── human_lymph_node_A1_RNA.h5ad
├── human_lymph_node_A1_ADT.h5ad
├── mouse_embryonic_E11_RNA.h5ad
├── mouse_embryonic_E11_ATAC.h5ad
├── sim_RNA_ADT/
│   ├── adata_RNA.h5ad
│   └── adata_ADT.h5ad
└── sim_RNA_ATAC/
    ├── adata_RNA.h5ad
    └── adata_ATAC.h5ad
```

The public-data filenames shown above are recommended local names rather than names imposed by the original repositories.

## Data-use and reproducibility notes

- Experimental datasets should be cited using their original publications and repository records.
- Users are responsible for complying with the licences and terms specified by GEO, 10x Genomics, UCSC, and BioSino NODE.
- Public raw data should be processed consistently before method comparison. GatorPrism expects processed feature matrices and does not perform dataset-specific count normalization or dimensionality reduction internally.
- The simulated modalities must retain identical barcodes and coordinates; only their molecular feature matrices and modality-specific random seeds differ.
- Ground-truth labels are provided for evaluation only and are not used to optimize GatorPrism.

## Reference

Townes, F. W. & Engelhardt, B. E. [Nonnegative spatial factorization applied to spatial genomics](https://doi.org/10.1038/s41592-022-01687-w). *Nature Methods* **20**, 229–238 (2023).
