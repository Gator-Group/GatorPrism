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

## Reference

Townes, F. W. & Engelhardt, B. E. [Nonnegative spatial factorization applied to spatial genomics](https://doi.org/10.1038/s41592-022-01687-w). *Nature Methods* **20**, 229–238 (2023).
