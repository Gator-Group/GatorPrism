# Datasets

GatorPrism was evaluated on publicly available spatial multi-omics datasets and two paired simulated datasets. This page documents the source of each public dataset, the simulation model and parameters, and the expected data organization. Public datasets are not redistributed by this repository; users should obtain them from the original providers and comply with the corresponding terms of use.

## Public datasets

All experimental datasets used in this study are publicly accessible.

| Dataset | Modalities | Tissue or stage | Public source |
|---|---|---|---|
| Human Lymph Node | RNA + protein (ADT) | Human lymph-node sections A1 and D1 | [NCBI Gene Expression Omnibus: GSE263617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617) |
| Human Tonsil | RNA + protein (ADT) | Human tonsil sections A1 and D1 | [NCBI Gene Expression Omnibus: GSE263617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617) |
| Spatial ATAC–RNA-seq Mouse Brain | RNA + ATAC | Postnatal day 22 (P22) mouse brain | [UCSC Cell Browser](https://brain-spatial-omics.cells.ucsc.edu/) |
| Mouse Embryonic MISAR-seq | RNA + ATAC | Developing mouse brain; E11.0 and E18.5 sections used in this study | [BioSino NODE: OEP003285](https://www.biosino.org/node/project/detail/OEP003285) |

## Simulated datasets

We generated two paired spatial multi-omics datasets following the nonnegative spatial factorization framework described by [Townes and Engelhardt](https://doi.org/10.1038/s41592-022-01687-w). Both simulations use a shared latent spatial-factor model but differ in geometry, feature dimensions, and domain number:

- **Simulated RNA + ADT:** four geometric shape factors plus background, giving five ground-truth domains;
- **Simulated RNA + ATAC:** six nested square-ring factors, giving six ground-truth domains without a background class.

The two ready-to-use simulated datasets are distributed with this repository as ZIP archives.


An expected local layout is:

```text
data/
├── human_lymph_node_A1_RNA.h5ad
├── human_lymph_node_A1_ADT.h5ad
├── mouse_embryonic_E11_RNA.h5ad
├── mouse_embryonic_E11_ATAC.h5ad
├── sim_RNA_ADT/
│   ├── adata_RNA.h5ad
│   ├── adata_ADT.h5ad
│   └── ground_truth.txt
└── sim_RNA_ATAC/
    ├── adata_RNA.h5ad
    ├── adata_ATAC.h5ad
    └── ground_truth.txt
```

## Reference

Townes, F. W. & Engelhardt, B. E. [Nonnegative spatial factorization applied to spatial genomics](https://doi.org/10.1038/s41592-022-01687-w). *Nature Methods* **20**, 229–238 (2023).
