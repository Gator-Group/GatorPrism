# Datasets

GatorPrism was evaluated on real dual-omics and tri-omics spatial datasets, as well as simulated dual-omics and tri-omics spatial datasets. This page lists the sources of the real datasets and summarizes the simulated datasets. All simulated datasets are provided as ZIP archives in this directory. Real datasets should be obtained from the original providers under their applicable terms of use.

## Public datasets

All experimental datasets used in this study are publicly accessible.

| Dataset | Modalities | Tissue or stage | Public source |
|---|---|---|---|
| Human Lymph Node | RNA + protein (ADT) | Human lymph-node sections A1 | [NCBI Gene Expression Omnibus: GSE263617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617) |
| Human Tonsil | RNA + protein (ADT) | Human tonsil sections S1 and S2 | [Zenodo record 18946723](https://doi.org/10.5281/zenodo.18946723) |
| Spatial ATAC–RNA-seq Mouse Brain | RNA + ATAC | Postnatal day 22 (P22) mouse brain | [UCSC Cell Browser](https://brain-spatial-omics.cells.ucsc.edu/) |
| Mouse Embryonic MISAR-seq | RNA + ATAC | Developing mouse brain; E11.0 and E18.5 sections used in this study | [BioSino NODE: OEP003285](https://www.biosino.org/node/project/detail/OEP003285) |
| Spatial-Mux-seq Mouse Brain | RNA + ATAC + protein (ADT) | 5-month-old EAE mouse brain; sample `5M_20um`, 20-µm spatial pixels | [NCBI Gene Expression Omnibus: GSE263333](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263333) |

## Simulated datasets

We generated three paired spatial multi-omics datasets following the nonnegative spatial factorization framework described by [Townes and Engelhardt](https://doi.org/10.1038/s41592-022-01687-w). They use a shared latent spatial-factor framework but differ in geometry, feature dimensions, modality number, and ground-truth domain number:

- **Simulated RNA + ADT:** four geometric shape factors plus background, giving five ground-truth domains.
- **Simulated RNA + ATAC:** six nested square-ring factors, giving six ground-truth domains without a background class.
- **Simulated RNA + ADT + ATAC:** five geometric shape factors plus background, giving six ground-truth domains. 

The three ready-to-use simulated datasets are distributed with this repository as ZIP archives:

- [sim_RNA_ADT.zip](./sim_RNA_ADT.zip)
- [sim_RNA_ATAC.zip](./sim_RNA_ATAC.zip)
- [sim_RNA_ADT_ATAC.zip](./sim_RNA_ADT_ATAC.zip)

## Reference

Townes, F. W. & Engelhardt, B. E. [Nonnegative spatial factorization applied to spatial genomics](https://doi.org/10.1038/s41592-022-01687-w). *Nature Methods* **20**, 229–238 (2023).
