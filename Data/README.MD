# Public datasets

All experimental datasets used in this study are publicly accessible. The Human Lymph Node dataset is available from the Gene Expression Omnibus under accession number GSE263617. The Human Tonsil dataset co-profiling transcriptomes and proteins is available at https://www.10xgenomics.com/resources/
datasets/gene-protein-expression-library-of-human-tonsil-cytassist
-ffpe-2-standard. The ATAC-RNA-seq mouse brain multi-omics dataset is available through the UCSC Cell Browser (https://brain-spatial-omics.cells.ucsc.edu). The mouse embryonic MISAR-seq dataset is deposited in the National Genomics Data Center under accession number OEP003285. 

# Simulated datasets
We further generated two simulated spatial multi-omics datasets following the nonnegative spatial factorization framework. 
\subsection{Simulated Datasets}
\label{sec:sim_data}

Following the nonnegative spatial factorization (NSF) framework of
Townes and Engelhardt~\cite{townes2023nonnegative}, we generated paired
simulated spatial multi-omics datasets from a shared latent factor model.
For each spot $i$ and feature $j$, the expected count is

\begin{equation}
\label{eq:nsf_mean}
\mu_{ij} \;=\; b \;+\; (F W^{\top})_{ij} \;+\; (U V^{\top})_{ij},
\end{equation}

\noindent where $F \in \mathbb{R}^{N \times L_{\mathrm{sp}}}$ are
\emph{spatial factors} derived from a preset geometric pattern and shared
across modalities, thereby defining the ground-truth spatial domains;
$W \in \mathbb{R}^{J \times L_{\mathrm{sp}}}$ are the spatial loadings;
$U \in \{0,1\}^{N \times L_{\mathrm{ns}}}$, with each entry drawn
$\sim\mathrm{Bernoulli}(p_{\mathrm{ns}})$, are \emph{non-spatial factors}
capturing spot-level variation; and
$V \in \mathbb{R}^{J \times L_{\mathrm{ns}}}$ are the corresponding
non-spatial loadings. The spatial and non-spatial loading blocks are
scaled by fractions $s_{\mathrm{sp}}$ and $1{-}s_{\mathrm{sp}}$,
respectively, so that every feature is driven by a mixture of
domain-level structure and spot-level noise. Counts are then sampled
from a shared discrete likelihood,

\begin{equation}
\label{eq:nsf_counts}
Y_{ij} \;\sim\; \mathrm{NegativeBinomial}\!\big(\,\mathrm{shape}=r,\; \mathrm{mean}=\mu_{ij}\,\big).
\end{equation}

\noindent We set the background mean to $b=0.2$, the mean expression level
to $\bar{\mu}=20$, and the dispersion to $r=10$ for all modalities. Raw
counts $Y$ were stored in \texttt{adata.layers["counts"]} and the
transformed values $\log(1+Y)$ in \texttt{adata.X} for downstream methods.

\textbf{Spatial structure.}
Spots are placed on a $36 \times 36$ grid, giving
$N = 36^{2} = 1{,}296$ spots, with coordinates rescaled to a square box.
The ground-truth domain label of each spot is
$\arg\max_{l} F_{il}$; spots whose factors are all zero are assigned to a
\emph{Background} class. The two paired datasets were designed with
\emph{different} geometries and \emph{different} numbers of domains so
that they are visually and statistically distinct
(Figure~\ref{fig:sim_overview}):
\begin{itemize}
    \item \textbf{RNA\,+\,ADT} uses four shape factors --- circle,
          triangle, square, and diamond --- together with a background
          region, yielding $K=5$ domains.
    \item \textbf{RNA\,+\,ATAC} uses six nested concentric square rings
          (Chebyshev-distance bins anchored at the top-left corner),
          yielding $K=6$ domains with no background region.
\end{itemize}

\textbf{Modality pairing.}
Within each dataset, the two modalities share the same spatial factor
matrix $F$ and the same spot coordinates, so they have identical
ground-truth spatial domains; only the loadings, the feature dimensions,
and the per-modality random seed differ, making the feature spaces
complementary while the spatial structure remains consistent. We assign
each modality an independent number of features --- $1{,}000$ genes for
RNA, $100$ protein features for ADT, and $1{,}000$ peaks for ATAC ---
following common spatial multi-omics conventions. The full parameter
setting is summarized in Table~\ref{tab:sim_params}.

\begin{table}[t]
\centering
\caption{Generation parameters for the paired simulated datasets.}
\label{tab:sim_params}
\small
\begin{tabular}{llcc}
\toprule
Parameter & Symbol & Value & Description \\
\midrule
Grid side            & ---             & $36$           & $N = 36^{2} = 1{,}296$ spots \\
Spatial factors      & $L_{\mathrm{sp}}$ & $4$ (ADT) / $6$ (ATAC) & \# spatial domains $-1$ (ADT) \\
Non-spatial factors  & $L_{\mathrm{ns}}$ & $3$            & non-spatial variation dim. \\
Non-spatial dropout  & $p_{\mathrm{ns}}$ & $0.2$          & $P(U_{il}=1)$ \\
Spatial share        & $s_{\mathrm{sp}}$ & $0.55$         & spatial vs.\ non-spatial weight \\
Background mean      & $b$             & $0.2$          & NB background \\
Mean expression      & $\bar{\mu}$     & $20$           & overall expression level \\
NB dispersion        & $r$             & $10$           & lower $\Rightarrow$ sparser \\
Base seed            & ---             & $101$          & omics $k$ uses seed $101{+}1000k$ \\
\midrule
\multicolumn{4}{l}{\textit{Per-dataset geometry and feature counts}} \\
RNA\,+\,ADT (shapes) & --- & RNA $1{,}000$ / ADT $100$ & $K=5$ domains \\
RNA\,+\,ATAC (nested)& --- & RNA $1{,}000$ / ATAC $1{,}000$ & $K=6$ domains \\
\bottomrule
\end{tabular}
\end{table}
