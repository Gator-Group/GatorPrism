# -*- coding: utf-8 -*-
"""
Simulate **paired two-omics** spatial data using the NSF generative model
(Townes & Engelhardt, Nature Methods 2023).

NSF generative scheme: Lambda = bkg_mean + F @ W.T + U @ V.T ; counts ~ NegBinom(shape=r, mean=Lambda)
  F : N x Lsp  spatial factors (geometric patterns; these define the spatial domains)
  W : J x Lsp  spatial loadings ; U : N x Lns non-spatial factors ; V : J x Lns non-spatial loadings

The key to pairing the two omics: they **share the same F, the same spatial coordinates and the same
spot order**; only the loadings, feature dimension and RNG seed differ.
  -> Both omics have identical true spatial domains but complementary feature spaces
     (= ground-truth data for vertical integration).

True spatial domain label = argmax(F); a spot that is zero across all spatial factors becomes
background. See the __main__ block for the geometry used by each shipped dataset.
"""
import os, numpy as np
from pandas import get_dummies
from anndata import AnnData
import scanpy as sc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

dtp = "float32"

# ---------------- NSF geometric patterns (verbatim from nsf-paper/simulations/sim.py) ----------------
def ggblocks():
    A = np.zeros([4, 36])
    A[0, [1, 6, 7, 8, 13]] = 1
    A[1, [3, 4, 5, 9, 11, 15, 16, 17]] = 1
    A[2, [18, 24, 25, 30, 31, 32]] = 1
    A[3, [21, 22, 23, 28, 34]] = 1
    return A  # 6x6 base, L=4

def squares():
    A = np.zeros([12, 12]); A[1:5, 1:5] = 1; A[7:11, 1:5] = 1; A[1:5, 7:11] = 1; A[7:11, 7:11] = 1; return A
def corners():
    B = np.zeros([6, 6])
    for i in range(6): B[i, i:] = 1
    A = np.flip(B, axis=1); AB = np.hstack((A, B)); CD = np.flip(AB, axis=0); return np.vstack((AB, CD))
def scotland():
    A = np.eye(12)
    for i in range(12): A[-i - 1, i] = 1
    return A
def checkers():
    A = np.zeros([4, 4]); B = np.ones([4, 4]); AB = np.hstack((A, B, A)); BA = np.hstack((B, A, B)); return np.vstack((AB, BA, AB))
def quilt():
    A = np.zeros([4, 144]); A[0] = squares().flatten(); A[1] = corners().flatten(); A[2] = scotland().flatten(); A[3] = checkers().flatten(); return A

def blocks6():
    """6x6 base: 5 clean contiguous factor blocks + background frame -> 6 spatial domains.
    Clean/blocky look (cf. NSF/SpatialGlue 'ground truth of factors' figure), distinct from ggblocks."""
    g = -np.ones((6, 6), dtype=int)          # -1 = background frame
    g[1:3, 1:3] = 0      # top-left block
    g[1:3, 3:5] = 1      # top-right block
    g[3:5, 1:3] = 2      # mid-left block
    g[3:5, 3:5] = 3      # mid-right block
    g[5, 1:5] = 4        # bottom stripe
    A = np.stack([(g.flatten() == f).astype(float) for f in range(5)])   # 5x36, L=5
    return A

# ---------------- Shape geometry (circle / triangle / diamond / square / plus) ----------------
def shape_factors(shapes, nside=36):
    """Each spatial factor (domain) is a geometric shape; build the binary N x L matrix F directly on
    an nside x nside grid. Spots not covered by any shape become background.
    shapes: list of (kind, params)."""
    ii, jj = np.meshgrid(np.arange(nside), np.arange(nside), indexing="ij")
    masks = []
    for kind, p in shapes:
        if kind == "circle":
            cy, cx, r = p; m = (ii - cy) ** 2 + (jj - cx) ** 2 <= r ** 2
        elif kind == "diamond":
            cy, cx, s = p; m = np.abs(ii - cy) + np.abs(jj - cx) <= s
        elif kind == "square":
            cy, cx, h = p; m = (np.abs(ii - cy) <= h) & (np.abs(jj - cx) <= h)
        elif kind == "triangle":                     # isosceles triangle, apex pointing up
            cy, cx, s = p; dy = ii - (cy - s)
            m = (dy >= 0) & (ii <= cy + s) & (np.abs(jj - cx) <= dy / 2 + 0.5)
        elif kind == "plus":
            cy, cx, s, w = p
            m = ((np.abs(ii - cy) <= s) & (np.abs(jj - cx) <= w)) | \
                ((np.abs(jj - cx) <= s) & (np.abs(ii - cy) <= w))
        else:
            raise ValueError(kind)
        masks.append(m.flatten().astype(float))
    F = np.stack(masks).T                            # N x L; the shapes are designed not to overlap
    return F


# Shape layout: RNA-ADT uses 4 shapes (K=5); nside=36
SHAPE_SCENARIOS = {
    "shapes_adt": [("circle", (9, 9, 7)), ("triangle", (10, 27, 7)),
                   ("square", (27, 9, 6)), ("diamond", (27, 27, 8))],
}


def nested_corner_factors(nside=36, edges=(6, 11, 16, 21, 26)):
    """Nested concentric square rings anchored at the top-left corner (binned Chebyshev distance)
    -> K = len(edges)+1. Domain-1 is the innermost small square in the corner, each successive
    L-shaped ring extends outward, and the outermost layer is one large region (no background).
    The default 5 edges give 6 distinct nested squares."""
    ii, jj = np.meshgrid(np.arange(nside), np.arange(nside), indexing="ij")
    d = np.maximum(ii, jj)
    bins = [0] + list(edges) + [nside + 1]
    K = len(bins) - 1
    masks = [((d >= bins[k]) & (d < bins[k + 1])).flatten().astype(float) for k in range(K)]
    return np.stack(masks).T                          # N x K (covers every spot, no background)


def sqrt_int(x):
    z = int(round(x ** .5))
    if x == z ** 2: return z
    raise ValueError("x must be a square integer")

def gen_spatial_factors(scenario="ggblocks", nside=36):
    if scenario in SHAPE_SCENARIOS:                  # shape geometry (circle/triangle/...)
        return shape_factors(SHAPE_SCENARIOS[scenario], nside=nside)
    if scenario == "nested5":                        # nested concentric square rings (top-left corner)
        return nested_corner_factors(nside=nside)
    A = {"ggblocks": ggblocks, "quilt": quilt, "blocks6": blocks6}[scenario]()
    unit = sqrt_int(A.shape[1]); assert nside % unit == 0
    ncopy = nside // unit; N = nside ** 2; L = A.shape[0]
    A = A.reshape((L, unit, unit)); A = np.kron(A, np.ones((1, ncopy, ncopy)))
    return A.reshape((L, N)).T  # N x L

def make_grid(N, xmin=-2, xmax=2):
    x = np.linspace(xmin, xmax, num=int(np.sqrt(N)), dtype=dtp)
    return np.stack([X.ravel() for X in np.meshgrid(x, x)], axis=1)

def rescale_spatial_coords(X, box_side=4):
    X = X.copy().astype(float); X -= X.min(0)
    X *= box_side / np.exp(np.mean(np.log(X.max(0)))); return X - X.mean(0)

def gen_spatial_coords(N):
    X = make_grid(N); X[:, 1] = -X[:, 1]; return rescale_spatial_coords(X)

def gen_nonspatial_factors(N, L=3, nzprob=0.2, seed=101):
    return np.random.default_rng(seed).binomial(1, nzprob, size=(N, L))

def gen_loadings(Lsp, Lns=3, Jmix=500, expr_mean=20.0, mix_frac_spat=0.55, seed=101):
    rng = np.random.default_rng(seed); J = Jmix
    W = get_dummies(rng.choice(Lsp, J, replace=True)).to_numpy(dtype=dtp) if Lsp > 0 else np.zeros((J, 0))
    V = get_dummies(rng.choice(Lns, J, replace=True)).to_numpy(dtype=dtp) if Lns > 0 else np.zeros((J, 0))
    # Every feature is "mixed" (driven by both spatial and non-spatial factors), matching the NSF default
    W *= (mix_frac_spat * expr_mean)
    V *= ((1 - mix_frac_spat) * expr_mean)
    return W, V

# ---------------- Joint simulation of the two omics ----------------
def sim_two_omics(scenario="ggblocks", nside=36, bkg_mean=0.2, base_seed=101,
                  omics=(("RNA", 1000, 20.0, 10.0), ("ADT", 100, 30.0, 10.0))):
    """Share the spatial factors F and the coordinates, then generate loadings, non-spatial factors
    and counts independently for each omics.
    Each entry of `omics` = (name, J number of features, expr_mean expression level,
    nb_shape dispersion [lower = sparser])."""
    F = gen_spatial_factors(scenario=scenario, nside=nside)      # N x Lsp, shared
    N, Lsp = F.shape
    X = gen_spatial_coords(N)                                    # shared coordinates
    # True spatial domain: argmax(F); a spot that is zero everywhere becomes background (= Lsp)
    gt = np.where(F.sum(1) > 0, F.argmax(1), Lsp).astype(int)    # 0..Lsp-1 plus background
    dom_names = [f"Domain-{i+1}" for i in range(Lsp)] + ["Background"]
    gt_str = np.array([dom_names[g] for g in gt])

    adatas = {}
    for k, (name, J, expr_mean, nb_shape) in enumerate(omics):
        seed = base_seed + 1000 * (k + 1)
        W, V = gen_loadings(Lsp, Lns=3, Jmix=J, expr_mean=expr_mean, seed=seed)
        U = gen_nonspatial_factors(N, L=V.shape[1], nzprob=0.2, seed=seed)
        Lambda = bkg_mean + F @ W.T + U @ V.T                    # N x J
        Y = np.random.default_rng(seed).negative_binomial(nb_shape, nb_shape / (Lambda + nb_shape)).astype(dtp)
        ad = AnnData(Y, obsm={"spatial": X.astype(dtp), "spfac": F.astype(dtp), "nsfac": U.astype(dtp)},
                     varm={"spload": W, "nsload": V})
        ad.var_names = [f"{name}_{j}" for j in range(J)]
        ad.obs_names = [str(i) for i in range(N)]
        ad.obs["ground_truth"] = gt_str
        ad.layers["counts"] = ad.X.copy()
        sc.pp.log1p(ad)
        adatas[name] = ad
    return adatas, X, F, gt, gt_str, dom_names

# ---------------- Run + save + visualize ----------------
def make_dataset(out, omics, scenario="ggblocks"):
    """Generate and save one paired two-omics dataset plus an overview figure.
    omics: 2 entries of (name, J, expr_mean, nb_shape)."""
    os.makedirs(out, exist_ok=True)
    adatas, X, F, gt, gt_str, dom_names = sim_two_omics(scenario=scenario, nside=36, omics=omics)
    K = len(np.unique(gt))
    names = list(adatas.keys())
    for name, ad in adatas.items():
        ad.write_h5ad(f"{out}/adata_{name}.h5ad")
        Y = np.asarray(ad.layers["counts"]); sparsity = float((Y == 0).mean())
        print(f"[saved] {out}/adata_{name}.h5ad  {ad.shape}  counts 0-{int(Y.max())}  zeros={sparsity:.0%}")
    np.savetxt(f"{out}/ground_truth.txt", gt, fmt="%d")
    print(f"  N={F.shape[0]} spots, K={K} domains {dict((d,int((gt_str==d).sum())) for d in dom_names)}")

    def col(ad, j): return np.asarray(ad.layers["counts"])[:, j]
    def marker(ad, factor):                       # representative feature dominated by this spatial factor
        W = ad.varm["spload"]; pure = W[:, factor] - (W.sum(1) - W[:, factor]); return int(np.argmax(pure))
    o1, o2 = names
    panels = [("Ground-truth domains", gt, "tab10", True),
              (f"{o1} total", np.log1p(adatas[o1].layers["counts"].sum(1)), "viridis", False),
              (f"{o2} total", np.log1p(adatas[o2].layers["counts"].sum(1)), "magma", False),
              (f"{o1} marker(f0)", np.log1p(col(adatas[o1], marker(adatas[o1], 0))), "inferno", False),
              (f"{o2} marker(f2)", np.log1p(col(adatas[o2], marker(adatas[o2], 2))), "inferno", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.3))
    for ax, (t, c, cmap, cat) in zip(axes, panels):
        sca = ax.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap, s=10, linewidths=0)
        if not cat: fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(t, fontsize=9); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"NSF-based paired simulation: {o1}-{o2} (shared spatial domains)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/overview.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}/overview.png")

if __name__ == "__main__":
    # NSF defaults (expr_mean=20, nb_shape=10, mix_frac_spat=0.55, bkg_mean=0.2, Lns=3, nzprob=0.2)
    # The two simulations use **different spatial geometry and a different number of domains**, so both
    # the shapes and the cluster counts differ (clean look, cf. the NSF/SpatialGlue figures):
    #   RNA-ADT  -> shapes_adt (circle/triangle/square/diamond + background = 5 domains)
    #   RNA-ATAC -> nested5    (6 nested concentric corner rings, no background = 6 domains)
    EXPR, SHAPE = 20.0, 10.0
    print("===== RNA-ADT (circle/triangle/square/diamond, K=5) =====")
    make_dataset("bench/data/sim_RNA_ADT",  (("RNA", 1000, EXPR, SHAPE), ("ADT", 100, EXPR, SHAPE)),
                 scenario="shapes_adt")
    print("\n===== RNA-ATAC (nested concentric corner squares, K=6) =====")
    make_dataset("bench/data/sim_RNA_ATAC", (("RNA", 1000, EXPR, SHAPE), ("ATAC", 1000, EXPR, SHAPE)),
                 scenario="nested5")
