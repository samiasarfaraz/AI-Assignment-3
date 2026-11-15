"""
CS 351 - Assignment 03, Part B Q1
Starter skeleton: Colour quantization with K-Means and DBSCAN.

Complete the TODO sections according to the assignment instructions.
Only NumPy and Matplotlib should be used for the algorithms.
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KM_MAX_SIDE = 300
MIN_PTS = 6

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_image(path):
    """Load an image as normalized RGB."""
    # TODO
    img = plt.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0  #normalise the rgb vals
    else:
        img = img.astype(np.float64) 
    if img.ndim == 2:   #greyscale image
        img = np.stack([img, img, img], axis=-1)
    return img[..., :3]


def sq_dists(X, C):
    """Calculate pairwise squared distances."""
    # TODO
    # (x - c)^2 = x^2 - 2.x.c + c^2
    d = (X**2).sum(1)[:, None] - 2.0 * X @ C.T + (C**2).sum(1)[None, :]
    return np.maximum(d, 0.0)


def downsample(img, max_side=80):
    """Downsample an image."""
    # TODO
    h, w = img.shape[:2]
    step = int(np.ceil(max(h, w) / max_side))
    if step <= 1:
        return img
    return img[::step, ::step]


def image_features(img, lam=None):
    """Extract RGB and optional spatial features."""
    # TODO
    h, w, _ = img.shape
    X = img.reshape(-1, 3)
    if lam is None:
        return X
    ys, xs = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w),
                         indexing="ij")
    spatial = np.stack([xs.ravel(), ys.ravel()], axis=1) * lam
    return np.hstack([X, spatial])


def compression_ratio(k, n_pixels):
    """Calculate compression ratio."""
    # TODO
    return 24.0 / (np.log2(k) + 24.0 * k / n_pixels)


def apply_palette(img, C):
    """Replace every pixel of img by its nearest colour in palette C."""
    h, w, _ = img.shape
    X = img.reshape(-1, 3)
    labels = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), 200000):
        labels[i:i + 200000] = sq_dists(X[i:i + 200000], C[:, :3]).argmin(1)
    return C[labels, :3].reshape(h, w, 3)


# ----------------------------------------------------------------------
# K-Means
# ----------------------------------------------------------------------

class KMeans:
    def __init__(self, k, init="random", max_iter=50, tol=0.001,
                 seed=0, snapshot_iters=()):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.snapshot_iters = set(snapshot_iters)

    def _init_random(self, X):
        # TODO
        idx = self.rng.choice(len(X), self.k, replace=False)
        return X[idx].astype(np.float64)

    def _init_kmeanspp(self, X):
        # TODO
        n = len(X)
        C = np.empty((self.k, X.shape[1]))
        C[0] = X[self.rng.integers(n)]
        d2 = sq_dists(X, C[:1]).ravel()
        for j in range(1, self.k):
            total = d2.sum()
            p = d2 / total if total > 0 else None
            C[j] = X[self.rng.choice(n, p=p)]
            d2 = np.minimum(d2, sq_dists(X, C[j:j+1]).ravel())
        return C

    def _assign(self, X, C, chunk=200000):
        labels = np.empty(len(X), dtype=np.int64)
        for i in range(0, len(X), chunk):
            labels[i:i + chunk] = sq_dists(X[i:i + chunk], C).argmin(1)
        return labels

    def fit(self, X):
        # TODO
        X = np.asarray(X, dtype=np.float64)
        if self.init == "random":
            C = self._init_random(X)
        else:
            C = self._init_kmeanspp(X)

        labels = None
        self.sse_history = []
        self.snapshots = {}

        for it in range(1, self.max_iter + 1):
            # 1) assignment step
            new_labels = self._assign(X, C)
            change = 1.0 if labels is None else np.mean(new_labels != labels)
            labels = new_labels

            # 2) update step (vectorised means with bincount)
            counts = np.bincount(labels, minlength=self.k)
            C_new = np.zeros_like(C)
            for ch in range(X.shape[1]):
                sums = np.bincount(labels, weights=X[:, ch],
                                   minlength=self.k)
                C_new[:, ch] = sums / np.maximum(counts, 1)

            # 3) empty clusters -> farthest pixel from its own centroid
            empty = np.flatnonzero(counts == 0)
            if len(empty) > 0:
                own = ((X - C[labels]) ** 2).sum(1)
                for j in empty:
                    far = own.argmax()
                    C_new[j] = X[far]
                    own[far] = -1.0

            C = C_new
            self.sse_history.append(float(((X - C[labels]) ** 2).sum()))
            if it in self.snapshot_iters:
                self.snapshots[it] = C.copy()

            # 4) stopping rule
            if change < self.tol:
                break

        self.centroids = C
        self.labels = labels
        self.n_iter = it
        return self

    def predict(self, X):
        # TODO
        return self._assign(np.asarray(X, dtype=np.float64), self.centroids)

    def quantize(self, img):
        # TODO
        return apply_palette(img, self.centroids)


# ----------------------------------------------------------------------
# DBSCAN
# ----------------------------------------------------------------------

CORE, BORDER, NOISE = 0, 1, 2


class DBSCAN:
    def __init__(self, eps, min_pts, chunk=1024):
        self.eps = eps
        self.min_pts = min_pts
        self.chunk = chunk

    def _neighbours(self, X):
        # TODO
        n = len(X)
        eps2 = self.eps ** 2
        nbrs = []
        for i in range(0, n, self.chunk):
            d = sq_dists(X[i:i + self.chunk], X)      # (chunk, n)
            within = d <= eps2
            for row in within:
                nbrs.append(np.flatnonzero(row))
        return nbrs

    def fit(self, X):
        # TODO
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        nbrs = self._neighbours(X)
        counts = np.array([len(a) for a in nbrs])
        is_core = counts >= self.min_pts

        labels = np.full(n, -1, dtype=np.int64)
        cid = 0
        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            labels[i] = cid
            queue = deque([i])
            while queue:
                p = queue.popleft()
                if not is_core[p]:          # border points do not expand
                    continue
                for q in nbrs[p]:
                    if labels[q] == -1:
                        labels[q] = cid
                        queue.append(q)
            cid += 1

        types = np.full(n, NOISE, dtype=np.int64)
        types[labels != -1] = BORDER
        types[is_core] = CORE

        self.labels_ = labels
        self.types_ = types
        self.n_clusters_ = cid
        return labels, types


def k_distance(X, k, chunk=1024):
    """Calculate sorted k-nearest-neighbour distances."""
    # TODO
    n = len(X)
    out = np.empty(n)
    for i in range(0, n, chunk):
        d = sq_dists(X[i:i + chunk], X)
        out[i:i + chunk] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return np.sort(np.sqrt(out))

def elbow_index(y):
    """Find the elbow point."""
    # TODO
    n = len(y)
    xs = np.arange(n) / max(n - 1, 1)
    ys = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    return int(np.argmax(np.abs(xs - ys)))


def choose_eps(X, min_pts, ax=None, title=""):
    # TODO
    y = k_distance(X, min_pts)
    i = elbow_index(y)
    eps = float(y[i])
    if ax is not None:
        ax.plot(y)
        ax.axhline(eps, color="r", ls="--")
        ax.plot(i, eps, "ro")
        ax.set_xlabel("points sorted by distance")
        ax.set_ylabel(f"distance to {min_pts}-th neighbour")
        ax.set_title(f"{title}  eps={eps:.4f}")
    return eps


def dbscan_image(img_small, labels):
    """Generate the clustered image."""
    # TODO
    h, w, _ = img_small.shape
    flat = img_small.reshape(-1, 3)
    out = np.zeros_like(flat)
    for c in np.unique(labels[labels >= 0]):
        m = labels == c
        out[m] = flat[m].mean(axis=0)
    return out.reshape(h, w, 3)


# ----------------------------------------------------------------------
# Toy data
# ----------------------------------------------------------------------

def make_rings(n=800, seed=0):
    """Generate concentric rings."""
    # TODO
    rng = np.random.default_rng(seed)
    n_in = n // 2
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.concatenate([np.full(n_in, 1.0), np.full(n - n_in, 3.0)])
    radius = radius + rng.normal(0, 0.08, n)
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = np.concatenate([np.zeros(n_in), np.ones(n - n_in)]).astype(int)
    return X, y


def make_blobs_outliers(n=600, seed=0):
    """Generate Gaussian blobs with outliers."""
    # TODO
    rng = np.random.default_rng(seed)
    centers = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])
    n_out = int(0.05 * n)
    n_blob = n - n_out
    per = n_blob // 3
    pts, ys = [], []
    for c, center in enumerate(centers):
        m = per if c < 2 else n_blob - 2 * per
        pts.append(center + rng.normal(0, 0.7, (m, 2)))
        ys.append(np.full(m, c))
    lo, hi = np.array([-8.0, -4.0]), np.array([8.0, 9.0])
    pts.append(rng.uniform(lo, hi, (n_out, 2)))
    ys.append(np.full(n_out, -1))
    return np.vstack(pts), np.concatenate(ys)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def save(fig, out, name):
    """Save a figure."""
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def save_table(rows, header, out, name, title=""):
    """Print a table, save it as a PNG (for report.pdf) and a CSV."""
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    cells = [[fmt(v) for v in r] for r in rows]
    fig, ax = plt.subplots(figsize=(1.7 * len(header) + 1,
                                    0.4 * len(rows) + 1.3))
    ax.axis("off")
    ax.set_title(title)
    tab = ax.table(cellText=cells, colLabels=header, loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.3)
    save(fig, out, name)
    with open(os.path.join(out, name.replace(".png", ".csv")), "w") as f:
        f.write(",".join(header) + "\n")
        for r in cells:
            f.write(",".join(r) + "\n")
    print("\n" + title)
    print(" | ".join(header))
    for r in cells:
        print(" | ".join(r))


def cluster_stats(labels):
    """(number of clusters, % noise) from a DBSCAN label array."""
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    return n_clusters, 100.0 * float(np.mean(labels == -1))


def scatter_labels(ax, X, labels, title):
    noise = labels == -1
    ax.scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise],
               cmap="tab10", s=6)
    ax.scatter(X[noise, 0], X[noise, 1], c="k", marker="x", s=12)
    ax.set_title(title)
    ax.set_aspect("equal")


def part_b_init(img, out, k=8, runs=5):
    # TODO
    X = image_features(img)
    rows = []
    for init in ("random", "kmeans++"):
        sses, iters = [], []
        for s in range(runs):
            km = KMeans(k, init=init, seed=s).fit(X)
            sses.append(km.sse_history[-1])
            iters.append(km.n_iter)
            rows.append((init, s, sses[-1], iters[-1]))
        rows.append((init, "mean", float(np.mean(sses)),
                     float(np.mean(iters))))
    save_table(rows, ["init", "seed", "final SSE", "iterations"], out,
               "init_table.png", f"Random vs K-Means++ (K={k})")
    return rows


def part_c_convergence(img, out, name, k=8):
    # TODO
    X = image_features(img)
    km = KMeans(k, init="kmeans++", seed=0, snapshot_iters=(1, 2, 5)).fit(X)

    shots = []
    for it in (1, 2, 5):
        if it in km.snapshots:
            shots.append((f"iteration {it}", km.snapshots[it]))
    shots.append((f"final (iter {km.n_iter})", km.centroids))

    fig, axes = plt.subplots(1, len(shots), figsize=(4 * len(shots), 4))
    for ax, (title, C) in zip(np.atleast_1d(axes), shots):
        ax.imshow(apply_palette(img, C))
        ax.set_title(title)
        ax.axis("off")
    save(fig, out, f"{name}_convergence_images.png")

    sse = np.array(km.sse_history)
    ok = bool(np.all(np.diff(sse) <= 1e-9 * max(sse[0], 1.0)))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.arange(1, len(sse) + 1), sse, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("SSE")
    ax.set_title(f"{name}: SSE per iteration (never increases: {ok})")
    save(fig, out, f"{name}_sse_curve.png")
    print(f"[{name}] SSE non-increasing: {ok}")
    return km.sse_history


def part_d_choose_k(img, out, name, ks=(2, 4, 8, 16, 32)):
    # TODO
    X = image_features(img)
    n = len(X)
    sses, crs, quant = [], [], []
    for k in ks:
        km = KMeans(k, init="kmeans++", seed=0).fit(X)
        sses.append(km.sse_history[-1])
        crs.append(compression_ratio(k, n))
        quant.append(km.quantize(img))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks, sses, "o-")
    ax.set_xticks(list(ks))
    ax.set_xlabel("K")
    ax.set_ylabel("final SSE")
    ax.set_title(f"{name}: elbow plot")
    save(fig, out, f"{name}_elbow.png")

    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3.4))
    axes[0].imshow(img)
    axes[0].set_title("original")
    for ax, k, q, cr in zip(axes[1:], ks, quant, crs):
        ax.imshow(q)
        ax.set_title(f"K={k}  CR={cr:.1f}x")
    for ax in axes:
        ax.axis("off")
    save(fig, out, f"{name}_k_images.png")

    rows = [(name, k, s, c) for k, s, c in zip(ks, sses, crs)]
    save_table(rows, ["image", "K", "final SSE", "CR"], out,
               f"{name}_k_table.png", f"{name}: SSE and compression ratio")
    return rows


def part_e_dbscan(img, out, name, min_pts=6):
    # TODO
    small = downsample(img, 80)
    X = image_features(small)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    eps = choose_eps(X, min_pts, ax=axes[0], title=f"{name} k-distance")
    labels, _ = DBSCAN(eps, min_pts).fit(X)
    axes[1].imshow(small)
    axes[1].set_title("downsampled original")
    axes[2].imshow(dbscan_image(small, labels))
    nc, npct = cluster_stats(labels)
    axes[2].set_title(f"DBSCAN: {nc} clusters, {npct:.1f}% noise")
    axes[1].axis("off")
    axes[2].axis("off")
    save(fig, out, f"{name}_dbscan.png")
    return eps, labels


def part_f_shapes(out):
    # TODO
    # eps=None -> choose from the k-distance elbow; the rings elbow is
    # too small (it splits the rings), so we hand-pick eps=0.3 for them.
    datasets = [("rings", make_rings(), 2, 0.3),
                ("blobs+outliers", make_blobs_outliers(), 3, None)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    rows = []
    for r, (dname, (X, _), k, eps) in enumerate(datasets):
        t = time.perf_counter()
        km = KMeans(k, init="kmeans++").fit(X)
        t_km = time.perf_counter() - t

        if eps is None:
            eps = choose_eps(X, 5)
        t = time.perf_counter()
        labels, _ = DBSCAN(eps, 5).fit(X)
        t_db = time.perf_counter() - t

        scatter_labels(axes[r, 0], X, km.labels, f"{dname}: K-Means (K={k})")
        scatter_labels(axes[r, 1], X, labels,
                       f"{dname}: DBSCAN (eps={eps:.2f})")
        rows.append((dname, "K-Means", k, 0.0, t_km))
        rows.append((dname, "DBSCAN", *cluster_stats(labels), t_db))
    save(fig, out, "shapes_comparison.png")
    return rows


def part_f_images(img, out, name, k, lam=0.5):
    # TODO
    small = downsample(img, 80)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    rows = []
    eps_rgb = None
    for r, (tag, l) in enumerate((("RGB", None), ("RGB+xy", lam))):
        X = image_features(small, l)

        t = time.perf_counter()
        km = KMeans(k, init="kmeans++").fit(X)
        t_km = time.perf_counter() - t

        eps = choose_eps(X, MIN_PTS)
        t = time.perf_counter()
        labels, _ = DBSCAN(eps, MIN_PTS).fit(X)
        t_db = time.perf_counter() - t

        axes[r, 0].imshow(small)
        axes[r, 0].set_title(f"{tag}: original")
        axes[r, 1].imshow(dbscan_image(small, km.labels))
        axes[r, 1].set_title(f"{tag}: K-Means K={k}")
        nc, npct = cluster_stats(labels)
        axes[r, 2].imshow(dbscan_image(small, labels))
        axes[r, 2].set_title(f"{tag}: DBSCAN eps={eps:.3f}, {nc} cl.")
        for ax in axes[r]:
            ax.axis("off")

        rows.append((f"{name} {tag}", "K-Means", k, 0.0, t_km))
        rows.append((f"{name} {tag}", "DBSCAN", nc, npct, t_db))
        if l is None:
            eps_rgb = eps
    save(fig, out, f"{name}_kmeans_vs_dbscan.png")
    return eps_rgb, rows


def part_f_sensitivity(img, name, k, eps, min_pts=6):
    # TODO
    small = downsample(img, 80)
    X = image_features(small)
    rows = []
    for f in (0.7, 1.0, 1.3):
        e = eps * f
        t = time.perf_counter()
        labels, _ = DBSCAN(e, min_pts).fit(X)
        dt = time.perf_counter() - t
        nc, npct = cluster_stats(labels)
        pct = round((f - 1) * 100)
        rows.append((name, f"DBSCAN eps={e:.4f} ({pct:+d}%)", nc, npct, dt))
    for kk in (max(k - 2, 2), k, k + 2):
        t = time.perf_counter()
        KMeans(kk, init="kmeans++").fit(X)
        dt = time.perf_counter() - t
        rows.append((name, f"K-Means K={kk}", kk, 0.0, dt))
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def find_images(folder):
    """Sorted list of real .jpg/.jpeg paths in folder.
    Skips hidden macOS junk files such as '._sample2.jpg'."""
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg"))
                   and not f.startswith("."))
    return [os.path.join(folder, f) for f in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images")
    parser.add_argument("--out", default="out/q1")
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # TODO
    paths = find_images(args.images)
    if not paths:
        raise SystemExit(f"No .jpg images found in '{args.images}'. "
                         "Run from the AI-Assignment-3 folder or pass --images.")
    print(f"Found {len(paths)} images: {[os.path.basename(p) for p in paths]}")

    # Load every image once (shrunk for the K-Means experiments)
    images = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        images[name] = downsample(load_image(path), KM_MAX_SIDE)
    names = list(images)

    comparison = []

    # Q1.2 K-Means++ vs random: one image is enough
    part_b_init(images[names[0]], args.out, k=8, runs=5)

    # Q1.3 - Q1.5: run on all five images
    for name in names:
        print(f"=== {name} ===")
        part_c_convergence(images[name], args.out, name, k=8)
        part_d_choose_k(images[name], args.out, name)
        part_e_dbscan(images[name], args.out, name)

    # Q1.6 (a) shapes
    comparison += part_f_shapes(args.out)

    # Q1.6 (b), (c): two sample images
    for name in names[:2]:
        eps_rgb, rows = part_f_images(images[name], args.out, name, args.k)
        comparison += rows
        comparison += part_f_sensitivity(images[name], name, args.k, eps_rgb)

    save_table(comparison,
               ["experiment", "method", "clusters", "noise %", "time (s)"],
               args.out, "comparison_table.png", "K-Means vs DBSCAN summary")
    print(f"Done. Results saved in '{args.out}'.")


if __name__ == "__main__":
    main()