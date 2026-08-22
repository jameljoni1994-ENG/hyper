"""Real-data logistic regression on LIBSVM-format binary datasets.

Same interface as problems.py classes: n, m, lam, L, mu, f, grad, hv, hess.
Objective:  f(w) = (1/m) sum_i log(1 + exp(-y_i x_i^T w)) + lam/2 ||w||^2,
with lam = 1/m (the convention used for a9a / mushrooms / w8a in the
LIBSVM benchmark literature).
"""
import os

import numpy as np


def load_libsvm(path, n_features):
    """Parse a LIBSVM sparse text file into dense X (m,n) and y in {-1,+1}."""
    rows, ys = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            ys.append(float(parts[0]))
            row = np.zeros(n_features)
            for tok in parts[1:]:
                idx, val = tok.split(":")
                row[int(idx) - 1] = float(val)
            rows.append(row)
    X = np.asarray(rows)
    y = np.where(np.asarray(ys) > 0, 1.0, -1.0)
    return X, y


class LibsvmLogistic:
    """L2-regularized logistic regression on a real LIBSVM dataset."""

    def __init__(self, data_dir, dataset="a9a", n_features=123, lam=None):
        path = os.path.join(data_dir, dataset)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing; run code/fetch_data.py first")
        X, y = load_libsvm(path, n_features)
        self.dataset = dataset
        self.n = n_features
        self.m = X.shape[0]
        self.lam = float(lam if lam is not None else 1.0 / self.m)
        self.Xy = X * y[:, None]
        self.X = X
        self.y = y
        rng = np.random.default_rng(2026)
        v = rng.standard_normal(self.n)
        nv = 0.0
        for _ in range(80):
            w = self.X.T @ (self.X @ v)
            nv = np.linalg.norm(w)
            if nv == 0:
                break
            v = w / nv
        self.L = 0.25 * nv / self.m + self.lam
        self.mu = self.lam

    def f(self, w):
        s = -(self.Xy @ w)
        return (np.logaddexp(0.0, s).sum()) / self.m \
            + 0.5 * self.lam * w @ w

    def grad(self, w):
        s = -(self.Xy @ w)
        p = 1.0 / (1.0 + np.exp(-s))
        return (-(self.Xy.T @ p)) / self.m + self.lam * w

    def hv(self, w, v):
        s = -(self.Xy @ w)
        p = 1.0 / (1.0 + np.exp(-s))
        return ((self.X.T @ (p * (1 - p) * (self.X @ v))) / self.m) \
            + self.lam * v

    def hess(self, w):
        s = -(self.Xy @ w)
        p = 1.0 / (1.0 + np.exp(-s))
        Wd = p * (1 - p)
        XW = self.X * Wd[:, None]
        return (self.X.T @ XW) / self.m + self.lam * np.eye(self.n)


DATASETS = {
    # name: (n_features, url)
    "a9a": (123, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"),
    "mushrooms": (112, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"),
    "w8a": (300, "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"),
}


def fetch_all(data_dir):
    import urllib.request
    os.makedirs(data_dir, exist_ok=True)
    for name, (_, url) in DATASETS.items():
        dst = os.path.join(data_dir, name)
        if os.path.exists(dst):
            print(f"  {name}: exists")
            continue
        print(f"  {name}: downloading ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(dst, "wb") as fh:
            fh.write(data)
        print(f"  {name}: saved {len(data)/1024:.0f} KB")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")
    fetch_all(data_dir)
    for name, (nf, _) in DATASETS.items():
        pb = LibsvmLogistic(data_dir, name, nf)
        g0 = np.linalg.norm(pb.grad(np.zeros(pb.n)))
        print(f"check {name}: m={pb.m} n={pb.n} L={pb.L:.4f} "
              f"lam={pb.lam:.2e} |grad(0)|={g0:.4f}")
