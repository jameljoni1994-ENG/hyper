"""Problem families for the hybrid switching study.

Each problem exposes:
    n            dimension
    f(x)         objective value
    grad(x)      gradient
    hv(x, v)     Hessian-vector product
    hess(x)      dense Hessian (used sparingly; expensive problems may return None)
    L, mu        smoothness and strong-convexity constants when known analytically
                 (mu = 0.0 means merely convex / unknown)
"""
import numpy as np


def _rot(n, seed):
    """Random orthogonal matrix via QR."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q


class Quadratic:
    """f(x) = 0.5 x'Qx - b'x with eigenvalues log-spaced in [1, kappa]."""

    name = "quadratic"

    def __init__(self, n=1000, kappa=1000.0, seed=0):
        self.n = n
        self.kappa = float(kappa)
        self.L = self.kappa
        self.mu = 1.0
        d = np.logspace(0.0, np.log10(self.kappa), n)
        V = _rot(n, 123 + seed)
        self.Q = (V * d) @ V.T
        rng = np.random.default_rng(777 + seed)
        self.xstar = rng.standard_normal(n)
        self.b = self.Q @ self.xstar
        self.fstar = -0.5 * self.b @ self.xstar

    def f(self, x):
        return 0.5 * x @ (self.Q @ x) - self.b @ x

    def grad(self, x):
        return self.Q @ x - self.b

    def hv(self, x, v):
        return self.Q @ v

    def hess(self, x):
        return self.Q


class Rosenbrock:
    """Classic curved valley; nonconvex globally, well-conditioned at optimum."""

    name = "rosenbrock"

    def __init__(self, n=50, a=1.0, b=100.0):
        self.n = n
        self.a = a
        self.b = b
        self.xstar = np.ones(n)
        self.L = None
        self.mu = None
        self.fstar = 0.0

    def f(self, x):
        t = x[1:] - x[:-1] ** 2
        return np.sum(self.b * t * t + (self.a - x[:-1]) ** 2)

    def grad(self, x):
        g = np.zeros_like(x)
        t = x[1:] - x[:-1] ** 2
        g[:-1] += -2.0 * (self.a - x[:-1]) - 4.0 * self.b * x[:-1] * t
        g[1:] += 2.0 * self.b * t
        return g

    def hv(self, x, v):
        return self.hess(x) @ v

    def hess(self, x):
        n, b = self.n, self.b
        H = np.zeros((n, n))
        t = x[1:] - x[:-1] ** 2
        H[np.arange(n), np.arange(n)] += 2.0
        H[np.arange(n - 1), np.arange(n - 1)] += -4.0 * b * t + 8.0 * b * x[:-1] ** 2
        H[np.arange(1, n), np.arange(1, n)] += 2.0 * b
        H[n - 1, n - 1] -= 2.0          # last variable has no (a-x)^2 term
        off = np.arange(n - 1)
        H[off, off + 1] += -4.0 * b * x[:-1]
        H[off + 1, off] += -4.0 * b * x[:-1]
        return H


class Logistic:
    """L2-regularized logistic regression on synthetic separable-ish data."""

    name = "logistic"

    def __init__(self, n=1000, m=5000, lam=0.01, seed=0):
        self.n, self.m, self.lam = n, m, lam
        rng = np.random.default_rng(2026 + seed)
        w_true = rng.standard_normal(n) / np.sqrt(n)
        X = rng.standard_normal((m, n))
        logits = X @ w_true
        y = np.where(logits + 0.3 * rng.standard_normal(m) > 0, 1.0, -1.0)
        self.Xy = X * y[:, None]
        self.X = X
        self.y = y
        # L = lambda_max(X'X)/(4m) + lam ; computed once via power iteration
        v = rng.standard_normal(n)
        for _ in range(50):
            w = self.X.T @ (self.X @ v)
            nv = np.linalg.norm(w)
            if nv == 0:
                break
            v = w / nv
        self.L = 0.25 * nv / m + lam
        self.mu = lam

    def f(self, w):
        s = -(self.Xy @ w)
        return (np.logaddexp(0.0, s).sum()) / self.m + 0.5 * self.lam * w @ w

    def grad(self, w):
        s = -(self.Xy @ w)
        p = 1.0 / (1.0 + np.exp(-s))
        return (-(self.Xy.T @ p)) / self.m + self.lam * w

    def hv(self, w, v):
        s = -(self.Xy @ w)
        p = 1.0 / (1.0 + np.exp(-s))
        return ((self.X.T @ (p * (1 - p) * (self.X @ v))) / self.m) + self.lam * v

    def hess(self, w):
        s = -(self.Xy @ w)
        p = 1.0 / (1.0 + np.exp(-s))
        Wd = p * (1 - p)
        XW = self.X * Wd[:, None]
        return (self.X.T @ XW) / self.m + self.lam * np.eye(self.n)


class LogSumExp:
    """f(x) = logsumexp(Ax - b); convex smooth, not strongly convex."""

    name = "logsumexp"

    def __init__(self, n=50, m=500, seed=0):
        rng = np.random.default_rng(31 + seed)
        self.n, self.m = n, m
        self.A = 0.5 * rng.standard_normal((m, n))
        self.b = rng.standard_normal(m)
        self.L = None
        self.mu = 0.0
        self.xstar = None
        self.fstar = None

    def _z(self, x):
        z = self.A @ x - self.b
        zm = z.max()
        e = np.exp(z - zm)
        se = e.sum()
        return z, e, se, zm

    def f(self, x):
        z, e, se, zm = self._z(x)
        return zm + np.log(se)

    def grad(self, x):
        z, e, se, zm = self._z(x)
        p = e / se
        return self.A.T @ p

    def hv(self, x, v):
        z, e, se, zm = self._z(x)
        p = e / se
        Av = self.A @ v
        return self.A.T @ (p * Av) - (p @ Av) * (self.A.T @ p)

    def hess(self, x):
        z, e, se, zm = self._z(x)
        p = e / se
        D = p - np.outer(p, p)
        return self.A.T @ (D @ self.A)


def make_problem(kind, **kw):
    cls = {"quadratic": Quadratic, "rosenbrock": Rosenbrock,
           "logistic": Logistic, "logsumexp": LogSumExp}[kind]
    return cls(**kw)
