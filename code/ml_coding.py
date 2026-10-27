"""ML coding interview reference implementations (NumPy / PyTorch).

Conventions
-----------
- Every function/method has type hints for all inputs and the return value.
- Docstrings state array shapes: N/M = number of samples/points, D = feature
  (coordinate) dim, K = clusters/neighbours/components, B = batch, C = classes/channels.
- Classification labels are non-negative integer arrays (np.int64), shape (N,).
- Every function here is exercised by test_ml_coding.py.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== 0. Chamfer distance ==============================
def pairwise_sq_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """All pairwise squared Euclidean distances via ||p-q||^2 = ||p||^2 + ||q||^2 - 2 p.q

    Args:
        P: (N, D) float.
        Q: (M, D) float.
    Returns:
        (N, M) float, entry [i, j] = ||P[i] - Q[j]||^2.
    """
    p_sq = np.sum(P ** 2, axis=1)                               # (N,)   ||p||^2
    q_sq = np.sum(Q ** 2, axis=1)                               # (M,)   ||q||^2
    cross = P @ Q.T                                             # (N, M) p . q
    d = p_sq[:, None] + q_sq[None, :] - 2 * cross               # (N, M)
    d = np.maximum(d, 0.0)                                      # fp error can give -1e-16
    return d


def chamfer_distance(P: np.ndarray, Q: np.ndarray, squared: bool = True) -> float:
    """Chamfer distance = mean(P->Q) + mean(Q->P).

    Args:
        P: (N, D) float, first point set.
        Q: (M, D) float, second point set.
        squared: use squared Euclidean distance if True.
    Returns:
        Scalar distance (float).
    """
    d = pairwise_sq_dist(P, Q)                                  # (N, M)
    if not squared:
        d = np.sqrt(d)
    p_to_q = d.min(axis=1)                                      # (N,) nearest Q point for each p
    q_to_p = d.min(axis=0)                                      # (M,) nearest P point for each q
    return float(p_to_q.mean() + q_to_p.mean())


def chamfer_distance_broadcast(P: np.ndarray, Q: np.ndarray) -> float:
    """Simple version (squared distance): easy to write, O(N*M*D) memory.

    Args:
        P: (N, D) float.
        Q: (M, D) float.
    Returns:
        Scalar distance (float).
    """
    d = np.sum((P[:, None, :] - Q[None, :, :]) ** 2, axis=-1)  # (N, M)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def batched_chamfer_distance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Batched squared Chamfer distance.

    Args:
        P: (B, N, D) float.
        Q: (B, M, D) float.
    Returns:
        (B,) float array, one distance per batch element.
    """
    p_sq = np.sum(P ** 2, axis=-1)                              # (B, N)
    q_sq = np.sum(Q ** 2, axis=-1)                              # (B, M)
    cross = P @ Q.swapaxes(1, 2)                                # (B, N, M)
    d = p_sq[:, :, None] + q_sq[:, None, :] - 2 * cross         # (B, N, M)
    d = np.maximum(d, 0.0)
    p_to_q = d.min(axis=2).mean(axis=1)                         # (B,)
    q_to_p = d.min(axis=1).mean(axis=1)                         # (B,)
    return p_to_q + q_to_p


# ============================== 1. KNN ==============================
class KNN:
    """Interview-length KNN classifier. Labels must be non-negative ints."""

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, D) float, y: (N,) int in [0, C)."""
        self.X, self.y = X, y

    def predict(self, Xt: np.ndarray) -> np.ndarray:
        """Xt: (M, D) float -> (M,) int predicted labels."""
        d = pairwise_sq_dist(Xt, self.X)                        # (M, N)
        idx = np.argsort(d, axis=1)[:, :self.k]                 # (M, k)
        labels = self.y[idx]                                    # (M, k)
        return np.array([np.bincount(r).argmax() for r in labels])


class KNNFast(KNN):
    """Matmul distance + argpartition (O(N) per query instead of O(N log N))."""

    def predict(self, Xt: np.ndarray) -> np.ndarray:
        """Xt: (M, D) float -> (M,) int predicted labels."""
        d = pairwise_sq_dist(Xt, self.X)                        # (M, N)
        k = min(self.k, d.shape[1])                             # k > N would be out of range
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]          # (M, k) kth = k-1, NOT k
        labels = self.y[idx]                                    # (M, k)
        return np.array([np.bincount(r).argmax() for r in labels])


# ============================== 2. K-Means ==============================
class KMeans:
    """K-Means. Steps of fit(): init -> [assign -> update -> check] x iters.

    init="random" picks K random samples; init="kmeans++" spreads the initial centers out.
    """

    def __init__(self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4,
                 init: str = "kmeans++", seed: Optional[int] = None) -> None:
        self.k, self.max_iters, self.tol, self.init = n_clusters, max_iters, tol, init
        self.rng = np.random.default_rng(seed)
        self.centroids: np.ndarray                              # (K, D), set by fit

    # ---- step 1a: random initialisation ----
    def _init_random(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (K, D), K distinct random samples."""
        idx = self.rng.choice(len(X), self.k, replace=False)    # (K,)
        return X[idx].copy()

    # ---- step 1b: K-Means++ initialisation ----
    def _init_plusplus(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (K, D). Next center is drawn with prob ∝ squared distance to nearest chosen center."""
        N, D = X.shape
        centroids = np.empty((self.k, D), dtype=X.dtype)
        centroids[0] = X[self.rng.integers(N)]                  # first center: uniform random
        for i in range(1, self.k):
            d = pairwise_sq_dist(X, centroids[:i])              # (N, i) to chosen centers
            nearest = d.min(axis=1)                             # (N,)
            total = nearest.sum()
            if total > 0:
                probs = nearest / total                         # (N,) sums to 1
            else:                                               # all points identical to a center
                probs = np.full(N, 1.0 / N)
            chosen = self.rng.choice(N, p=probs)
            centroids[i] = X[chosen]
        return centroids

    # ---- step 2: assignment ----
    def _assign(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (N,) index of nearest centroid."""
        d = pairwise_sq_dist(X, self.centroids)                 # (N, K)
        return d.argmin(axis=1)                                 # (N,)

    # ---- step 3: update ----
    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """X: (N, D), labels: (N,) -> new centroids (K, D). Empty cluster -> re-seed with a random point."""
        new = np.empty_like(self.centroids)                     # (K, D)
        for j in range(self.k):
            members = X[labels == j]                            # (n_j, D)
            if len(members) == 0:                               # empty cluster: mean of [] would be NaN
                new[j] = X[self.rng.integers(len(X))]
            else:
                new[j] = members.mean(axis=0)
        return new

    def fit(self, X: np.ndarray) -> "KMeans":
        """X: (N, D) float. Returns self."""
        if self.init == "random":
            self.centroids = self._init_random(X)
        else:
            self.centroids = self._init_plusplus(X)
        for _ in range(self.max_iters):
            labels = self._assign(X)                            # (N,)
            new = self._update(X, labels)                       # (K, D)
            shift = np.linalg.norm(new - self.centroids)        # how far the centers moved
            self.centroids = new
            if shift < self.tol:                                # step 4: converged
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) float -> (N,) int cluster ids."""
        return self._assign(X)


# ============================== 3. Logistic regression ==============================
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe sigmoid. z: any shape float -> same shape, values in (0, 1)."""
    z = np.clip(z, -500, 500)                                   # exp(500) ~ 1e217 still fits in float64
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 1000, reg: float = 0.0) -> None:
        self.lr, self.epochs, self.reg = lr, epochs, reg
        self.w: np.ndarray                                      # (D,), set by fit
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """X: (N, D) float, y: (N,) in {0, 1}. Returns self."""
        N, D = X.shape
        self.w, self.b = np.zeros(D), 0.0
        for _ in range(self.epochs):
            err = sigmoid(X @ self.w + self.b) - y              # (N,)  dL/dz = p - y
            self.w -= self.lr * (X.T @ err / N + self.reg * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) float -> (N,) P(y=1)."""
        return sigmoid(X @ self.w + self.b)


# ============================== 4. Decision tree (CART, Gini) ==============================
class _Node:
    def __init__(self, feature: Optional[int] = None, thr: Optional[float] = None,
                 left: Optional["_Node"] = None, right: Optional["_Node"] = None,
                 value: Optional[int] = None) -> None:
        self.feature, self.thr, self.left, self.right, self.value = feature, thr, left, right, value


class DecisionTree:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 max_features: Optional[int] = None) -> None:
        self.max_depth, self.min_split, self.max_features = max_depth, min_samples_split, max_features
        self.root: _Node

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """y: (n,) int labels -> Gini impurity."""
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return float(1.0 - np.sum(p ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Returns (feature, threshold, gain); feature is None if no split has gain > 0."""
        N, D = X.shape
        feats = np.arange(D)
        if self.max_features:                                   # random-forest feature subsampling
            feats = np.random.choice(D, self.max_features, replace=False)
        best: Tuple[Optional[int], Optional[float], float] = (None, None, -1.0)  # gain may be 0 (XOR)
        parent = self._gini(y)
        for f in feats:
            vals = np.unique(X[:, f])
            for t in (vals[:-1] + vals[1:]) / 2:                # midpoints; last value never splits
                l = X[:, f] <= t
                nl = l.sum()
                gain = parent - (nl * self._gini(y[l]) + (N - nl) * self._gini(y[~l])) / N
                if gain > best[2]:
                    best = (int(f), float(t), float(gain))
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        if depth >= self.max_depth or len(y) < self.min_split or len(np.unique(y)) == 1:
            return _Node(value=int(np.bincount(y).argmax()))
        f, t, gain = self._best_split(X, y)
        if f is None:
            return _Node(value=int(np.bincount(y).argmax()))
        l = X[:, f] <= t
        return _Node(f, t, self._build(X[l], y[l], depth + 1), self._build(X[~l], y[~l], depth + 1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """X: (N, D) float, y: (N,) int in [0, C). Returns self."""
        self.root = self._build(X, y, 0)
        return self

    def _one(self, x: np.ndarray, n: _Node) -> int:
        while n.value is None:
            n = n.left if x[n.feature] <= n.thr else n.right
        return n.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) float -> (N,) int labels."""
        return np.array([self._one(x, self.root) for x in X])


class RandomForest:
    def __init__(self, n_trees: int = 10, max_depth: int = 5, seed: int = 0) -> None:
        self.n_trees, self.max_depth = n_trees, max_depth
        self.rng = np.random.default_rng(seed)
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """X: (N, D) float, y: (N,) int. Returns self."""
        N, D = X.shape
        self.trees = []
        for _ in range(self.n_trees):
            idx = self.rng.choice(N, N, replace=True)           # bootstrap
            t = DecisionTree(self.max_depth, max_features=max(1, int(np.sqrt(D))))
            self.trees.append(t.fit(X[idx], y[idx]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) float -> (N,) int labels (majority vote)."""
        votes = np.stack([t.predict(X) for t in self.trees])    # (T, N)
        return np.array([np.bincount(c).argmax() for c in votes.T])


# ============================== 5. GMM (EM) ==============================
class GMM:
    def __init__(self, n_components: int = 3, max_iters: int = 100, tol: float = 1e-4,
                 reg: float = 1e-6, seed: int = 0) -> None:
        self.k, self.max_iters, self.tol, self.reg = n_components, max_iters, tol, reg
        self.rng = np.random.default_rng(seed)
        self.mu: np.ndarray                                     # (K, D)
        self.cov: np.ndarray                                    # (K, D, D)
        self.pi: np.ndarray                                     # (K,)
        self.ll: float                                          # final total log-likelihood

    def _log_pdf(self, X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """X: (N, D), mu: (D,), cov: (D, D) -> (N,) log N(x | mu, cov)."""
        D = X.shape[1]
        cov = cov + self.reg * np.eye(D)                        # keep invertible
        _, logdet = np.linalg.slogdet(cov)
        diff = X - mu                                           # (N, D)
        maha = np.einsum('nd,de,ne->n', diff, np.linalg.inv(cov), diff)  # (N,)
        return -0.5 * (D * np.log(2 * np.pi) + logdet + maha)

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Responsibilities in the log domain. X: (N, D) -> (gamma (N, K), total log-likelihood)."""
        log_prob = np.empty((len(X), self.k))                   # (N, K) log(pi_k * N(x|k))
        for j in range(self.k):
            log_prior = np.log(self.pi[j] + 1e-12)
            log_prob[:, j] = log_prior + self._log_pdf(X, self.mu[j], self.cov[j])
        m = log_prob.max(axis=1, keepdims=True)                 # (N, 1) for log-sum-exp
        log_norm = m + np.log(np.exp(log_prob - m).sum(axis=1, keepdims=True))  # (N, 1) log p(x)
        gamma = np.exp(log_prob - log_norm)                     # (N, K) rows sum to 1
        return gamma, float(log_norm.sum())

    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """Re-estimate pi, mu, cov from responsibilities. X: (N, D), gamma: (N, K)."""
        N = len(X)
        Nk = gamma.sum(axis=0) + 1e-12                          # (K,) effective count per component
        self.pi = Nk / N                                        # (K,)
        self.mu = gamma.T @ X / Nk[:, None]                     # (K, D)
        for j in range(self.k):
            diff = X - self.mu[j]                               # (N, D)
            weighted = gamma[:, j, None] * diff                 # (N, D)
            self.cov[j] = weighted.T @ diff / Nk[j]             # (D, D)

    def fit(self, X: np.ndarray) -> "GMM":
        """X: (N, D) float. Returns self."""
        N, D = X.shape
        self.mu = X[self.rng.choice(N, self.k, replace=False)]  # (K, D) random data points
        self.cov = np.stack([np.eye(D)] * self.k)               # (K, D, D)
        self.pi = np.full(self.k, 1.0 / self.k)                 # (K,)
        prev = -np.inf
        for _ in range(self.max_iters):
            gamma, ll = self._e_step(X)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
            self._m_step(X, gamma)
        self.ll = ll
        return self


# ============================== 6. PCA ==============================
class PCA:
    def __init__(self, n_components: int = 2) -> None:
        self.n = n_components
        self.mean: np.ndarray                                   # (D,)
        self.components: np.ndarray                             # (D, n)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) float -> (N, n) projected data."""
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)       # Xc = U S Vt
        self.components = Vt[:self.n].T                         # (D, n)
        return Xc @ self.components


# ============================== 7. Attention ==============================
# Mask convention everywhere: bool tensor, True = may attend, False = blocked.
KVCache = Tuple[torch.Tensor, torch.Tensor]                     # (k, v), each (B, H, S, dk)


def causal_mask(T: int, S: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower-triangular mask aligned to the END of the keys (works with a KV cache).

    Returns:
        (T, S) bool; query i (the i-th of the last T positions) sees keys 0..S-T+i.
    """
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 dropout_p: float = 0.0, training: bool = False
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax(q k^T / sqrt(dk)) v

    Args:
        q: (..., T, dk) queries.
        k: (..., S, dk) keys.
        v: (..., S, dv) values.
        mask: bool, broadcastable to (..., T, S); True = keep. None = no mask.
    Returns:
        out: (..., T, dv); weights: (..., T, S).
    """
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5        # (..., T, S)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    w = F.softmax(scores, dim=-1)
    w = F.dropout(w, dropout_p, training)
    return w @ v, w                                             # (..., T, dv), (..., T, S)


def attention(x_q: torch.Tensor, x_kv: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
              Wv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Single-head attention. Self-attention if x_kv is x_q; cross-attention otherwise.

    Args:
        x_q: (B, T, Dq) the sequence that asks (decoder side).
        x_kv: (B, S, Dkv) the sequence that is looked up (encoder side / same as x_q).
        Wq: (Dq, dk), Wk: (Dkv, dk), Wv: (Dkv, dv) projection matrices.
        mask: bool broadcastable to (B, T, S), or None.
    Returns:
        (B, T, dv).
    """
    q, k, v = x_q @ Wq, x_kv @ Wk, x_kv @ Wv                    # (B,T,dk) (B,S,dk) (B,S,dv)
    return scaled_dot_product_attention(q, k, v, mask)[0]


def self_attention(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
                   causal: bool = False) -> torch.Tensor:
    """x: (B, T, D); W*: (D, d) -> (B, T, d). Queries, keys, values all come from x."""
    mask = causal_mask(x.size(1), x.size(1), x.device) if causal else None
    return attention(x, x, Wq, Wk, Wv, mask)


def cross_attention(x: torch.Tensor, context: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                    Wv: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Queries from x, keys/values from context (e.g. decoder attends to encoder output).

    Args:
        x: (B, T, Dq); context: (B, S, Dkv) -- T and S may differ.
        context_mask: (B, S) bool, True = real token, False = padding; or None.
    Returns:
        (B, T, dv). No causal mask: every query may see the whole context.
    """
    mask = context_mask[:, None, :] if context_mask is not None else None  # (B, 1, S)
    return attention(x, context, Wq, Wk, Wv, mask)


def multi_head_attention(x_q: torch.Tensor, x_kv: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                         Wv: torch.Tensor, Wo: torch.Tensor, num_heads: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Functional multi-head attention (self if x_kv is x_q, cross otherwise).

    Args:
        x_q: (B, T, C); x_kv: (B, S, C).
        Wq, Wk, Wv, Wo: (C, C) projections, C divisible by num_heads.
        mask: bool broadcastable to (B, 1, T, S) (shared across heads), or None.
    Returns:
        (B, T, C).
    """
    B, T, C = x_q.shape
    S, H = x_kv.size(1), num_heads
    q = (x_q @ Wq).view(B, T, H, C // H).transpose(1, 2)        # (B, H, T, dk)
    k = (x_kv @ Wk).view(B, S, H, C // H).transpose(1, 2)       # (B, H, S, dk)
    v = (x_kv @ Wv).view(B, S, H, C // H).transpose(1, 2)       # (B, H, S, dk)
    out, _ = scaled_dot_product_attention(q, k, v, mask)        # (B, H, T, dk)
    out = out.transpose(1, 2).contiguous().view(B, T, C)        # (B, T, C), heads concatenated
    return out @ Wo


class MultiHeadAttention(nn.Module):
    """Module version: self-attention (context=None) or cross-attention, with KV cache."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.dk = num_heads, d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B, H, L, dk)."""
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, is_causal: bool = False,
                kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        """
        Args:
            x: (B, T, C) queries' source (T=1 when decoding step by step).
            context: (B, S, C) keys/values' source for cross-attention; None = self-attention.
            mask: bool broadcastable to (B, 1, T, S); True = keep (e.g. padding mask).
            is_causal: add the causal mask (self-attention decoding).
            kv_cache: self-attn: previous (k, v) to append to; cross-attn: the (k, v) of
                the context computed on the first step, reused as-is.
        Returns:
            y: (B, T, C); cache: (k, v), each (B, H, S_total, dk).
        """
        B, T, C = x.shape
        q = self._split(self.q_proj(x))                         # (B, H, T, dk)
        if context is not None and kv_cache is not None:        # cross-attn: reuse cached context K/V
            k, v = kv_cache
        else:
            src = x if context is None else context
            k, v = self._split(self.k_proj(src)), self._split(self.v_proj(src))  # (B, H, S, dk)
            if kv_cache is not None:                            # self-attn: append new K/V
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
        if is_causal:
            cm = causal_mask(T, k.size(2), x.device)            # (T, S)
            mask = cm if mask is None else mask & cm
        out, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout_p, self.training)
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, C)
        return self.out_proj(out), (k, v)


# ============================== 8. Norm layers ==============================
class BatchNorm1d(nn.Module):
    def __init__(self, n: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(1, n))
        self.beta = nn.Parameter(torch.zeros(1, n))
        self.register_buffer('running_mean', torch.zeros(1, n))
        self.register_buffer('running_var', torch.ones(1, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C) -> (B, C)."""
        if self.training:
            mean = x.mean(0, keepdim=True)                      # (1, C)
            var_biased = x.var(0, keepdim=True, unbiased=False) # (1, C) used to normalise
            var_unbiased = x.var(0, keepdim=True, unbiased=True)  # (1, C) used for running stats
            with torch.no_grad():                               # buffers: no gradient
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var_unbiased)
            var = var_biased
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)         # (B, C)
        return self.gamma * x_hat + self.beta


class LayerNorm(nn.Module):
    def __init__(self, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma, self.beta = nn.Parameter(torch.ones(n)), nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n) -> (..., n); normalises over the last dim."""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps, self.gamma = eps, nn.Parameter(torch.ones(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n) -> (..., n)."""
        return self.gamma * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ============================== 9. Conv2d / MaxPool ==============================
def _windows(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    """x: (B, C, H, W) -> zero-copy view (B, C, H_out, W_out, kh, kw)."""
    return np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]


def conv2d(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """Cross-correlation, same as PyTorch.

    Args:
        x: (B, Cin, H, W) float.
        weight: (Cout, Cin, kh, kw) float.
        bias: (Cout,) float or None.
    Returns:
        (B, Cout, Ho, Wo) with Ho = (H + 2*padding - kh) // stride + 1.
    """
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    w = _windows(x, weight.shape[2], weight.shape[3], stride)   # (B, Cin, Ho, Wo, kh, kw)
    out = np.einsum('bchwkl,ockl->bohw', w, weight)             # (B, Cout, Ho, Wo)
    return out + bias[None, :, None, None] if bias is not None else out


def maxpool2d(x: np.ndarray, k: int = 2, stride: Optional[int] = None) -> np.ndarray:
    """x: (B, C, H, W) -> (B, C, Ho, Wo). stride defaults to k."""
    stride = stride or k
    return _windows(x, k, k, stride).max(axis=(-2, -1))


# ============================== 10. Dropout / LoRA ==============================
class InvertedDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: any shape -> same shape. Identity in eval mode."""
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        return x * (torch.rand_like(x) < keep).to(x.dtype) / keep


class LoRALinear(nn.Module):
    def __init__(self, din: int, dout: int, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = nn.Linear(din, dout)
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, din) / r ** 0.5)   # random
        self.B = nn.Parameter(torch.zeros(dout, r))             # zeros => delta W = 0 at start
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., din) -> (..., dout)."""
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the low-rank update into the base weight (zero inference overhead)."""
        self.base.weight += self.scale * self.B @ self.A        # (dout, din)


# ============================== 11. Losses ==============================
def stable_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """logits: (N, C) float, targets: (N,) int in [0, C) -> mean CE (float)."""
    shifted = logits - logits.max(axis=1, keepdims=True)        # (N, C) max is 0 -> exp never overflows
    log_sum = np.log(np.exp(shifted).sum(axis=1, keepdims=True))  # (N, 1)
    log_probs = shifted - log_sum                               # (N, C) log-softmax
    rows = np.arange(len(targets))
    picked = log_probs[rows, targets]                           # (N,) log-prob of the true class
    return float(-picked.mean())


def binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                      alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """logits: (N,) raw scores, targets: (N,) in {0, 1} -> scalar tensor."""
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """z1, z2: (B, D) paired embeddings (row i of z1 matches row i of z2) -> scalar tensor."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature                            # (B, B), diagonal = positives
    return F.cross_entropy(logits, torch.arange(len(z1), device=z1.device))


# ============================== 12. AUC ==============================
def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Sort + cumsum + trapezoid. Handles ties.

    Args:
        y_true: (N,) in {0, 1}.
        y_score: (N,) float, higher = more positive.
    Returns:
        AUC in [0, 1]; 0.5 if only one class is present.
    """
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # 1. sort by score, high to low
    order = np.argsort(-y_score, kind='stable')
    y_sorted = y_true[order]                                    # (N,)
    s_sorted = y_score[order]                                   # (N,)
    # 2. thresholds: last index of each distinct score (so ties move together)
    change = np.where(np.diff(s_sorted))[0]                     # positions where the score changes
    cut = np.r_[change, len(y_sorted) - 1]                      # (T,)
    # 3. counts at each threshold
    tps = np.cumsum(y_sorted)[cut]                              # (T,) positives predicted positive
    fps = (cut + 1) - tps                                       # (T,) negatives predicted positive
    # 4. rates, with the (0, 0) starting point
    tpr = np.r_[0, tps] / n_pos                                 # (T+1,)
    fpr = np.r_[0, fps] / n_neg                                 # (T+1,)
    # 5. trapezoid area under the curve
    widths = np.diff(fpr)                                       # (T,)
    heights = (tpr[:-1] + tpr[1:]) / 2                          # (T,)
    return float(np.sum(widths * heights))


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U. Correct only when there are no tied scores.

    Args/Returns: same as auc_roc (assumes both classes present).
    """
    ranks = np.argsort(np.argsort(y_score)) + 1
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ============================== 13. IoU / GIoU / NMS ==============================
def box_iou_giou(b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise IoU and GIoU.

    Args:
        b1: (N, 4) float, [x1, y1, x2, y2] with x1<x2, y1<y2.
        b2: (M, 4) float, same format.
    Returns:
        (iou, giou), each (N, M).
    """
    # areas
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])       # (N,)
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])       # (M,)
    # intersection: top-left is the max of the corners, bottom-right is the min
    inter_lt = np.maximum(b1[:, None, :2], b2[None, :, :2])     # (N, M, 2)
    inter_rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])     # (N, M, 2)
    inter_wh = np.clip(inter_rb - inter_lt, 0, None)            # (N, M, 2) 0 if no overlap
    inter = inter_wh[..., 0] * inter_wh[..., 1]                 # (N, M)
    # IoU
    union = area1[:, None] + area2[None, :] - inter             # (N, M)
    iou = inter / np.maximum(union, 1e-8)
    # smallest enclosing box -> GIoU
    enc_lt = np.minimum(b1[:, None, :2], b2[None, :, :2])       # (N, M, 2)
    enc_rb = np.maximum(b1[:, None, 2:], b2[None, :, 2:])       # (N, M, 2)
    enc_wh = np.clip(enc_rb - enc_lt, 0, None)
    enc = enc_wh[..., 0] * enc_wh[..., 1]                       # (N, M)
    giou = iou - (enc - union) / np.maximum(enc, 1e-8)
    return iou, giou


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> List[int]:
    """Greedy non-maximum suppression.

    Args:
        boxes: (N, 4) float, [x1, y1, x2, y2].
        scores: (N,) float.
        thr: suppress boxes with IoU > thr against a kept box.
    Returns:
        Indices of kept boxes, in descending score order.
    """
    x1, y1, x2, y2 = boxes.T                                    # each (N,)
    areas = (x2 - x1) * (y2 - y1)                               # (N,)
    order = scores.argsort()[::-1]                              # indices, best score first
    keep: List[int] = []
    while order.size > 0:
        best = order[0]                                         # highest remaining score
        rest = order[1:]                                        # everything else still in play
        keep.append(int(best))
        # overlap of `best` with each remaining box
        w = np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest])
        h = np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])
        inter = np.clip(w, 0, None) * np.clip(h, 0, None)       # (len(rest),)
        union = areas[best] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-8)
        order = rest[iou <= thr]                                # drop boxes that overlap too much
    return keep


# ============================== 14. Sampling ==============================
def softmax(z: np.ndarray) -> np.ndarray:
    """z: (V,) float -> (V,) probabilities. Subtract the max first so exp never overflows."""
    e = np.exp(z - z.max())
    return e / e.sum()


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits, set the rest to -inf. logits: (V,) -> (V,)."""
    kth_largest = np.sort(logits)[-k]                           # value of the k-th largest
    return np.where(logits < kth_largest, -np.inf, logits)


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter: keep the smallest set of tokens whose mass >= p, renormalise. probs: (V,) -> (V,)."""
    order = np.argsort(-probs)                                  # token ids, most likely first
    sorted_probs = probs[order]                                 # (V,)
    cum_before = np.cumsum(sorted_probs) - sorted_probs         # mass BEFORE each token
    remove = cum_before > p                                     # token not needed to reach p
    out = probs.copy()
    out[order[remove]] = 0.0                                    # the top token is never removed
    return out / out.sum()


def sample_top_k_top_p(logits: np.ndarray, k: int = 0, p: float = 1.0, temperature: float = 1.0,
                       rng: Optional[np.random.Generator] = None) -> int:
    """Sample one token id: temperature -> top-k -> softmax -> top-p -> draw.

    Args:
        logits: (V,) float raw scores.
        k: keep the top-k tokens (0 = disabled).
        p: nucleus mass in (0, 1] (1.0 = disabled).
        temperature: > 0; lower = sharper.
        rng: optional numpy Generator for reproducibility.
    Returns:
        Sampled token id in [0, V).
    """
    rng = rng or np.random.default_rng()
    z = logits / temperature                                    # (V,)
    if k > 0:
        z = top_k_filter(z, k)
    probs = softmax(z)                                          # (V,)
    if p < 1.0:
        probs = top_p_filter(probs, p)
    return int(rng.choice(len(probs), p=probs))


def pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """X: (N, D) float -> (N, n_components) projection onto the top principal components."""
    return PCA(n_components).fit_transform(X)
