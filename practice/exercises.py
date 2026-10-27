"""空白题：在这里手写，不要看 ../code 和 ../notes。

用法（在 practice/ 目录下）：
    python run.py list                 # 题目列表
    python run.py mock                 # 随机抽一题，计时模拟面试
    python run.py mock kmeans          # 指定题目
    python run.py check kmeans         # 只检查一题
    python run.py check all            # 检查全部
做完一题就删掉/重置：python run.py reset kmeans   （恢复成空白模板）

要求：类型注解写全；一行一个想法；行尾注释 shape；先说边界条件再写代码。
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- chamfer
def chamfer_distance(P: np.ndarray, Q: np.ndarray, squared: bool = True) -> float:
    """P: (N, D), Q: (M, D). Return mean(P->Q nearest) + mean(Q->P nearest)."""
    raise NotImplementedError


# ---------------------------------------------------------------- knn
class KNN:
    """k-nearest-neighbour classifier; labels are non-negative ints."""

    def __init__(self, k: int = 3) -> None:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (N, D), y: (N,)."""
        raise NotImplementedError

    def predict(self, Xt: np.ndarray) -> np.ndarray:
        """Xt: (M, D) -> (M,) int labels."""
        raise NotImplementedError


# ---------------------------------------------------------------- kmeans
class KMeans:
    """K-Means，分两阶段做（对应 notes/classic_ml/02_KMeans.md）：
      阶段 1：init="random"（随机选 K 个样本做初始中心），写通整个流程；
      阶段 2：加 init="kmeans++"，只新增 _init_plusplus，其余不动。
    随机数直接用 np.random（不需要 rng / seed 参数）。要处理空簇。Expose .centroids (K, D)。
    距离矩阵可自己写一个 pairwise_sq_dist（和 chamfer 题同款）。
    下面的私有方法是建议的拆分，签名可自行调整；判分只调用 fit / predict / .centroids。
    """

    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4,
                 init: str = "kmeans++") -> None:
        raise NotImplementedError

    # ---- 阶段 1 ----
    def _init_random(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (K, D)，K 个不重复的随机样本。"""
        raise NotImplementedError

    def _assign(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (N,) 每个点最近的中心编号。"""
        raise NotImplementedError

    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """X: (N, D), labels: (N,) -> (K, D) 新中心。空簇不能出 NaN。"""
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> "KMeans":
        """X: (N, D). 初始化 -> 循环(分配、更新、收敛判断)。Return self."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (N,) int cluster ids."""
        raise NotImplementedError

    # ---- 阶段 2 ----
    def _init_plusplus(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (K, D)。下一个中心被选中的概率 ∝ 到已选最近中心的平方距离；
        所有点都与已选中心重合时（概率和为 0）要退化为均匀。"""
        raise NotImplementedError


# ---------------------------------------------------------------- logistic regression
class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent + L2 on w (not b)."""

    def __init__(self, lr: float = 0.1, epochs: int = 1000, reg: float = 0.0) -> None:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """X: (N, D), y: (N,) in {0, 1}. Return self. Must not overflow for large |z|."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (N,) P(y=1)."""
        raise NotImplementedError


# ---------------------------------------------------------------- decision tree
class DecisionTree:
    """CART classification tree with Gini impurity."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """X: (N, D), y: (N,) int. Return self."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> (N,) int."""
        raise NotImplementedError


# ---------------------------------------------------------------- pca
def pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """X: (N, D) -> (N, n_components) projection onto the top principal components (use SVD)."""
    raise NotImplementedError


# ---------------------------------------------------------------- cross entropy
def stable_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """logits: (N, C), targets: (N,) int. Mean CE; must survive logits of +-1000."""
    raise NotImplementedError


# ---------------------------------------------------------------- focal / infonce
def binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                      alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """logits: (N,), targets: (N,) in {0,1}. Mean focal loss, numerically stable."""
    raise NotImplementedError


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """z1, z2: (B, D); row i of z1 pairs with row i of z2. Scalar loss."""
    raise NotImplementedError


# ---------------------------------------------------------------- auc
def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """y_true: (N,) in {0,1}, y_score: (N,). Must handle tied scores; 0.5 if one class only."""
    raise NotImplementedError


# ---------------------------------------------------------------- iou / nms
def box_iou_giou(b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """b1: (N, 4), b2: (M, 4), [x1, y1, x2, y2]. Return (iou, giou), each (N, M). No loops."""
    raise NotImplementedError


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> List[int]:
    """boxes: (N, 4), scores: (N,). Indices kept, best score first."""
    raise NotImplementedError


# ---------------------------------------------------------------- attention
def causal_mask(T: int, S: int) -> torch.Tensor:
    """(T, S) bool, True = may attend. T queries are the LAST T positions of S keys."""
    raise NotImplementedError


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """q:(...,T,dk) k:(...,S,dk) v:(...,S,dv), mask bool True=keep. Return (out, weights)."""
    raise NotImplementedError


def self_attention(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
                   causal: bool = False) -> torch.Tensor:
    """x: (B, T, D), W*: (D, d) -> (B, T, d)."""
    raise NotImplementedError


def cross_attention(x: torch.Tensor, context: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                    Wv: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x: (B, T, Dq), context: (B, S, Dkv), context_mask: (B, S) bool True=real token."""
    raise NotImplementedError


def multi_head_attention(x_q: torch.Tensor, x_kv: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                         Wv: torch.Tensor, Wo: torch.Tensor, num_heads: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x_q: (B,T,C), x_kv: (B,S,C), W*: (C,C), mask broadcastable to (B,1,T,S). -> (B,T,C)."""
    raise NotImplementedError


# ---------------------------------------------------------------- norm layers
class BatchNorm1d(nn.Module):
    """x: (B, C). Train: batch stats + running stats (momentum 0.1, unbiased running var). Eval: running."""

    def __init__(self, n: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LayerNorm(nn.Module):
    """Normalise over the last dim, learnable gamma/beta."""

    def __init__(self, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------- dropout / lora
class InvertedDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LoRALinear(nn.Module):
    """Frozen nn.Linear(din, dout) + trainable low-rank update, scaled by alpha/r. Expose .base, .A, .B."""

    def __init__(self, din: int, dout: int, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------- conv
def conv2d(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """x: (B, Cin, H, W), weight: (Cout, Cin, kh, kw) -> (B, Cout, Ho, Wo). Same as torch (cross-correlation)."""
    raise NotImplementedError


# ---------------------------------------------------------------- sampling
def sample_top_k_top_p(logits: np.ndarray, k: int = 0, p: float = 1.0,
                       temperature: float = 1.0) -> int:
    """logits: (V,). Temperature -> top-k -> top-p -> draw one token id (use np.random.choice)."""
    raise NotImplementedError
