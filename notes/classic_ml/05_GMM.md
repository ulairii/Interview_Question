# GMM（EM）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


**E-step**：`γᵢₖ = πₖ N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ,Σⱼ)`
**M-step**：`Nₖ=Σᵢγᵢₖ`，`πₖ=Nₖ/N`，`μₖ=Σᵢγᵢₖxᵢ/Nₖ`，`Σₖ=Σᵢγᵢₖ(xᵢ-μₖ)(xᵢ-μₖ)ᵀ/Nₖ`

必须在 **log 域**做 E-step + LogSumExp：

```python
lp = log(pi_k) + log_gaussian(X, mu_k, cov_k)    # (N,K)
lse = m + log(sum(exp(lp - m), axis=1))          # m = lp.max(axis=1)
gamma = exp(lp - lse)
```

马氏距离：`np.einsum('nd,de,ne->n', diff, inv_cov, diff)`。

- **奇异协方差**：`Σ += 1e-6·I`。
- 初始化：随机选点 / K-Means 结果。
- 对数似然单调不降，用 `|ll - prev| < tol` 判停。
- K-Means 是 GMM 的特例（`Σ=σ²I`、σ→0、硬分配）。
- 单个分量坍缩到一个点（`Σ→0`，似然→∞）是 GMM 的病态，靠 reg 缓解。

<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np


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
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



