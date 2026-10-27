# Conv2d / MaxPool

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


```python
def conv2d(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray] = None,
           stride: int = 1, padding: int = 0) -> np.ndarray:  # x:(B,Cin,H,W) w:(Cout,Cin,kh,kw) -> (B,Cout,Ho,Wo)
    if padding: x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    win = np.lib.stride_tricks.sliding_window_view(x, w.shape[2:], axis=(2,3))[:, :, ::stride, ::stride]
                                                         # (B,Cin,Ho,Wo,kh,kw) 零拷贝
    out = np.einsum('bchwkl,ockl->bohw', win, w)         # (B,Cout,Ho,Wo)
    return out + b[None,:,None,None] if b is not None else out
```

- `Ho = (H + 2p - kh)//stride + 1`。
- 深度学习里的"卷积"实际是互相关（不翻转核），与 PyTorch 一致。
- MaxPool：同样取窗口视图，`.max(axis=(-2,-1))`。
- 若要求手写 im2col：把每个窗口展开成列 `(Cin·kh·kw, Ho·Wo)`，权重 reshape 成 `(Cout, Cin·kh·kw)`，一次矩阵乘法。
- 滑动窗口循环版（四重循环）先写出来也可，再说明向量化。
- 已与 `F.conv2d`（stride 1/2、padding 0/1）和 `F.max_pool2d` 对照通过。


<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np


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
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



