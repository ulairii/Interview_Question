# IoU / GIoU / NMS

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


```python
def box_iou_giou(b1, b2):                                       # (N,4), (M,4): [x1,y1,x2,y2]
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])       # (N,)
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])       # (M,)
    # 交集：左上角取 max，右下角取 min
    inter_lt = np.maximum(b1[:, None, :2], b2[None, :, :2])     # (N, M, 2)
    inter_rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])     # (N, M, 2)
    inter_wh = np.clip(inter_rb - inter_lt, 0, None)            # (N, M, 2) 不相交为 0
    inter = inter_wh[..., 0] * inter_wh[..., 1]                 # (N, M)
    union = area1[:, None] + area2[None, :] - inter             # (N, M)
    iou = inter / np.maximum(union, 1e-8)
    # 最小外接框 → GIoU
    enc_lt = np.minimum(b1[:, None, :2], b2[None, :, :2])
    enc_rb = np.maximum(b1[:, None, 2:], b2[None, :, 2:])
    enc_wh = np.clip(enc_rb - enc_lt, 0, None)
    enc = enc_wh[..., 0] * enc_wh[..., 1]                       # (N, M)
    giou = iou - (enc - union) / np.maximum(enc, 1e-8)
    return iou, giou
```

NMS 按「取最优 → 算重叠 → 筛掉」三步写：

```python
def nms(boxes, scores, thr=0.5):
    x1, y1, x2, y2 = boxes.T                                    # 各 (N,)
    areas = (x2 - x1) * (y2 - y1)                               # (N,)
    order = scores.argsort()[::-1]                              # 分数从高到低的下标
    keep = []
    while order.size > 0:
        best, rest = order[0], order[1:]                        # 当前最高分框 / 其余候选
        keep.append(int(best))
        w = np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest])
        h = np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])
        inter = np.clip(w, 0, None) * np.clip(h, 0, None)       # (len(rest),)
        union = areas[best] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-8)
        order = rest[iou <= thr]                                # 保留重叠不大的
    return keep
```

- GIoU ∈ (-1, 1]，不相交时仍有梯度；IoU 在不相交时恒为 0，无梯度。
- **NMS**：按分数降序，取最高分框，删掉与其 IoU>阈值的框，循环。注意：`order = rest[iou <= thr]` 直接用 `rest` 索引，避免 `+1` 偏移出错。多类别则按类别分别做（或 batched NMS，坐标加类别偏移）。复杂度最坏 O(N²)。
- **3D axis-aligned**：BEV 面积交集 × `max(0, min(zmax) - max(zmin))` 的高度交集。
- **3D 旋转框（BEV + yaw）**：旋转矩阵得 4 角点 → Sutherland-Hodgman 裁剪两个凸四边形得交集多边形 → 按极角排序 → 鞋带公式 `S=½|Σ(xᵢyᵢ₊₁-xᵢ₊₁yᵢ)|` → 乘高度交集。
- 与 torchvision 的 `box_iou / generalized_box_iou / nms` 对照（本机装有时）。

<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np


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
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



