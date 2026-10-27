"""Run: python test_ml_coding.py   (CPU only)"""
import numpy as np
import torch
import torch.nn.functional as F
from ml_coding import *

rng = np.random.default_rng(0)
torch.manual_seed(0)


def close(a, b, tol=1e-5):
    assert np.allclose(a, b, atol=tol), (a, b)


# Chamfer
P, Q = rng.random((100, 3)), rng.random((150, 3))
close(chamfer_distance(P, Q), chamfer_distance_broadcast(P, Q))
Pb, Qb = rng.random((4, 100, 3)), rng.random((4, 150, 3))
close(batched_chamfer_distance(Pb, Qb), [chamfer_distance(a, b) for a, b in zip(Pb, Qb)])
# one-way collapse counter-example: P all at one point of Q -> P->Q is 0 but full CD is not
Qs = rng.random((50, 3)); Pc = np.repeat(Qs[:1], 50, 0)
assert chamfer_distance(Pc, Qs) > 0.05
print("chamfer ok")

# KNN
Xtr = np.r_[rng.normal(0, .3, (50, 2)), rng.normal(3, .3, (50, 2))]; ytr = np.r_[np.zeros(50), np.ones(50)].astype(int)
Xte = np.r_[rng.normal(0, .3, (20, 2)), rng.normal(3, .3, (20, 2))]; yte = np.r_[np.zeros(20), np.ones(20)].astype(int)
for cls in (KNN, KNNFast):
    m = cls(5); m.fit(Xtr, ytr); assert (m.predict(Xte) == yte).all()
m = KNNFast(500); m.fit(Xtr, ytr); m.predict(Xte)  # k > N must not crash
print("knn ok")

# KMeans
np.random.seed(0); km = KMeans(2).fit(Xtr); lab = km.predict(Xtr)
assert (lab == ytr).all() or (lab == 1 - ytr).all()
print("kmeans ok")

# Logistic regression
lr = LogisticRegression(lr=0.5, epochs=500).fit(Xtr, ytr)
assert ((lr.predict_proba(Xte) > .5) == yte).all()
close(sigmoid(np.array([-1000., 0., 1000.])), [0, .5, 1])
print("logreg ok")

# Tree / forest
assert (DecisionTree(3).fit(Xtr, ytr).predict(Xte) == yte).all()
assert (RandomForest(5, 3).fit(Xtr, ytr).predict(Xte) == yte).all()
print("tree/forest ok")

# GMM vs sklearn if present
np.random.seed(0); g = GMM(2).fit(Xtr)
assert np.isfinite(g.ll)
try:
    from sklearn.mixture import GaussianMixture
    sk = GaussianMixture(2, covariance_type='full', random_state=0, reg_covar=1e-6, tol=1e-8, max_iter=200).fit(Xtr)
    assert abs(g.ll / len(Xtr) - sk.score(Xtr)) < 0.2
except ImportError:
    pass
print("gmm ok")

# PCA vs eig of covariance
Xp = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
Z = PCA(2).fit_transform(Xp)
w = np.linalg.eigvalsh(np.cov(Xp.T))[::-1]
close(Z.var(0, ddof=1), w[:2], 1e-6)
print("pca ok")

# Attention
B, T, S, C, H = 2, 6, 9, 32, 4
x, ctx = torch.randn(B, T, C), torch.randn(B, S, C)
q_, k_, v_ = (torch.randn(B, H, n, 8) for n in (T, S, S))
close(scaled_dot_product_attention(q_, k_, v_)[0].numpy(), F.scaled_dot_product_attention(q_, k_, v_).numpy(), 1e-5)
qc = torch.randn(B, H, T, 8); kc, vc = (torch.randn(B, H, T, 8) for _ in range(2))
close(scaled_dot_product_attention(qc, kc, vc, causal_mask(T, T))[0].numpy(),
      F.scaled_dot_product_attention(qc, kc, vc, is_causal=True).numpy(), 1e-5)
W = [torch.randn(C, C) / C ** 0.5 for _ in range(4)]
mha = MultiHeadAttention(C, H, dropout=0.0).eval()
with torch.no_grad():
    for lin, w in zip((mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj), W):
        lin.weight.copy_(w.T); lin.bias.zero_()
    # functional == module, self and cross, with causal / padding masks
    cm = causal_mask(T, T)
    close(multi_head_attention(x, x, *W, H, cm).numpy(), mha(x, is_causal=True)[0].numpy(), 1e-5)
    pad = torch.ones(B, S, dtype=torch.bool); pad[:, -3:] = False         # last 3 context tokens are padding
    close(multi_head_attention(x, ctx, *W, H, pad[:, None, None, :]).numpy(),
          mha(x, context=ctx, mask=pad[:, None, None, :])[0].numpy(), 1e-5)
    # padding really is ignored: changing padded context entries changes nothing
    ctx2 = ctx.clone(); ctx2[:, -3:] += 100
    close(mha(x, context=ctx, mask=pad[:, None, None, :])[0].numpy(),
          mha(x, context=ctx2, mask=pad[:, None, None, :])[0].numpy(), 1e-5)
    # self-attention == cross-attention with context=x
    close(mha(x)[0].numpy(), mha(x, context=x)[0].numpy(), 1e-5)
    # single-head functions
    Wq, Wk, Wv = (torch.randn(C, 8) for _ in range(3))
    close(self_attention(x, Wq, Wk, Wv, causal=True).numpy(),
          F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, is_causal=True).numpy(), 1e-5)
    close(cross_attention(x, ctx, Wq, Wk, Wv).numpy(),
          F.scaled_dot_product_attention(x @ Wq, ctx @ Wk, ctx @ Wv).numpy(), 1e-5)
    # KV-cache decode == full causal forward
    y_full, _ = mha(x, is_causal=True)
    cache, outs = None, []
    for t_ in range(T):
        o, cache = mha(x[:, t_:t_ + 1], is_causal=True, kv_cache=cache)
        outs.append(o)
    close(torch.cat(outs, 1).numpy(), y_full.numpy(), 1e-5)
    # cross-attn cache: reuse context K/V on later steps
    _, ccache = mha(x[:, :1], context=ctx)
    close(mha(x[:, 1:2], context=ctx, kv_cache=ccache)[0].numpy(), mha(x[:, 1:2], context=ctx)[0].numpy(), 1e-5)
print("attention (self/cross/multi-head/kv cache) ok")

# Norms vs torch
xb = torch.randn(16, 8)
bn, ref = BatchNorm1d(8), torch.nn.BatchNorm1d(8)
close(bn(xb).detach().numpy(), ref(xb).detach().numpy(), 1e-5)
close(bn.running_var.numpy(), ref.running_var.numpy(), 1e-6)
bn.eval(); ref.eval()
close(bn(xb).detach().numpy(), ref(xb).detach().numpy(), 1e-5)
close(LayerNorm(8)(xb).detach().numpy(), torch.nn.LayerNorm(8)(xb).detach().numpy(), 1e-5)
r = RMSNorm(8)(xb)
close(r.pow(2).mean(-1).detach().numpy(), np.ones(16), 1e-3)
print("norm ok")

# Conv / pool vs torch
xi, wi, bi = rng.normal(size=(2, 3, 9, 9)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
for s, p in [(1, 0), (2, 1)]:
    ref = F.conv2d(torch.tensor(xi), torch.tensor(wi), torch.tensor(bi), stride=s, padding=p).numpy()
    close(conv2d(xi, wi, bi, s, p), ref, 1e-8)
close(maxpool2d(xi, 2), F.max_pool2d(torch.tensor(xi), 2).numpy())
print("conv/pool ok")

# Dropout, LoRA
d = InvertedDropout(0.3); xd = torch.ones(200000)
assert abs(d(xd).mean().item() - 1) < 0.01
d.eval(); assert (d(xd) == xd).all()
lo = LoRALinear(16, 8); xl = torch.randn(4, 16)
close(lo(xl).detach().numpy(), lo.base(xl).detach().numpy())      # delta W = 0 at init
lo.B.data.normal_(); before = lo(xl).detach().clone(); lo.merge(); lo.B.data.zero_()
close(lo(xl).detach().numpy(), before.numpy(), 1e-5)              # merged weight reproduces output
assert lo.A.requires_grad and not lo.base.weight.requires_grad
print("dropout/lora ok")

# Losses
lg, tg = rng.normal(size=(10, 5)) * 50, rng.integers(0, 5, 10)
close(stable_cross_entropy(lg, tg), F.cross_entropy(torch.tensor(lg), torch.tensor(tg)).item())
lb, tb = torch.randn(32), torch.randint(0, 2, (32,))
assert binary_focal_loss(lb, tb, 0.5, 0.0).item() > 0
close(binary_focal_loss(lb, tb, 0.5, 0.0).item(), 0.5 * F.binary_cross_entropy_with_logits(lb, tb.float()).item())
z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
assert infonce_loss(z1, z1 + 1e-3 * z2) < infonce_loss(z1, z2)
print("loss ok")

# AUC
from itertools import product
yt = rng.integers(0, 2, 200); ys = rng.random(200)
brute = np.mean([(a > b) + 0.5 * (a == b) for a, b in product(ys[yt == 1], ys[yt == 0])])
close(auc_roc(yt, ys), brute); close(auc_rank(yt, ys), brute)
ys_t = np.round(ys, 1)                                             # heavy ties
brute_t = np.mean([(a > b) + 0.5 * (a == b) for a, b in product(ys_t[yt == 1], ys_t[yt == 0])])
close(auc_roc(yt, ys_t), brute_t)
try:
    from sklearn.metrics import roc_auc_score
    close(auc_roc(yt, ys_t), roc_auc_score(yt, ys_t))
except ImportError:
    pass
print("auc ok")

# IoU / GIoU / NMS vs torchvision if present
b = np.sort(rng.random((6, 2, 2)), axis=1).reshape(6, 4)[:, [0, 2, 1, 3]]  # x1<x2? build properly
b = np.c_[np.minimum(b[:, 0], b[:, 1]), np.minimum(b[:, 2], b[:, 3]), np.maximum(b[:, 0], b[:, 1]) + .1, np.maximum(b[:, 2], b[:, 3]) + .1]
iou, giou = box_iou_giou(b, b)
close(np.diag(iou), 1); close(np.diag(giou), 1); assert (giou <= iou + 1e-9).all()
sc = rng.random(6)
try:
    import torchvision
    t = torch.tensor(b)
    close(iou, torchvision.ops.box_iou(t, t).numpy())
    close(giou, torchvision.ops.generalized_box_iou(t, t).numpy())
    assert sorted(nms(b, sc, .3)) == sorted(torchvision.ops.nms(t, torch.tensor(sc), .3).tolist())
except ImportError:
    assert nms(b, sc, 1.0)[0] == int(sc.argmax())
print("box ok")

# Sampling
lg = np.array([5., 4., 0., -1.])
assert {(np.random.seed(i), sample_top_k_top_p(lg, k=2))[1] for i in range(200)} <= {0, 1}
assert {(np.random.seed(i), sample_top_k_top_p(lg, p=0.5))[1] for i in range(200)} == {0}
print("sampling ok")
print("ALL PASSED")
