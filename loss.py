from argparse import Namespace

import torch.nn.functional as F
from torch import nn

# 保持语义一致性
class ZSLViTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, recon_v, recon_s, logits, atts, labels):
        # logits = self.mlp_g(x) @ self.atts.T
        L_pre = F.cross_entropy(logits, labels)

        L_SR = sum(F.l1_loss(recon_v[i], x) for i in range(3))
        # L_SR = sum(F.l1_loss(self.projs[i](recon_v[i]), x) for i in range(3))

        L_VR = sum(F.l1_loss(recon_s[i], atts[labels]) for i in range(3))

        return L_pre, L_SR, L_VR

# 关注哈希空间判别结构
class SCPLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.n_classes = args.n_classes
        self.normalized = "normalize" in args.backbone

        # proxy
        self.weight = nn.Parameter(torch.FloatTensor(self.n_classes, args.n_bits))
        nn.init.xavier_uniform_(self.weight)

        self.ls_eps = 0  # for label smoothing

        self.tau = 0.38  # tau in Eq. 3
        self.psi = 0.2  # psi in Eq. 3
        self.sp = 9.0  # scaling hyperparameter in Eq. 2
        self.sn = 9.0  # scaling hyperparameter in Eq. 4
        self.mu = 0.0  # mu in Eq. 4
        self.b = 2  # b in Eq. 6

    def forward(self, batch, labels):
        if not self.normalized:
            batch = F.normalize(batch)

        # B x K, C x K -> B x C
        sim_mat = F.linear(batch, F.normalize(self.weight))

        tp = ((sim_mat.clamp(min=0.0) * labels) * 2).sum() + self.b

        if self.ls_eps > 0:
            labels = (1 - self.ls_eps) * labels + self.ls_eps / self.n_classes

        # Eq. 1, * labels means only positives
        lossp = ((1.0 - sim_mat) * torch.exp((1.0 - sim_mat) * self.sp).detach() * labels).sum()

        # a threshold 𝜏 is used
        # to exclude negative pairs with a cosine similarity lower than it.
        mask = sim_mat > self.tau
        sim_mat = sim_mat[mask]  # B x C -> mask.sum()
        # Eq. 3
        lossn = ((sim_mat - self.psi) * torch.exp((sim_mat - self.mu) * self.sn).detach() * (1 - labels[mask])).sum()

        # Eq. 6
        loss = 1.0 - tp / (tp + lossp + lossn)

        return loss

# 改进版损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

# class ZSLViTLoss(nn.Module):
#     """
#     Progressive Semantic-Guided ZSL Loss with CPFLoss-inspired enhancement.
#
#     输出依然保持原来的 (L_pre, L_SR, L_VR)
#     """
#
#     def __init__(self, args: Namespace):
#         super().__init__()
#
#         self.n_classes = args.n_classes
#         self.n_bits = getattr(args, "n_bits", 16)
#
#         # 原始损失权重
#         self.lambda_proxy = getattr(args, "lambda_proxy", 0.1)  # CPFLoss 风格约束权重
#
#         # CPFLoss 风格参数
#         self.tau = 0.38  # 正负样本阈值
#         self.psi = 0.2   # 负样本偏移
#         self.sp = 9.0    # 正样本放大系数
#         self.sn = 9.0    # 负样本放大系数
#         self.mu = 0.0
#         self.b = 2.0
#
#         # proxy 权重
#         self.proxy_weight = nn.Parameter(torch.randn(self.n_classes, self.n_bits))
#         nn.init.xavier_uniform_(self.proxy_weight)
#
#     def forward(self, x, recon_v, recon_s, logits, atts, labels):
#         device = x.device
#         atts = atts.to(device)
#         labels = labels.to(device)
#         """
#         Args:
#             x: B x n_bits, encoded visual feature
#             recon_v: list of 3 visual reconstruction tensors
#             recon_s: list of 3 semantic reconstruction tensors
#             logits: B x n_classes, classification logits
#             atts: class attribute matrix (n_classes x att_dim)
#             labels: B tensor, class indices
#         """
#
#         # -------------------------------
#         # 1. 原始损失
#         L_pre = F.cross_entropy(logits, labels)  # 分类损失
#         L_SR = sum(F.l1_loss(recon_v[i], x) for i in range(3))  # 视觉重建
#         L_VR = sum(F.l1_loss(recon_s[i], atts[labels]) for i in range(3))  # 语义重建
#
#         # -------------------------------
#         # 2. CPFLoss 风格增强
#         x_norm = F.normalize(x, dim=1)             # B x n_bits
#         proxy_weight = self.proxy_weight.to(device)  # 🔥 关键防御
#         proxy_norm = F.normalize(self.proxy_weight, dim=1)  # n_classes x n_bits
#
#         sim_mat = F.linear(x_norm, proxy_norm)    # B x C 相似度矩阵
#
#         # one-hot mask
#         labels_onehot = F.one_hot(labels, num_classes=self.n_classes).float()  # B x C
#
#         # 正样本部分
#         tp = ((sim_mat.clamp(min=0.0) * labels_onehot) * 2).sum() + self.b
#         lossp = ((1.0 - sim_mat) * torch.exp((1.0 - sim_mat) * self.sp).detach() * labels_onehot).sum()
#
#         # 负样本部分
#         mask = sim_mat > self.tau
#         sim_mat_masked = sim_mat[mask]
#         labels_masked = labels_onehot[mask]
#         if sim_mat_masked.numel() > 0:
#             lossn = ((sim_mat_masked - self.psi) *
#                      torch.exp((sim_mat_masked - self.mu) * self.sn).detach() *
#                      (1 - labels_masked)).sum()
#         else:
#             lossn = 0.0
#
#         L_proxy = 1.0 - tp / (tp + lossp + lossn)
#
#         # -------------------------------
#         # 3. 融合增强到原始重建损失
#         L_SR = L_SR + self.lambda_proxy * L_proxy
#         L_VR = L_VR + self.lambda_proxy * L_proxy
#
#         # -------------------------------
#         return L_pre, L_SR, L_VR


if __name__ == "__main__":
    import torch
    from _utils import gen_test_data

    B, C, K = 9, 717, 16
    A = 102
    e, sl, ol = gen_test_data(B, C, K)
    recon_v = [torch.randn(B, K) for i in range(3)]
    recon_s = [torch.randn(B, A) for i in range(3)]

    criterion = ZSLViTLoss(Namespace(dataset="sun", data_dir="../_datasets_zs", n_classes=C, n_bits=K))

    loss = criterion(e, recon_v, recon_s, sl)
    print(loss)
