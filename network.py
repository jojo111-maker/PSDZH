import logging
import os
from functools import partial

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from timm.models import register_model, build_model_with_cfg


import math
import torch
import torch.nn.functional as F
from torch import nn


from timm import create_model
import EViT.evit as evit

_logger = logging.getLogger(__name__)


class Attention(evit.Attention):
    def forward(self, x, keep_rate=None, tokens=None, aux_sem=None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)

        # Note: changed
        if aux_sem is not None:
            q[:, :, 0, :] = 0.9 * q[:, :, 0, :] + 0.1 * aux_sem # 重点，0.9视觉为主导，0.1语义为辅助
        token_cls = q[:, :, 0, :].reshape(B, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double-check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens, token_cls

        return x, None, None, None, left_tokens, token_cls

# transformer单元
class Block(evit.Block):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        keep_rate=0.0,
        fuse_token=False,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
            keep_rate,
            fuse_token,
        )
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate
        )

    # def forward(self, x, keep_rate=None, tokens=None, get_idx=False, aux_sem=None, return_attn=False, **kwargs):
    #     if keep_rate is None:
    #         keep_rate = self.keep_rate  # inference uses default keep rate
    #     B, N, C = x.shape
    #
    #     tmp, index, idx, cls_attn, left_tokens, token_cls = self.attn(self.norm1(x), keep_rate, tokens, aux_sem)
    #     x = x + self.drop_path(tmp)
    #
    #     if index is not None:
    #         non_cls = x[:, 1:]
    #         x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]
    #
    #         if self.fuse_token:
    #             compl = evit.complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
    #             non_topk = torch.gather(
    #                 non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C)
    #             )  # [B, N-1-left_tokens, C]
    #
    #             non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
    #             extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
    #             x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
    #         else:
    #             x = torch.cat([x[:, 0:1], x_others], dim=1)
    #
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     n_tokens = x.shape[1] - 1
    #
    #     # ✅ 默认只返回 3 个值，兼容 EViT/evit.py 里：x, left_token, idx = blk(...)
    #     if not return_attn:
    #         # 如果 index 为空，idx 也应为空，避免上层误用
    #         return x, n_tokens, (idx if index is not None else None)
    #
    #     # ✅ 需要 token_cls 时才返回第 4 个值
    #     return x, n_tokens, (idx if index is not None else None), token_cls

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False, aux_sem=None):
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        B, N, C = x.shape

        tmp, index, idx, cls_attn, left_tokens, token_cls = self.attn(self.norm1(x), keep_rate, tokens, aux_sem)
        x = x + self.drop_path(tmp)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = evit.complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(
                    non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C)
                )  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, idx, token_cls
        return x, n_tokens, None, token_cls


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

# 视觉与语义对齐的桥梁
# 语义引导的视觉表示
class S2V(nn.Module):
    def __init__(self, att_dim, embed_dim) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(att_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        out = self.fc(x)
        return out

#视觉约束下的语义重建
class V2S(nn.Module):
    def __init__(self, embed_dim, att_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, att_dim),
            nn.ReLU(),
            nn.Linear(att_dim, att_dim),
            nn.ReLU(),
        )
        self.apply(weights_init)  # TODO: add?

    def forward(self, x):
        return self.fc(x)


class ZSLViT(evit.EViT):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=evit.PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        keep_rate=(1,),
        fuse_token=False,
        dataset=None,
        n_bits=None,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            representation_size,
            distilled,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            embed_layer,
            norm_layer,
            act_layer,
            weight_init,
            keep_rate,
            fuse_token,
        )
        self.num_heads = num_heads

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    keep_rate=keep_rate[i],
                    fuse_token=fuse_token,
                )
                for i in range(depth)
            ]
        )

        self.dataset = dataset
        attr_dim = {"cub": 312, "awa2": 85, "sun": 102}[dataset]

        self.S2V = nn.ModuleList([S2V(attr_dim, self.embed_dim) for _ in range(3)])
        self.V2S = nn.ModuleList([V2S(self.embed_dim, attr_dim) for _ in range(3)])

        # MyCode: support n_bits
        if n_bits is not None:
            self.mlp_g = nn.Linear(n_bits, attr_dim, bias=False)
            self.fc = nn.Linear(self.embed_dim, n_bits, bias=False)
            self.fc_list = nn.ModuleList([nn.Linear(self.embed_dim, n_bits, bias=False) for _ in range(3)])
        else:
            self.mlp_g = nn.Linear(self.embed_dim, attr_dim, bias=False)

    def forward_features(self, x, keep_rate=None, tokens=None, get_idx=False, labels=None, atts=None):
        _, _, h, w = x.shape
        if not isinstance(keep_rate, (tuple, list)):
            keep_rate = (keep_rate,) * self.depth
        if not isinstance(tokens, (tuple, list)):
            tokens = (tokens,) * self.depth
        assert len(keep_rate) == self.depth
        assert len(tokens) == self.depth
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # for input with another resolution, interpolate the positional embedding.
        # used for finetuning a ViT on images with larger size.
        pos_embed = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            assert h == w  # for simplicity assume h == w
            real_pos = pos_embed[:, self.num_tokens :]
            hw = int(math.sqrt(real_pos.shape[1]))
            true_hw = int(math.sqrt(x.shape[1] - self.num_tokens))
            real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
            new_pos = F.interpolate(real_pos, size=true_hw, mode="bicubic", align_corners=False)
            new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
            pos_embed = torch.cat([pos_embed[:, : self.num_tokens], new_pos], dim=1)

        x = self.pos_drop(x + pos_embed)

        left_tokens = []
        idxs = []

        # Note: add 3 vars
        aux = []
        p_count = 0
        token_cls_reg = []
        for i, blk in enumerate(self.blocks):
            aux_sem = None  # 先初始化，避免未定义
            if self.keep_rate[i] < 1 and self.training and atts is not None:
                aux_sem = self.S2V[p_count](atts[labels]).reshape(-1, self.num_heads, self.embed_dim // self.num_heads)
                aux.append(aux_sem.reshape(-1, self.embed_dim))
                x, left_token, idx, token_cls, _, _ = blk(x, keep_rate[i], tokens[i], get_idx, aux_sem,
                                                          return_attn=True)
                token_cls_reg_p = self.V2S[p_count](token_cls)
                token_cls_reg.append(token_cls_reg_p)
                p_count += 1
            else:
                x, left_token, idx, token_cls, _, _ = blk(x, keep_rate[i], tokens[i], get_idx, aux_sem=None,
                                                          return_attn=True)

            left_tokens.append(left_token)

            if idx is not None:
                idxs.append(idx)

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False, labels=None, atts=None):
        x, _, idxs, aux, token_cls_reg = self.forward_features(x, keep_rate, tokens, get_idx, labels, atts)
        if get_idx:
            return x, idxs
        # return x, aux, token_cls_reg
        if hasattr(self, "fc"):
            x = self.fc(x)
            aux = [self.fc_list[i](x) for i, x in enumerate(aux)]

        if atts is not None:
            logits = self.mlp_g(x) @ atts.T
        else:
            logits = None

        # x, recon_v, recon_s, logits
        return x, aux, token_cls_reg, logits
# def forward_features(self, x, keep_rate=None, tokens=None, get_idx=False, labels=None, atts=None):
#     _, _, h, w = x.shape
#     if not isinstance(keep_rate, (tuple, list)):
#         keep_rate = (keep_rate,) * self.depth
#     if not isinstance(tokens, (tuple, list)):
#         tokens = (tokens,) * self.depth
#     assert len(keep_rate) == self.depth
#     assert len(tokens) == self.depth
#
#     x = self.patch_embed(x)
#     cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#     if self.dist_token is None:
#         x = torch.cat((cls_token, x), dim=1)
#     else:
#         x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#
#     pos_embed = self.pos_embed
#     if x.shape[1] != pos_embed.shape[1]:
#         assert h == w
#         real_pos = pos_embed[:, self.num_tokens:]
#         hw = int(math.sqrt(real_pos.shape[1]))
#         true_hw = int(math.sqrt(x.shape[1] - self.num_tokens))
#         real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
#         new_pos = F.interpolate(real_pos, size=true_hw, mode="bicubic", align_corners=False)
#         new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
#         pos_embed = torch.cat([pos_embed[:, : self.num_tokens], new_pos], dim=1)
#
#     x = self.pos_drop(x + pos_embed)
#
#     left_tokens = []
#     idxs = []
#     aux = []
#     p_count = 0
#     token_cls_reg = []
#
#     for i, blk in enumerate(self.blocks):
#         aux_sem = None
#         if self.keep_rate[i] < 1 and self.training and atts is not None:
#             aux_sem = self.S2V[p_count](atts[labels]).reshape(
#                 -1, self.num_heads, self.embed_dim // self.num_heads
#             )
#             aux.append(aux_sem.reshape(-1, self.embed_dim))
#
#             # ✅ 和 Block.forward 对齐：return_attn=True -> 返回4个
#             x, n_tokens, idx, token_cls = blk(
#                 x, keep_rate[i], tokens[i], get_idx, aux_sem, return_attn=True
#             )
#
#             token_cls_reg_p = self.V2S[p_count](token_cls)
#             token_cls_reg.append(token_cls_reg_p)
#             p_count += 1
#         else:
#             # ✅ 和 Block.forward 对齐：return_attn=True -> 返回4个
#             x, n_tokens, idx, token_cls = blk(
#                 x, keep_rate[i], tokens[i], get_idx, aux_sem=None, return_attn=True
#             )
#
#         left_tokens.append(n_tokens)
#         if idx is not None:
#             idxs.append(idx)
#
#     x_cls = x[:, 0]
#     return x_cls, left_tokens, idxs, aux, token_cls_reg
#
#
# def forward(self, x, keep_rate=None, tokens=None, get_idx=False, labels=None, atts=None):
#     x, _, idxs, aux, token_cls_reg = self.forward_features(x, keep_rate, tokens, get_idx, labels, atts)
#
#     if get_idx:
#         return x, idxs
#
#     if hasattr(self, "fc"):
#         x = self.fc(x)
#         aux = [self.fc_list[i](z) for i, z in enumerate(aux)]  # ✅ 不覆盖x
#
#     logits = (self.mlp_g(x) @ atts.T) if atts is not None else None
#     return x, aux, token_cls_reg, logits


def _create_evit(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or evit.default_cfgs[variant]
    default_cfg.update(kwargs)
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg["num_classes"]
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        ZSLViT,
        variant,
        pretrained,
        # default_cfg=default_cfg,
        pretrained_strict=False,
        representation_size=repr_size,
        pretrained_filter_fn=evit.checkpoint_filter_fn,
        # pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs,
    )
    return model


@register_model
def deit_small_patch16_shrink_base2(pretrained=False, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
    keep_rate = [1] * 12
    for loc in drop_loc:
        keep_rate[loc] = base_keep_rate
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, keep_rate=keep_rate)
    model_kwargs.update(kwargs)
    model = _create_evit("deit_small_patch16_224", pretrained=pretrained, **model_kwargs)
    return model


@register_model
# def deit_base_patch16_shrink_base2(pretrained=False, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
#     keep_rate = [1] * 12
#     for loc in drop_loc:
#         keep_rate[loc] = base_keep_rate
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, keep_rate=keep_rate)
#     model_kwargs.update(kwargs)
#     model = _create_evit("deit_base_patch16_224", pretrained=pretrained, **model_kwargs)
#     return model


def build_model(args, pretrained=True):

    backbone, scale = args.backbone.split("_")

    if backbone == "evit":
        net = create_model(
            f"deit_{scale}_patch16_shrink_base2",
            base_keep_rate=0.9,
            drop_loc=(3, 6, 9),
            pretrained=pretrained,
            num_classes=args.n_classes,
            drop_rate=0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            fuse_token=True,
            img_size=(evit._cfg()["input_size"][1:]),
            dataset=args.dataset,
            # representation_size=args.n_bits,
            n_bits=args.n_bits,
        )
    else:
        raise NotImplementedError(f"Not support: {args.backbone}")
    return net.to(args.device), 0


if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K, A = 10, 100, 8, 312

    e, s, l = gen_test_data(B, C, K, False, True)
    net = create_model(
        "deit_small_patch16_shrink_base2",
        base_keep_rate=0.9,
        drop_loc=(3, 6, 9),
        pretrained=True,
        num_classes=C,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        fuse_token=True,
        img_size=(evit._cfg()["input_size"][1:]),
        dataset="cub",
        # representation_size=K,
        n_bits=K,
    )

    net.to("cuda:1")

    x = torch.randn(B, 3, 224, 224).to("cuda:1")
    a = torch.randn(C, A).to("cuda:1")
    out = net(x, labels=s, atts=a)
    for z in out:
        if isinstance(z, torch.Tensor):
            print(z.shape)
        elif isinstance(z, list):
            print(len(z), z[0].shape)
        else:
            print(z)

    for name, p in net.named_parameters():
        if name.startswith("fc") or name.startswith("mlp_g"):
            print(name, "###")
        else:
            print(name)
