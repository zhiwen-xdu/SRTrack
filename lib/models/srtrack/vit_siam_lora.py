from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from .utils import combine_tokens, token2feature, feature2token
from lib.models.layers.patch_embed import PatchEmbed
from lib.models.vipt.base_backbone import BaseBackbone
import loralib as lora


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, return_attention=False):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         if return_attention:
#             return x, attn
#         return x


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)
        self.qkv = lora.Linear(dim, dim * 3, bias=qkv_bias, r=8)
        self.proj = lora.Linear(dim, dim, r=16)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x




# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x, return_attention=False):
#         if return_attention:
#             feat, attn = self.attn(self.norm1(x), True)
#             x = x + self.drop_path(feat)
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#             return x, attn
#         else:
#             x = x + self.drop_path(self.attn(self.norm1(x)))
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#             return x


class Block_LoRA(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformerSiam_LoRA(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',search_size=None, template_size=None,
                 new_patch_size=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed_rgb = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_event = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_rgb = nn.Sequential(*[
            Block_LoRA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm_rgb = norm_layer(embed_dim)

        self.blocks_event = nn.Sequential(*[
            Block_LoRA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm_event = norm_layer(embed_dim)

        cross_index = [2,5,8,11]
        self.blocks_cross = nn.Sequential(*[
            Block_LoRA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cross_index[i]], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(len(cross_index))])

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb image + event image
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        x_event = x[:, 3:, :, :]
        z_event = z[:, 3:, :, :]

        z_rgb = self.patch_embed_rgb(z_rgb)
        x_rgb = self.patch_embed_rgb(x_rgb)
        z_event = self.patch_embed_event(z_event)
        x_event = self.patch_embed_event(x_event)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z_rgb += self.pos_embed_z
        x_rgb += self.pos_embed_x
        z_event += self.pos_embed_z
        x_event += self.pos_embed_x

        if self.add_sep_seg:
            x_rgb += self.search_segment_pos_embed
            z_rgb += self.template_segment_pos_embed

        x_rgb = combine_tokens(z_rgb, x_rgb, mode=self.cat_mode)
        x_event = combine_tokens(z_event, x_event, mode=self.cat_mode)
        if self.add_cls_token:
            x_rgb = torch.cat([cls_tokens, x_rgb], dim=1)

        x_rgb = self.pos_drop(x_rgb)
        x_event = self.pos_drop(x_event)

        x_dict = {'rgb': x_rgb, 'event': x_event, 'rgb_event': None, }

        len_rgb = x_dict['rgb'].shape[1]
        cross_index = [2,5,8,11]
        for i, blk_rgb in enumerate(self.blocks_rgb):
            # for mode in ['rgb', 'event']:
            #     if mode == "rgb":
            #         x_dict["rgb"] = blk_rgb(x_dict["rgb"])
            #     elif mode == "event":
            #         blk_event = self.blocks_event[i]
            #         x_dict["event"] = blk_event(x_dict["event"])

            for mode in ['rgb', 'event', 'rgb_event']:
                if mode == "rgb":
                    x_dict["rgb"] = blk_rgb(x_dict["rgb"])

                elif mode == "event":
                    blk_event = self.blocks_event[i]
                    x_dict["event"] = blk_event(x_dict["event"])

                elif mode == "rgb_event":
                    if i not in cross_index:
                        continue
                    blk_cross = self.blocks_cross[cross_index.index(i)]
                    x_dict["rgb_event"] = combine_tokens(x_dict['rgb'], x_dict['event'], mode=self.cat_mode)
                    x_dict["rgb_event"] = blk_cross(x_dict["rgb_event"])
                    x_dict['rgb'] = x_dict["rgb_event"][:, :len_rgb, :]
                    x_dict['event'] = x_dict["rgb_event"][:, len_rgb:, :]


        x_rgb = self.norm_rgb(x_dict["rgb"])
        x_event = self.norm_event(x_dict["event"])
        x_fusion = sum([x_rgb + x_event]) / 2
        aux_dict = {"attn": None}
        return x_fusion, aux_dict

        # return x_rgb,x_event,x_fusion,aux_dict


    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x_fusion, aux_dict = self.forward_features(z, x)
        return x_fusion, aux_dict

        # x_rgb, x_event, x_fusion, aux_dict = self.forward_features(z, x)
        # return x_rgb, x_event, x_fusion, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerSiam_LoRA(**kwargs)
    return model


def vit_base_patch16_224_siam_lora(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
