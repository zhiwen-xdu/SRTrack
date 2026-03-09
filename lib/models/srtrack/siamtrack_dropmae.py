"""
    RGB和Event分别一个独立的Feature Encoder
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.vipt.vit_siam_dropmae import vit_base_patch16_224_siam_dropmae
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SIAMTrack_DropMAE(nn.Module):
    """ This is the base class for SIAMTrack """
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        # template：[32,6,192,192]，search：[32,6,384,384]
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out


    # def forward(self, template: torch.Tensor,
    #             search: torch.Tensor,
    #             ce_template_mask=None,
    #             ce_keep_rate=None,
    #             return_last_attn=False,
    #             ):
    #     # template：[32,6,192,192]，search：[32,6,384,384]
    #     x_rgb,x_event,x_fusion, aux_dict = self.backbone(z=template, x=search,
    #                                 ce_template_mask=ce_template_mask,
    #                                 ce_keep_rate=ce_keep_rate,
    #                                 return_last_attn=return_last_attn,)
    #
    #     # Forward head
    #     out_rgb = self.forward_head(x_rgb, None)
    #     out_event = self.forward_head(x_event, None)
    #     out_fusion = self.forward_head(x_fusion, None)
    #
    #     out_fusion.update(aux_dict)
    #     out_fusion['backbone_feat'] = x_fusion
    #
    #     return out_rgb, out_event, out_fusion



    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)


        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def weight_diff(checkpoint_pretrained,checkpoint_trained):
    pretrained_weight_l2 = []
    diff_weight_l2 = []
    for key, pretrained_value in checkpoint_pretrained.items():
        if key in checkpoint_trained and "num" not in key:
            trained_value = checkpoint_trained[key]
            pre = torch.sum(torch.norm(pretrained_value))
            diff = torch.sum(torch.norm(trained_value - pretrained_value))
            pretrained_weight_l2.append(float(pre))
            diff_weight_l2.append(float(diff))

    ratio = sum(diff_weight_l2)/sum(pretrained_weight_l2)

    return ratio


def build_siamtrack_dropmae(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')  # use pretrained OSTrack as initialization
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_prompt':
        backbone = vit_base_patch16_224_siam_dropmae(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                               template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                               new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                               )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError
    """For prompt no need, because we have OSTrack as initialization"""
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    box_head = build_box_head(cfg, hidden_dim)

    model = SIAMTrack_DropMAE(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    model_state_dict = model.state_dict()

    if training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")["net"]
        for key, value in checkpoint.items():
            if "backbone.patch_embed." in key:
                model_state_dict[key.replace("backbone.patch_embed.", "backbone.patch_embed_rgb.")] = value
                model_state_dict[key.replace("backbone.patch_embed.", "backbone.patch_embed_event.")] = value
            elif "backbone.norm." in key:
                model_state_dict[key.replace("backbone.norm.", "backbone.norm_rgb.")] = value
                model_state_dict[key.replace("backbone.norm.", "backbone.norm_event.")] = value
            elif "backbone.blocks." in key:
                model_state_dict[key.replace("backbone.blocks.", "backbone.blocks_rgb.")] = value
                model_state_dict[key.replace("backbone.blocks.", "backbone.blocks_event.")] = value
                if "backbone.blocks.2." in key:
                    model_state_dict[key.replace("backbone.blocks.2.", "backbone.blocks_cross.0.")] = value
                elif "backbone.blocks.5." in key:
                    model_state_dict[key.replace("backbone.blocks.5.", "backbone.blocks_cross.1.")] = value
                elif "backbone.blocks.8." in key:
                    model_state_dict[key.replace("backbone.blocks.8.", "backbone.blocks_cross.2.")] = value
                elif "backbone.blocks.11." in key:
                    model_state_dict[key.replace("backbone.blocks.11.", "backbone.blocks_cross.3.")] = value
            else:
                model_state_dict[key] = value

        model.load_state_dict(model_state_dict,strict=False)


    return model
