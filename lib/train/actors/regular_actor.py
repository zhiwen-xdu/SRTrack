import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu

from lib.utils.box_ops import calculate_giou
import torch.nn.functional as F

class RegularActor(BaseActor):
    """ Actor for training ViPT models """
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)
        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # data['template_images']: [1, B, 6, 256, 256]
        # data['template_anno']: [1, B, 4]
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,*data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 256, 256)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict


    # def compute_similarity(self, pred_dict, gt_dict):
    #     pred_features = pred_dict['backbone_feat']                   # [B,64+256,768]
    #     enc_opt = pred_features[:, -256:]                            # [B,256,768]
    #     opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
    #     bs, Nq, C, HW = opt.size()
    #     feature_map = opt.view(-1, C, 16, 16)                        # [B,768,16,16]
    #     pred_boxes = pred_dict['pred_boxes']
    #     pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # [B,4]
    #
    #     pred_boxes_vec_1 = pred_boxes_vec[:, None, :].repeat((1, bs, 1)).view(-1, 4)       # [B,B,4]->[B*B,4]
    #     pred_boxes_vec_2 = pred_boxes_vec[None,:,:].repeat((bs, 1, 1)).view(-1, 4)         # [B,B,4]->[B*B,4]
    #     giou = calculate_giou(pred_boxes_vec_1, pred_boxes_vec_2)                          # [B*B,]
    #     giou = giou.view(-1, bs)                                                           # [B,B]
    #
    #     #  使用topk函数找到每行的前两大值
    #     values, indices = giou.topk(2, dim=-1, largest=True)                               # [B,2],[B,2]
    #     bbox_similarity, bbox_index = values[:, -1],indices[:, -1]                              # [B,],[B,]
    #
    #     # similar_boxes_vec = torch.cat([pred_boxes_vec_2.view(bs, bs, 4)[i, bbox_index[i], :][None,:] for i in range(bs)], dim=0)  # [B,4]
    #     similar_feature_map = torch.cat([feature_map[bbox_index[i], :, :, :][None, :, :, :] for i in range(bs)], dim=0)  # [B,768,16,16]
    #
    #     gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
    #     gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)                                                             # [B,1,16,16]
    #     feature_map_idx = torch.nonzero(gt_gaussian_maps == 1.0)          # [B,4]-[batch_idx,channel_idx,h_idx,w_idx]
    #
    #     feature_vector = torch.cat([feature_map[i, :, feature_map_idx[i,2], feature_map_idx[i,3]][None, :] for i in range(bs)],dim=0)  # [B,768]
    #     similar_feature_vector = torch.cat([similar_feature_map[i, :, feature_map_idx[i, 2], feature_map_idx[i, 3]][None, :] for i in range(bs)],dim=0)  # [B,768]
    #     feature_similarity = F.cosine_similarity(feature_vector,similar_feature_vector)  # 【B,】
    #
    #     similarity_metric = (bbox_similarity * feature_similarity).mean()
    #
    #     return similarity_metric


    def compute_similarity(self, pred_dict, gt_dict):
        pred_features = pred_dict['backbone_feat']                   # [B,64+256,768]
        enc_opt = pred_features[:, -256:]                            # [B,256,768]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        feature_map = opt.view(-1, C, 16, 16)                        # [B,768,16,16]
        pred_boxes = pred_dict['pred_boxes']
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # [B,4]

        pred_boxes_vec_1 = pred_boxes_vec[:, None, :].repeat((1, bs, 1)).view(-1, 4)       # [B,B,4]->[B*B,4]
        pred_boxes_vec_2 = pred_boxes_vec[None,:,:].repeat((bs, 1, 1)).view(-1, 4)         # [B,B,4]->[B*B,4]
        giou = calculate_giou(pred_boxes_vec_1, pred_boxes_vec_2)                          # [B*B,]
        giou = giou.view(-1, bs)                                                           # [B,B]

        #  使用topk函数找到每行的前两大值
        values, indices = giou.topk(2, dim=-1, largest=True)                               # [B,2],[B,2]
        bbox_similarity, bbox_index = values[:, -1],indices[:, -1]                         # [B,],[B,]
        # 只保留相似度超过0.8的值
        bbox_similarity[bbox_similarity<0.8] = 0.0
        bbox_similarity[bbox_similarity>=0.8] = 1.0

        # similar_boxes_vec = torch.cat([pred_boxes_vec_2.view(bs, bs, 4)[i, bbox_index[i], :][None,:] for i in range(bs)], dim=0)  # [B,4]
        similar_feature_map = torch.cat([feature_map[bbox_index[i], :, :, :][None, :, :, :] for i in range(bs)], dim=0)  # [B,768,16,16]

        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)                                                             # [B,1,16,16]
        feature_map_idx = torch.nonzero(gt_gaussian_maps == 1.0)          # [B,4]-[batch_idx,channel_idx,h_idx,w_idx]

        feature_vector = torch.cat([feature_map[i, :, feature_map_idx[i,2], feature_map_idx[i,3]][None, :] for i in range(bs)],dim=0)  # [B,768]
        similar_feature_vector = torch.cat([similar_feature_map[i, :, feature_map_idx[i, 2], feature_map_idx[i, 3]][None, :] for i in range(bs)],dim=0)  # [B,768]
        feature_similarity = F.cosine_similarity(feature_vector,similar_feature_vector)  # [B,]

        print("==1==",bbox_similarity, feature_similarity, bbox_similarity * feature_similarity)
        similarity_metric = (bbox_similarity * feature_similarity).sum() / (bbox_similarity>=0.8).sum()

        return similarity_metric


    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        similarity_loss = - self.compute_similarity(pred_dict, gt_dict)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + similarity_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/similarity": similarity_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss