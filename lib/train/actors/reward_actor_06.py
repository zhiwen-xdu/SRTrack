# Deterministic Policy Gradient（DPG）方法:
# 1) 对GT Bbox随机采样，使用Critic获到Reward最高的Bbox,把它当做GT BBox;
# 2) 使用采样得到的最好BBox中心坐标生成对应Map,把它当做GT Map;
import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_xywh
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu

from lib.utils.box_ops import calculate_giou
l1_elementwise = torch.nn.L1Loss(reduction='none')


class RewardActor06(BaseActor):
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


    def add_noise(self, gt_boxes, num_samples=256, box_scale=0.02):
        """
        Sample from a gaussian noise the size and offset near the given boxes.
        :param gt_boxes: (B,4) [x,y,w,h]
        :param box_scale: (float)
        :output: (B, N, 4) N-Sampled Bbox Num
        """
        assert gt_boxes.shape[-1] == 4
        batch_size = gt_boxes.shape[0]
        boxes_noise = torch.randn(batch_size, num_samples, 2).to(gt_boxes.dtype).to(gt_boxes.device) * box_scale   # [B,N,4]
        boxes_noise = torch.clamp(boxes_noise, min=-0.02, max=0.02)
        boxes_out = gt_boxes.detach()
        boxes_out = boxes_out[:, None, :].repeat((1, num_samples, 1))                # [B,N,4]
        boxes_out[:,:,0:2] = boxes_out[:,:,0:2] + boxes_noise
        boxes_out = torch.clamp(boxes_out, min=0.0, max=1.0)
        boxes_out = boxes_out.view(-1, 4)                          # [B*N,4]

        return boxes_out

    # Note: box_xywh_to_xyxy, box_cxcywh_to_xyxy两个函数不能弄错
    def reward_function(self, gt_boxes, pred_boxes, num_samples=255):
        # gt_boxes: [B*N,4]- [x,y,w,h], N表示采样的Bbox数;
        # pred_boxes: [B,1,4]- [x,y,w,h];
        # best_boxes: [B,4]- [x,y,x,y];
        # best_maps: [B,1,H,W]
        gt_boxes = box_xywh_to_xyxy(gt_boxes)  # [B*N,4]
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes).repeat((1, num_samples, 1)).view(-1, 4)        # [B*N,4]
        boxes = gt_boxes.view(-1, num_samples, 4)  # [B,N,4]

        # 取N个Sampled Bboxs中GIoU+L1 Reward最高的Bbox
        giou = calculate_giou(gt_boxes, pred_boxes)   # [B*N,]
        giou = giou.view(-1, num_samples)             # [B,N]
        l1 = l1_elementwise(gt_boxes, pred_boxes)     # [B*N,4]
        l1 = l1.mean(dim=1)                           # [B*N,]
        l1 = - l1.view(-1, num_samples)               # [B,N]
        box_reward = giou + 2.5 * l1                  # [B,N]
        bbox_index = torch.argmax(box_reward, dim=1)  # [B,]
        best_boxes = torch.cat([boxes[i, bbox_index[i], :][None, :] for i in range(len(bbox_index))], dim=0)  # [B,4]

        best_center_boxes = box_xyxy_to_xywh(best_boxes)
        best_maps = generate_heatmap([best_center_boxes], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        best_maps = best_maps[-1].unsqueeze(1)

        return best_boxes, best_maps


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
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # [B*N,4]
        sampled_boxes = self.add_noise(gt_bbox, num_samples=512)
        # [B,4]-[x,y,x,y], [B,1,H,W]
        best_boxes,best_maps = self.reward_function(sampled_boxes, pred_boxes, num_samples=512)
        # compute loss with sample-label
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, best_boxes)    # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, best_boxes)                 # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], best_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # compute loss with gt-label
        try:
            giou_loss_gt, iou_gt = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss_gt, iou_gt = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss_gt = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss_gt = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss_gt = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        # loss = self.loss_weight['giou'] * giou_loss_gt + self.loss_weight['l1'] * l1_loss_gt + self.loss_weight['focal'] * location_loss_gt
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        if return_status:
            # status for log
            mean_iou = iou_gt.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/giou-gt": giou_loss_gt.item(),
                      "Loss/l1-gt": l1_loss_gt.item(),
                      "Loss/location-gt": location_loss_gt.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss