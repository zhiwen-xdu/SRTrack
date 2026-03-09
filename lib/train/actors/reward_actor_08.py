import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu


from lib.utils.box_ops import giou_loss_elementwise
from lib.utils.focal_loss import FocalLoss_Elementwise
focal_loss_elementwise = FocalLoss_Elementwise()
l1_loss_elementwise = torch.nn.L1Loss(reduction='none')



class RewardActor08(BaseActor):
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
        # compute giou and iou
        try:
            giou_loss, iou = giou_loss_elementwise(pred_boxes_vec, gt_boxes_vec)  # [B,],[B,]
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = l1_loss_elementwise(pred_boxes_vec, gt_boxes_vec)             # [B,4]
        l1_loss = l1_loss.sum(dim=1)                                            # [B,]
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = focal_loss_elementwise(pred_dict['score_map'], gt_gaussian_maps)  # [B,]
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # [B,]
        loss_vector = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        B = loss_vector.shape[0]
        # # Normalization Importance
        importance = B * loss_vector / torch.sum(loss_vector)
        importance_clamp = importance.clamp(min=0.75, max=1.25)
        importance_vector = B * importance_clamp / torch.sum(importance_clamp)
        # # sample-level importance weighted sum
        loss = (loss_vector * importance_vector).mean()

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": (giou_loss * importance_vector).mean().item(),
                      "Loss/l1": (l1_loss * importance_vector).mean().item(),
                      "Loss/location": (location_loss * importance_vector).mean().item(),
                      "Loss/giou_nore": giou_loss.mean().item(),
                      "Loss/l1_nore": l1_loss.mean().item(),
                      "Loss/location_nore": location_loss.mean().item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss