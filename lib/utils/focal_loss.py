from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-12)

        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


def calculate_focal(prediction_maps, target_maps):
    """
    :param prediction_maps: (B*N,1,16,16)
    :param target_maps: (B*N,1,16,16)
    :return:
    """
    alpha = 2
    beta = 4

    positive_index = target_maps.eq(1).float()
    negative_index = target_maps.lt(1).float()

    negative_weights = torch.pow(1 - target_maps, beta)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    prediction_maps = torch.clamp(prediction_maps, 1e-12)

    # [B*N,1,16,16]
    positive_loss = torch.log(prediction_maps) * torch.pow(1 - prediction_maps, alpha) * positive_index
    # [B*N,1,16,16]
    negative_loss = torch.log(1 - prediction_maps) * torch.pow(prediction_maps, alpha) * negative_weights * negative_index

    # [B*N,],[B*N,],[B*N,]
    num_positive = positive_index.float().sum(dim=(1,2,3))
    positive_loss = positive_loss.sum(dim=(1,2,3))
    negative_loss = negative_loss.sum(dim=(1,2,3))
    # [B*N,]
    location = (positive_loss + negative_loss) / num_positive

    return location



class FocalLoss_Elementwise(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss_Elementwise, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        # [B,1,16,16],[B,1,16,16]
        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-12)

        # [B,1,16,16],[B,1,16,16]
        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,self.alpha) * negative_weights * negative_index

        # [B,],[B,],[B,]
        num_positive = positive_index.float().sum(dim=(1,2,3))
        positive_loss = positive_loss.sum(dim=(1,2,3))
        negative_loss = negative_loss.sum(dim=(1,2,3))

        # loss: [B,]
        if num_positive.sum() == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss