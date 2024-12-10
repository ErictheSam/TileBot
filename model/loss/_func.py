import math
import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage.morphology import distance_transform_edt as edt

__all__ = [
    "focal_neg_loss_with_logits",
    "weighted_bce_with_logits",
    "hausdorff_recall_loss"
]

device = 0

def __calculate_value(gt_tensor, alpha, beta):
    with torch.no_grad():
        field = np.zeros_like(gt_tensor)
        for i in range(len(gt_tensor)):
            bg_mask = 1 - gt_tensor[i]
            bg_field = edt(bg_mask)
            bg_field[bg_field < 10] = 0
            field[i] = bg_field ** alpha
        return field * beta


def focal_neg_loss_with_logits(preds, gt, alpha=2, beta=4):
    """
    borrow from https://github.com/princeton-vl/CornerNet
    """
    epsilon = 1e-8

    preds = torch.sigmoid(preds)

    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
#     pos_inds = gt.gt(0)
#     neg_inds = gt.eq(0)

    neg_weights = torch.pow(1 - gt[neg_inds], beta)

    loss = 0
    pos_pred = preds[pos_inds]
    neg_pred = preds[neg_inds]

    # print(pos_pred, neg_pred, pos_pred.max(), neg_pred.max())


    pos_loss = torch.log(pos_pred + epsilon) * torch.pow(1 - pos_pred, alpha)
    neg_loss = torch.log(1 - neg_pred + epsilon) * torch.pow(neg_pred, alpha) * neg_weights
    
    field_map = torch.from_numpy(__calculate_value(gt.detach().cpu().numpy(), 2.0, 0.5)).cuda(device).float()[neg_inds]
    # print(field_map.shape, neg_loss.shape)
    neg_loss = neg_loss * (1. + field_map)

    num_pos = pos_inds.float().sum()
    pos_loss = 10 * pos_loss.sum()
    # print(neg_loss)
    neg_loss = neg_loss.sum()
    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


# def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
#     pos_mask = torch.where(gt == 1, torch.ones_like(gt), torch.zeros_like(gt))
#     neg_mask = torch.ones_like(pos_mask) - pos_mask

#     losses = F.binary_cross_entropy_with_logits(out, gt, reduction='none')

#     loss_neg = (losses * neg_mask).sum() / (torch.sum(neg_mask))
#     loss_v = loss_neg * neg_w

#     pos_sum = torch.sum(pos_mask)
#     if pos_sum != 0:
#         loss_pos = (losses * pos_mask).sum() / pos_sum
#         loss_v += (loss_pos * pos_w)
#     return loss_v


def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
    pos_mask = torch.where(gt != 0.0, torch.ones_like(gt), torch.zeros_like(gt))
    #pos_mask = torch.where(gt == 1, torch.ones_like(gt), torch.zeros_like(gt))
    neg_mask = torch.ones_like(pos_mask) - pos_mask
    loss = F.binary_cross_entropy_with_logits(out, gt, reduction='none')
    loss_pos = (loss * pos_mask).sum() / ( torch.sum(pos_mask) + 1e-5)
    loss_neg = (loss * neg_mask).sum() / ( torch.sum(neg_mask) + 1e-5)
    loss = loss_pos * pos_w + loss_neg * neg_w
    return loss

def __distance_field(img):
    """
    Thanks to github base
    """
    with torch.no_grad():

        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5
            
            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

def hausdorff_recall_loss(pred, target, alpha=0.01):
    """
    Uses one binary channel: 1 - fg, 0 - bg
    pred: (b, x, y, z) or (b, x, y)
    target: (b, x, y, z) or (b, x, y)
    """
    # assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
    assert (
        pred.dim() == target.dim()
    ), "Prediction and target need to be of same dimension"

    pred = torch.sigmoid(pred)

    target_dt = torch.from_numpy(__distance_field(target.detach().cpu().numpy())).cuda(device).float()

    pred_error = (pred - target) ** 2
    distance = target_dt ** alpha
    # print(pred_error.mean(), distance.mean())
    dt_field = pred_error * distance
    loss = dt_field.mean()
    return loss