import torch
import torch.nn as nn
import torch.nn.functional as F

def bce_loss(input, target, reduce=True):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.
    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    if reduce:
        return loss.mean()
    else:
        return loss


def calculate_model_losses(args, pred, target, name, angles=None, angles_pred=None, mu=None, logvar=None,
                           KL_weight=None, writer=None, counter=None, withangles=False):
    #args, orig_gt_box, orig_box, name='box', withangles=args.with_angles, angles_pred=orig_angle,
    #mu=mu_box, logvar=logvar_box, angles=orig_gt_angle,
    #KL_weight=0.1, writer=writer, counter=counter
    total_loss = 0.0
    losses = {}
    rec_loss = F.l1_loss(pred, target)#! 计算重建损失（rec_loss）为预测值（pred）与目标值（target）之间的L1损失
    total_loss = add_loss(total_loss, rec_loss, losses, name, 1)
    #if withangles:
    #    angle_loss = F.nll_loss(angles_pred, angles)
    #    total_loss = add_loss(total_loss, angle_loss, losses, 'angle_pred', 1)

    try: #! KL散度损失，用于度量预测分布与目标分布之间的差异。
        loss_gauss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    except:
        print("blowup!!!")
        print("logvar", torch.sum(logvar.data), torch.sum(torch.abs(logvar.data)), torch.max(logvar.data),
              torch.min(logvar.data))
        print("mu", torch.sum(mu.data), torch.sum(torch.abs(mu.data)), torch.max(mu.data), torch.min(mu.data))
        return total_loss, losses
    total_loss = add_loss(total_loss, loss_gauss, losses, 'KLD_Gauss', KL_weight)

    writer.add_scalar('Train Loss KL {}'.format(name), loss_gauss, counter)
    writer.add_scalar('Train Loss Rec {}'.format(name), rec_loss, counter)
    #if withangles:
    #    writer.add_scalar('Train Loss Angle {}'.format(name), angle_loss, counter)
    return total_loss, losses


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss_weighted = curr_loss * weight
    loss_dict[loss_name] = curr_loss_weighted.item()
    if total_loss is not None:
        return total_loss + curr_loss_weighted
    else:
        return curr_loss_weighted
