import torch
import torch.nn as nn


class TransRTLoss(nn.Module):
    def __init__(self, norm=2):
        super(TransRTLoss, self).__init__()
        self.norm = norm
    def forward(self, user, tag, item, require_pow=False):
        if require_pow:
            pass
        else:
            transrt_loss = torch.norm(user + tag - item, p=2)
            return transrt_loss


class CosSim(nn.Module):
    def __init__(self, norm=2):
        super(CosSim, self).__init__()
        self.norm = norm

    def forward(self, user, item):
        user_norm = torch.norm(user, p=2, dim=1).reshape(-1, 1)
        item_norm = torch.norm(item, p=2, dim=1).reshape(-1, 1)

        norm_mat = user_norm.matmul(item_norm.T)
        score = user.matmul(item.T)
        cos_sim = score / norm_mat
        return cos_sim
