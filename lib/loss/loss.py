import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = output
        loss = F.cross_entropy(output, target)
        return loss


class VBLoss(nn.Module):
    def __init__(self, para_dict=None):
        super(VBLoss, self).__init__()

    def weight_softmax(self, output, weight, dim=1):
        exp_out = torch.exp(output)
        weight_exp_out = exp_out * weight
        softmax_out = weight_exp_out / (torch.unsqueeze(weight_exp_out.sum(dim=dim), dim=dim))
        return softmax_out

        pass

    # 作用于内部
    def forward(self, output, target, weight=None):
        if weight is not None:
            probs = self.weight_softmax(output=output, weight=weight, dim=1)
            log_probs = torch.log(probs)
            tmp1 = -log_probs.gather(1, target.unsqueeze(-1)).squeeze()
            loss = torch.mean(tmp1)
        else:
            loss = F.cross_entropy(output, target)
        return loss


    # # 作用于内部，这个代码有问题
    # def forward(self, output, target, weight=None):
    #     if weight is not None:
    #         probs = torch.softmax(output, dim=1)
    #         weighted_probs = probs * weight.unsqueeze(0)
    #         log_probs = torch.log(weighted_probs)
    #         tmp1 = -log_probs.gather(1, target.unsqueeze(-1)).squeeze()
    #         loss = torch.sum(tmp1)
    #         loss = torch.mean(loss)
    #     else:
    #         loss = F.cross_entropy(output, target)
        
    #     if loss.data.item() < 0:
    #         print(loss.data.item())

    #     return loss

    # target = torch.tensor([0,1,0,1,0,1,0,1]).to("cuda")
    # tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()

    # # 作用于log外部
    # def forward(self, output, target, weight=None):
    #     output = output
    #     if weight is not None:
    #         loss = F.cross_entropy(output, target, weight)
    #     else:
    #         loss = F.cross_entropy(output, target)
    #     return loss

class CSCE(nn.Module):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        scheduler = cfg.LOSS.CSCE.SCHEDULER
        self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH

        if scheduler == "drw":
            self.betas = [0, 0.999999]
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]
        self.weight = None

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, x, target, **kwargs):
        return F.cross_entropy(x, target, weight= self.weight)


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).
class LDAMLoss(nn.Module):

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0

        self.s = s
        self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight= self.weight)