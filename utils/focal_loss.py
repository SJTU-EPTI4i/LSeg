import torch
from torch import nn
from torch.nn import functional as F

ps = [0.8, 0.14, 0.06]
gammas = [5.0, 3.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1

class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, size_average=True, device='cuda'):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.device = device

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        loss = -1 * (1-pt) ** self.gamma * logpt
        loss = -logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()