import torch
import torch.nn as nn
import torch.nn.functional as F


def reward_shaping(pp, threshold=0.02):
    # 将 pp 缩放到 threshold 附近开始收敛
    # 当 pp = 0.02 时，pp/threshold = 1, tanh(1) ≈ 0.76
    # 当 pp 远大于 0.02 时，值趋近于 1.0
    return threshold * torch.tanh(pp / threshold)


class ProfitLoss(nn.Module):
    def __init__(self, dataset, k=10):
        super().__init__()
        self.data = dataset
        self.k = k

    """
    Loss function for
    :parameter
        x:[batch, D]
        x_k: [batch_size, k, D]
        logits: [batch_size] 
    """

    def forward(self, y_prob, indices):

        last_one_in_seq = indices[-1]  # [B, 1 ]

        x = self.data[last_one_in_seq] #[B, D]

        x_k = self.data[last_one_in_seq, self.k + last_one_in_seq]  # [B, k, D]

        # ave(sum((3* (xt-x0)/x0) -1)/t+3)
        #todo 0 represent high price
        # todo 1 represent low price
        high = x_k[: , :, 0] # shape [batch, k]
        low = x_k[: , :, 1]
        middle_price = ((high+low)/2) #[batch, k]

        middle_price_x0 = ((x[:, 0]+x[:, 1])/2).unsqueeze(1) # [batch, 1]


        # x.shape[0] == batch
        # 3 is a hyperparameter
        k = x_k.size(1)
        i_indices = torch.arange(k, device=x.device, dtype=x.dtype)
        weights = 3.0 / (i_indices + 3.0)  # [k]
        profit_k = (middle_price - middle_price_x0) / middle_price_x0 #[batch, k]
        unrealized_profit_mean_after_k = (profit_k * weights).mean(dim=1) # [batch]
        #todo stop loss set to 0.01
        stop_loss_sign = torch.sign(y_prob.unsqueeze(1)) * profit_k < - 0.01
        stop_loss_sign = torch.any(stop_loss_sign, dim=1)
        stop_loss_penalty = (stop_loss_sign * (-weights.unsqueeze(0))).sum(dim=1)
        pp = unrealized_profit_mean_after_k + stop_loss_penalty
        #todo marigin
        y_prob_steep = torch.pow(y_prob, 3)
        loss = - F.softplus(y_prob_steep * reward_shaping(pp)).mean()

        return loss






