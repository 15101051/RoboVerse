import torch
import torch.nn as nn
import torch.nn.functional as F

class StateContrastor(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(StateContrastor, self).__init__()
        # 特征提取网络，将输入特征映射到低维空间
        self.contrastor = nn.Sequential(
            nn.Linear(input_dim - state_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.temperature = 0.1  # 对比损失的温度参数
        self.input_dim = input_dim
        self.state_dim = state_dim

    def forward(self, global_cond, labeled_state):
        """
        global_cond: 全局观测状态，Shape: (B, input_dim - state_dim)
        labeled_state: 标记状态，Shape: (B, state_dim)
        """
        # 删除 global_cond 的最后 state_dim 维
        global_cond = global_cond[:, :-self.state_dim]

        # 计算对比损失
        loss = self.compute_contrastive_loss(global_cond, labeled_state)
        return loss

    def find_positive(self, labeled_state):
        """
        随机选择一个状态作为正样本对应状态，找到批次中另一个状态相同的组成正样本对
        如果找不到相同状态样本，则更换选择的状态，重复查找
        return: 正样本对的索引, positive_idxs, 为长度为2的列表
        """
        B = labeled_state.size(0)
        positive_idxs = torch.randint(0, B, (2,))

        # 如果两个正样本状态不同，则重新选择
        while not torch.equal(labeled_state[positive_idxs[0]], labeled_state[positive_idxs[1]]):
            positive_idxs = torch.randint(0, B, (2,))


        return positive_idxs.tolist()

    def find_negative(self, labeled_state, positive_idxs):
        """
        找到批次中与选中正样本状态不同的所有样本作为负样本
        return: 负样本索引, negatives_idxs, 为长度为N的列表
        """
        B = labeled_state.size(0)
        negatives_idxs = []

        for i in range(B):
            if i not in positive_idxs:
                negatives_idxs.append(i)
        
        return negatives_idxs

    def compute_contrastive_loss(self, global_cond, labeled_state):
        """
        计算对比损失
        使用 nfoNCE loss 来计算损失
        """
        B = labeled_state.size(0)
        
        # 将全局条件特征映射到低维空间
        features = self.contrastor(global_cond)  # Shape: (B, 128)

        # 找到正样本和负样本
        positive_idxs = self.find_positive(labeled_state) 
        negative_idxs = self.find_negative(labeled_state, positive_idxs)

        # 获取正样本和负样本的特征
        positive_features = features[positive_idxs] # Shape: (2, 128)
        negative_features = features[negative_idxs] # Shape: (N, 128)

        # 如果负样本数量为0，则返回0
        if len(negative_features) == 0:
            return torch.tensor(0.0)

        # 计算 nfoNCE loss
        positive_similarity = F.cosine_similarity(positive_features[0].unsqueeze(0), positive_features[1].unsqueeze(0), dim=1)
        negative_similarity = F.cosine_similarity(positive_features[0].unsqueeze(0), negative_features, dim=1)

        # 计算对比损失
        denominator = torch.exp(positive_similarity / self.temperature) + torch.sum(torch.exp(negative_similarity / self.temperature))
        contrastive_loss = -torch.log(torch.exp(positive_similarity / self.temperature) / denominator)
        
        return contrastive_loss