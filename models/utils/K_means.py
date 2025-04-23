import torch
from torch import nn


class KMeans(nn.Module):
    def __init__(self, num_clusters, max_iters=100, tol=1e-4):
        super(KMeans, self).__init__()
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.tol = tol

    def initialize_clusters(self, data):
        indices = torch.randperm(len(data))[:self.num_clusters]
        return data[indices]

    def forward(self, data):
        # 初始化聚类中心
        global labels
        self.centroids = self.initialize_clusters(data)

        for _ in range(self.max_iters):
            # 计算每个样本到各聚类中心的距离
            distances = torch.cdist(data, self.centroids)

            # 分配每个样本到最近的聚类中心
            labels = torch.argmin(distances, dim=1)

            # 计算新的聚类中心
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(self.num_clusters)])

            # 检查是否收敛
            if torch.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return labels
