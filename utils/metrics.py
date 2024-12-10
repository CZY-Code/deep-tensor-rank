import torch

def chamfer_distance_and_f_score(P, Q, threshold=0.01, scaler=1.0):
    """
    计算两个点云之间的 Chamfer 距离和 F-score。

    参数:
    P (torch.Tensor): 形状为 [N, D] 的点云，其中 N 是点的数量，D 是点的维度。
    Q (torch.Tensor): 形状为 [M, D] 的点云，其中 M 是点的数量，D 是点的维度。
    threshold (float): 匹配点的最大距离阈值。

    返回:
    chamfer_dist (float): Chamfer 距离。
    f_score (float): F-score。
    precision (float): 精确率。
    recall (float): 召回率。
    """
    # 计算最近距离
    dist_PQ = torch.cdist(P, Q, p=2.0)  # 形状为 [N, M]
    min_dist_PQ, _ = dist_PQ.min(dim=1) # 形状为 [N]
    dist_QP = torch.cdist(Q, P, p=2.0)  # 形状为 [M, N]
    min_dist_QP, _ = dist_QP.min(dim=1) # 形状为 [M]

    # 计算 Chamfer 距离
    sum_min_dist_PQ = min_dist_PQ.sum()
    sum_min_dist_QP = min_dist_QP.sum()
    chamfer_dist = (sum_min_dist_PQ / P.shape[0] + sum_min_dist_QP / Q.shape[0]) * (scaler**2)
    # chamfer_dist = (sum_min_dist_PQ / P.shape[0] + sum_min_dist_QP / Q.shape[0]) * (1.0**2)

    # 计算 F-score
    correct_pred = (min_dist_PQ < (threshold*scaler)).float()
    precision = correct_pred.mean().item()

    correct_gt = (min_dist_QP < (threshold*scaler)).float()
    recall = correct_gt.mean().item()

    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0.0


    return chamfer_dist.item(), f_score, precision, recall