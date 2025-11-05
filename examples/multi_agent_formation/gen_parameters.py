# 这个文件用于生成参数
import torch
import networkx as nx

def generate_initial_state_batch(n_agents, n_dim, batch_size=5, noise_scale=0.5):
    """生成批量初始状态（带随机扰动）"""
    state_dim = n_agents * n_dim
    x0_batch = []
    for _ in range(batch_size):
        x0 = torch.randn(1, 1, state_dim) * noise_scale
        x0_batch.append(x0)
    return x0_batch

def generate_laplacian(n_agents):
    """生成完全图拉普拉斯矩阵（与main.py逻辑一致，统一管理）"""
    G = nx.complete_graph(n_agents)
    return torch.tensor(nx.laplacian_matrix(G).todense(), dtype=torch.float32)

def generate_desired_distances(x_star_reshaped):
    """根据目标编队生成期望距离矩阵（统一管理）"""
    N = x_star_reshaped.shape[0]
    n_dim = x_star_reshaped.shape[1]
    d = torch.zeros(N, N, n_dim)
    for i in range(N):
        for j in range(N):
            d[i, j] = x_star_reshaped[i] - x_star_reshaped[j]
    return d