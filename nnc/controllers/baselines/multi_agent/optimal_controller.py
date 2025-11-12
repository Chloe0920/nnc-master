import torch
import networkx as nx
from nnc.controllers.base import BaseController

class TraditionalOptimalController(BaseController):
    def __init__(self, laplacian, n_dim, desired_distances, device=None):
        super().__init__()
        self.L = laplacian
        self.N = laplacian.shape[0]
        self.n_dim = n_dim
        self.d = desired_distances.to(device)
        self.device = device or torch.device('cpu')

        # 特征值计算（数值稳定）
        eigenvalues, _ = torch.linalg.eigh(self.L)  # 输出纯实数，按升序排列
        eigenvalues = torch.clamp(eigenvalues, min=1e-3)  # 过滤数值误差的微小负值
        self.lambda_ = eigenvalues[1] if self.N > 1 else 1.0
        self.sqrt_lambda = torch.sqrt(self.lambda_)
        self.A = -self.L + torch.diag(torch.diag(self.L))

    def forward(self, t, x):
        """
        修复维度逻辑：用 view_as 自动匹配输出维度
        x输入形状：[batch_size, 1, 1, 1, 1, N*n_dim]
        """
        batch_size = x.shape[0]
        # 强制展平x（保留batch维度）→ [batch_size, N*n_dim]
        x_flat = torch.flatten(x, start_dim=1)
        # 重塑为 [batch_size, N, n_dim]
        x_reshaped = x_flat.view(batch_size, self.N, self.n_dim)

        sqrt_lambda = torch.sqrt(torch.clamp(self.lambda_, min=1e-6))
        # 初始化u：与x维度完全一致
        u = torch.zeros_like(x, device=self.device)
        # 展平u用于计算 → [batch_size, N*n_dim]
        u_flat = u.view(batch_size, -1)

        for i in range(self.N):
            sum_term = torch.zeros(batch_size, self.n_dim, device=self.device)
            for j in range(self.N):
                if self.A[i, j] > 0:
                    diff = x_reshaped[:, i] - x_reshaped[:, j] - self.d[i, j]
                    sum_term += self.A[i, j] * diff
            # 赋值给u_flat的对应切片
            u_flat[:, i * self.n_dim : (i + 1) * self.n_dim] = -sum_term / sqrt_lambda

        # 恢复u的原始维度（与x一致）
        return u_flat.view_as(x)