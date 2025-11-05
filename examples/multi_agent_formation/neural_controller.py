import torch
from nnc.controllers.neural_network.nnc_controllers import NeuralNetworkController  # 继承原仓库NNC基类

class NeuralFormationController(NeuralNetworkController):
    def __init__(self, n_agents, n_dim, hidden_sizes=(256, 256), time_encoding=True):
        # 调用原仓库基类初始化（确保接口与原仓库一致）
        super().__init__(neural_net=self._build_network(n_agents, n_dim, hidden_sizes, time_encoding))
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.time_encoding = time_encoding
        self.input_dim = n_agents * n_dim + (2 if time_encoding else 1)  # 时间编码用2维（sin+cos）
        self.output_dim = n_agents * n_dim

    def _build_network(self, n_agents, n_dim, hidden_sizes, time_encoding):
        """完全对齐原仓库EluTimeControl的网络结构"""
        input_dim = n_agents * n_dim + (2 if time_encoding else 1)
        output_dim = n_agents * n_dim

        layers = []
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(torch.nn.ELU())  # 原仓库默认激活
        # 移除最后一层ELU，添加tanh激活（原仓库核心：限制控制输入平滑性）
        if layers and layers[-1].__class__.__name__ == 'ELU':
            layers.pop()
        layers.append(torch.nn.Tanh())  # 关键：避免控制输入突变，让轨迹平滑
        return torch.nn.Sequential(*layers)

    def forward(self, t, x):
        """严格对齐原仓库输入输出逻辑"""
        batch_size = x.shape[0]
        # 1. 展平状态x：[batch,1,1,1,D] → [batch,D]（原仓库标准操作）
        x_flat = torch.flatten(x, start_dim=1)  # 原仓库用flatten，而非多次squeeze

        # 2. 时间编码（原仓库核心：提升时变控制能力）
        if self.time_encoding:
            t = t.to(x.device)
            t_expand = t.expand(batch_size, 1)
            t_feat = torch.cat([torch.sin(t_expand), torch.cos(t_expand)], dim=1)  # 周期特征
        else:
            t_feat = t.expand(batch_size, 1).to(x.device)

        # 3. 拼接输入特征（状态+时间编码）
        input_feat = torch.cat([x_flat, t_feat], dim=1)

        # 4. 前向计算+恢复维度（与x完全一致）
        u = self.neural_net(input_feat)
        return u.view_as(x)  # 原仓库用view_as确保维度对齐，而非手动unsqueeze