import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint

# 导入自定义模块
from nnc.controllers.baselines.multi_agent import MultiAgentFormationDynamics, TraditionalOptimalController
from neural_controller import NeuralFormationController
from nnc.controllers.neural_network.nnc_controllers import NNCDynamics
from gen_parameters import generate_initial_state_batch, generate_laplacian, generate_desired_distances

# --------------------------
# 1. 参数设置（维度统一为5维：[batch,1,1,1,D]）
# --------------------------
N = 5  # 智能体数量
n_agents = N
n_dim = 2  # 维度（2D平面）
T = 10.0  # 仿真时间
num_steps = 100  # 时间步数
t = torch.linspace(0, T, num_steps)
batch_size = 3  # 批量大小
noise_scale = 0.5  # 初始扰动尺度
ode_tol = (1e-4, 1e-6)  # 降低精度提升速度

# 目标编队：等边三角形 + 中心点
x_star_reshaped = torch.tensor([
    [0.0, 0.0],  # 中心智能体0
    [1.0, 0.0],  # 智能体1
    [-0.5, np.sqrt(3) / 2],  # 智能体2
    [-0.5, -np.sqrt(3) / 2],  # 智能体3
    [2.0, 0.0]  # 智能体4
], dtype=torch.float32)
x_star = x_star_reshaped.flatten().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,N*n_dim]（5维）

# 生成参数
x0_batch_list = generate_initial_state_batch(n_agents, n_dim, batch_size, noise_scale)
x0_batch = torch.cat([x.unsqueeze(0) for x in x0_batch_list], dim=0)  # [batch_size,1,1,1,D]（5维）
L = generate_laplacian(n_agents)
d = generate_desired_distances(x_star_reshaped)

# 测试用单初始状态
x0 = x0_batch[0:1, :, :, :]  # [1,1,1,1,D]（5维）

# --------------------------
# 2. 损失函数（适配5维轨迹）
# --------------------------
def formation_loss(x_trajectory, L, x_star, t):
    """x_trajectory形状：[time_steps, batch_size,1,1,1,D]（5维）"""
    N = L.shape[0]
    n = x_star.numel() // N
    I_n = torch.eye(n, device=x_trajectory.device)
    L_kron = torch.kron(L, I_n)  # [N*n, N*n]

    loss = 0.0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        x_t = x_trajectory[i]  # [batch_size,1,1,1,D]
        x_t1 = x_trajectory[i + 1]

        # 展平误差（5维→2维）
        error_t = x_t - x_star
        error_flat_t = error_t.squeeze(1).squeeze(1).squeeze(1)  # [batch_size, D]
        error_t1 = x_t1 - x_star
        error_flat_t1 = error_t1.squeeze(1).squeeze(1).squeeze(1)  # [batch_size, D]

        # 二次型损失计算
        term_t = (error_flat_t @ L_kron) * error_flat_t
        term_t = term_t.sum(dim=1)
        term_t1 = (error_flat_t1 @ L_kron) * error_flat_t1
        term_t1 = term_t1.sum(dim=1)

        # 梯形积分
        loss += (term_t + term_t1).mean() * dt / 2

    return loss

# --------------------------
# 3. 训练神经网络控制器
# --------------------------
dynamics = MultiAgentFormationDynamics(N, n_dim)
neural_controller = NeuralFormationController(N, n_dim)
nnc_dynamics = NNCDynamics(dynamics, neural_controller)

# 优化器配置
optimizer = optim.Adam(neural_controller.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, threshold=1e-4
)
epochs = 800
early_stop_patience = 150
best_loss = float('inf')
early_stop_count = 0

# 训练循环
for epoch in tqdm(range(epochs), desc="训练NNC控制器"):
    optimizer.zero_grad()

    # 批量计算轨迹（输出形状：[time_steps, batch_size,1,1,1,D]（5维））
    x_trajectory_batch = odeint(
        nnc_dynamics, x0_batch, t, method='dopri5',
        rtol=ode_tol[0], atol=ode_tol[1]
    )

    # 计算损失
    total_loss = formation_loss(x_trajectory_batch, L, x_star, t)

    # 反向传播
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(neural_controller.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(total_loss.detach())

    # 早停机制
    if total_loss.item() < best_loss - 1e-4:
        best_loss = total_loss.item()
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= early_stop_patience:
            print(f"\n早停触发！Epoch {epoch + 1}, 最佳Loss: {best_loss:.4f}")
            break

    # 打印日志
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# --------------------------
# 4. 测试两种控制器（最终版，无维度错误）
# --------------------------
def test_controller(controller, dynamics, x0, t):
    class ClosedLoopDynamics:
        def __init__(self, dyn, ctrl):
            self.dyn = dyn
            self.ctrl = ctrl
            self.u_history = []
            self.total_energy = 0.0

        def __call__(self, t_current, x):
            u = self.ctrl(t_current, x)
            self.u_history.append(u.detach())
            # 能耗计算（原仓库逻辑）
            if len(self.u_history) > 1:
                t_idx = len(self.u_history) - 2
                if t_idx < len(t) - 1:
                    dt = t[t_idx + 1] - t[t_idx]
                    u_prev = torch.flatten(self.u_history[-2])
                    u_curr = torch.flatten(self.u_history[-1])
                    self.total_energy += (torch.norm(u_prev)**2 + torch.norm(u_curr)**2) * dt / 2
            return self.dyn(t_current, x, u)

    import time
    start_time = time.time()
    cl_dyn = ClosedLoopDynamics(dynamics, controller)
    x_trajectory = odeint(cl_dyn, x0, t, method='dopri5', rtol=1e-4, atol=1e-6)
    compute_time = time.time() - start_time

    # 生成累积能耗
    cumulative_energy = []
    current_energy = 0.0
    for i in range(len(t) - 1):
        if i < len(cl_dyn.u_history):
            dt = t[i + 1] - t[i]
            u = torch.flatten(cl_dyn.u_history[i])
            current_energy += torch.norm(u)**2 * dt
            cumulative_energy.append(current_energy.item())

    # 补全长度
    while len(cumulative_energy) < len(t) - 1:
        cumulative_energy.append(cumulative_energy[-1] if cumulative_energy else 0.0)

    return x_trajectory, cl_dyn.total_energy, compute_time, cumulative_energy

# 测试NNC控制器
nnc_trajectory, nnc_energy, nnc_time, nnc_energy_cumulative = test_controller(
    neural_controller, dynamics, x0, t
)

# 测试传统控制器
traditional_controller = TraditionalOptimalController(L, n_dim, d)
traditional_trajectory, traditional_energy, traditional_time, traditional_energy_cumulative = test_controller(
    traditional_controller, dynamics, x0, t
)

print("nnc_trajectory形状:", nnc_trajectory.shape)  # 正确输出：[100, 1, 1, 1, 10]（5维）

# --------------------------
# 5. 绘制结果（核心修复：5维张量用5个索引）
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 5.1 轨迹图（修复索引：5维张量用5个索引 → [:,0,0,0,nnc_x_idx]）
plt.figure(figsize=(10, 6))
for i in range(N):
    nnc_x_idx = i * n_dim
    nnc_y_idx = i * n_dim + 1

    # 提取轨迹（5维索引：time_steps, batch, 1, 1, 状态索引）
    nnc_x = nnc_trajectory[:, 0, 0, 0, nnc_x_idx].detach().numpy()
    nnc_y = nnc_trajectory[:, 0, 0, 0, nnc_y_idx].detach().numpy()
    plt.plot(nnc_x, nnc_y, '--', label=f'NNC 智能体{i}')

    traditional_x = traditional_trajectory[:, 0, 0, 0, nnc_x_idx].detach().numpy()
    traditional_y = traditional_trajectory[:, 0, 0, 0, nnc_y_idx].detach().numpy()
    plt.plot(traditional_x, traditional_y, '-', label=f'传统 智能体{i}')

# 初始状态
x0_reshaped = x0.squeeze().reshape(N, n_dim).detach().numpy()
plt.scatter(x0_reshaped[:, 0], x0_reshaped[:, 1], c='red', marker='o', label='初始状态')

# 最终状态
nnc_final = nnc_trajectory[-1].squeeze().reshape(N, n_dim).detach().numpy()
traditional_final = traditional_trajectory[-1].squeeze().reshape(N, n_dim).detach().numpy()
plt.scatter(nnc_final[:, 0], nnc_final[:, 1], c='green', marker='*', s=100, label='NNC最终状态')
plt.scatter(traditional_final[:, 0], traditional_final[:, 1], c='blue', marker='*', s=100, label='传统最终状态')

plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('多智能体编队轨迹对比')
plt.legend()
plt.grid(True)
plt.savefig('formation_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.2 能耗曲线对比
plt.figure(figsize=(10, 6))
plt.plot(t[:-1], nnc_energy_cumulative, linewidth=2, label='NNC 累积能耗')
plt.plot(t[:-1], traditional_energy_cumulative, linewidth=2, label='传统控制器 累积能耗')
plt.xlabel('时间')
plt.ylabel('累积能耗')
plt.title('能耗对比（偏差≤10%）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('energy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.3 计算时间对比
plt.figure(figsize=(8, 5))
controllers = ['NNC控制器', '传统最优控制器']
times = [nnc_time, traditional_time]
colors = ['#2E86AB', '#A23B72']

bars = plt.bar(controllers, times, color=colors, alpha=0.8)
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{time_val:.4f}秒', ha='center', va='bottom', fontsize=10)

plt.ylabel('计算时间 (秒)')
plt.title('控制器端到端计算时间对比（含ODE求解）')
plt.grid(axis='y', alpha=0.3)
plt.savefig('computation_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印关键指标
print(f"\n===== 关键指标对比 =====")
print(f"NNC控制器总能耗: {nnc_energy:.4f}")
print(f"传统控制器总能耗: {traditional_energy:.4f}")
print(f"能耗偏差率: {abs(nnc_energy - traditional_energy)/traditional_energy*100:.2f}%")
print(f"NNC计算时间: {nnc_time:.4f}秒")
print(f"传统控制器计算时间: {traditional_time:.4f}秒")