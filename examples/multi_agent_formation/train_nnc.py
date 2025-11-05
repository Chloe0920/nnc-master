import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint

# 导入自定义模块
from nnc.controllers.baselines.multi_agent import MultiAgentFormationDynamics
from neural_controller import NeuralFormationController
from nnc.controllers.neural_network.nnc_controllers import NNCDynamics
from gen_parameters import generate_initial_state_batch, generate_laplacian, generate_desired_distances


# --------------------------
# 修复：手动实现克罗内克积（修正维度广播逻辑，兼容2D矩阵）
# --------------------------
def kron(A, B):
    A = A.squeeze().view(-1, A.shape[-1])  # [N, N]
    B = B.squeeze().view(-1, B.shape[-1])  # [n, n]

    N = A.shape[0]
    n = B.shape[0]

    A_expanded = A.unsqueeze(1).unsqueeze(3)  # [N, 1, N, 1]
    B_expanded = B.unsqueeze(0).unsqueeze(2)  # [1, n, 1, n]
    kron_prod = A_expanded * B_expanded
    kron_prod = kron_prod.permute(0, 1, 2, 3).reshape(N * n, N * n)
    return kron_prod


# --------------------------
# 1. 训练参数设置（核心：提高学习率+优化调度策略）
# --------------------------
N = 10  # 10个智能体
n_dim = 2  # 2D平面（n=2）
T = 10.0  # 仿真时间
num_steps = 100  # 时间步数
t = torch.linspace(0, T, num_steps)
batch_size = 16  # 批量大小
noise_scale = 0.5  # 大初始扰动（±2.5）
ode_tol = (1e-4, 1e-6)  # ODE求解精度
epochs = 20000  # 训练轮次（不变）
early_stop_patience = 300  # 早停耐心值（不变）

# 核心修改：提高初始学习率（从0.001→0.005）
initial_lr = 0.001
weight_decay = 1e-5  # 保持权重衰减，防过拟合

# 目标编队：半径5的正圆形（10智能体均匀分布）
x_star_reshaped = []
radius = 0.5
for i in range(N):
    angle = 2 * np.pi * i / N
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    x_star_reshaped.append([x, y])
x_star_reshaped = torch.tensor(x_star_reshaped, dtype=torch.float32)  # [10,2]
x_star = x_star_reshaped.flatten().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,20]

# 生成关键参数
x0_batch_list = generate_initial_state_batch(N, n_dim, batch_size, noise_scale)
x0_batch = torch.cat([x.unsqueeze(0) for x in x0_batch_list], dim=0)
L = generate_laplacian(N)  # 10×10拉普拉斯矩阵
I_n = torch.eye(n_dim, dtype=torch.float32)  # 2×2单位矩阵
L_kron_I = kron(L, I_n)  # 20×20核心加权矩阵
d = generate_desired_distances(x_star_reshaped)

# 保存测试参数
test_params = {
    'N': N,
    'n_dim': n_dim,
    'T': T,
    'num_steps': num_steps,
    'x_star': x_star,
    'x_star_reshaped': x_star_reshaped,
    'x0': x0_batch[0:1, :, :, :],
    'L': L,
    'd': d
}
torch.save(test_params, 'nnc_params.pth')
print("测试参数已保存到 'nnc_params.pth'")
print(f"拉普拉斯矩阵L维度：{L.shape}")
print(f"克罗内克积L⊗I_n维度：{L_kron_I.shape}")


# --------------------------
# 2. 核心：损失函数（不变）
# --------------------------
def nnc_formation_loss(x_trajectory, t, L_kron_I, x_star):
    total_loss = 0.0
    L_kron_I = L_kron_I.to(x_trajectory.device)
    x_star = x_star.to(x_trajectory.device)

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        x_t = x_trajectory[i].squeeze().view(-1, L_kron_I.shape[0])  # [batch,20]
        x_t1 = x_trajectory[i + 1].squeeze().view(-1, L_kron_I.shape[0])  # [batch,20]

        error_t = x_t - x_star.squeeze()
        error_t1 = x_t1 - x_star.squeeze()

        quadratic_t = torch.matmul(torch.matmul(error_t.unsqueeze(1), L_kron_I), error_t.unsqueeze(-1)).squeeze()
        quadratic_t1 = torch.matmul(torch.matmul(error_t1.unsqueeze(1), L_kron_I), error_t1.unsqueeze(-1)).squeeze()

        loss_t = quadratic_t.mean()
        loss_t1 = quadratic_t1.mean()
        total_loss += (loss_t + loss_t1) * dt / 2

    return total_loss


# --------------------------
# 3. 初始化训练组件（核心：优化器+调度器调整）
# --------------------------
dynamics = MultiAgentFormationDynamics(N, n_dim)
neural_controller = NeuralFormationController(N, n_dim, hidden_sizes=(256, 256))
nnc_dynamics = NNCDynamics(dynamics, neural_controller)

# 优化器：提高初始学习率（0.005）
optimizer = optim.Adam(
    neural_controller.parameters(),
    lr=initial_lr,
    weight_decay=weight_decay  # 保持权重衰减，防止高学习率过拟合
)

# 学习率调度器：优化策略（加速收敛）
# factor=0.7（衰减更快）、patience=30（更早衰减）、threshold=1e-4（对损失变化更敏感）
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=30, threshold=1e-4,
    min_lr=1e-7, verbose=True  # verbose=True：打印学习率变化日志
)

best_loss = float('inf')
early_stop_count = 0

# --------------------------
# 4. 训练循环（新增：梯度裁剪，可选但推荐）
# --------------------------
print(f"开始训练NNC控制器（{epochs}轮，10智能体圆形编队）...")
print(f"初始学习率：{initial_lr}，学习率调度：factor=0.7, patience=30")
print(f"损失函数：J_NNC = ∫₀^T (x(t)-x*)^T (L⊗I_n) (x(t)-x*) dt")

for epoch in tqdm(range(epochs), desc="训练进度", unit="epoch"):
    optimizer.zero_grad()

    # 批量计算轨迹
    x_trajectory_batch = odeint(
        nnc_dynamics, x0_batch, t, method='dopri5',
        rtol=ode_tol[0], atol=ode_tol[1]
    )

    # 计算损失
    total_loss = nnc_formation_loss(x_trajectory_batch, t, L_kron_I, x_star)

    # 反向传播
    total_loss.backward()

    # 可选但推荐：梯度裁剪（防止高学习率导致梯度爆炸）
    torch.nn.utils.clip_grad_norm_(neural_controller.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step(total_loss.detach())

    # 早停机制+保存最优模型
    if total_loss.item() < best_loss - 1e-6:
        best_loss = total_loss.item()
        early_stop_count = 0
        torch.save(neural_controller.state_dict(), 'best_nnc_controller.pth')
    else:
        early_stop_count += 1
        if early_stop_count >= early_stop_patience:
            print(f"\n早停触发！Epoch {epoch + 1}，最佳Loss={best_loss:.6f}")
            break

    # 每100轮输出一次日志（更频繁监控收敛情况）
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1} | NNC Formation Loss: {total_loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.7f}")

# 训练完成日志
print("\n训练完成！")
print(f"最优模型已保存到 'best_nnc_controller.pth'")
print(f"最佳NNC损失（J_NNC）：{best_loss:.6f}")