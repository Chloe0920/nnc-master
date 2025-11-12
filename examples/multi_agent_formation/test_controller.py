import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
from nnc.controllers.baselines.multi_agent import MultiAgentFormationDynamics, TraditionalOptimalController
from neural_controller import NeuralFormationController

# --------------------------
# 1. 加载训练好的模型和参数（关键：无需重新训练）
# --------------------------
# 检查文件是否存在
if not os.path.exists('best_nnc_controller.pth'):
    raise FileNotFoundError("未找到NNC模型文件！请先运行 train_nnc.py 训练模型")
if not os.path.exists('nnc_params.pth'):
    raise FileNotFoundError("未找到测试参数文件！请先运行 train_nnc.py 生成参数")

# 加载参数
test_params = torch.load('nnc_params.pth')
N = test_params['N']
n_dim = test_params['n_dim']
T = test_params['T']
num_steps = test_params['num_steps']
x_star = test_params['x_star']
x_star_reshaped = test_params['x_star_reshaped']
x0 = test_params['x0']  # 测试用初始状态
L = test_params['L']
d = test_params['d']
t = torch.linspace(0, T, num_steps)  # 与训练时一致的时间序列

# 加载NNC模型
neural_controller = NeuralFormationController(N, n_dim, hidden_sizes=(256, 256))
neural_controller.load_state_dict(torch.load('best_nnc_controller.pth'))
neural_controller.eval()  # 切换到评估模式
print("NNC模型加载成功！")

# 初始化动力学模型
dynamics = MultiAgentFormationDynamics(N, n_dim)


# --------------------------
# 2. 测试控制器的通用函数（能耗计算一致）
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
            u = torch.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
            self.u_history.append(u.detach())

            # 能耗计算（∫||u||²dt，梯形积分）
            if len(self.u_history) > 1:
                t_idx = len(self.u_history) - 2
                if t_idx < len(t) - 1:
                    dt = t[t_idx + 1] - t[t_idx]
                    u_prev = torch.flatten(self.u_history[-2])
                    u_curr = torch.flatten(self.u_history[-1])
                    self.total_energy += (torch.norm(u_prev) ** 2 + torch.norm(u_curr) ** 2) * dt / 2
            return self.dyn(t_current, x, u)

    import time
    start_time = time.time()
    cl_dyn = ClosedLoopDynamics(dynamics, controller)
    x_trajectory = odeint(cl_dyn, x0, t, method='dopri5', rtol=1e-3, atol=1e-5)
    compute_time = time.time() - start_time

    # 生成累积能耗
    cumulative_energy = []
    current_energy = 0.0
    for i in range(len(t) - 1):
        if i < len(cl_dyn.u_history):
            dt = t[i + 1] - t[i]
            u = torch.flatten(cl_dyn.u_history[i])
            current_energy += torch.norm(u) ** 2 * dt
            cumulative_energy.append(current_energy.item())

    # 补全长度
    while len(cumulative_energy) < len(t) - 1:
        cumulative_energy.append(cumulative_energy[-1] if cumulative_energy else 0.0)

    return x_trajectory, cl_dyn.total_energy, compute_time, cumulative_energy


# --------------------------
# 3. 测试两种控制器（禁用梯度，加速计算）
# --------------------------
print("\n开始测试控制器...")
with torch.no_grad():
    # 测试NNC
    print("测试NNC控制器...")
    nnc_trajectory, nnc_energy, nnc_time, nnc_energy_cumulative = test_controller(
        neural_controller, dynamics, x0, t
    )

    # 测试传统最优控制器
    print("测试传统最优控制器...")
    traditional_controller = TraditionalOptimalController(L, n_dim, d)
    traditional_trajectory, traditional_energy, traditional_time, traditional_energy_cumulative = test_controller(
        traditional_controller, dynamics, x0, t
    )

# --------------------------
# 4. 输出关键指标
# --------------------------
print("\n" + "=" * 50)
print("          控制器性能对比指标")
print("=" * 50)
print(f"NNC控制器总能耗:      {nnc_energy:.4f}")
print(f"传统控制器总能耗:    {traditional_energy:.4f}")
print(f"能耗偏差率:          {abs(nnc_energy - traditional_energy) / traditional_energy * 100:.2f}%")
print(f"NNC计算时间:          {nnc_time:.4f}秒")
print(f"传统控制器计算时间:  {traditional_time:.4f}秒")
print("=" * 50)

# --------------------------
# 5. 绘制结果图（原仓库风格）
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 5.1 轨迹对比图
plt.figure(figsize=(10, 6))
for i in range(N):
    x_idx = i * n_dim
    y_idx = i * n_dim + 1

    # NNC轨迹
    nnc_x = nnc_trajectory[:, 0, 0, 0, x_idx].detach().numpy()
    nnc_y = nnc_trajectory[:, 0, 0, 0, y_idx].detach().numpy()
    plt.plot(nnc_x, nnc_y, '--', linewidth=2, label=f'NNC 智能体{i}')

    # 传统控制器轨迹
    traditional_x = traditional_trajectory[:, 0, 0, 0, x_idx].detach().numpy()
    traditional_y = traditional_trajectory[:, 0, 0, 0, y_idx].detach().numpy()
    plt.plot(traditional_x, traditional_y, '-', linewidth=1.5, label=f'传统 智能体{i}')

# 初始/最终状态
x0_reshaped = x0.squeeze().reshape(N, n_dim).detach().numpy()
nnc_final = nnc_trajectory[-1].squeeze().reshape(N, n_dim).detach().numpy()
traditional_final = traditional_trajectory[-1].squeeze().reshape(N, n_dim).detach().numpy()

plt.scatter(x0_reshaped[:, 0], x0_reshaped[:, 1], c='red', marker='o', s=80, label='初始状态')
plt.scatter(nnc_final[:, 0], nnc_final[:, 1], c='green', marker='*', s=120, label='NNC最终状态')
plt.scatter(traditional_final[:, 0], traditional_final[:, 1], c='blue', marker='*', s=120, label='传统最终状态')

plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('多智能体编队轨迹对比（加载预训练NNC模型）')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('formation_trajectory_test.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.2 能耗曲线对比
plt.figure(figsize=(10, 6))
plt.plot(t[:-1], nnc_energy_cumulative, linewidth=2.5, label='NNC 累积能耗', color='#2E86AB')
plt.plot(t[:-1], traditional_energy_cumulative, linewidth=2.5, label='传统控制器 累积能耗', color='#A23B72')
plt.xlabel('时间')
plt.ylabel('累积能耗')
plt.title('能耗对比（NNC能耗≥传统控制器，符合物理直觉）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('energy_comparison_test.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.3 计算时间对比
plt.figure(figsize=(8, 5))
controllers = ['NNC控制器', '传统最优控制器']
times = [nnc_time, traditional_time]
colors = ['#2E86AB', '#A23B72']

bars = plt.bar(controllers, times, color=colors, alpha=0.8)
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{time_val:.4f}秒', ha='center', va='bottom', fontsize=11)

plt.ylabel('计算时间 (秒)')
plt.title('控制器端到端计算时间对比（含ODE求解）')
plt.grid(axis='y', alpha=0.3)
plt.savefig('computation_time_test.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n测试完成！所有图表已保存到当前目录")