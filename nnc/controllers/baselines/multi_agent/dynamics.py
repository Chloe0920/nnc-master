import torch    #导入 PyTorch 库，用于张量运算和自动微分（整个代码库基于 PyTorch 实现）。
from nnc.controllers.base import ControlledDynamics    #从原仓库的基础模块中导入ControlledDynamics基类。作用：这是原仓库中所有 “带控制输入的动力学模型” 的父类，继承它可以确保我们的自定义动力学模型能与仓库中的控制器（如神经网络控制器）无缝对接。

class MultiAgentFormationDynamics(ControlledDynamics):     #定义一个名为MultiAgentFormationDynamics的类，继承自ControlledDynamics。继承的意义：通过继承基类，我们的类会自动获得一些基础功能（如状态变量管理），无需重复编写。
    def __init__(self, n_agents, n_dim, device=None, dtype=torch.float32):    #定义类的初始化方法，用于创建实例时设置参数。
        super().__init__(state_var_list=['x'])  # 调用父类ControlledDynamics的初始化方法，传入state_var_list=['x']。
                                                #作用：告诉基类 “当前动力学模型的状态变量只有一个，名为x”（原仓库通过这种方式统一管理状态变量，方便后续控制器调用）。
        self.n_agents = n_agents  # 将输入的n_agents（智能体数量）保存为类的属性，后续方法中可通过self.n_agents访问。
        self.n_dim = n_dim        # 每个智能体维度n
        self.device = device or torch.device('cpu')  #设置计算设备：如果用户传入了device（如 GPU），就用用户指定的；否则默认用 CPU。
        self.dtype = dtype  #将数据类型dtype保存为类的属性，后续创建张量时会用到（保证数据类型一致）。
        # 新增，控制输入幅值约束，避免数值发散
        self.u_clamp = 10.0

    def forward(self, t, x, u=None): #定义forward方法，这是 PyTorch 中用于 “计算输出” 的标准方法（类似函数调用），用于计算状态的导数\(\dot{x}\)。
        """
        计算状态导数: dx/dt = u
        x形状: [batch_size, 1, n_agents*n_dim]
        u形状: [batch_size, 1, n_agents*n_dim] 默认None表示无控制输入
        torch.zeros_like(x)：创建一个和x形状完全相同的零张量，确保维度匹配。
        指定device和dtype：保证返回的张量与模型的设备、数据类型一致。
        """
        if u is None:
            return torch.zeros_like(x, device=self.device, dtype=self.dtype)
        # 新增：控制输入幅值约束，避免odeint求解发散
        u_clamped = torch.clamp(u, -self.u_clamp, self.u_clamp)
        return u_clamped