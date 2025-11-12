#%% md
# # Small NNC example with noise
#%%
# To edit source files and automatically refresh code
%load_ext autoreload
%autoreload 2

# Load custom modules path
import sys
sys.path.append('../../')


# Custom modules path
import nnc.controllers.baselines.ct_lti.dynamics as dynamics
import nnc.controllers.baselines.ct_lti.optimal_controllers as oc
from nnc.controllers.neural_network.nnc_controllers import\
     NeuralNetworkController, NNCDynamics

# Computation and plot helpers for this example
from small_example_helpers import evaluate_trajectory, todf
from small_example_helpers import compare_trajectories, grand_animation

# progress bar
from tqdm.notebook import tqdm

# Other libraries for computing
import torch
import numpy as np
import pandas as pd

# ODE solvers with gradient flow
from torchdiffeq import odeint

# plots
import plotly
plotly.offline.init_notebook_mode()


#import cairo
#import plotly.express as px
#from graph_tool import draw
import matplotlib.pyplot as plt
import numpy as np
#%%
import torch
class EluTimeControl(torch.nn.Module):
    """
    Very simple Elu architecture for control of linear systems
    """
    def __init__(self, n_nodes, n_drivers):
        super().__init__()
        self.input_layer = torch.nn.Linear(1, n_nodes+4)
        self.hidden_1 = torch.nn.Linear(n_nodes+4,n_nodes+4)
        self.output_layer = torch.nn.Linear(n_nodes+4, n_drivers)
        self.total_nodes = 1 + n_nodes+4 + n_nodes+4 + n_drivers +4 # 2 bias terms

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Control calculation via a fully connected NN.
        :param t: A scalar or a batch with scalars, shape: `[b, 1]` or '[1]'
        :param x: input_states for all nodes, shape `[b, m, n_nodes]`
        :return:
        """
        # sanity check to make sure we don't propagate through time
        t = t.detach() # we do not want to learn time :)
        # check for batch size and if t is scalar:
        if len(t.shape)  in {0 , 1} :
            if x is not None and len(list(x.shape)) > 1:
                t = t.repeat(x.shape[0], 1)
            else:
                # add single sample dimension if t is scalar or single dimension tensor
                # scalars are expected to have 0 dims, if i remember right?
                t = t.unsqueeze(0)
        u = self.input_layer(t)
        u = torch.nn.functional.elu(u)
        u = self.hidden_1(u)
        u = torch.nn.functional.elu(u)
        u = self.output_layer(u)
        return u
#%%
device = 'gpu' #for gpu -> go to runtime -> change runtime type -> GPU
#device = 'cpu'
# get device info
# print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.device)
#%% md
# ## Continuous Time Time-Invariant Dynamics
# First we define our dynamics as:
# \begin{equation}\label{eq:ld}
# \dfrac{dx}{dt} = \langle A, x^{\intercal} \rangle + \langle B, u^{\intercal}\rangle
# \end{equation}
# where:
# \begin{align}
# x &: \text{a state vector over $N$ nodes.}\\
# A &: \text{an $N\times N$ interaction matrix, indicating nodal interactions.}\\
# u &: \text{a control signal vector of $M\leq N$ independent control signals.}\\
# B &: \text{an $N\times M$ driver matrix, indicating control signal effects on nodes.}\\
# \end{align}
# 
# In this example we parametrize the system as:
# 
# ### System Parametrization
# 
# \begin{align}
# A &= 
# \begin{pmatrix}
#     1 & 0 \\
#     1 & 0 \\
# \end{pmatrix}\\
# B &= 
# \begin{pmatrix}
#     1 \\
#     0 \\
# \end{pmatrix}
# \end{align}
# 
# Essentially, we evaluate control over a 2-node undirected graph with control signal applied over one node (single driver node).
#%%
# Basic setup for calculations, graph, number of nodes, etc.
dtype = torch.float32
training_session = True

# interaction matrix
A = torch.tensor([
    [1., 0.],
    [1., 0.]
])

# driver matrix
B = torch.tensor([
    [1.],
    [0.]
])

# interaction matrix dimensions denote how many nodes we have in the network
n_nodes = A.shape[-1]
# column dimension implies the number of driver nodes.
n_drivers = B.shape[-1]

# implementing the dynamics
linear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype, device)
print(n_drivers)
#%% md
# ### Experimental Setting
# We aim to control the system from initial state $x(0)=(1., 0.5)$ to $x^* = (0,0)$ within time $T=1$.
#%%
# total time for control
T = 1
# we evaluate two points in time, first point is matched to initial
# state and  second one is matched to the terminal state
t = torch.linspace(0, T, 2)

# initial state is set as follows, but can be chosen arbirtarily:
x0 = torch.tensor([[
    1.0, 0.5
]])

# same applies for target state:
x_target = torch.tensor([[
    0, 0
]])
#%%
print(x_target)

#%% md
# ### Control  Baseline
# Here we use the minimum energy optimal control as baseline based on the following work:
# - Yan, G., Ren, J., Lai, Y. C., Lai, C. H., & Li, B. (2012). Controlling complex networks: How much energy is needed?. Physical review letters, 108(21), 218703.
# 
# For CT-LTI systems that satisfy controllability assumption, this baseline achieves the maximum theoretical performance, i.e. it cannot be surpassed by any other.
#%%
# baseline definition
oc_baseline = oc.ControllabiltyGrammianController(
    alpha = A,
    beta = B,
    t_infs = T,
    x0s = x0, # a batch with one sample
    x_infs = x_target, # a batch with one sample 
    simpson_evals = 100,
    progress_bar=None,
    use_inverse=False,
)
#%% md
# Below, we create on line 2 a `lambda` expression function to make the dynamics compatible with required method signatue, and then we evolve the system with `odeint` method from `torchdiffeq` package (line 3).
# The result of `odeint` is a tensor of shape `[timesteps, state_size]`.
# The reached state corresponds to the last index of the result `[-1]`.
# As we print on line 4, optimal control reached the target state after control $x(T)\approx x^*$.
#%%
timesteps = torch.linspace(0, T, 2)
oc_dynamics = lambda t,x: linear_dynamics(t, x, oc_baseline(t, x))
xT_oc = odeint(oc_dynamics, x0.unsqueeze(0), t=timesteps)[-1]
print(str(xT_oc.flatten().detach().cpu().numpy()))
#%% md
# ### Neural Network Control
# We define a custom neural network for learning contro.
# In this example we pick a fully connected architecture with 2 layers of n+4 neurons
# and expotential linear unit activation.
# The content of the class can be found below:
#%%
#what happens if we do not train NN -- random init for u(t)
torch.manual_seed(0)
neural_net = EluTimeControl(n_nodes, n_drivers)
nnc_model = NeuralNetworkController(neural_net)
nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)

t = torch.linspace(0, T, 2)
x = x0.detach()
ld_controlled_lambda = lambda t, x_in: linear_dynamics(t, u=neural_net(t, x_in), x=x_in)
n_timesteps = 40
x_all_nn = odeint(ld_controlled_lambda, x0, t, method='rk4', 
                  options = dict(step_size=T/n_timesteps))
x_T = x_all_nn[-1, :]
print(str(x_T.flatten().detach().cpu().numpy()))

print(n_nodes)
#%%
target_sigma = 0.05
x_target_noise = x_target+(target_sigma*torch.randn(1,2))
print(x_target_noise)
#%% md
# We define a training routine for the neural networks.
# - Note the `x0.detach()` and `t.detach()` on lines 7,8 respectively to avoid unexpected gradient flows.
# - It is important is to notice on the next cell, in line 11 that we include no energy regularization term in the loss.
# - We then do the backpropagation with autograd on line 13 and then let the optimizer (Adam) to update the network parameters.
# - We prefer to train with variable step method `dopri5` on line 8, but we evaluate with constant step size on line 16, to limit performance advantages due to step size selection.
#%%
def train(nnc_dyn, epochs, lr, t, n_timesteps=40, target_sigma = 0.0): #simple training
    optimizer = torch.optim.Adam(nnc_dyn.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(nnc_dyn.parameters(), lr=lr)
    trajectories = [] # keep trajectories for plots
    loss_list = []
    for i in tqdm(range(epochs)):
        optimizer.zero_grad() # do not accumulate gradients over epochs
        
        x = x0.detach()
        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')
        x_T = x_nnc[-1, :] # reached state at T
        
        loss_original = ((x_target - x_T)**2).sum().detach() # !no noise
        loss_list.append(loss_original)

        x_target_noise = (x_target + (target_sigma*torch.randn(1,2))).detach()
        loss = ((x_target_noise - x_T)**2).sum() # with target noise 
        loss.backward() # learning is done on corrupted version of target
        optimizer.step() 

        trajectory = evaluate_trajectory(linear_dynamics, 
                                         nnc_dyn.nnc, 
                                         x0, 
                                         T, 
                                         n_timesteps, 
                                         method='rk4',
                                         options=dict(step_size = T/n_timesteps)
                                        )
        trajectories.append(trajectory)
    return torch.stack(trajectories).squeeze(-2), loss_list
#%% md
# ## Experimental Results
# We can now apply training, collect trajctories and save the model.
# Notice that we decrease learning rate on lines 11 and 14.
# NNC is very sensitive to learning rate and often requires several lr-adapaptions to converge to desirable performance.
# Such, adaptions can be automated as discussed in the paper.
#%%
n_timesteps = 40 #relevant for plotting, ode solver is variable step
linear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype, device)

my_sigma = [0.0, 0.05, 0.1, 0.5]
loss_sigma = []

for sigma_tmp in my_sigma:
  print("Current target noise "+ str(sigma_tmp))
  torch.manual_seed(0)
  neural_net = EluTimeControl(n_nodes, n_drivers)
  nnc_model = NeuralNetworkController(neural_net)
  nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)
  # time to train now:
  t = torch.linspace(0, T, n_timesteps)
  t1,loss_list1 = train(nnc_dyn, 200, 0.1, t, target_sigma = sigma_tmp) # , 200 epochs, learning rate 0.1
  t2,loss_list2 = train(nnc_dyn, 200, 0.01, t, target_sigma = sigma_tmp) # , 200 epochs, learning rate 0.01
  loss_list = loss_list1 + loss_list2
  loss_sigma.append(loss_list)
#%% md
# Now we can check the reached for NNC, and as we observe it is close to the target.
# More epochs on lower learnig rates will further improve this result.
#%%
t = torch.linspace(0, T, 2)
x = x0.detach()
ld_controlled_lambda = lambda t, x_in: linear_dynamics(t, u=neural_net(t, x_in), x=x_in)
x_all_nn = odeint(ld_controlled_lambda, x0, t, method='rk4', 
                  options = dict(step_size=T/n_timesteps))
x_T = x_all_nn[-1, :]
print(str(x_T.flatten().detach().cpu().numpy()))
#%%
import pandas as pd

y0 = [float(x) for x in loss_sigma[0]]
y1 = [float(x) for x in loss_sigma[1]]
y2 = [float(x) for x in loss_sigma[2]]
y3 = [float(x) for x in loss_sigma[3]]
x = [x for x in range(0,len(y0))]

fig, axs = plt.subplots(2, 2, figsize=(16,8))
axs[0, 0].plot(x, y0)
axs[0, 0].set_title("Zero target noise")
axs[0, 0].set(xlabel='epoch', ylabel='loss')
axs[0, 0].set_ylim((0, 1))
df0 = pd.DataFrame(list(zip(x, y0)),
               columns =['epoch', 'loss'])
print(df0.head())
my_csv = "two_lr_sigma_"+str(my_sigma[0])+".csv"
print(my_csv)
df0.to_csv(my_csv)
#axs[0, 0].grid(which='both', linestyle='--')


axs[0, 1].plot(x, y1, 'tab:orange')
axs[0, 1].set_title(f"target noise ~ N(0,$\sigma$ = {my_sigma[1]})")
axs[0, 1].set(xlabel='epoch', ylabel='loss')
axs[0, 1].set_ylim((0, 1))

df1 = pd.DataFrame(list(zip(x, y1)),
               columns =['epoch', 'loss'])
print(df1.head())
my_csv = "two_lr_sigma_"+str(my_sigma[1])+".csv"
print(my_csv)
df1.to_csv(my_csv)

axs[1, 0].plot(x, y2, 'tab:green')
axs[1, 0].set_title(f"target noise ~ N(0,$\sigma$ = {my_sigma[2]})")
axs[1, 0].set(xlabel='epoch', ylabel='loss')
axs[1, 0].set_ylim((0, 2))

df2 = pd.DataFrame(list(zip(x, y2)),
               columns =['epoch', 'loss'])
print(df2.head())
my_csv = "two_lr_sigma_"+str(my_sigma[2])+".csv"
print(my_csv)
df2.to_csv(my_csv)

axs[1, 1].plot(x, y3, 'tab:red')
axs[1, 1].set_title(f"target noise ~ N(0,$\sigma$ = {my_sigma[3]})")
axs[1, 1].set(xlabel='epoch', ylabel='loss')
axs[1, 1].set_ylim((0, 2))
fig.tight_layout()

df3 = pd.DataFrame(list(zip(x, y3)),
               columns =['epoch', 'loss'])
print(df3.head())
my_csv = "two_lr_sigma_"+str(my_sigma[3])+".csv"
print(my_csv)
df3.to_csv(my_csv)

fig.savefig("LTI_target_noise_two_lr.pdf")
#%%
from google.colab import files
files.download("LTI_target_noise_two_lr.pdf") 
#%%
from google.colab import files
for idx in range(0,4):
  my_csv = "two_lr_sigma_"+str(my_sigma[idx])+".csv"
  files.download(my_csv) 
#%% md
# Now we compare NNC to optimal control (OC) in terms of: (i) controlled trajectories (left) and (ii) total energy (right). As we see both methods are extremely close.
#%%
from IPython.display import HTML
fig_comparison, _, _ = compare_trajectories(linear_dynamics,
                     oc_baseline,
                     nnc_model,
                     x0,
                     x_target,
                     T,
                     x1_min=-3,
                     x1_max=3,
                     x2_min=-1.5,
                     x2_max=1.5,
                     n_points=200,
                    )
fig_comparison
HTML(fig_comparison.to_html())