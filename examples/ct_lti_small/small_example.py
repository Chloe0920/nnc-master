<<<<<<< HEAD
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "# Small NODEC example\nA small example to compare NODEC with optimal control on a toy dynamical system."
  },
  {
   "cell_type": "code",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-11-07T03:11:42.524088Z",
     "start_time": "2025-11-07T03:11:38.314389Z"
    }
   },
   "source": "# To edit source files and automatically refresh code\n%load_ext autoreload\n%autoreload 2\n\n# Load custom modules path\nimport sys\nsys.path.append('../../')\n\n# Custom modules path\nimport nnc.controllers.baselines.ct_lti.dynamics as dynamics\nimport nnc.controllers.baselines.ct_lti.optimal_controllers as oc\nfrom nnc.controllers.neural_network.nnc_controllers import\\\n     NeuralNetworkController, NNCDynamics\n\n# Computation and plot helpers for this example\nfrom small_example_helpers import EluTimeControl, evaluate_trajectory, todf\nfrom small_example_helpers import compare_trajectories, grand_animation\n\n# progress bar\nfrom tqdm.cli import tqdm\n\n# Other libraries for computing\nimport torch\nimport numpy as np\nimport pandas as pd\n\n# ODE solvers with gradient flow\nfrom torchdiffeq import odeint\n\n# plots\nimport plotly\nplotly.offline.init_notebook_mode()",
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The autoreload extension is already loaded. To reload it, use:\n",
      "  %reload_ext autoreload\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hp\\.conda\\envs\\ANNF\\lib\\site-packages\\pandas\\_testing.py:24: FutureWarning: In the future `np.bool` will be defined as the corresponding NumPy scalar.\n",
      "  import pandas._libs.testing as _testing\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "module 'numpy' has no attribute 'bool'.\n`np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.\nThe aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:\n    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations",
     "output_type": "error",
     "traceback": [
      "\u001B[1;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[1;31mAttributeError\u001B[0m                            Traceback (most recent call last)",
      "Cell \u001B[1;32mIn[2], line 16\u001B[0m\n\u001B[0;32m     12\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mnnc\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mcontrollers\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mneural_network\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mnnc_controllers\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m\\\n\u001B[0;32m     13\u001B[0m      NeuralNetworkController, NNCDynamics\n\u001B[0;32m     15\u001B[0m \u001B[38;5;66;03m# Computation and plot helpers for this example\u001B[39;00m\n\u001B[1;32m---> 16\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01msmall_example_helpers\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m EluTimeControl, evaluate_trajectory, todf\n\u001B[0;32m     17\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01msmall_example_helpers\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m compare_trajectories, grand_animation\n\u001B[0;32m     19\u001B[0m \u001B[38;5;66;03m# progress bar\u001B[39;00m\n",
      "File \u001B[1;32mE:\\004NNC\\nnc-master\\examples\\ct_lti_small\\small_example_helpers.py:2\u001B[0m\n\u001B[0;32m      1\u001B[0m \u001B[38;5;28;01mimport\u001B[39;00m \u001B[38;5;21;01mtorch\u001B[39;00m\n\u001B[1;32m----> 2\u001B[0m \u001B[38;5;28;01mimport\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m \u001B[38;5;28;01mas\u001B[39;00m \u001B[38;5;21;01mpd\u001B[39;00m\n\u001B[0;32m      3\u001B[0m \u001B[38;5;28;01mimport\u001B[39;00m \u001B[38;5;21;01mnumpy\u001B[39;00m \u001B[38;5;28;01mas\u001B[39;00m \u001B[38;5;21;01mnp\u001B[39;00m\n\u001B[0;32m      4\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mtorchdiffeq\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m odeint\n",
      "File \u001B[1;32m~\\.conda\\envs\\ANNF\\lib\\site-packages\\pandas\\__init__.py:180\u001B[0m\n\u001B[0;32m    177\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mio\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mjson\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m _json_normalize \u001B[38;5;28;01mas\u001B[39;00m json_normalize\n\u001B[0;32m    179\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mutil\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01m_tester\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m test\n\u001B[1;32m--> 180\u001B[0m \u001B[38;5;28;01mimport\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mtesting\u001B[39;00m\n\u001B[0;32m    181\u001B[0m \u001B[38;5;28;01mimport\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01marrays\u001B[39;00m\n\u001B[0;32m    183\u001B[0m \u001B[38;5;66;03m# use the closest tagged version if possible\u001B[39;00m\n",
      "File \u001B[1;32m~\\.conda\\envs\\ANNF\\lib\\site-packages\\pandas\\testing.py:5\u001B[0m\n\u001B[0;32m      1\u001B[0m \u001B[38;5;124;03m\"\"\"\u001B[39;00m\n\u001B[0;32m      2\u001B[0m \u001B[38;5;124;03mPublic testing utility functions.\u001B[39;00m\n\u001B[0;32m      3\u001B[0m \u001B[38;5;124;03m\"\"\"\u001B[39;00m\n\u001B[1;32m----> 5\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01m_testing\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m (\n\u001B[0;32m      6\u001B[0m     assert_extension_array_equal,\n\u001B[0;32m      7\u001B[0m     assert_frame_equal,\n\u001B[0;32m      8\u001B[0m     assert_index_equal,\n\u001B[0;32m      9\u001B[0m     assert_series_equal,\n\u001B[0;32m     10\u001B[0m )\n\u001B[0;32m     12\u001B[0m __all__ \u001B[38;5;241m=\u001B[39m [\n\u001B[0;32m     13\u001B[0m     \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124massert_extension_array_equal\u001B[39m\u001B[38;5;124m\"\u001B[39m,\n\u001B[0;32m     14\u001B[0m     \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124massert_frame_equal\u001B[39m\u001B[38;5;124m\"\u001B[39m,\n\u001B[0;32m     15\u001B[0m     \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124massert_series_equal\u001B[39m\u001B[38;5;124m\"\u001B[39m,\n\u001B[0;32m     16\u001B[0m     \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124massert_index_equal\u001B[39m\u001B[38;5;124m\"\u001B[39m,\n\u001B[0;32m     17\u001B[0m ]\n",
      "File \u001B[1;32m~\\.conda\\envs\\ANNF\\lib\\site-packages\\pandas\\_testing.py:24\u001B[0m\n\u001B[0;32m     16\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mnumpy\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mrandom\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m rand, randn\n\u001B[0;32m     18\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01m_config\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mlocalization\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m (  \u001B[38;5;66;03m# noqa:F401\u001B[39;00m\n\u001B[0;32m     19\u001B[0m     can_set_locale,\n\u001B[0;32m     20\u001B[0m     get_locales,\n\u001B[0;32m     21\u001B[0m     set_locale,\n\u001B[0;32m     22\u001B[0m )\n\u001B[1;32m---> 24\u001B[0m \u001B[38;5;28;01mimport\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01m_libs\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mtesting\u001B[39;00m \u001B[38;5;28;01mas\u001B[39;00m \u001B[38;5;21;01m_testing\u001B[39;00m\n\u001B[0;32m     25\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01m_typing\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m FilePathOrBuffer, FrameOrSeries\n\u001B[0;32m     26\u001B[0m \u001B[38;5;28;01mfrom\u001B[39;00m \u001B[38;5;21;01mpandas\u001B[39;00m\u001B[38;5;21;01m.\u001B[39;00m\u001B[38;5;21;01mcompat\u001B[39;00m \u001B[38;5;28;01mimport\u001B[39;00m _get_lzma_file, _import_lzma\n",
      "File \u001B[1;32mpandas\\_libs\\testing.pyx:10\u001B[0m, in \u001B[0;36minit pandas._libs.testing\u001B[1;34m()\u001B[0m\n",
      "File \u001B[1;32m~\\.conda\\envs\\ANNF\\lib\\site-packages\\numpy\\__init__.py:305\u001B[0m, in \u001B[0;36m__getattr__\u001B[1;34m(attr)\u001B[0m\n\u001B[0;32m    300\u001B[0m     warnings\u001B[38;5;241m.\u001B[39mwarn(\n\u001B[0;32m    301\u001B[0m         \u001B[38;5;124mf\u001B[39m\u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mIn the future `np.\u001B[39m\u001B[38;5;132;01m{\u001B[39;00mattr\u001B[38;5;132;01m}\u001B[39;00m\u001B[38;5;124m` will be defined as the \u001B[39m\u001B[38;5;124m\"\u001B[39m\n\u001B[0;32m    302\u001B[0m         \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mcorresponding NumPy scalar.\u001B[39m\u001B[38;5;124m\"\u001B[39m, \u001B[38;5;167;01mFutureWarning\u001B[39;00m, stacklevel\u001B[38;5;241m=\u001B[39m\u001B[38;5;241m2\u001B[39m)\n\u001B[0;32m    304\u001B[0m \u001B[38;5;28;01mif\u001B[39;00m attr \u001B[38;5;129;01min\u001B[39;00m __former_attrs__:\n\u001B[1;32m--> 305\u001B[0m     \u001B[38;5;28;01mraise\u001B[39;00m \u001B[38;5;167;01mAttributeError\u001B[39;00m(__former_attrs__[attr])\n\u001B[0;32m    307\u001B[0m \u001B[38;5;66;03m# Importing Tester requires importing all of UnitTest which is not a\u001B[39;00m\n\u001B[0;32m    308\u001B[0m \u001B[38;5;66;03m# cheap import Since it is mainly used in test suits, we lazy import it\u001B[39;00m\n\u001B[0;32m    309\u001B[0m \u001B[38;5;66;03m# here to save on the order of 10 ms of import time for most users\u001B[39;00m\n\u001B[0;32m    310\u001B[0m \u001B[38;5;66;03m#\u001B[39;00m\n\u001B[0;32m    311\u001B[0m \u001B[38;5;66;03m# The previous way Tester was imported also had a side effect of adding\u001B[39;00m\n\u001B[0;32m    312\u001B[0m \u001B[38;5;66;03m# the full `numpy.testing` namespace\u001B[39;00m\n\u001B[0;32m    313\u001B[0m \u001B[38;5;28;01mif\u001B[39;00m attr \u001B[38;5;241m==\u001B[39m \u001B[38;5;124m'\u001B[39m\u001B[38;5;124mtesting\u001B[39m\u001B[38;5;124m'\u001B[39m:\n",
      "\u001B[1;31mAttributeError\u001B[0m: module 'numpy' has no attribute 'bool'.\n`np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.\nThe aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:\n    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations"
     ]
    }
   ],
   "execution_count": 2
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## Continuous Time Time-Invariant Dynamics\nFirst we define our dynamics as:\n\\begin{equation}\\label{eq:ld}\n\\dfrac{dx}{dt} = \\langle A, x^{\\intercal} \\rangle + \\langle B, u^{\\intercal}\\rangle\n\\end{equation}\nwhere:\n\\begin{align}\nx &: \\text{a state vector over $N$ nodes.}\\\\\nA &: \\text{an $N\\times N$ interaction matrix, indicating nodal interactions.}\\\\\nu &: \\text{a control signal vector of $M\\leq N$ independent control signals.}\\\\\nB &: \\text{an $N\\times M$ driver matrix, indicating control signal effects on nodes.}\\\\\n\\end{align}\n\nIn this example we parametrize the system as:\n\n### System Parametrization\n\n\\begin{align}\nA &= \n\\begin{pmatrix}\n    0 & 1 \\\\\n    1 & 0 \\\\\n\\end{pmatrix}\\\\\nB &= \n\\begin{pmatrix}\n    1 \\\\\n    0 \\\\\n\\end{pmatrix}\n\\end{align}\n\nEssentially, we evaluate control over a 2-node undirected graph with control signal applied over one node (single driver node)."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "# Basic setup for calculations, graph, number of nodes, etc.\ndtype = torch.float32\ndevice = 'cpu'\ntraining_session = True\n\n# interaction matrix\nA = torch.tensor([\n    [0., 1.],\n    [1., 0.]\n])\n\n# driver matrix\nB = torch.tensor([\n    [1.],\n    [0.]\n])\n\n# interaction matrix dimensions denote how many nodes we have in the network\nn_nodes = A.shape[-1]\n# column dimension implies the number of driver nodes.\nn_drivers = B.shape[-1]\n\n# implementing the dynamics\nlinear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype, device)\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "The exact implementation of the dynamics:"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "dynamics.ContinuousTimeInvariantDynamics??"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### Experimental Setting\nWe aim to control the system from initial state $x(0)=(0.5, 0.5)$ to $x^* = (1,-1)$ within time $T=1$."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "# total time for control\nT = 1\n# we evaluate two points in time, first point is matched to initial\n# state and  second one is matched to the terminal state\nt = torch.linspace(0, T, 2)\n\n# initial state is set as follows, but can be chosen arbirtarily:\nx0 = torch.tensor([[\n    0.5, 0.5\n]])\n\n# same applies for target state:\nx_target = torch.tensor([[\n    1, -1\n]])"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### Control  Baseline\nHere we use the minimum energy optimal control as baseline based on the following work:\n- Yan, G., Ren, J., Lai, Y. C., Lai, C. H., & Li, B. (2012). Controlling complex networks: How much energy is needed?. Physical review letters, 108(21), 218703.\n\nFor CT-LTI systems that satisfy controllability assumption, this baseline achieves the maximum theoretical performance, i.e. it cannot be surpassed by any other."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "# baseline definition\noc_baseline = oc.ControllabiltyGrammianController(\n    alpha = A,\n    beta = B,\n    t_infs = T,\n    x0s = x0, # a batch with one sample\n    x_infs = x_target, # a batch with one sample \n    simpson_evals = 100,\n    progress_bar=None,\n    use_inverse=False,\n)"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "Below, we create on line 2 a `lambda` expression function to make the dynamics compatible with required method signatue, and then we evolve the system with `odeint` method from `torchdiffeq` package (line 3).\nThe result of `odeint` is a tensor of shape `[timesteps, state_size]`.\nThe reached state corresponds to the last index of the result `[-1]`.\nAs we print on line 4, optimal control reached the target state after control $x(T)\\approx x^*$."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "timesteps = torch.linspace(0, T, 2)\noc_dynamics = lambda t,x: linear_dynamics(t, x, oc_baseline(t, x))\nxT_oc = odeint(oc_dynamics, x0.unsqueeze(0), t=timesteps)[-1]\nprint(str(xT_oc.flatten().detach().cpu().numpy()))"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### Neural Network Control\nWe define a custom neural network for learning contro.\nIn this example we pick a fully connected architecture with 2 layers of n+4 neurons\nand exponential linear unit (ELU) activation.\nThe content of the class can be found below:"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "EluTimeControl??"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "We define a training routine for the neural networks.\n- Note the `x0.detach()` and `t.detach()` on lines 7,8 respectively to avoid unexpected gradient flows.\n- It is important is to notice on the next cell, in line 11 that we include no energy regularization term in the loss.\n- We then do the backpropagation with autograd on line 13 and then let the optimizer (Adam) to update the network parameters.\n- We prefer to train with variable step method `dopri5` on line 8, but we evaluate with constant step size on line 16, to limit performance advantages due to step size selection."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "def train(nnc_dyn, epochs, lr, t, n_timesteps=40): #simple training\n    optimizer = torch.optim.Adam(nnc_dyn.parameters(), lr=lr)\n    trajectories = [] # keep trajectories for plots\n    for i in tqdm(range(epochs)):\n        optimizer.zero_grad() # do not accumulate gradients over epochs\n        \n        x = x0.detach()\n        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')\n        x_T = x_nnc[-1, :] # reached state at T\n        \n        loss = ((x_target - x_T)**2).sum() # !No energy regularization\n\n        loss.backward()\n        optimizer.step() \n\n        trajectory = evaluate_trajectory(linear_dynamics, \n                                         nnc_dyn.nnc, \n                                         x0, \n                                         T, \n                                         n_timesteps, \n                                         method='rk4',\n                                         options=dict(step_size = T/n_timesteps)\n                                        )\n        trajectories.append(trajectory)\n    return torch.stack(trajectories).squeeze(-2)"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## Experimental Results\nWe can now apply training, collect trajctories and save the model.\nNotice that we decrease learning rate on lines 11 and 14.\nNODEC is very sensitive to learning rate and often requires several lr-adapaptions to converge to desirable performance.\nSuch, adaptions can be automated as discussed in the paper."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "is_executing": true
    }
   },
   "outputs": [],
   "source": "n_timesteps = 40 #relevant for plotting, ode solver is variable step\nlinear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype, device)\n\nif training_session:\n    torch.manual_seed(0)\n    neural_net = EluTimeControl(n_nodes, n_drivers)\n    nnc_model = NeuralNetworkController(neural_net)\n    nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)\n    # time to train now:\n    t = torch.linspace(0, T, n_timesteps)\n    t1 = train(nnc_dyn, 200, 0.1, t) # , 200 epochs, learning rate 0.1\n    df1 = todf(t1, lr=0.1)\n    t2 = train(nnc_dyn, 300, 0.01, t) # , 300 epochs, learning rate 0.01\n    df2 = todf(t2, lr=0.01)\n    torch.save(neural_net, 'trained_elu_net.torch')\n    alldf = pd.concat([df1, df2], ignore_index=True)\n    \n    alldf.to_csv('all_trajectories.csv')   \nelse: \n    neural_net = torch.load('trained_elu_net.torch')\n    alldf = pd.read_csv('all_trajectories.csv', index_col=0)\n    nnc_model = NeuralNetworkController(neural_net)\n    nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "Now we can check the reached for NODEC, and as we observe it is close to the target.\nMore epochs on lower learnig rates will further improve this result."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "is_executing": true
    }
   },
   "outputs": [],
   "source": "t = torch.linspace(0, T, 2)\nx = x0.detach()\nld_controlled_lambda = lambda t, x_in: linear_dynamics(t, u=neural_net(t, x_in), x=x_in)\nx_all_nn = odeint(ld_controlled_lambda, x0, t, method='rk4', \n                  options = dict(step_size=T/n_timesteps))\nx_T = x_all_nn[-1, :]\nprint(str(x_T.flatten().detach().cpu().numpy()))"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "Now we compare NODEC to optimal control (OC) in terms of: (i) controlled trajectories (left) and (ii) total energy (right). As we see both methods are extremely close."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "is_executing": true
    }
   },
   "outputs": [],
   "source": "fig_comparison, _, _ = compare_trajectories(linear_dynamics,\n                     oc_baseline,\n                     nnc_model,\n                     x0,\n                     x_target,\n                     T,\n                     x1_min=-3,\n                     x1_max=3,\n                     x2_min=-1.5,\n                     x2_max=1.5,\n                     n_points=200,\n                    )\nfig_comparison"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "Finally we animate the learning effort of NODEC through the training process. We observe how it slowly converges to the OC trajectory, both in terms of state values and energy."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "is_executing": true
    }
   },
   "outputs": [],
   "source": "animation =  grand_animation(alldf,\n                             linear_dynamics, \n                             oc_baseline, \n                             nnc_model, \n                             x0,\n                             x_target, \n                             T,\n                            frame_duration=50) # 50-100 should be ok.\nanimation"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## Conclusion\nThis notebook gave us a quick yet insightful demonstration of NODEC and its performance compared to optimal control.\nResults conclude that NODEC can achieve the control goal. Implicit energy regularization is also observed.\nLarger scale demonstrations on complex dynamics are found in the paper results."
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  },
  "latex_metadata": {
   "affiliation": "ETH Z\\\"urich, Switzerland",
   "author": "Thomas Asikis",
   "title": "Small NNC example"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {
     "002d59f567ca46ba9eb8418ca1d20490": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_644605e0c20844e4baa9b9298c09968c",
       "max": 200,
       "style": "IPY_MODEL_c655374a178c44ffa49f2d0031c31e97",
       "value": 200
      }
     },
     "005b3706ebb94e9998e51963ff182dd5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "014695a4b90149eba71f03516e18c255": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_f7587edf77524260871b5aea2498895f",
        "IPY_MODEL_d0c8d1bddecf4ec7a0066a9f54bd0c6a"
       ],
       "layout": "IPY_MODEL_d1078b1205534916a518d952f3cc4ddc"
      }
     },
     "01746170b5794cd2942e0d5ed3c1d8ae": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_17acb6d2c97f4d90a1c45ee05a2359f5",
       "max": 200,
       "style": "IPY_MODEL_793f24a97a18438fa113cd308e77ab1a",
       "value": 200
      }
     },
     "0201f287dd224e82a7068749ccdffbfd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "023346bb70044e5682f05da2f85399c9": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "02bf9bf329aa4b0797fdc4a77a2e3f71": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "0353ae8ac4ac41c5aa99006e56ec6b58": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_23e050c8c9e94829a056496054627932",
       "max": 300,
       "style": "IPY_MODEL_431385120eab4d78991f200420d5957c",
       "value": 300
      }
     },
     "03574a0bf42d479e95121eb8745a1799": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_cf1052676d2646ec8aa389f65f52a6af",
        "IPY_MODEL_b3f9bcfb2ac44ea8b58b9f4e2e9c98d4"
       ],
       "layout": "IPY_MODEL_1aaaa6d9b92a4551a68b78b5e795df14"
      }
     },
     "039cef2ba0cc4544963144e7132a4303": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "03f0b92465f244729fd9ffe68657a746": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "040e16ebba49457f9b5d6a315c9554fb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "041c4577cd154a11bd203ba8b9ef0814": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "04abc016a87541f48a7df9b6a35ba8c9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_8381899ede4f4c13a9deb3e18fa709e6",
       "style": "IPY_MODEL_08501b532ed14926bc2265787bf2db65",
       "value": " 300/300 [11:06&lt;00:00,  2.22s/it]"
      }
     },
     "05746fa0e0ec4b9e9809f872a2409e61": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_bd23b9e83dde4ff3bdcf82f41950d6cc",
       "max": 2,
       "style": "IPY_MODEL_7d33a349fd384f33b124d8ab9a6a541e",
       "value": 2
      }
     },
     "05a4bf23395d4669a621da2340f2dcc2": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "05e2a4880d8c42db8432e75813854c6e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "064efbed9d8a4b51ab87448f4b7e30d6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "06d91eea70394a0db970c57b44dee804": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "075d72f8ad794bcaac1ebcaabfc020b8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "07c8e7fd75f6499bb4fcff6160060c9c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "08501b532ed14926bc2265787bf2db65": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "0853d1531b1e409fa2f076a62fa41a7f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "08ff5f75450d4e0bb0ec92a75114d641": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "094c74f539854be0ad88c73cb32f8913": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "096d21a66da64056806c52542d700b25": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_ab0893c752c84b97b740a08a34c89a2a",
       "max": 2,
       "style": "IPY_MODEL_80e8324050854bd790bcfeb8eb647b81",
       "value": 2
      }
     },
     "09871093305140deb2c16d792d5708df": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "09a069c8c4594fa6ad5d6e76cfe83701": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "09c00bf002a04d31b5d0c72d40ffe090": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "09e4fb21849e403e8bfeac92ee20a09b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_825638143d814b42b24892ce66d290c4",
        "IPY_MODEL_9dcd5b65fa6c42cea1fb1cb2e41d7642"
       ],
       "layout": "IPY_MODEL_37d7b125e9724f1fa1c9c074166d30e3"
      }
     },
     "0bf81f2702d3476ab3f53fcec853a9ac": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_d39585ef35ae46bdb29a4fe03ea7f80e",
       "max": 200,
       "style": "IPY_MODEL_cf69a28284744cb19eb4962e17a61157",
       "value": 200
      }
     },
     "0c2b0e00bec64bd2a4feb98540a38153": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_8ec0ea1cfc28409591bc5c620bb3a366",
       "max": 2,
       "style": "IPY_MODEL_9a72932bbd2f4487acabc05224f0eecd",
       "value": 2
      }
     },
     "0d469214d3774a6293b71ec9ee990f0a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_e34a0470f4c4414381a0f5d8dd584c70",
        "IPY_MODEL_cc41d24433854b0aaa2d0339cf5667fd"
       ],
       "layout": "IPY_MODEL_8829c85b980846c2b000aced08613a3a"
      }
     },
     "0d6eb4e6a1cf46f9a4adfd566b264bc7": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_1ac92d4647a44df38a3008214bae177b",
       "style": "IPY_MODEL_064efbed9d8a4b51ab87448f4b7e30d6",
       "value": " 300/300 [00:56&lt;00:00,  5.32it/s]"
      }
     },
     "0dbf5b71324a475cbb0a1d054390839f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "0dc2668c69e04a19b846b7952323ddf4": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "0e539d49dff34320bea9d52213f3efb9": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "0ee0c2d234284cd2b367bdc3e54fa07a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "0f5ec697bf1240d4a75402a6951a5797": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "108357b9ba4740da8b5c689fb34e9141": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "10daa34f258648a0a27d847a9ad48e34": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_bdfcde8b3a0d4b60a43ce46c9cb4f490",
        "IPY_MODEL_d8b42274c88045ec8eeff13f5f009f17"
       ],
       "layout": "IPY_MODEL_af1bc4e3b0274dd78db5ddf0e3b451f0"
      }
     },
     "123bd18e6c7d495598abbf8b10d9d9dc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_ed8b40ea79eb412096bdec71e6873bf0",
        "IPY_MODEL_ce06ac6e90fd4bf99b8ed1574455f7ed"
       ],
       "layout": "IPY_MODEL_278401ec4a3c4083a6b859ceed816e41"
      }
     },
     "12d68e6f9fae46da8dfdd6fb467efa8a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_95744244bde34b35a990ff69d32b0a5b",
       "style": "IPY_MODEL_479d33fb42b24f48a2d27ea6d7285351",
       "value": " 200/200 [00:32&lt;00:00,  6.14it/s]"
      }
     },
     "12d9758aab96413b843a5cb49c49cf86": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "12d9ec781da44d96afbce411ff6dd654": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_6b9add0f90054681b3649651e4c27477",
       "style": "IPY_MODEL_8d4b6c6c2c904fe39a50a16bf0ee1702",
       "value": " 200/200 [09:46&lt;00:00,  2.93s/it]"
      }
     },
     "12fdd2ec99da4ac080d2a24b30d777be": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "143a9654993a482995b6f0d8c1a82676": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "156d230fa70f4fb2bbc7efed3ce55f50": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "15a8b9830465461fa93ced228d4195f3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "15fa43a5da0a4906855ceac366431d5b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_ac30774fcfe24e3c8888cf18558a6bfc",
       "style": "IPY_MODEL_3334e79aa3aa4e5ebaae2e641317302d",
       "value": 100
      }
     },
     "16aefc0ee1b84260b38fb4cada31e8c4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "16cdec1d589e4b84b9e1432f9d326da8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1707ea997b1e4b208d4792020417f510": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_904203b941fc4bd6844520979fa109de",
       "style": "IPY_MODEL_094c74f539854be0ad88c73cb32f8913",
       "value": " 2/2 [04:26&lt;00:00, 133.26s/it]"
      }
     },
     "17acb6d2c97f4d90a1c45ee05a2359f5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "187aab75554742e88ef8704a832fcc3f": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_b88644021e7f43928d5ae08371a892fd",
        "IPY_MODEL_e1cd50413e864ede8fcee5b696107db3"
       ],
       "layout": "IPY_MODEL_569422e860bd43f2a4d5c2edb626b240"
      }
     },
     "1972bb4bd9624659913178b3cf980e79": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_37801dec522f47658bf3b461698cfcd9",
        "IPY_MODEL_1a49429d9da84577958af3077a2f38c0"
       ],
       "layout": "IPY_MODEL_b71f99c19c3c402dafd7644523e5b5b1"
      }
     },
     "199f54999a9f4b818fa156890b76ef05": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "19a8df789fd944cca9081036753dcb38": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "19bb316c90624805acb7f93073376ac4": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1a49429d9da84577958af3077a2f38c0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_0853d1531b1e409fa2f076a62fa41a7f",
       "style": "IPY_MODEL_b88f3049eeee41e8bfd65d3fea6811f5",
       "value": " 2/2 [00:06&lt;00:00,  3.04s/it]"
      }
     },
     "1aaaa6d9b92a4551a68b78b5e795df14": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1ac92d4647a44df38a3008214bae177b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1afedb0c361141458e03934c2466e097": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1b645a854bed4edda33cf10cde01d963": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "danger",
       "description": "  0%",
       "layout": "IPY_MODEL_4be5e471a19542dc949292ffae9d71a9",
       "max": 200,
       "style": "IPY_MODEL_09c00bf002a04d31b5d0c72d40ffe090"
      }
     },
     "1c4ae29996b641dbafe41bdc8a622b70": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "1c77ddbe02434eb89476dbe1bd5d0d74": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1e1c21bba5bd442cabb00ed2cb4910d7": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1f13321849cb4357bdbbdf65400bd224": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "1f8284eac1014b25b3f21f78d27c0ae9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_b304c23114fc4b368054fdedebe98bec",
        "IPY_MODEL_1707ea997b1e4b208d4792020417f510"
       ],
       "layout": "IPY_MODEL_3d50c97cce8f46b3861741b4d9e32b0a"
      }
     },
     "1f9157b23a874a0b9bef49634ce2abd4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_7f865c763c074d128c345864ccb0aaa3",
        "IPY_MODEL_b9f67e0cac1742e4b245b4a27f45c5d1"
       ],
       "layout": "IPY_MODEL_795629f364aa4de2b1fc6a673c728c94"
      }
     },
     "1fcb0d10fd7b47f6abdf37241e196efb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_096d21a66da64056806c52542d700b25",
        "IPY_MODEL_cc46fc787dfc46e1933532edba52b4d1"
       ],
       "layout": "IPY_MODEL_f07b336518c14a58a43692379d3e77d6"
      }
     },
     "204400c2b64b4a8baba57f92ba638205": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "20a08026ac5d49a0b14aa41635e3ea96": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_f75172c01ff34e14af386a5c891bcd88",
        "IPY_MODEL_e2a4e0fc30514b20adaee2fdee557be3"
       ],
       "layout": "IPY_MODEL_59ae1ef2028a48e9b1643ffa40c51a42"
      }
     },
     "20eb1f87d25547a3a9936ff64fda12fb": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "2142b21e4b614286b407f8b963edbed0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "214edccc75174a0a9ef6b47be16bceb3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_4198b858cbdd4d52801d65adce4e7aea",
       "style": "IPY_MODEL_d6c4f7c9159c4e539c747b6e2983ed2b",
       "value": " 2/2 [03:39&lt;00:00, 109.94s/it]"
      }
     },
     "2166534bcde14ddda56d9d1c106bcdfb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_916530172abc4fb3b7190ea2a11c7e3a",
       "style": "IPY_MODEL_ab4173afdbbf4496bfd4fa490f6ff84b",
       "value": " 2/2 [01:13&lt;00:00, 36.87s/it]"
      }
     },
     "21789dfc9274478489fdd046e7d8f445": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "21ce6253ddab4eb58de3a8e84e943e6d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "22abd941f4214f6688eacea066cd083e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "22b5281f57f44b3d81b4484e1328abaf": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_1afedb0c361141458e03934c2466e097",
       "style": "IPY_MODEL_4cb2da30f6334ef39b20ea61a77779db",
       "value": " 100/100 [00:57&lt;00:00,  1.73it/s]"
      }
     },
     "2360820a5f7c48a3923e868e63951d53": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "2390f0aec01441a598460829b3aa4e96": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "23e050c8c9e94829a056496054627932": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "23fe8f5977a44673ad7dd704fee8dc31": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_0f5ec697bf1240d4a75402a6951a5797",
       "max": 200,
       "style": "IPY_MODEL_84455b0a64a74294bd5814ded73c05f5",
       "value": 200
      }
     },
     "24486927ecf1462d9deafc8721c5a67a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "258346915b2c40ecabeceec670bff283": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "2676372c45bd45fa95569015aa8b4068": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "26e29cd8d57340a789e8c700fdf4b144": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_be819e0b4cc24b29aa88c4ce487888c2",
        "IPY_MODEL_0d6eb4e6a1cf46f9a4adfd566b264bc7"
       ],
       "layout": "IPY_MODEL_8242093b80f842448baf3b0d3e1298ac"
      }
     },
     "278401ec4a3c4083a6b859ceed816e41": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "280f8786057d40aaa970861990da1c18": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "289e351478334a148ce6924f6bde30b7": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "28e9c87517254377a37144a64d9fab82": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "293c45fd969748568a082c8e78917138": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "2a3c47712e9b4db6a46d31fdb8e60d8c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "2a9591799c3544b894b018fa5e15a680": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "2bd914e6301341aab54384c9e5d9e40c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "2bdaffc829654f17a037026956f08917": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "2beaafc1a6584294ba6d6d51d675e9cc": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "2cfee9448f2e4870874d83d1f0177f04": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "2e19164d8cd34b8a803b01964f58e141": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_c190511f14da4810aec9df4f2c3a41f5",
       "style": "IPY_MODEL_06d91eea70394a0db970c57b44dee804",
       "value": " 300/300 [00:55&lt;00:00,  5.44it/s]"
      }
     },
     "2ea7ad1fc71547758997763659ee3752": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "2eac5e4754a04871ad5524b632bedfb5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "30c1af7383b34066a34b851405c9224b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "30c37a13ee3c400bbe33e366fc614d7b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "30f63f246ab148118e394f6d33ec1f92": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "31d01089215047c698efb8e13106d57d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "32d071ff0e994878a3ede66e8fc1a5ff": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "32ed88399c784b8589c41ef22264a6f7": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "3334e79aa3aa4e5ebaae2e641317302d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "336c2f60995e4601bc967b86e8516d99": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_05a4bf23395d4669a621da2340f2dcc2",
       "max": 300,
       "style": "IPY_MODEL_fac8c672dd3a4fb09444daa2c5220741",
       "value": 300
      }
     },
     "33c1904fd85d471c81cf350dfd006e00": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "34197b6f340045a58a6e3c83cda8ecef": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "347f9d470b754d66b877ea7932a69209": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_361d1fed3dc541518cb862d0831cba13",
       "max": 300,
       "style": "IPY_MODEL_3abb725c08d54936afcc72031712940f",
       "value": 300
      }
     },
     "35a0e7ff2c5b41569870b35ccc9f7bb2": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "361d1fed3dc541518cb862d0831cba13": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "36285193f99749d381423663b551b044": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "36b984a22422429b926bda8dac248a67": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "36e1baf5fb8b43a79f4b692e5d6f2b2c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "371c0a34975d4df6be3632a3b5689d2a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "danger",
       "description": "  0%",
       "layout": "IPY_MODEL_c090dc7ad12340789ba2d7d72f995729",
       "max": 200,
       "style": "IPY_MODEL_2cfee9448f2e4870874d83d1f0177f04"
      }
     },
     "37801dec522f47658bf3b461698cfcd9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_2a3c47712e9b4db6a46d31fdb8e60d8c",
       "max": 2,
       "style": "IPY_MODEL_813e680a9ae84b68bf3d3dd81f2cb483",
       "value": 2
      }
     },
     "37d7b125e9724f1fa1c9c074166d30e3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "37e80dca0f3844519e288819d200a5a8": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "391fcfa48f1b4e36997ef0702f10ad5d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_65c16eee1eb94d27913f27e95d6bafd6",
       "style": "IPY_MODEL_8d6012b4c1384571ad3f4902f5738572",
       "value": " 36/100 [00:06&lt;00:11,  5.57it/s]"
      }
     },
     "399a9dc5fe7a4441b207c7afbaa5d214": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_7f81e141eb3b4fd88a3934316bf11824",
        "IPY_MODEL_40a48ff94e424b2484bd08d6d7ae7398"
       ],
       "layout": "IPY_MODEL_535c9cc66b084e52ad494e745ec9a23f"
      }
     },
     "39dcca616a374486bb6cc48db683bf55": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_8b3b79ffdd8f41bdb0cc20498532c99a",
       "max": 300,
       "style": "IPY_MODEL_a8195f66e3bf4b13a69a4f8c5d875b07",
       "value": 300
      }
     },
     "3a2eef88782e423d803d7d7a7f39802b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "3aaa8d3bdb4a44d090453fa1a0567b05": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_5283fa9e66614ea6a7145d7ab1f580bb",
        "IPY_MODEL_b2c3096e6e9a4a6d9da80fd1ceae1b78"
       ],
       "layout": "IPY_MODEL_0ee0c2d234284cd2b367bdc3e54fa07a"
      }
     },
     "3abb725c08d54936afcc72031712940f": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "3bb25574b9824b02a1895a42cd36573e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_3c76607a28734c9a899657f7f039ac36",
       "max": 200,
       "style": "IPY_MODEL_16aefc0ee1b84260b38fb4cada31e8c4",
       "value": 200
      }
     },
     "3bdb33b3c17045189cf6f154ab6f948c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_93deaf444cbe4c4f9fa4a89d899badc3",
       "max": 2,
       "style": "IPY_MODEL_31d01089215047c698efb8e13106d57d",
       "value": 2
      }
     },
     "3c648575362040eab27a7212d28a5aa7": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "3c648d21e8604436933da0b69ae4c3d6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_ce9b73bc52684b538566f7f6f8964eb3",
       "max": 2,
       "style": "IPY_MODEL_8309c3cbff684aaaa1bc8a88524d6aa9",
       "value": 2
      }
     },
     "3c76607a28734c9a899657f7f039ac36": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "3d50c97cce8f46b3861741b4d9e32b0a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "3f39e1a5b59d4340aa580cc7117f5cef": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "3fe6aecfd5c14291b52f3a2b9fa844b2": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "401575bb4d4d4bfd8a2db2a25a3c54c3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "40a48ff94e424b2484bd08d6d7ae7398": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_f13c7ab103aa49c0a2e8dd0d139264f8",
       "style": "IPY_MODEL_f7d17faed5744373988335dd6badeec6",
       "value": " 400/400 [01:51&lt;00:00,  3.59it/s]"
      }
     },
     "40cc03073d25468e91fecf6f1913cfac": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_2ea7ad1fc71547758997763659ee3752",
       "style": "IPY_MODEL_e42fb4a4d5a943cc8104e4a4354bbf9b",
       "value": " 2/2 [04:20&lt;00:00, 130.24s/it]"
      }
     },
     "418e4881b88d4d50bef32241d7a2f663": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_d1714b06f9b24fdcb4fb40ede36d9149",
       "max": 2,
       "style": "IPY_MODEL_32ed88399c784b8589c41ef22264a6f7",
       "value": 2
      }
     },
     "4190847bd0bc4376a15cb591aba5b902": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_b1daf4b4191a4a80928799a74dae3829",
        "IPY_MODEL_c863855be51e48aea32d3acbed5e906b"
       ],
       "layout": "IPY_MODEL_78f413bba67c4397814759d60fc7aa64"
      }
     },
     "4198b858cbdd4d52801d65adce4e7aea": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "41a1041df1e24f0082fd32d3e28f8289": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_8e4c6a49b90344e1ad5197a5aa9df450",
       "max": 2,
       "style": "IPY_MODEL_12fdd2ec99da4ac080d2a24b30d777be",
       "value": 2
      }
     },
     "41e84ffaca0a4bb596d9bf970ed70752": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_9e7e7006e76347cfb21f1453cead6cc0",
       "max": 300,
       "style": "IPY_MODEL_eb8fa51d45c04b60a7122ae24b6e445a",
       "value": 300
      }
     },
     "422bb40b54c347ef8a0ed9f5187eb8ac": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4238d2986e9b4efeb6d8e5f9d5695c18": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_9fd93d0caa8b43d38438bbc8a8268516",
        "IPY_MODEL_ab916a934ef043efa57547624f11ad39"
       ],
       "layout": "IPY_MODEL_1c77ddbe02434eb89476dbe1bd5d0d74"
      }
     },
     "431385120eab4d78991f200420d5957c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "4320bbdf72a348a38496d24de28574b4": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4332d5782bf84883a614d7dd535644d4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_91900a0131ea47018a6185433cfe384e",
       "style": "IPY_MODEL_c7c2c9410b444ca990b6a0a5ca9fac96",
       "value": " 0/200 [00:00&lt;?, ?it/s]"
      }
     },
     "442a19f8ce1f4cb4b277182c54f37412": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_ded6bbcf69174565a96a3434936805aa",
       "style": "IPY_MODEL_6820cd5fd8154861a28f5271de3fa23e",
       "value": " 400/400 [51:31&lt;00:00,  7.73s/it]"
      }
     },
     "447a462ad35d4cc7af1ae9f408b79704": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "44a22c1eaafc4672a029097656db33ff": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_98ab0e6f34664c63b13604fcf4d54850",
       "max": 300,
       "style": "IPY_MODEL_4685d1d27e924c01b65ff420ce6292e4",
       "value": 300
      }
     },
     "4590898e33714706a0c729e472bdc936": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "46509eeb153e46f1a91b8cbcae039e25": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4685d1d27e924c01b65ff420ce6292e4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "46a01765cf944f7f94a59406439d9b8c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_a7b71edb1f18461182063ba01e2e8153",
       "style": "IPY_MODEL_f082d1f6c2d3423f8dd5f01d75157924",
       "value": " 200/200 [02:53&lt;00:00,  1.15it/s]"
      }
     },
     "477bcad1abed42ada6dbf67c640d0efb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_611d39563ef846db93fd9f7cd814a803",
        "IPY_MODEL_391fcfa48f1b4e36997ef0702f10ad5d"
       ],
       "layout": "IPY_MODEL_075d72f8ad794bcaac1ebcaabfc020b8"
      }
     },
     "479d33fb42b24f48a2d27ea6d7285351": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "491d6d6139fa47119156657b8334d046": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_0bf81f2702d3476ab3f53fcec853a9ac",
        "IPY_MODEL_6fb4e131d37a4198a1d9ef56f80f87df"
       ],
       "layout": "IPY_MODEL_b9131bbb84164a8d80b478d19d0a24f0"
      }
     },
     "494f8306d0d04f449b2fc8ed1e026fb6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_41e84ffaca0a4bb596d9bf970ed70752",
        "IPY_MODEL_864f93670a134cd9bbd23b54cbf7176e"
       ],
       "layout": "IPY_MODEL_fce90cf031fc4aa8b3db5822bb26f01f"
      }
     },
     "49583b10f48e4c2cbd0faa71dbc222f0": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4afd2928c81b4097b24f56435aa94cae": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4b11d7442c63475285daa788767f0d8d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "danger",
       "description": "  0%",
       "layout": "IPY_MODEL_fba80bacb5824f839d556bbf55c1aad6",
       "max": 200,
       "style": "IPY_MODEL_c3a28bf75ebc47099aa29cd99190f58d"
      }
     },
     "4b21bd8461824452a2124200792830d9": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4b2d8003d1fe4fcf801c921c740ae6dc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_8c46653c18cc447bac0c070b04a39364",
       "max": 300,
       "style": "IPY_MODEL_70deed10e21e47fabe60f9e5d9d46aa2",
       "value": 300
      }
     },
     "4b393b795b7d42f9968a8916cbaa56e5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4bd0762e7a8d42fa9f432c8ac4629e94": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_34197b6f340045a58a6e3c83cda8ecef",
       "style": "IPY_MODEL_74fd0b4380af445b8aa4987236bd7fe0",
       "value": " 200/200 [11:44&lt;00:00,  3.52s/it]"
      }
     },
     "4be5e471a19542dc949292ffae9d71a9": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4cb2da30f6334ef39b20ea61a77779db": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "4cd41f6f88f8437698197b6ada89e1fc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_023346bb70044e5682f05da2f85399c9",
       "style": "IPY_MODEL_7f16ba386f644f51b27ad7fe4c08c093",
       "value": " 27/200 [3:37:03&lt;23:10:46, 482.35s/it]"
      }
     },
     "4df2431a6ef24100b4aa8c9db5494243": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_6ad5f23bfb0d4a998496c686b04872de",
        "IPY_MODEL_4f2708cc04d84c51b9762cd2bc86e847"
       ],
       "layout": "IPY_MODEL_d0bb16bcc3944068b9ca2dacbee2fad6"
      }
     },
     "4f0e78a0191b422990ef3965e3388c69": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "4f2708cc04d84c51b9762cd2bc86e847": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_736b343facfe4239b3719694a3db919f",
       "style": "IPY_MODEL_039cef2ba0cc4544963144e7132a4303",
       "value": " 200/200 [13:25&lt;00:00,  4.03s/it]"
      }
     },
     "4fc6aefe8f1044c1926884ef06d5a104": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "500c6483b18c4b2080976519239678d1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "5120d219c8f74d518e5e23f3746c0834": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_3c648575362040eab27a7212d28a5aa7",
       "style": "IPY_MODEL_d21c9a4c45304ff987cba8674fd6d44e",
       "value": " 300/300 [15:06&lt;00:00,  3.02s/it]"
      }
     },
     "51f0f7beefea4906801eb4798f190109": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_c5cd27b908e74622b3ef9c7d705e1e76",
       "style": "IPY_MODEL_a4364c35d04048f3902ef7c9be092fc2",
       "value": " 200/200 [03:29&lt;00:00,  1.05s/it]"
      }
     },
     "52211e5e56d947f1adabe6905bc1b20b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_3c648d21e8604436933da0b69ae4c3d6",
        "IPY_MODEL_78a68dd9fc314f3794813ca48fd7bad2"
       ],
       "layout": "IPY_MODEL_db82d5edcfdf447484a4c45a0c7b2599"
      }
     },
     "5283fa9e66614ea6a7145d7ab1f580bb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_7ab2d7e0745f492c95fdcc01ea526660",
       "max": 2,
       "style": "IPY_MODEL_09a069c8c4594fa6ad5d6e76cfe83701",
       "value": 2
      }
     },
     "52b4e3149f5243eba5136b9da6965740": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "52ded505a1e6408a8513729d78bf8e79": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "5314c6a70ae348b488a4bd5397fac7f3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "535c9cc66b084e52ad494e745ec9a23f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "54959e533bd045608c4e15e7246583d5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "54e532ae266b47b3b10e7b402bae0a26": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_93813a7f86d54a01bb419e5f7380577c",
       "max": 200,
       "style": "IPY_MODEL_fc3ac025200c4c4099b4abaca46ff3d5",
       "value": 200
      }
     },
     "565672af0927472c8ed34f8a27e1a83a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_eac6fbed32cd4f1b810ebf1ac66399b4",
       "max": 2,
       "style": "IPY_MODEL_d35d78bfaf124a9f92e65158e75ba90e",
       "value": 2
      }
     },
     "5691525b552147a497f463da43c63c9f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "569422e860bd43f2a4d5c2edb626b240": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "56f7f51f8ec44a99b6f7c19ed799593c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_8a4ec07820304bb3959a4886572e256b",
        "IPY_MODEL_f36de1e438dc4730bacab82a601fbeca"
       ],
       "layout": "IPY_MODEL_964f3401d730419a997c6932c0556d68"
      }
     },
     "572e65ba13f44f318f361d9cccf84154": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_4b11d7442c63475285daa788767f0d8d",
        "IPY_MODEL_7d42b0973dc54e289cccd7fbeff101e4"
       ],
       "layout": "IPY_MODEL_4fc6aefe8f1044c1926884ef06d5a104"
      }
     },
     "5883fca2f9f346469e0f2ea2e5b68bc7": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "58ae4c3efb8744728741939dcfd59907": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_1b645a854bed4edda33cf10cde01d963",
        "IPY_MODEL_a86ffc03083f4b2096f4ae5ba888f322"
       ],
       "layout": "IPY_MODEL_bedbd74cbd0a43f088571061a48186c9"
      }
     },
     "590112e86fa14ec0bd038269e1041e38": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "59ae1ef2028a48e9b1643ffa40c51a42": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "5a6b5846697d4ae9855f14d0629ca318": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "5aefb71b9c0e4565a3216fc3fff96d1a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_65294d58f59545a891e172385b40b7de",
       "style": "IPY_MODEL_9e89d0ce3d7b415b9c0968949ee04fa4",
       "value": " 2/2 [01:59&lt;00:00, 59.74s/it]"
      }
     },
     "5ba27fbd2e424400a21c37819f03e8e0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_07c8e7fd75f6499bb4fcff6160060c9c",
       "style": "IPY_MODEL_4590898e33714706a0c729e472bdc936",
       "value": " 2/2 [01:00&lt;00:00, 30.22s/it]"
      }
     },
     "5c44d8acf6824e51a956821b66798c88": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_78944d80568e4373939e650709180a93",
        "IPY_MODEL_b2c4d415c8cc42789e11e8d70e8af8af"
       ],
       "layout": "IPY_MODEL_a3ac1454cf3b4efea900927dfda0ced6"
      }
     },
     "5c79de21221744f892925486a0c51fbc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_30f63f246ab148118e394f6d33ec1f92",
       "style": "IPY_MODEL_63d4e9636d9b44c79e6300352f305236",
       "value": " 200/200 [01:32&lt;00:00,  2.16it/s]"
      }
     },
     "5d043d8a7273486e98db0c49f0e36ee2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_8e07d701daab4f5db990c0de63b01cf1",
        "IPY_MODEL_7ecfbfb9854e4fda99a5c91f66c8fbc2"
       ],
       "layout": "IPY_MODEL_447a462ad35d4cc7af1ae9f408b79704"
      }
     },
     "5d4d5c8fef9e438e9a69e8b23d7576e8": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_c0e9ac1df48a481c816244727794efe8",
       "style": "IPY_MODEL_143a9654993a482995b6f0d8c1a82676",
       "value": " 300/300 [00:56&lt;00:00,  5.33it/s]"
      }
     },
     "5e1ed0ae189a484cb35f7ae415afd4ae": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "5fcd3a9c4c3d4a1bb0ba90c09dd7cc98": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_4b2d8003d1fe4fcf801c921c740ae6dc",
        "IPY_MODEL_bc79e296edfc4f84a6a9f50f47dc9625"
       ],
       "layout": "IPY_MODEL_49583b10f48e4c2cbd0faa71dbc222f0"
      }
     },
     "60aca04caa9843e3b49884243e023069": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "611d39563ef846db93fd9f7cd814a803": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "danger",
       "description": " 36%",
       "layout": "IPY_MODEL_7b20403c7afc4e0f8c2e12a7b586663a",
       "style": "IPY_MODEL_401575bb4d4d4bfd8a2db2a25a3c54c3",
       "value": 36
      }
     },
     "6171321a457b422b8131f9d1db191a56": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "633c27ebb4784f588dd25de209252431": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "63d4e9636d9b44c79e6300352f305236": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "63dc0d0ac074454ba16035252ef76a91": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "644605e0c20844e4baa9b9298c09968c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "64ba619722a34f2183ef532677aeeb07": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "65294d58f59545a891e172385b40b7de": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "65c16eee1eb94d27913f27e95d6bafd6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "66b052767be94bffb302f2a7c936aa16": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "6820cd5fd8154861a28f5271de3fa23e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "69537409303a4f40a56a9522306db183": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_a58ee14276f14bb88a5da2ece8f19b48",
        "IPY_MODEL_b825c71657ab40379288da89a629def1"
       ],
       "layout": "IPY_MODEL_fd228f36d47146c69e21c197412552c5"
      }
     },
     "69edbcef48c943f0993dc38834253bb9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "6ad5f23bfb0d4a998496c686b04872de": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_9676cfc9acb24bb7abce84754d2a2ae3",
       "max": 200,
       "style": "IPY_MODEL_b9cc6ca808bc49faaad07328c1e7c4d7",
       "value": 200
      }
     },
     "6b9add0f90054681b3649651e4c27477": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "6be987b06cd74f418111f30115cb0667": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_371c0a34975d4df6be3632a3b5689d2a",
        "IPY_MODEL_a1537d90ea9b415bad04cc56840529da"
       ],
       "layout": "IPY_MODEL_c3f601ca181a41c6a633d66aa8ad849b"
      }
     },
     "6d4d7ca5046e43988c3b0963efce41e1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "danger",
       "description": "  0%",
       "layout": "IPY_MODEL_d1b128436ddb4a2c9dcbef3b4c13f463",
       "max": 200,
       "style": "IPY_MODEL_30c37a13ee3c400bbe33e366fc614d7b"
      }
     },
     "6dea30e9d6744d1fb4873bcc0309d95d": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "6e053ade1bba42469303278ad0a8709d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_4b21bd8461824452a2124200792830d9",
       "max": 2,
       "style": "IPY_MODEL_77ac39df370c442fbdb75dfe327ca68d",
       "value": 2
      }
     },
     "6ee8e580ec0a4dbc84d20ca5cdce3659": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_9347752520f446a0ac46a244c7c90c23",
       "style": "IPY_MODEL_041c4577cd154a11bd203ba8b9ef0814",
       "value": 100
      }
     },
     "6ee9907fb4e34df1adf991591c3b6ac8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "6eef5afda12242458b9ae0f5dd6d62a0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "6f0cc152ae8a497888b014ab82d7593e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "6f2c91d254e441e8a51ccd861d5e5e25": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "6fa11001c48e4cdea94a4046fa3d7ff4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "6fb4e131d37a4198a1d9ef56f80f87df": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_5691525b552147a497f463da43c63c9f",
       "style": "IPY_MODEL_75850f26227847abaa5d329486d25869",
       "value": " 200/200 [01:48&lt;00:00,  1.84it/s]"
      }
     },
     "70671b44b73f4fab81e8e23d0d615cb7": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "70a2219d96794c6497563586ffd94141": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "70deed10e21e47fabe60f9e5d9d46aa2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "71ce339aeb8e4bb0826cd240ce32a924": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "736b343facfe4239b3719694a3db919f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "73c21c21c4524f5f8363e90b465737d1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_0e539d49dff34320bea9d52213f3efb9",
       "max": 2,
       "style": "IPY_MODEL_69edbcef48c943f0993dc38834253bb9",
       "value": 2
      }
     },
     "74ea9583a7da4266a428df9f5aafd8d5": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_3f39e1a5b59d4340aa580cc7117f5cef",
       "style": "IPY_MODEL_2bd914e6301341aab54384c9e5d9e40c",
       "value": " 300/300 [09:12&lt;00:00,  1.84s/it]"
      }
     },
     "74fd0b4380af445b8aa4987236bd7fe0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "75850f26227847abaa5d329486d25869": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "75b3e310ef8f4a54a3597cd7b372bf30": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "76abbd606f7a485dbe40ef58285f3e0e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "776d0e21db534eacb29f09a921e34b91": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "77ac39df370c442fbdb75dfe327ca68d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "77bd90f973014f17a865ccd14b15d3d3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "77cf50eb56be46f1acc17774e03d1846": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "7838d64623284d3d8a0c4b705c5705c8": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_05746fa0e0ec4b9e9809f872a2409e61",
        "IPY_MODEL_5ba27fbd2e424400a21c37819f03e8e0"
       ],
       "layout": "IPY_MODEL_4320bbdf72a348a38496d24de28574b4"
      }
     },
     "78391f741a51400fbfdde626a3c24ba6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_cb2b8c6b4e844840ab180e0e1ec37dfd",
        "IPY_MODEL_2e19164d8cd34b8a803b01964f58e141"
       ],
       "layout": "IPY_MODEL_f33c23a7e9b64d809207a1c01934f351"
      }
     },
     "786a96942a914f69b8fcead6f535a674": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "7877849a0b5f4525aa1d6ecfdf19bb81": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "78944d80568e4373939e650709180a93": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_6dea30e9d6744d1fb4873bcc0309d95d",
       "max": 300,
       "style": "IPY_MODEL_f5a34d7312774abfbd381767fee34b15",
       "value": 300
      }
     },
     "78a68dd9fc314f3794813ca48fd7bad2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_f28c596da1c34a89bd55e760aa3ba280",
       "style": "IPY_MODEL_2142b21e4b614286b407f8b963edbed0",
       "value": " 2/2 [00:00&lt;00:00,  4.92it/s]"
      }
     },
     "78d133fcdcc64f0e9c157104a07498a8": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_a1621714596f4f69b3eee003ca2ff0c2",
        "IPY_MODEL_5d4d5c8fef9e438e9a69e8b23d7576e8"
       ],
       "layout": "IPY_MODEL_cf7d571e90a248c18c36bc126415f8bb"
      }
     },
     "78f413bba67c4397814759d60fc7aa64": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "793f24a97a18438fa113cd308e77ab1a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "795629f364aa4de2b1fc6a673c728c94": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "7ab2d7e0745f492c95fdcc01ea526660": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "7affcb4f3c014e098d4ef3935319cca2": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "7b20403c7afc4e0f8c2e12a7b586663a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "7b5fedaeb68b4e109a132f6024c00758": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_500c6483b18c4b2080976519239678d1",
       "max": 200,
       "style": "IPY_MODEL_30c1af7383b34066a34b851405c9224b",
       "value": 200
      }
     },
     "7bb28a9ed2d2490e947bd5131ad4cfd7": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "7c622b04a67d4563af59c80b6eebe4c1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "7cb22a856cd54450afa5386cbb319a61": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_99df768f86b647fcb0c82b2fdf07e378",
       "max": 200,
       "style": "IPY_MODEL_99f392a21bc24b25ae9ee17aba3cb896",
       "value": 200
      }
     },
     "7cf8f00f1d424d52b7311349c99bedcf": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_ae5addb52d224935935dc16673c8ed0c",
       "style": "IPY_MODEL_fb0322676ef6410dae53a96e5a08123b",
       "value": " 2/2 [00:59&lt;00:00, 29.94s/it]"
      }
     },
     "7d07aae6fd7043bc88023fa95168af3f": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "7d33a349fd384f33b124d8ab9a6a541e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "7d42b0973dc54e289cccd7fbeff101e4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_a6cd2e74e01d40ed965693b1d5a40a2b",
       "style": "IPY_MODEL_f105b822ebfb43909ef152d96bb9ab00",
       "value": " 0/200 [4:53:27&lt;?, ?it/s]"
      }
     },
     "7dcbc753322043c9bcb3da800e75db79": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "7ecfbfb9854e4fda99a5c91f66c8fbc2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_22abd941f4214f6688eacea066cd083e",
       "style": "IPY_MODEL_258346915b2c40ecabeceec670bff283",
       "value": " 300/300 [01:00&lt;00:00,  4.99it/s]"
      }
     },
     "7f16ba386f644f51b27ad7fe4c08c093": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "7f81e141eb3b4fd88a3934316bf11824": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_c6e54443904e410abb351dccc23bca5b",
       "max": 400,
       "style": "IPY_MODEL_75b3e310ef8f4a54a3597cd7b372bf30",
       "value": 400
      }
     },
     "7f865c763c074d128c345864ccb0aaa3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_54959e533bd045608c4e15e7246583d5",
       "max": 200,
       "style": "IPY_MODEL_77cf50eb56be46f1acc17774e03d1846",
       "value": 200
      }
     },
     "80e8324050854bd790bcfeb8eb647b81": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "813e680a9ae84b68bf3d3dd81f2cb483": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "8177175ee5f84a5bb4e4495284e3329d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "81eb122cc31e436fa2518540b509d67f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "81f90fd6c6494070b2749c48b49563c4": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "8242093b80f842448baf3b0d3e1298ac": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "825638143d814b42b24892ce66d290c4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_1f13321849cb4357bdbbdf65400bd224",
       "max": 200,
       "style": "IPY_MODEL_040e16ebba49457f9b5d6a315c9554fb",
       "value": 200
      }
     },
     "8309c3cbff684aaaa1bc8a88524d6aa9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "8381899ede4f4c13a9deb3e18fa709e6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "84455b0a64a74294bd5814ded73c05f5": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "85062f5084a34c53b9070bf241f0f6eb": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "85347a3174794baaab724fb8c308f963": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_ecdb0901f6a64234afe87251cc2861d9",
        "IPY_MODEL_22b5281f57f44b3d81b4484e1328abaf"
       ],
       "layout": "IPY_MODEL_2676372c45bd45fa95569015aa8b4068"
      }
     },
     "864f93670a134cd9bbd23b54cbf7176e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_422bb40b54c347ef8a0ed9f5187eb8ac",
       "style": "IPY_MODEL_bd2510ebcf1343b391e8b14656725e21",
       "value": " 300/300 [02:41&lt;00:00,  1.85it/s]"
      }
     },
     "87f5c560a4b647578512d1f4883a73ed": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "8823c082c8ed493cb3178e340ab8f996": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "8829c85b980846c2b000aced08613a3a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "894399c1f4c94c3ebde809aed58fcc0d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_6d4d7ca5046e43988c3b0963efce41e1",
        "IPY_MODEL_4332d5782bf84883a614d7dd535644d4"
       ],
       "layout": "IPY_MODEL_c694c077c4ce45c79d5a89fd2150a82a"
      }
     },
     "8a3108b7fa6243b094d6e159f76bb71b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "8a4ec07820304bb3959a4886572e256b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_f2537df39ba74b7baf88efccc73102c5",
       "max": 300,
       "style": "IPY_MODEL_d7c788e387144ff58717ac2f5add29fd",
       "value": 300
      }
     },
     "8a539cd815044bdcb9e3effc992898aa": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_a46db3dcc2824167bdc2eea2cb1a67dd",
        "IPY_MODEL_442a19f8ce1f4cb4b277182c54f37412"
       ],
       "layout": "IPY_MODEL_16cdec1d589e4b84b9e1432f9d326da8"
      }
     },
     "8a574bff7bb944aa9c3f2d4f7cefd001": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "8ab38bd0c0104b04bcf7e37d5d451dca": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_41a1041df1e24f0082fd32d3e28f8289",
        "IPY_MODEL_5aefb71b9c0e4565a3216fc3fff96d1a"
       ],
       "layout": "IPY_MODEL_f33d9d8677b342a5b82d7fad684589f9"
      }
     },
     "8b3b79ffdd8f41bdb0cc20498532c99a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "8c46653c18cc447bac0c070b04a39364": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "8ccc549d96f74e5aa9205459c1e3b345": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "8d4b6c6c2c904fe39a50a16bf0ee1702": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "8d6012b4c1384571ad3f4902f5738572": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "8e07d701daab4f5db990c0de63b01cf1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_df585ffddcb848aea7591f62a6c563ba",
       "max": 300,
       "style": "IPY_MODEL_32d071ff0e994878a3ede66e8fc1a5ff",
       "value": 300
      }
     },
     "8e4c6a49b90344e1ad5197a5aa9df450": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "8e8af27098434fa680e7c0b40eef899f": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_6ee8e580ec0a4dbc84d20ca5cdce3659",
        "IPY_MODEL_bab03ad950694dfb8972b97a47a2270a"
       ],
       "layout": "IPY_MODEL_907d3e0a03af405cae7c2e34c33026be"
      }
     },
     "8e9e9e15459c4b69b13450b434a7cbab": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "8ec0ea1cfc28409591bc5c620bb3a366": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "8fabdcdea24b4b54a9bbdd56da662bcf": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "904203b941fc4bd6844520979fa109de": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "9049915c466449bf9fc9e8234f8810dd": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "907d3e0a03af405cae7c2e34c33026be": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "90af23406818480c89f896fe463aef3b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_24486927ecf1462d9deafc8721c5a67a",
       "style": "IPY_MODEL_36285193f99749d381423663b551b044",
       "value": " 300/300 [00:54&lt;00:00,  5.51it/s]"
      }
     },
     "916530172abc4fb3b7190ea2a11c7e3a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "918bb4b8c1a64800a0f68742d0a4eb03": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "91900a0131ea47018a6185433cfe384e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "92964a4277cf4e1db0ab54de566c53f1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "92df9886389744c68470d97e09e556e1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "9339a5d8fab34838a642627ec736ee7d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "9347752520f446a0ac46a244c7c90c23": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "93813a7f86d54a01bb419e5f7380577c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "93deaf444cbe4c4f9fa4a89d899badc3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "947ccf849d4a40c9a5ae8e6633876809": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_b41cb8612e2e4cb0a7fd4051a069dafd",
        "IPY_MODEL_f4af073fe03e408bab08676718edc754"
       ],
       "layout": "IPY_MODEL_c82e33abb4ff434cbed06ce9530bb95c"
      }
     },
     "95107cb944314535a49dd5085f2f5947": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_b5ec8fc3592743daabebcec33adc0384",
       "style": "IPY_MODEL_fb63e87baae64d1bb51747a09af59d0c",
       "value": " 400/400 [58:19&lt;00:00,  8.75s/it]"
      }
     },
     "95744244bde34b35a990ff69d32b0a5b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "95bdc7a7fc5842cf9eeb07f811fa338e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "9637f0b173ee4e4b8666817f3c144134": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_2eac5e4754a04871ad5524b632bedfb5",
       "style": "IPY_MODEL_aa055abd62e04115949e37da49b6a023",
       "value": " 0/200 [3:45:23&lt;?, ?it/s]"
      }
     },
     "964f3401d730419a997c6932c0556d68": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "9676cfc9acb24bb7abce84754d2a2ae3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "96a94721e56e42119b08ff65b78d2269": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_cba25e88e2c54b72885b0c4e85288bab",
        "IPY_MODEL_40cc03073d25468e91fecf6f1913cfac"
       ],
       "layout": "IPY_MODEL_6171321a457b422b8131f9d1db191a56"
      }
     },
     "970d6f561807474e8170da3045e3220d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "9716614967bc4d30859edab335b605e3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "97283ed56c1043e0ae3331d195230a0b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "98ab0e6f34664c63b13604fcf4d54850": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "990a36284a4540c4b05288f2eb363276": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "99973e1e8c19420f94305738043628a6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "99df768f86b647fcb0c82b2fdf07e378": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "99f392a21bc24b25ae9ee17aba3cb896": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "9a2aeb063e5141259cafc1a099cb2b09": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "9a72932bbd2f4487acabc05224f0eecd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "9b436a9d3c21459f9311bd4e5208449b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_347f9d470b754d66b877ea7932a69209",
        "IPY_MODEL_04abc016a87541f48a7df9b6a35ba8c9"
       ],
       "layout": "IPY_MODEL_77bd90f973014f17a865ccd14b15d3d3"
      }
     },
     "9da400dd12594b29beb1069ddf179b62": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "danger",
       "description": "  0%",
       "layout": "IPY_MODEL_156d230fa70f4fb2bbc7efed3ce55f50",
       "max": 200,
       "style": "IPY_MODEL_199f54999a9f4b818fa156890b76ef05"
      }
     },
     "9db566bdbaea46f2a3677493b9e227c6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_23fe8f5977a44673ad7dd704fee8dc31",
        "IPY_MODEL_ed98421ab07449bdbfdad6f6a7125cc2"
       ],
       "layout": "IPY_MODEL_8ccc549d96f74e5aa9205459c1e3b345"
      }
     },
     "9dcd5b65fa6c42cea1fb1cb2e41d7642": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_d28f565b70934cfda746a97502cc06f1",
       "style": "IPY_MODEL_be0b46731a1b452299192868703f53cc",
       "value": " 200/200 [01:54&lt;00:00,  1.75it/s]"
      }
     },
     "9e7e7006e76347cfb21f1453cead6cc0": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "9e89d0ce3d7b415b9c0968949ee04fa4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "9fd93d0caa8b43d38438bbc8a8268516": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_1e1c21bba5bd442cabb00ed2cb4910d7",
       "max": 200,
       "style": "IPY_MODEL_7dcbc753322043c9bcb3da800e75db79",
       "value": 200
      }
     },
     "a1537d90ea9b415bad04cc56840529da": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_6f2c91d254e441e8a51ccd861d5e5e25",
       "style": "IPY_MODEL_09871093305140deb2c16d792d5708df",
       "value": " 0/200 [3:44:54&lt;?, ?it/s]"
      }
     },
     "a1621714596f4f69b3eee003ca2ff0c2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_6ee9907fb4e34df1adf991591c3b6ac8",
       "max": 300,
       "style": "IPY_MODEL_87f5c560a4b647578512d1f4883a73ed",
       "value": 300
      }
     },
     "a1a17343c6444227bfe3a68f3550e33e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a26eef444a184f29ad42e8f10f5addca": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a2a60c4849794768a5fe96ce5161b892": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_ba2e3c2689634917be04cb879a4fdebb",
       "style": "IPY_MODEL_70a2219d96794c6497563586ffd94141",
       "value": " 300/300 [00:56&lt;00:00,  5.30it/s]"
      }
     },
     "a2c373dfca9546e2b3a82023a08586b0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_c88c41d9f4ca4f60b313c112668ec5c2",
        "IPY_MODEL_95107cb944314535a49dd5085f2f5947"
       ],
       "layout": "IPY_MODEL_19bb316c90624805acb7f93073376ac4"
      }
     },
     "a3044cd2d6b64d789fee113ff02def66": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_6e053ade1bba42469303278ad0a8709d",
        "IPY_MODEL_c03d1ebf8f3448e38b306b1e212b5b3d"
       ],
       "layout": "IPY_MODEL_3a2eef88782e423d803d7d7a7f39802b"
      }
     },
     "a3ac1454cf3b4efea900927dfda0ced6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a3c1d79d501a4402a640747905909e85": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a4364c35d04048f3902ef7c9be092fc2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "a46db3dcc2824167bdc2eea2cb1a67dd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_6f0cc152ae8a497888b014ab82d7593e",
       "max": 400,
       "style": "IPY_MODEL_85062f5084a34c53b9070bf241f0f6eb",
       "value": 400
      }
     },
     "a570aa43fec443929ccbb865f206f8c4": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a58d484527db4b7f9d001c119bef2616": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_39dcca616a374486bb6cc48db683bf55",
        "IPY_MODEL_a2a60c4849794768a5fe96ce5161b892"
       ],
       "layout": "IPY_MODEL_f6808c1d7f9d45d597266787677f1b58"
      }
     },
     "a58ee14276f14bb88a5da2ece8f19b48": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_4afd2928c81b4097b24f56435aa94cae",
       "max": 2,
       "style": "IPY_MODEL_8823c082c8ed493cb3178e340ab8f996",
       "value": 2
      }
     },
     "a638af5e7bc1411e9a56bd4eef41bd67": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_336c2f60995e4601bc967b86e8516d99",
        "IPY_MODEL_5120d219c8f74d518e5e23f3746c0834"
       ],
       "layout": "IPY_MODEL_66b052767be94bffb302f2a7c936aa16"
      }
     },
     "a6cd2e74e01d40ed965693b1d5a40a2b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a7b71edb1f18461182063ba01e2e8153": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a8195f66e3bf4b13a69a4f8c5d875b07": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "a86ffc03083f4b2096f4ae5ba888f322": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_81f90fd6c6494070b2749c48b49563c4",
       "style": "IPY_MODEL_6eef5afda12242458b9ae0f5dd6d62a0",
       "value": " 0/200 [3:42:48&lt;?, ?it/s]"
      }
     },
     "a9552da95dbd437c97870dc5b83b1617": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "a9cec229ecb34482afb6187a5dcdd6ee": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "aa055abd62e04115949e37da49b6a023": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "aa5b5994bc4b4223a0e41e85912767c1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "aaaca3205a8d4e84a874b80e6919b9d0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_565672af0927472c8ed34f8a27e1a83a",
        "IPY_MODEL_214edccc75174a0a9ef6b47be16bceb3"
       ],
       "layout": "IPY_MODEL_e3f6e67162e84c7f98ef7e57a9bdf590"
      }
     },
     "ab0893c752c84b97b740a08a34c89a2a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "ab4173afdbbf4496bfd4fa490f6ff84b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "ab916a934ef043efa57547624f11ad39": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_f2550e33567d4b1da968234e96877090",
       "style": "IPY_MODEL_d3d55d8460d84b0ea1cb32bf5b4b2222",
       "value": " 200/200 [03:11&lt;00:00,  1.04it/s]"
      }
     },
     "ac30774fcfe24e3c8888cf18558a6bfc": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "ae5addb52d224935935dc16673c8ed0c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "af1bc4e3b0274dd78db5ddf0e3b451f0": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "b041420243ef426dbdc382115119e8e3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_aa5b5994bc4b4223a0e41e85912767c1",
       "max": 2,
       "style": "IPY_MODEL_dcb1f674394c46fbb2e695ef53cab115",
       "value": 2
      }
     },
     "b0478dd6470e47edad71ab2b04f29fb9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "danger",
       "description": " 14%",
       "layout": "IPY_MODEL_4b393b795b7d42f9968a8916cbaa56e5",
       "max": 200,
       "style": "IPY_MODEL_b1a566a7afba4166b132134be2081cc7",
       "value": 27
      }
     },
     "b09dde67a7fa4db78b8b19cf55b71ac3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_cfb1ee1152404722b268bcf4ed2e5d76",
        "IPY_MODEL_d3b0d092bc394271b9199b4b12254184"
       ],
       "layout": "IPY_MODEL_a3c1d79d501a4402a640747905909e85"
      }
     },
     "b1a566a7afba4166b132134be2081cc7": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "b1daf4b4191a4a80928799a74dae3829": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_05e2a4880d8c42db8432e75813854c6e",
       "max": 2,
       "style": "IPY_MODEL_d37c4542f4c04ca8aaac8c1d8d4f3b90",
       "value": 2
      }
     },
     "b22b22f3d70541108992de0120e6d5d3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "b27d7e46c5d74f459c5dae636ba0e3f0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_3bb25574b9824b02a1895a42cd36573e",
        "IPY_MODEL_c09bfccae89c4c1893db9ee52091d294"
       ],
       "layout": "IPY_MODEL_204400c2b64b4a8baba57f92ba638205"
      }
     },
     "b2c3096e6e9a4a6d9da80fd1ceae1b78": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_3fe6aecfd5c14291b52f3a2b9fa844b2",
       "style": "IPY_MODEL_12d9758aab96413b843a5cb49c49cf86",
       "value": " 2/2 [00:58&lt;00:00, 29.16s/it]"
      }
     },
     "b2c4d415c8cc42789e11e8d70e8af8af": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_7bb28a9ed2d2490e947bd5131ad4cfd7",
       "style": "IPY_MODEL_8177175ee5f84a5bb4e4495284e3329d",
       "value": " 300/300 [02:04&lt;00:00,  2.42it/s]"
      }
     },
     "b304c23114fc4b368054fdedebe98bec": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_a9552da95dbd437c97870dc5b83b1617",
       "max": 2,
       "style": "IPY_MODEL_92964a4277cf4e1db0ab54de566c53f1",
       "value": 2
      }
     },
     "b3f9bcfb2ac44ea8b58b9f4e2e9c98d4": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_f33f34cb34fe4013ba87ffeb6803b8ec",
       "style": "IPY_MODEL_d604818302984961a7c16895b9889a07",
       "value": " 300/300 [02:38&lt;00:00,  1.89it/s]"
      }
     },
     "b41171bb1fcd40d8b615d813192e749f": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_e9a87ce747114b1d8ab1d79318ce8d58",
        "IPY_MODEL_4bd0762e7a8d42fa9f432c8ac4629e94"
       ],
       "layout": "IPY_MODEL_2beaafc1a6584294ba6d6d51d675e9cc"
      }
     },
     "b41cb8612e2e4cb0a7fd4051a069dafd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_a26eef444a184f29ad42e8f10f5addca",
       "max": 2,
       "style": "IPY_MODEL_fe813fa0b6104b20bf37b7cf22fdb994",
       "value": 2
      }
     },
     "b46f9aa612b543249c5f1ba788f40d73": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "b4f2c7a1dece4fc9b36c0268d4861790": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_01746170b5794cd2942e0d5ed3c1d8ae",
        "IPY_MODEL_12d68e6f9fae46da8dfdd6fb467efa8a"
       ],
       "layout": "IPY_MODEL_108357b9ba4740da8b5c689fb34e9141"
      }
     },
     "b5ec8fc3592743daabebcec33adc0384": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "b71f99c19c3c402dafd7644523e5b5b1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "b74cbed1f8d74b04a642eb4ea6020ea6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "b7650c9c6576451f995c8c5fd7259417": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_3bdb33b3c17045189cf6f154ab6f948c",
        "IPY_MODEL_2166534bcde14ddda56d9d1c106bcdfb"
       ],
       "layout": "IPY_MODEL_33c1904fd85d471c81cf350dfd006e00"
      }
     },
     "b825c71657ab40379288da89a629def1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_f9373a82cc314bd292402a8d8cae8748",
       "style": "IPY_MODEL_6fa11001c48e4cdea94a4046fa3d7ff4",
       "value": " 2/2 [00:27&lt;00:00, 13.69s/it]"
      }
     },
     "b88644021e7f43928d5ae08371a892fd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_5e1ed0ae189a484cb35f7ae415afd4ae",
       "max": 200,
       "style": "IPY_MODEL_280f8786057d40aaa970861990da1c18",
       "value": 200
      }
     },
     "b88f3049eeee41e8bfd65d3fea6811f5": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "b9131bbb84164a8d80b478d19d0a24f0": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "b9cc6ca808bc49faaad07328c1e7c4d7": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "b9f67e0cac1742e4b245b4a27f45c5d1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_005b3706ebb94e9998e51963ff182dd5",
       "style": "IPY_MODEL_7d07aae6fd7043bc88023fa95168af3f",
       "value": " 200/200 [00:33&lt;00:00,  6.05it/s]"
      }
     },
     "ba2e3c2689634917be04cb879a4fdebb": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "bab03ad950694dfb8972b97a47a2270a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_786a96942a914f69b8fcead6f535a674",
       "style": "IPY_MODEL_28e9c87517254377a37144a64d9fab82",
       "value": " 100/100 [02:52&lt;00:00,  1.73s/it]"
      }
     },
     "bc79e296edfc4f84a6a9f50f47dc9625": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_8e9e9e15459c4b69b13450b434a7cbab",
       "style": "IPY_MODEL_71ce339aeb8e4bb0826cd240ce32a924",
       "value": " 300/300 [00:57&lt;00:00,  5.17it/s]"
      }
     },
     "bd1cf0ada0ef47028539a8fa794d0120": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "bd23b9e83dde4ff3bdcf82f41950d6cc": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "bd2510ebcf1343b391e8b14656725e21": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "bdfcde8b3a0d4b60a43ce46c9cb4f490": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_52b4e3149f5243eba5136b9da6965740",
       "style": "IPY_MODEL_ed08e5b49c174c7198b4c3c89a5ef5cc",
       "value": 100
      }
     },
     "be0b46731a1b452299192868703f53cc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "be2508cb7ce64495acd34436c96ac58d": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "be819e0b4cc24b29aa88c4ce487888c2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_e07d90e463f94f0280640cfb3340ec4a",
       "max": 300,
       "style": "IPY_MODEL_e5e7d83c4ebc41b9a04a04ab7029d9a6",
       "value": 300
      }
     },
     "be8a848b0dbb47fc8f4b15a55244c94f": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "bebc345d8cf74999b6e2a25609d49062": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "bec0f0776ec14a43ab44f654baddbcd3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "bedbd74cbd0a43f088571061a48186c9": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c03d1ebf8f3448e38b306b1e212b5b3d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_b46f9aa612b543249c5f1ba788f40d73",
       "style": "IPY_MODEL_be8a848b0dbb47fc8f4b15a55244c94f",
       "value": " 2/2 [00:00&lt;00:00,  4.73it/s]"
      }
     },
     "c07fe3c2d25a4975975b15fe228884ee": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c090dc7ad12340789ba2d7d72f995729": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c09bfccae89c4c1893db9ee52091d294": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_be2508cb7ce64495acd34436c96ac58d",
       "style": "IPY_MODEL_ee0aca055d704c9ab122e6ffaae095bc",
       "value": " 200/200 [00:35&lt;00:00,  5.56it/s]"
      }
     },
     "c0e9ac1df48a481c816244727794efe8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c1406aebf4fc4d61b1cef0cdc516ff6e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_e03b05e652664c869023fc1aa132c7cb",
       "style": "IPY_MODEL_b74cbed1f8d74b04a642eb4ea6020ea6",
       "value": " 300/300 [08:12&lt;00:00,  1.64s/it]"
      }
     },
     "c190511f14da4810aec9df4f2c3a41f5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c194d3c1193846c18c83ae2a09ec55d0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_81eb122cc31e436fa2518540b509d67f",
       "style": "IPY_MODEL_1c4ae29996b641dbafe41bdc8a622b70",
       "value": " 2/2 [03:16&lt;00:00, 98.37s/it]"
      }
     },
     "c1feae097d3d4da9921a1a7e303f0a61": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_44a22c1eaafc4672a029097656db33ff",
        "IPY_MODEL_90af23406818480c89f896fe463aef3b"
       ],
       "layout": "IPY_MODEL_d6ce30b404ce4ff997807acd0d47b554"
      }
     },
     "c3a0ccbb943a47b6a510887ca7b7962a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_03f0b92465f244729fd9ffe68657a746",
       "max": 300,
       "style": "IPY_MODEL_5a6b5846697d4ae9855f14d0629ca318",
       "value": 300
      }
     },
     "c3a28bf75ebc47099aa29cd99190f58d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "c3f601ca181a41c6a633d66aa8ad849b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c55539f896074ba1ab57f267bbb2ce1e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "c5cd27b908e74622b3ef9c7d705e1e76": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c655374a178c44ffa49f2d0031c31e97": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "c694c077c4ce45c79d5a89fd2150a82a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c6e54443904e410abb351dccc23bca5b": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c7c2c9410b444ca990b6a0a5ca9fac96": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "c82e33abb4ff434cbed06ce9530bb95c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "c863855be51e48aea32d3acbed5e906b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_4f0e78a0191b422990ef3965e3388c69",
       "style": "IPY_MODEL_a9cec229ecb34482afb6187a5dcdd6ee",
       "value": " 2/2 [01:43&lt;00:00, 51.76s/it]"
      }
     },
     "c88c41d9f4ca4f60b313c112668ec5c2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_efb0d67b135840279979a935cadb1a63",
       "max": 400,
       "style": "IPY_MODEL_8fabdcdea24b4b54a9bbdd56da662bcf",
       "value": 400
      }
     },
     "ca6e65dc37c249308499fec40f7f5873": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_73c21c21c4524f5f8363e90b465737d1",
        "IPY_MODEL_7cf8f00f1d424d52b7311349c99bedcf"
       ],
       "layout": "IPY_MODEL_46509eeb153e46f1a91b8cbcae039e25"
      }
     },
     "cb2b8c6b4e844840ab180e0e1ec37dfd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_ee97729d91f842bab8609f16067bcdb0",
       "max": 300,
       "style": "IPY_MODEL_37e80dca0f3844519e288819d200a5a8",
       "value": 300
      }
     },
     "cba25e88e2c54b72885b0c4e85288bab": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_7c622b04a67d4563af59c80b6eebe4c1",
       "max": 2,
       "style": "IPY_MODEL_c55539f896074ba1ab57f267bbb2ce1e",
       "value": 2
      }
     },
     "cc41d24433854b0aaa2d0339cf5667fd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_36b984a22422429b926bda8dac248a67",
       "style": "IPY_MODEL_21ce6253ddab4eb58de3a8e84e943e6d",
       "value": " 2/2 [00:06&lt;00:00,  3.17s/it]"
      }
     },
     "cc46fc787dfc46e1933532edba52b4d1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_36e1baf5fb8b43a79f4b692e5d6f2b2c",
       "style": "IPY_MODEL_8a574bff7bb944aa9c3f2d4f7cefd001",
       "value": " 2/2 [02:32&lt;00:00, 76.12s/it]"
      }
     },
     "ce06ac6e90fd4bf99b8ed1574455f7ed": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_918bb4b8c1a64800a0f68742d0a4eb03",
       "style": "IPY_MODEL_64ba619722a34f2183ef532677aeeb07",
       "value": " 200/200 [00:33&lt;00:00,  5.88it/s]"
      }
     },
     "ce9b73bc52684b538566f7f6f8964eb3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "cf1052676d2646ec8aa389f65f52a6af": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_a570aa43fec443929ccbb865f206f8c4",
       "max": 300,
       "style": "IPY_MODEL_776d0e21db534eacb29f09a921e34b91",
       "value": 300
      }
     },
     "cf69a28284744cb19eb4962e17a61157": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "cf7d571e90a248c18c36bc126415f8bb": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "cfb1ee1152404722b268bcf4ed2e5d76": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_7877849a0b5f4525aa1d6ecfdf19bb81",
       "max": 2,
       "style": "IPY_MODEL_9716614967bc4d30859edab335b605e3",
       "value": 2
      }
     },
     "d0bb16bcc3944068b9ca2dacbee2fad6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d0c8d1bddecf4ec7a0066a9f54bd0c6a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_a1a17343c6444227bfe3a68f3550e33e",
       "style": "IPY_MODEL_19a8df789fd944cca9081036753dcb38",
       "value": " 100/100 [01:13&lt;00:00,  1.37it/s]"
      }
     },
     "d1078b1205534916a518d952f3cc4ddc": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d117a8e1cd4f424289121c0e45aab561": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_0353ae8ac4ac41c5aa99006e56ec6b58",
        "IPY_MODEL_d590950d75b64408b08754ce95ec9f03"
       ],
       "layout": "IPY_MODEL_5883fca2f9f346469e0f2ea2e5b68bc7"
      }
     },
     "d1714b06f9b24fdcb4fb40ede36d9149": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d1b128436ddb4a2c9dcbef3b4c13f463": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d21c9a4c45304ff987cba8674fd6d44e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "d28f565b70934cfda746a97502cc06f1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d2f6efe846834a8da725c90a3a84fc9d": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_60aca04caa9843e3b49884243e023069",
       "style": "IPY_MODEL_970d6f561807474e8170da3045e3220d",
       "value": " 2/2 [01:58&lt;00:00, 59.47s/it]"
      }
     },
     "d35d78bfaf124a9f92e65158e75ba90e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "d37c4542f4c04ca8aaac8c1d8d4f3b90": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "d39585ef35ae46bdb29a4fe03ea7f80e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d3b0d092bc394271b9199b4b12254184": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_35a0e7ff2c5b41569870b35ccc9f7bb2",
       "style": "IPY_MODEL_0201f287dd224e82a7068749ccdffbfd",
       "value": " 2/2 [02:20&lt;00:00, 70.15s/it]"
      }
     },
     "d3d55d8460d84b0ea1cb32bf5b4b2222": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "d590950d75b64408b08754ce95ec9f03": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_bd1cf0ada0ef47028539a8fa794d0120",
       "style": "IPY_MODEL_289e351478334a148ce6924f6bde30b7",
       "value": " 300/300 [00:57&lt;00:00,  5.21it/s]"
      }
     },
     "d604818302984961a7c16895b9889a07": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "d6c4f7c9159c4e539c747b6e2983ed2b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "d6ce30b404ce4ff997807acd0d47b554": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d6e3fa47d15946abb6215174e3dc3847": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "d778a8e621f148768319876eee2276c5": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_7affcb4f3c014e098d4ef3935319cca2",
       "max": 300,
       "style": "IPY_MODEL_76abbd606f7a485dbe40ef58285f3e0e",
       "value": 300
      }
     },
     "d77e70209e394ad58c7746153dd65374": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_b041420243ef426dbdc382115119e8e3",
        "IPY_MODEL_d2f6efe846834a8da725c90a3a84fc9d"
       ],
       "layout": "IPY_MODEL_d6e3fa47d15946abb6215174e3dc3847"
      }
     },
     "d7b570e9ece448e7ac427b5f9a9d7aa0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "d7c788e387144ff58717ac2f5add29fd": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "d8b42274c88045ec8eeff13f5f009f17": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_95bdc7a7fc5842cf9eeb07f811fa338e",
       "style": "IPY_MODEL_fb2a06c4e11b417bb5bcd7259d494d26",
       "value": " 100/100 [00:26&lt;00:00,  3.80it/s]"
      }
     },
     "db33162ecd0d485eb50bd270b76c41cf": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_0c2b0e00bec64bd2a4feb98540a38153",
        "IPY_MODEL_dd11d43f6c434e84ab32f114371d30c6"
       ],
       "layout": "IPY_MODEL_2360820a5f7c48a3923e868e63951d53"
      }
     },
     "db82d5edcfdf447484a4c45a0c7b2599": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "dcb1f674394c46fbb2e695ef53cab115": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "dd11d43f6c434e84ab32f114371d30c6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_bec0f0776ec14a43ab44f654baddbcd3",
       "style": "IPY_MODEL_92df9886389744c68470d97e09e556e1",
       "value": " 2/2 [00:00&lt;00:00,  5.27it/s]"
      }
     },
     "de54816d45594d3e9ebf6a4dc355b7d9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_15fa43a5da0a4906855ceac366431d5b",
        "IPY_MODEL_edd3b4540b1c484fb88b5e7e818751e9"
       ],
       "layout": "IPY_MODEL_bebc345d8cf74999b6e2a25609d49062"
      }
     },
     "ded6bbcf69174565a96a3434936805aa": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "df585ffddcb848aea7591f62a6c563ba": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "e03b05e652664c869023fc1aa132c7cb": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "e07d90e463f94f0280640cfb3340ec4a": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "e1cd50413e864ede8fcee5b696107db3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_2a9591799c3544b894b018fa5e15a680",
       "style": "IPY_MODEL_f256104b4cf14ad082651ea23bfcd2c7",
       "value": " 200/200 [15:59&lt;00:00,  4.80s/it]"
      }
     },
     "e2a4e0fc30514b20adaee2fdee557be3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_97283ed56c1043e0ae3331d195230a0b",
       "style": "IPY_MODEL_8a3108b7fa6243b094d6e159f76bb71b",
       "value": " 200/200 [12:17&lt;00:00,  3.69s/it]"
      }
     },
     "e34a0470f4c4414381a0f5d8dd584c70": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_63dc0d0ac074454ba16035252ef76a91",
       "max": 2,
       "style": "IPY_MODEL_e4e71bff8119478888b409a852f8bac8",
       "value": 2
      }
     },
     "e3f6e67162e84c7f98ef7e57a9bdf590": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "e42fb4a4d5a943cc8104e4a4354bbf9b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "e4e31e648cf549dda28df8b1056041db": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_b0478dd6470e47edad71ab2b04f29fb9",
        "IPY_MODEL_4cd41f6f88f8437698197b6ada89e1fc"
       ],
       "layout": "IPY_MODEL_15a8b9830465461fa93ced228d4195f3"
      }
     },
     "e4e71bff8119478888b409a852f8bac8": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "e5e7d83c4ebc41b9a04a04ab7029d9a6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "e9a87ce747114b1d8ab1d79318ce8d58": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_f8f2b91deea14c9a8dd8466153b4f512",
       "max": 200,
       "style": "IPY_MODEL_edda0eeaca644ead8c16fb3963a5c597",
       "value": 200
      }
     },
     "ea52ffff5d864f0da3a799987550cf50": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_418e4881b88d4d50bef32241d7a2f663",
        "IPY_MODEL_c194d3c1193846c18c83ae2a09ec55d0"
       ],
       "layout": "IPY_MODEL_590112e86fa14ec0bd038269e1041e38"
      }
     },
     "eac6fbed32cd4f1b810ebf1ac66399b4": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "eb8fa51d45c04b60a7122ae24b6e445a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "ecdb0901f6a64234afe87251cc2861d9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_f77cb21b5317409d8d413902a82f66a1",
       "style": "IPY_MODEL_70671b44b73f4fab81e8e23d0d615cb7",
       "value": 100
      }
     },
     "ed08e5b49c174c7198b4c3c89a5ef5cc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "ed8b40ea79eb412096bdec71e6873bf0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_9049915c466449bf9fc9e8234f8810dd",
       "max": 200,
       "style": "IPY_MODEL_990a36284a4540c4b05288f2eb363276",
       "value": 200
      }
     },
     "ed98421ab07449bdbfdad6f6a7125cc2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_52ded505a1e6408a8513729d78bf8e79",
       "style": "IPY_MODEL_08ff5f75450d4e0bb0ec92a75114d641",
       "value": " 200/200 [01:31&lt;00:00,  2.19it/s]"
      }
     },
     "edd3b4540b1c484fb88b5e7e818751e9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_9a2aeb063e5141259cafc1a099cb2b09",
       "style": "IPY_MODEL_d7b570e9ece448e7ac427b5f9a9d7aa0",
       "value": " 100/100 [01:11&lt;00:00,  1.41it/s]"
      }
     },
     "edda0eeaca644ead8c16fb3963a5c597": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "ee0aca055d704c9ab122e6ffaae095bc": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "ee97729d91f842bab8609f16067bcdb0": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "efb0d67b135840279979a935cadb1a63": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f07b336518c14a58a43692379d3e77d6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f082d1f6c2d3423f8dd5f01d75157924": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "f105b822ebfb43909ef152d96bb9ab00": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "f13c7ab103aa49c0a2e8dd0d139264f8": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f151632da6144645be164cc6702fc1a6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_7cb22a856cd54450afa5386cbb319a61",
        "IPY_MODEL_5c79de21221744f892925486a0c51fbc"
       ],
       "layout": "IPY_MODEL_633c27ebb4784f588dd25de209252431"
      }
     },
     "f2537df39ba74b7baf88efccc73102c5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f2550e33567d4b1da968234e96877090": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f256104b4cf14ad082651ea23bfcd2c7": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "f28c596da1c34a89bd55e760aa3ba280": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f33c23a7e9b64d809207a1c01934f351": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f33d9d8677b342a5b82d7fad684589f9": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f33f34cb34fe4013ba87ffeb6803b8ec": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f36de1e438dc4730bacab82a601fbeca": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_2390f0aec01441a598460829b3aa4e96",
       "style": "IPY_MODEL_293c45fd969748568a082c8e78917138",
       "value": " 300/300 [00:57&lt;00:00,  5.17it/s]"
      }
     },
     "f4af073fe03e408bab08676718edc754": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "layout": "IPY_MODEL_0dc2668c69e04a19b846b7952323ddf4",
       "style": "IPY_MODEL_2bdaffc829654f17a037026956f08917",
       "value": " 2/2 [00:05&lt;00:00,  2.78s/it]"
      }
     },
     "f52e487e3f99409498b49dbb43d9b93e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f5a34d7312774abfbd381767fee34b15": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "f5edf0c878d04bb1b535a32bef2adb49": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_9da400dd12594b29beb1069ddf179b62",
        "IPY_MODEL_9637f0b173ee4e4b8666817f3c144134"
       ],
       "layout": "IPY_MODEL_c07fe3c2d25a4975975b15fe228884ee"
      }
     },
     "f6808c1d7f9d45d597266787677f1b58": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f75172c01ff34e14af386a5c891bcd88": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_99973e1e8c19420f94305738043628a6",
       "max": 200,
       "style": "IPY_MODEL_02bf9bf329aa4b0797fdc4a77a2e3f71",
       "value": 200
      }
     },
     "f7587edf77524260871b5aea2498895f": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "IntProgressModel",
      "state": {
       "bar_style": "success",
       "description": "100%",
       "layout": "IPY_MODEL_0dbf5b71324a475cbb0a1d054390839f",
       "style": "IPY_MODEL_9339a5d8fab34838a642627ec736ee7d",
       "value": 100
      }
     },
     "f77cb21b5317409d8d413902a82f66a1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f7d17faed5744373988335dd6badeec6": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "f8f2b91deea14c9a8dd8466153b4f512": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f9373a82cc314bd292402a8d8cae8748": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "f98d2e5125f14fe99fd543e9a684bb9e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_d778a8e621f148768319876eee2276c5",
        "IPY_MODEL_74ea9583a7da4266a428df9f5aafd8d5"
       ],
       "layout": "IPY_MODEL_5314c6a70ae348b488a4bd5397fac7f3"
      }
     },
     "f993f6c919fc4a499d65ea43db97928a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_002d59f567ca46ba9eb8418ca1d20490",
        "IPY_MODEL_51f0f7beefea4906801eb4798f190109"
       ],
       "layout": "IPY_MODEL_21789dfc9274478489fdd046e7d8f445"
      }
     },
     "f99b9c7433c9409eac358cd028ada4f3": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_7b5fedaeb68b4e109a132f6024c00758",
        "IPY_MODEL_12d9ec781da44d96afbce411ff6dd654"
       ],
       "layout": "IPY_MODEL_b22b22f3d70541108992de0120e6d5d3"
      }
     },
     "fac8c672dd3a4fb09444daa2c5220741": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "fb0322676ef6410dae53a96e5a08123b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "fb2a06c4e11b417bb5bcd7259d494d26": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "fb63e87baae64d1bb51747a09af59d0c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "description_width": ""
      }
     },
     "fba80bacb5824f839d556bbf55c1aad6": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "fc3ac025200c4c4099b4abaca46ff3d5": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "fc8e3b8d2097451697e84893872d9ef0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_c3a0ccbb943a47b6a510887ca7b7962a",
        "IPY_MODEL_c1406aebf4fc4d61b1cef0cdc516ff6e"
       ],
       "layout": "IPY_MODEL_20eb1f87d25547a3a9936ff64fda12fb"
      }
     },
     "fce90cf031fc4aa8b3db5822bb26f01f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "fd228f36d47146c69e21c197412552c5": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {}
     },
     "fe813fa0b6104b20bf37b7cf22fdb994": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "description_width": "initial"
      }
     },
     "ffff294dbe154cb390777d4aa121847b": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "children": [
        "IPY_MODEL_54e532ae266b47b3b10e7b402bae0a26",
        "IPY_MODEL_46a01765cf944f7f94a59406439d9b8c"
       ],
       "layout": "IPY_MODEL_f52e487e3f99409498b49dbb43d9b93e"
      }
     }
    },
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
=======
{"cells":[{"cell_type":"markdown","metadata":{},"source":"# Small NODEC example\nA small example to compare NODEC with optimal control on a toy dynamical system."},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"# To edit source files and automatically refresh code\n%load_ext autoreload\n%autoreload 2\n\n# Load custom modules path\nimport sys\nsys.path.append('../../')\n\n# Custom modules path\nimport nnc.controllers.baselines.ct_lti.dynamics as dynamics\nimport nnc.controllers.baselines.ct_lti.optimal_controllers as oc\nfrom nnc.controllers.neural_network.nnc_controllers import\\\n     NeuralNetworkController, NNCDynamics\n\n# Computation and plot helpers for this example\nfrom small_example_helpers import EluTimeControl, evaluate_trajectory, todf\nfrom small_example_helpers import compare_trajectories, grand_animation\n\n# progress bar\nfrom tqdm.cli import tqdm\n\n# Other libraries for computing\nimport torch\nimport numpy as np\nimport pandas as pd\n\n# ODE solvers with gradient flow\nfrom torchdiffeq import odeint\n\n# plots\nimport plotly\nplotly.offline.init_notebook_mode()"},{"cell_type":"markdown","metadata":{},"source":"## Continuous Time Time-Invariant Dynamics\nFirst we define our dynamics as:\n\\begin{equation}\\label{eq:ld}\n\\dfrac{dx}{dt} = \\langle A, x^{\\intercal} \\rangle + \\langle B, u^{\\intercal}\\rangle\n\\end{equation}\nwhere:\n\\begin{align}\nx &: \\text{a state vector over $N$ nodes.}\\\\\nA &: \\text{an $N\\times N$ interaction matrix, indicating nodal interactions.}\\\\\nu &: \\text{a control signal vector of $M\\leq N$ independent control signals.}\\\\\nB &: \\text{an $N\\times M$ driver matrix, indicating control signal effects on nodes.}\\\\\n\\end{align}\n\nIn this example we parametrize the system as:\n\n### System Parametrization\n\n\\begin{align}\nA &= \n\\begin{pmatrix}\n    0 & 1 \\\\\n    1 & 0 \\\\\n\\end{pmatrix}\\\\\nB &= \n\\begin{pmatrix}\n    1 \\\\\n    0 \\\\\n\\end{pmatrix}\n\\end{align}\n\nEssentially, we evaluate control over a 2-node undirected graph with control signal applied over one node (single driver node)."},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"# Basic setup for calculations, graph, number of nodes, etc.\ndtype = torch.float32\ndevice = 'cpu'\ntraining_session = True\n\n# interaction matrix\nA = torch.tensor([\n    [0., 1.],\n    [1., 0.]\n])\n\n# driver matrix\nB = torch.tensor([\n    [1.],\n    [0.]\n])\n\n# interaction matrix dimensions denote how many nodes we have in the network\nn_nodes = A.shape[-1]\n# column dimension implies the number of driver nodes.\nn_drivers = B.shape[-1]\n\n# implementing the dynamics\nlinear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype, device)\n"},{"cell_type":"markdown","metadata":{},"source":"The exact implementation of the dynamics:"},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"dynamics.ContinuousTimeInvariantDynamics??"},{"cell_type":"markdown","metadata":{},"source":"### Experimental Setting\nWe aim to control the system from initial state $x(0)=(0.5, 0.5)$ to $x^* = (1,-1)$ within time $T=1$."},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"# total time for control\nT = 1\n# we evaluate two points in time, first point is matched to initial\n# state and  second one is matched to the terminal state\nt = torch.linspace(0, T, 2)\n\n# initial state is set as follows, but can be chosen arbirtarily:\nx0 = torch.tensor([[\n    0.5, 0.5\n]])\n\n# same applies for target state:\nx_target = torch.tensor([[\n    1, -1\n]])"},{"cell_type":"markdown","metadata":{},"source":"### Control  Baseline\nHere we use the minimum energy optimal control as baseline based on the following work:\n- Yan, G., Ren, J., Lai, Y. C., Lai, C. H., & Li, B. (2012). Controlling complex networks: How much energy is needed?. Physical review letters, 108(21), 218703.\n\nFor CT-LTI systems that satisfy controllability assumption, this baseline achieves the maximum theoretical performance, i.e. it cannot be surpassed by any other."},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"# baseline definition\noc_baseline = oc.ControllabiltyGrammianController(\n    alpha = A,\n    beta = B,\n    t_infs = T,\n    x0s = x0, # a batch with one sample\n    x_infs = x_target, # a batch with one sample \n    simpson_evals = 100,\n    progress_bar=None,\n    use_inverse=False,\n)"},{"cell_type":"markdown","metadata":{},"source":"Below, we create on line 2 a `lambda` expression function to make the dynamics compatible with required method signatue, and then we evolve the system with `odeint` method from `torchdiffeq` package (line 3).\nThe result of `odeint` is a tensor of shape `[timesteps, state_size]`.\nThe reached state corresponds to the last index of the result `[-1]`.\nAs we print on line 4, optimal control reached the target state after control $x(T)\\approx x^*$."},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"timesteps = torch.linspace(0, T, 2)\noc_dynamics = lambda t,x: linear_dynamics(t, x, oc_baseline(t, x))\nxT_oc = odeint(oc_dynamics, x0.unsqueeze(0), t=timesteps)[-1]\nprint(str(xT_oc.flatten().detach().cpu().numpy()))"},{"cell_type":"markdown","metadata":{},"source":"### Neural Network Control\nWe define a custom neural network for learning contro.\nIn this example we pick a fully connected architecture with 2 layers of n+4 neurons\nand exponential linear unit (ELU) activation.\nThe content of the class can be found below:"},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"EluTimeControl??"},{"cell_type":"markdown","metadata":{},"source":"We define a training routine for the neural networks.\n- Note the `x0.detach()` and `t.detach()` on lines 7,8 respectively to avoid unexpected gradient flows.\n- It is important is to notice on the next cell, in line 11 that we include no energy regularization term in the loss.\n- We then do the backpropagation with autograd on line 13 and then let the optimizer (Adam) to update the network parameters.\n- We prefer to train with variable step method `dopri5` on line 8, but we evaluate with constant step size on line 16, to limit performance advantages due to step size selection."},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":"def train(nnc_dyn, epochs, lr, t, n_timesteps=40): #simple training\n    optimizer = torch.optim.Adam(nnc_dyn.parameters(), lr=lr)\n    trajectories = [] # keep trajectories for plots\n    for i in tqdm(range(epochs)):\n        optimizer.zero_grad() # do not accumulate gradients over epochs\n        \n        x = x0.detach()\n        x_nnc = odeint(nnc_dyn, x, t.detach(), method='dopri5')\n        x_T = x_nnc[-1, :] # reached state at T\n        \n        loss = ((x_target - x_T)**2).sum() # !No energy regularization\n\n        loss.backward()\n        optimizer.step() \n\n        trajectory = evaluate_trajectory(linear_dynamics, \n                                         nnc_dyn.nnc, \n                                         x0, \n                                         T, \n                                         n_timesteps, \n                                         method='rk4',\n                                         options=dict(step_size = T/n_timesteps)\n                                        )\n        trajectories.append(trajectory)\n    return torch.stack(trajectories).squeeze(-2)"},{"cell_type":"markdown","metadata":{},"source":"## Experimental Results\nWe can now apply training, collect trajctories and save the model.\nNotice that we decrease learning rate on lines 11 and 14.\nNODEC is very sensitive to learning rate and often requires several lr-adapaptions to converge to desirable performance.\nSuch, adaptions can be automated as discussed in the paper."},{"cell_type":"code","execution_count":null,"metadata":{"pycharm":{"is_executing":true}},"outputs":[],"source":"n_timesteps = 40 #relevant for plotting, ode solver is variable step\nlinear_dynamics = dynamics.ContinuousTimeInvariantDynamics(A, B, dtype, device)\n\nif training_session:\n    torch.manual_seed(0)\n    neural_net = EluTimeControl(n_nodes, n_drivers)\n    nnc_model = NeuralNetworkController(neural_net)\n    nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)\n    # time to train now:\n    t = torch.linspace(0, T, n_timesteps)\n    t1 = train(nnc_dyn, 200, 0.1, t) # , 200 epochs, learning rate 0.1\n    df1 = todf(t1, lr=0.1)\n    t2 = train(nnc_dyn, 300, 0.01, t) # , 300 epochs, learning rate 0.01\n    df2 = todf(t2, lr=0.01)\n    torch.save(neural_net, 'trained_elu_net.torch')\n    alldf = pd.concat([df1, df2], ignore_index=True)\n    \n    alldf.to_csv('all_trajectories.csv')   \nelse: \n    neural_net = torch.load('trained_elu_net.torch')\n    alldf = pd.read_csv('all_trajectories.csv', index_col=0)\n    nnc_model = NeuralNetworkController(neural_net)\n    nnc_dyn = NNCDynamics(linear_dynamics, nnc_model)"},{"cell_type":"markdown","metadata":{},"source":"Now we can check the reached for NODEC, and as we observe it is close to the target.\nMore epochs on lower learnig rates will further improve this result."},{"cell_type":"code","execution_count":null,"metadata":{"pycharm":{"is_executing":true}},"outputs":[],"source":"t = torch.linspace(0, T, 2)\nx = x0.detach()\nld_controlled_lambda = lambda t, x_in: linear_dynamics(t, u=neural_net(t, x_in), x=x_in)\nx_all_nn = odeint(ld_controlled_lambda, x0, t, method='rk4', \n                  options = dict(step_size=T/n_timesteps))\nx_T = x_all_nn[-1, :]\nprint(str(x_T.flatten().detach().cpu().numpy()))"},{"cell_type":"markdown","metadata":{},"source":"Now we compare NODEC to optimal control (OC) in terms of: (i) controlled trajectories (left) and (ii) total energy (right). As we see both methods are extremely close."},{"cell_type":"code","execution_count":null,"metadata":{"pycharm":{"is_executing":true}},"outputs":[],"source":"fig_comparison, _, _ = compare_trajectories(linear_dynamics,\n                     oc_baseline,\n                     nnc_model,\n                     x0,\n                     x_target,\n                     T,\n                     x1_min=-3,\n                     x1_max=3,\n                     x2_min=-1.5,\n                     x2_max=1.5,\n                     n_points=200,\n                    )\nfig_comparison"},{"cell_type":"markdown","metadata":{},"source":"Finally we animate the learning effort of NODEC through the training process. We observe how it slowly converges to the OC trajectory, both in terms of state values and energy."},{"cell_type":"code","execution_count":null,"metadata":{"pycharm":{"is_executing":true}},"outputs":[],"source":"animation =  grand_animation(alldf,\n                             linear_dynamics, \n                             oc_baseline, \n                             nnc_model, \n                             x0,\n                             x_target, \n                             T,\n                            frame_duration=50) # 50-100 should be ok.\nanimation"},{"cell_type":"markdown","metadata":{},"source":"## Conclusion\nThis notebook gave us a quick yet insightful demonstration of NODEC and its performance compared to optimal control.\nResults conclude that NODEC can achieve the control goal. Implicit energy regularization is also observed.\nLarger scale demonstrations on complex dynamics are found in the paper results."}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"codemirror_mode":{"name":"ipython","version":3},"file_extension":".py","mimetype":"text/x-python","name":"python","nbconvert_exporter":"python","pygments_lexer":"ipython3","version":"3.7.3"},"latex_metadata":{"affiliation":"ETH Z\\\"urich, Switzerland","author":"Thomas Asikis","title":"Small NNC example"},"widgets":{"application/vnd.jupyter.widget-state+json":{"state":{"002d59f567ca46ba9eb8418ca1d20490":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_644605e0c20844e4baa9b9298c09968c","max":200,"style":"IPY_MODEL_c655374a178c44ffa49f2d0031c31e97","value":200}},"005b3706ebb94e9998e51963ff182dd5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"014695a4b90149eba71f03516e18c255":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_f7587edf77524260871b5aea2498895f","IPY_MODEL_d0c8d1bddecf4ec7a0066a9f54bd0c6a"],"layout":"IPY_MODEL_d1078b1205534916a518d952f3cc4ddc"}},"01746170b5794cd2942e0d5ed3c1d8ae":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_17acb6d2c97f4d90a1c45ee05a2359f5","max":200,"style":"IPY_MODEL_793f24a97a18438fa113cd308e77ab1a","value":200}},"0201f287dd224e82a7068749ccdffbfd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"023346bb70044e5682f05da2f85399c9":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"02bf9bf329aa4b0797fdc4a77a2e3f71":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"0353ae8ac4ac41c5aa99006e56ec6b58":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_23e050c8c9e94829a056496054627932","max":300,"style":"IPY_MODEL_431385120eab4d78991f200420d5957c","value":300}},"03574a0bf42d479e95121eb8745a1799":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_cf1052676d2646ec8aa389f65f52a6af","IPY_MODEL_b3f9bcfb2ac44ea8b58b9f4e2e9c98d4"],"layout":"IPY_MODEL_1aaaa6d9b92a4551a68b78b5e795df14"}},"039cef2ba0cc4544963144e7132a4303":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"03f0b92465f244729fd9ffe68657a746":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"040e16ebba49457f9b5d6a315c9554fb":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"041c4577cd154a11bd203ba8b9ef0814":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"04abc016a87541f48a7df9b6a35ba8c9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_8381899ede4f4c13a9deb3e18fa709e6","style":"IPY_MODEL_08501b532ed14926bc2265787bf2db65","value":" 300/300 [11:06&lt;00:00,  2.22s/it]"}},"05746fa0e0ec4b9e9809f872a2409e61":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_bd23b9e83dde4ff3bdcf82f41950d6cc","max":2,"style":"IPY_MODEL_7d33a349fd384f33b124d8ab9a6a541e","value":2}},"05a4bf23395d4669a621da2340f2dcc2":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"05e2a4880d8c42db8432e75813854c6e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"064efbed9d8a4b51ab87448f4b7e30d6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"06d91eea70394a0db970c57b44dee804":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"075d72f8ad794bcaac1ebcaabfc020b8":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"07c8e7fd75f6499bb4fcff6160060c9c":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"08501b532ed14926bc2265787bf2db65":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"0853d1531b1e409fa2f076a62fa41a7f":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"08ff5f75450d4e0bb0ec92a75114d641":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"094c74f539854be0ad88c73cb32f8913":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"096d21a66da64056806c52542d700b25":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_ab0893c752c84b97b740a08a34c89a2a","max":2,"style":"IPY_MODEL_80e8324050854bd790bcfeb8eb647b81","value":2}},"09871093305140deb2c16d792d5708df":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"09a069c8c4594fa6ad5d6e76cfe83701":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"09c00bf002a04d31b5d0c72d40ffe090":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"09e4fb21849e403e8bfeac92ee20a09b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_825638143d814b42b24892ce66d290c4","IPY_MODEL_9dcd5b65fa6c42cea1fb1cb2e41d7642"],"layout":"IPY_MODEL_37d7b125e9724f1fa1c9c074166d30e3"}},"0bf81f2702d3476ab3f53fcec853a9ac":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_d39585ef35ae46bdb29a4fe03ea7f80e","max":200,"style":"IPY_MODEL_cf69a28284744cb19eb4962e17a61157","value":200}},"0c2b0e00bec64bd2a4feb98540a38153":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_8ec0ea1cfc28409591bc5c620bb3a366","max":2,"style":"IPY_MODEL_9a72932bbd2f4487acabc05224f0eecd","value":2}},"0d469214d3774a6293b71ec9ee990f0a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_e34a0470f4c4414381a0f5d8dd584c70","IPY_MODEL_cc41d24433854b0aaa2d0339cf5667fd"],"layout":"IPY_MODEL_8829c85b980846c2b000aced08613a3a"}},"0d6eb4e6a1cf46f9a4adfd566b264bc7":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_1ac92d4647a44df38a3008214bae177b","style":"IPY_MODEL_064efbed9d8a4b51ab87448f4b7e30d6","value":" 300/300 [00:56&lt;00:00,  5.32it/s]"}},"0dbf5b71324a475cbb0a1d054390839f":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"0dc2668c69e04a19b846b7952323ddf4":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"0e539d49dff34320bea9d52213f3efb9":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"0ee0c2d234284cd2b367bdc3e54fa07a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"0f5ec697bf1240d4a75402a6951a5797":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"108357b9ba4740da8b5c689fb34e9141":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"10daa34f258648a0a27d847a9ad48e34":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_bdfcde8b3a0d4b60a43ce46c9cb4f490","IPY_MODEL_d8b42274c88045ec8eeff13f5f009f17"],"layout":"IPY_MODEL_af1bc4e3b0274dd78db5ddf0e3b451f0"}},"123bd18e6c7d495598abbf8b10d9d9dc":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_ed8b40ea79eb412096bdec71e6873bf0","IPY_MODEL_ce06ac6e90fd4bf99b8ed1574455f7ed"],"layout":"IPY_MODEL_278401ec4a3c4083a6b859ceed816e41"}},"12d68e6f9fae46da8dfdd6fb467efa8a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_95744244bde34b35a990ff69d32b0a5b","style":"IPY_MODEL_479d33fb42b24f48a2d27ea6d7285351","value":" 200/200 [00:32&lt;00:00,  6.14it/s]"}},"12d9758aab96413b843a5cb49c49cf86":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"12d9ec781da44d96afbce411ff6dd654":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_6b9add0f90054681b3649651e4c27477","style":"IPY_MODEL_8d4b6c6c2c904fe39a50a16bf0ee1702","value":" 200/200 [09:46&lt;00:00,  2.93s/it]"}},"12fdd2ec99da4ac080d2a24b30d777be":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"143a9654993a482995b6f0d8c1a82676":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"156d230fa70f4fb2bbc7efed3ce55f50":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"15a8b9830465461fa93ced228d4195f3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"15fa43a5da0a4906855ceac366431d5b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_ac30774fcfe24e3c8888cf18558a6bfc","style":"IPY_MODEL_3334e79aa3aa4e5ebaae2e641317302d","value":100}},"16aefc0ee1b84260b38fb4cada31e8c4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"16cdec1d589e4b84b9e1432f9d326da8":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1707ea997b1e4b208d4792020417f510":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_904203b941fc4bd6844520979fa109de","style":"IPY_MODEL_094c74f539854be0ad88c73cb32f8913","value":" 2/2 [04:26&lt;00:00, 133.26s/it]"}},"17acb6d2c97f4d90a1c45ee05a2359f5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"187aab75554742e88ef8704a832fcc3f":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_b88644021e7f43928d5ae08371a892fd","IPY_MODEL_e1cd50413e864ede8fcee5b696107db3"],"layout":"IPY_MODEL_569422e860bd43f2a4d5c2edb626b240"}},"1972bb4bd9624659913178b3cf980e79":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_37801dec522f47658bf3b461698cfcd9","IPY_MODEL_1a49429d9da84577958af3077a2f38c0"],"layout":"IPY_MODEL_b71f99c19c3c402dafd7644523e5b5b1"}},"199f54999a9f4b818fa156890b76ef05":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"19a8df789fd944cca9081036753dcb38":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"19bb316c90624805acb7f93073376ac4":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1a49429d9da84577958af3077a2f38c0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_0853d1531b1e409fa2f076a62fa41a7f","style":"IPY_MODEL_b88f3049eeee41e8bfd65d3fea6811f5","value":" 2/2 [00:06&lt;00:00,  3.04s/it]"}},"1aaaa6d9b92a4551a68b78b5e795df14":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1ac92d4647a44df38a3008214bae177b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1afedb0c361141458e03934c2466e097":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1b645a854bed4edda33cf10cde01d963":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"danger","description":"  0%","layout":"IPY_MODEL_4be5e471a19542dc949292ffae9d71a9","max":200,"style":"IPY_MODEL_09c00bf002a04d31b5d0c72d40ffe090"}},"1c4ae29996b641dbafe41bdc8a622b70":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"1c77ddbe02434eb89476dbe1bd5d0d74":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1e1c21bba5bd442cabb00ed2cb4910d7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1f13321849cb4357bdbbdf65400bd224":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"1f8284eac1014b25b3f21f78d27c0ae9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_b304c23114fc4b368054fdedebe98bec","IPY_MODEL_1707ea997b1e4b208d4792020417f510"],"layout":"IPY_MODEL_3d50c97cce8f46b3861741b4d9e32b0a"}},"1f9157b23a874a0b9bef49634ce2abd4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_7f865c763c074d128c345864ccb0aaa3","IPY_MODEL_b9f67e0cac1742e4b245b4a27f45c5d1"],"layout":"IPY_MODEL_795629f364aa4de2b1fc6a673c728c94"}},"1fcb0d10fd7b47f6abdf37241e196efb":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_096d21a66da64056806c52542d700b25","IPY_MODEL_cc46fc787dfc46e1933532edba52b4d1"],"layout":"IPY_MODEL_f07b336518c14a58a43692379d3e77d6"}},"204400c2b64b4a8baba57f92ba638205":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"20a08026ac5d49a0b14aa41635e3ea96":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_f75172c01ff34e14af386a5c891bcd88","IPY_MODEL_e2a4e0fc30514b20adaee2fdee557be3"],"layout":"IPY_MODEL_59ae1ef2028a48e9b1643ffa40c51a42"}},"20eb1f87d25547a3a9936ff64fda12fb":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"2142b21e4b614286b407f8b963edbed0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"214edccc75174a0a9ef6b47be16bceb3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_4198b858cbdd4d52801d65adce4e7aea","style":"IPY_MODEL_d6c4f7c9159c4e539c747b6e2983ed2b","value":" 2/2 [03:39&lt;00:00, 109.94s/it]"}},"2166534bcde14ddda56d9d1c106bcdfb":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_916530172abc4fb3b7190ea2a11c7e3a","style":"IPY_MODEL_ab4173afdbbf4496bfd4fa490f6ff84b","value":" 2/2 [01:13&lt;00:00, 36.87s/it]"}},"21789dfc9274478489fdd046e7d8f445":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"21ce6253ddab4eb58de3a8e84e943e6d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"22abd941f4214f6688eacea066cd083e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"22b5281f57f44b3d81b4484e1328abaf":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_1afedb0c361141458e03934c2466e097","style":"IPY_MODEL_4cb2da30f6334ef39b20ea61a77779db","value":" 100/100 [00:57&lt;00:00,  1.73it/s]"}},"2360820a5f7c48a3923e868e63951d53":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"2390f0aec01441a598460829b3aa4e96":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"23e050c8c9e94829a056496054627932":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"23fe8f5977a44673ad7dd704fee8dc31":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_0f5ec697bf1240d4a75402a6951a5797","max":200,"style":"IPY_MODEL_84455b0a64a74294bd5814ded73c05f5","value":200}},"24486927ecf1462d9deafc8721c5a67a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"258346915b2c40ecabeceec670bff283":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"2676372c45bd45fa95569015aa8b4068":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"26e29cd8d57340a789e8c700fdf4b144":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_be819e0b4cc24b29aa88c4ce487888c2","IPY_MODEL_0d6eb4e6a1cf46f9a4adfd566b264bc7"],"layout":"IPY_MODEL_8242093b80f842448baf3b0d3e1298ac"}},"278401ec4a3c4083a6b859ceed816e41":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"280f8786057d40aaa970861990da1c18":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"289e351478334a148ce6924f6bde30b7":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"28e9c87517254377a37144a64d9fab82":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"293c45fd969748568a082c8e78917138":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"2a3c47712e9b4db6a46d31fdb8e60d8c":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"2a9591799c3544b894b018fa5e15a680":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"2bd914e6301341aab54384c9e5d9e40c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"2bdaffc829654f17a037026956f08917":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"2beaafc1a6584294ba6d6d51d675e9cc":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"2cfee9448f2e4870874d83d1f0177f04":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"2e19164d8cd34b8a803b01964f58e141":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_c190511f14da4810aec9df4f2c3a41f5","style":"IPY_MODEL_06d91eea70394a0db970c57b44dee804","value":" 300/300 [00:55&lt;00:00,  5.44it/s]"}},"2ea7ad1fc71547758997763659ee3752":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"2eac5e4754a04871ad5524b632bedfb5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"30c1af7383b34066a34b851405c9224b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"30c37a13ee3c400bbe33e366fc614d7b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"30f63f246ab148118e394f6d33ec1f92":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"31d01089215047c698efb8e13106d57d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"32d071ff0e994878a3ede66e8fc1a5ff":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"32ed88399c784b8589c41ef22264a6f7":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"3334e79aa3aa4e5ebaae2e641317302d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"336c2f60995e4601bc967b86e8516d99":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_05a4bf23395d4669a621da2340f2dcc2","max":300,"style":"IPY_MODEL_fac8c672dd3a4fb09444daa2c5220741","value":300}},"33c1904fd85d471c81cf350dfd006e00":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"34197b6f340045a58a6e3c83cda8ecef":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"347f9d470b754d66b877ea7932a69209":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_361d1fed3dc541518cb862d0831cba13","max":300,"style":"IPY_MODEL_3abb725c08d54936afcc72031712940f","value":300}},"35a0e7ff2c5b41569870b35ccc9f7bb2":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"361d1fed3dc541518cb862d0831cba13":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"36285193f99749d381423663b551b044":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"36b984a22422429b926bda8dac248a67":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"36e1baf5fb8b43a79f4b692e5d6f2b2c":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"371c0a34975d4df6be3632a3b5689d2a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"danger","description":"  0%","layout":"IPY_MODEL_c090dc7ad12340789ba2d7d72f995729","max":200,"style":"IPY_MODEL_2cfee9448f2e4870874d83d1f0177f04"}},"37801dec522f47658bf3b461698cfcd9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_2a3c47712e9b4db6a46d31fdb8e60d8c","max":2,"style":"IPY_MODEL_813e680a9ae84b68bf3d3dd81f2cb483","value":2}},"37d7b125e9724f1fa1c9c074166d30e3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"37e80dca0f3844519e288819d200a5a8":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"391fcfa48f1b4e36997ef0702f10ad5d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_65c16eee1eb94d27913f27e95d6bafd6","style":"IPY_MODEL_8d6012b4c1384571ad3f4902f5738572","value":" 36/100 [00:06&lt;00:11,  5.57it/s]"}},"399a9dc5fe7a4441b207c7afbaa5d214":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_7f81e141eb3b4fd88a3934316bf11824","IPY_MODEL_40a48ff94e424b2484bd08d6d7ae7398"],"layout":"IPY_MODEL_535c9cc66b084e52ad494e745ec9a23f"}},"39dcca616a374486bb6cc48db683bf55":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_8b3b79ffdd8f41bdb0cc20498532c99a","max":300,"style":"IPY_MODEL_a8195f66e3bf4b13a69a4f8c5d875b07","value":300}},"3a2eef88782e423d803d7d7a7f39802b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"3aaa8d3bdb4a44d090453fa1a0567b05":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_5283fa9e66614ea6a7145d7ab1f580bb","IPY_MODEL_b2c3096e6e9a4a6d9da80fd1ceae1b78"],"layout":"IPY_MODEL_0ee0c2d234284cd2b367bdc3e54fa07a"}},"3abb725c08d54936afcc72031712940f":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"3bb25574b9824b02a1895a42cd36573e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_3c76607a28734c9a899657f7f039ac36","max":200,"style":"IPY_MODEL_16aefc0ee1b84260b38fb4cada31e8c4","value":200}},"3bdb33b3c17045189cf6f154ab6f948c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_93deaf444cbe4c4f9fa4a89d899badc3","max":2,"style":"IPY_MODEL_31d01089215047c698efb8e13106d57d","value":2}},"3c648575362040eab27a7212d28a5aa7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"3c648d21e8604436933da0b69ae4c3d6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_ce9b73bc52684b538566f7f6f8964eb3","max":2,"style":"IPY_MODEL_8309c3cbff684aaaa1bc8a88524d6aa9","value":2}},"3c76607a28734c9a899657f7f039ac36":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"3d50c97cce8f46b3861741b4d9e32b0a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"3f39e1a5b59d4340aa580cc7117f5cef":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"3fe6aecfd5c14291b52f3a2b9fa844b2":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"401575bb4d4d4bfd8a2db2a25a3c54c3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"40a48ff94e424b2484bd08d6d7ae7398":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_f13c7ab103aa49c0a2e8dd0d139264f8","style":"IPY_MODEL_f7d17faed5744373988335dd6badeec6","value":" 400/400 [01:51&lt;00:00,  3.59it/s]"}},"40cc03073d25468e91fecf6f1913cfac":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_2ea7ad1fc71547758997763659ee3752","style":"IPY_MODEL_e42fb4a4d5a943cc8104e4a4354bbf9b","value":" 2/2 [04:20&lt;00:00, 130.24s/it]"}},"418e4881b88d4d50bef32241d7a2f663":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_d1714b06f9b24fdcb4fb40ede36d9149","max":2,"style":"IPY_MODEL_32ed88399c784b8589c41ef22264a6f7","value":2}},"4190847bd0bc4376a15cb591aba5b902":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_b1daf4b4191a4a80928799a74dae3829","IPY_MODEL_c863855be51e48aea32d3acbed5e906b"],"layout":"IPY_MODEL_78f413bba67c4397814759d60fc7aa64"}},"4198b858cbdd4d52801d65adce4e7aea":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"41a1041df1e24f0082fd32d3e28f8289":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_8e4c6a49b90344e1ad5197a5aa9df450","max":2,"style":"IPY_MODEL_12fdd2ec99da4ac080d2a24b30d777be","value":2}},"41e84ffaca0a4bb596d9bf970ed70752":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_9e7e7006e76347cfb21f1453cead6cc0","max":300,"style":"IPY_MODEL_eb8fa51d45c04b60a7122ae24b6e445a","value":300}},"422bb40b54c347ef8a0ed9f5187eb8ac":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4238d2986e9b4efeb6d8e5f9d5695c18":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_9fd93d0caa8b43d38438bbc8a8268516","IPY_MODEL_ab916a934ef043efa57547624f11ad39"],"layout":"IPY_MODEL_1c77ddbe02434eb89476dbe1bd5d0d74"}},"431385120eab4d78991f200420d5957c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"4320bbdf72a348a38496d24de28574b4":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4332d5782bf84883a614d7dd535644d4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_91900a0131ea47018a6185433cfe384e","style":"IPY_MODEL_c7c2c9410b444ca990b6a0a5ca9fac96","value":" 0/200 [00:00&lt;?, ?it/s]"}},"442a19f8ce1f4cb4b277182c54f37412":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_ded6bbcf69174565a96a3434936805aa","style":"IPY_MODEL_6820cd5fd8154861a28f5271de3fa23e","value":" 400/400 [51:31&lt;00:00,  7.73s/it]"}},"447a462ad35d4cc7af1ae9f408b79704":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"44a22c1eaafc4672a029097656db33ff":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_98ab0e6f34664c63b13604fcf4d54850","max":300,"style":"IPY_MODEL_4685d1d27e924c01b65ff420ce6292e4","value":300}},"4590898e33714706a0c729e472bdc936":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"46509eeb153e46f1a91b8cbcae039e25":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4685d1d27e924c01b65ff420ce6292e4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"46a01765cf944f7f94a59406439d9b8c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_a7b71edb1f18461182063ba01e2e8153","style":"IPY_MODEL_f082d1f6c2d3423f8dd5f01d75157924","value":" 200/200 [02:53&lt;00:00,  1.15it/s]"}},"477bcad1abed42ada6dbf67c640d0efb":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_611d39563ef846db93fd9f7cd814a803","IPY_MODEL_391fcfa48f1b4e36997ef0702f10ad5d"],"layout":"IPY_MODEL_075d72f8ad794bcaac1ebcaabfc020b8"}},"479d33fb42b24f48a2d27ea6d7285351":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"491d6d6139fa47119156657b8334d046":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_0bf81f2702d3476ab3f53fcec853a9ac","IPY_MODEL_6fb4e131d37a4198a1d9ef56f80f87df"],"layout":"IPY_MODEL_b9131bbb84164a8d80b478d19d0a24f0"}},"494f8306d0d04f449b2fc8ed1e026fb6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_41e84ffaca0a4bb596d9bf970ed70752","IPY_MODEL_864f93670a134cd9bbd23b54cbf7176e"],"layout":"IPY_MODEL_fce90cf031fc4aa8b3db5822bb26f01f"}},"49583b10f48e4c2cbd0faa71dbc222f0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4afd2928c81b4097b24f56435aa94cae":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4b11d7442c63475285daa788767f0d8d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"danger","description":"  0%","layout":"IPY_MODEL_fba80bacb5824f839d556bbf55c1aad6","max":200,"style":"IPY_MODEL_c3a28bf75ebc47099aa29cd99190f58d"}},"4b21bd8461824452a2124200792830d9":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4b2d8003d1fe4fcf801c921c740ae6dc":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_8c46653c18cc447bac0c070b04a39364","max":300,"style":"IPY_MODEL_70deed10e21e47fabe60f9e5d9d46aa2","value":300}},"4b393b795b7d42f9968a8916cbaa56e5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4bd0762e7a8d42fa9f432c8ac4629e94":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_34197b6f340045a58a6e3c83cda8ecef","style":"IPY_MODEL_74fd0b4380af445b8aa4987236bd7fe0","value":" 200/200 [11:44&lt;00:00,  3.52s/it]"}},"4be5e471a19542dc949292ffae9d71a9":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4cb2da30f6334ef39b20ea61a77779db":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"4cd41f6f88f8437698197b6ada89e1fc":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_023346bb70044e5682f05da2f85399c9","style":"IPY_MODEL_7f16ba386f644f51b27ad7fe4c08c093","value":" 27/200 [3:37:03&lt;23:10:46, 482.35s/it]"}},"4df2431a6ef24100b4aa8c9db5494243":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_6ad5f23bfb0d4a998496c686b04872de","IPY_MODEL_4f2708cc04d84c51b9762cd2bc86e847"],"layout":"IPY_MODEL_d0bb16bcc3944068b9ca2dacbee2fad6"}},"4f0e78a0191b422990ef3965e3388c69":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"4f2708cc04d84c51b9762cd2bc86e847":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_736b343facfe4239b3719694a3db919f","style":"IPY_MODEL_039cef2ba0cc4544963144e7132a4303","value":" 200/200 [13:25&lt;00:00,  4.03s/it]"}},"4fc6aefe8f1044c1926884ef06d5a104":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"500c6483b18c4b2080976519239678d1":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"5120d219c8f74d518e5e23f3746c0834":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_3c648575362040eab27a7212d28a5aa7","style":"IPY_MODEL_d21c9a4c45304ff987cba8674fd6d44e","value":" 300/300 [15:06&lt;00:00,  3.02s/it]"}},"51f0f7beefea4906801eb4798f190109":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_c5cd27b908e74622b3ef9c7d705e1e76","style":"IPY_MODEL_a4364c35d04048f3902ef7c9be092fc2","value":" 200/200 [03:29&lt;00:00,  1.05s/it]"}},"52211e5e56d947f1adabe6905bc1b20b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_3c648d21e8604436933da0b69ae4c3d6","IPY_MODEL_78a68dd9fc314f3794813ca48fd7bad2"],"layout":"IPY_MODEL_db82d5edcfdf447484a4c45a0c7b2599"}},"5283fa9e66614ea6a7145d7ab1f580bb":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_7ab2d7e0745f492c95fdcc01ea526660","max":2,"style":"IPY_MODEL_09a069c8c4594fa6ad5d6e76cfe83701","value":2}},"52b4e3149f5243eba5136b9da6965740":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"52ded505a1e6408a8513729d78bf8e79":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"5314c6a70ae348b488a4bd5397fac7f3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"535c9cc66b084e52ad494e745ec9a23f":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"54959e533bd045608c4e15e7246583d5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"54e532ae266b47b3b10e7b402bae0a26":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_93813a7f86d54a01bb419e5f7380577c","max":200,"style":"IPY_MODEL_fc3ac025200c4c4099b4abaca46ff3d5","value":200}},"565672af0927472c8ed34f8a27e1a83a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_eac6fbed32cd4f1b810ebf1ac66399b4","max":2,"style":"IPY_MODEL_d35d78bfaf124a9f92e65158e75ba90e","value":2}},"5691525b552147a497f463da43c63c9f":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"569422e860bd43f2a4d5c2edb626b240":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"56f7f51f8ec44a99b6f7c19ed799593c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_8a4ec07820304bb3959a4886572e256b","IPY_MODEL_f36de1e438dc4730bacab82a601fbeca"],"layout":"IPY_MODEL_964f3401d730419a997c6932c0556d68"}},"572e65ba13f44f318f361d9cccf84154":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_4b11d7442c63475285daa788767f0d8d","IPY_MODEL_7d42b0973dc54e289cccd7fbeff101e4"],"layout":"IPY_MODEL_4fc6aefe8f1044c1926884ef06d5a104"}},"5883fca2f9f346469e0f2ea2e5b68bc7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"58ae4c3efb8744728741939dcfd59907":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_1b645a854bed4edda33cf10cde01d963","IPY_MODEL_a86ffc03083f4b2096f4ae5ba888f322"],"layout":"IPY_MODEL_bedbd74cbd0a43f088571061a48186c9"}},"590112e86fa14ec0bd038269e1041e38":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"59ae1ef2028a48e9b1643ffa40c51a42":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"5a6b5846697d4ae9855f14d0629ca318":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"5aefb71b9c0e4565a3216fc3fff96d1a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_65294d58f59545a891e172385b40b7de","style":"IPY_MODEL_9e89d0ce3d7b415b9c0968949ee04fa4","value":" 2/2 [01:59&lt;00:00, 59.74s/it]"}},"5ba27fbd2e424400a21c37819f03e8e0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_07c8e7fd75f6499bb4fcff6160060c9c","style":"IPY_MODEL_4590898e33714706a0c729e472bdc936","value":" 2/2 [01:00&lt;00:00, 30.22s/it]"}},"5c44d8acf6824e51a956821b66798c88":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_78944d80568e4373939e650709180a93","IPY_MODEL_b2c4d415c8cc42789e11e8d70e8af8af"],"layout":"IPY_MODEL_a3ac1454cf3b4efea900927dfda0ced6"}},"5c79de21221744f892925486a0c51fbc":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_30f63f246ab148118e394f6d33ec1f92","style":"IPY_MODEL_63d4e9636d9b44c79e6300352f305236","value":" 200/200 [01:32&lt;00:00,  2.16it/s]"}},"5d043d8a7273486e98db0c49f0e36ee2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_8e07d701daab4f5db990c0de63b01cf1","IPY_MODEL_7ecfbfb9854e4fda99a5c91f66c8fbc2"],"layout":"IPY_MODEL_447a462ad35d4cc7af1ae9f408b79704"}},"5d4d5c8fef9e438e9a69e8b23d7576e8":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_c0e9ac1df48a481c816244727794efe8","style":"IPY_MODEL_143a9654993a482995b6f0d8c1a82676","value":" 300/300 [00:56&lt;00:00,  5.33it/s]"}},"5e1ed0ae189a484cb35f7ae415afd4ae":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"5fcd3a9c4c3d4a1bb0ba90c09dd7cc98":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_4b2d8003d1fe4fcf801c921c740ae6dc","IPY_MODEL_bc79e296edfc4f84a6a9f50f47dc9625"],"layout":"IPY_MODEL_49583b10f48e4c2cbd0faa71dbc222f0"}},"60aca04caa9843e3b49884243e023069":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"611d39563ef846db93fd9f7cd814a803":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"danger","description":" 36%","layout":"IPY_MODEL_7b20403c7afc4e0f8c2e12a7b586663a","style":"IPY_MODEL_401575bb4d4d4bfd8a2db2a25a3c54c3","value":36}},"6171321a457b422b8131f9d1db191a56":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"633c27ebb4784f588dd25de209252431":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"63d4e9636d9b44c79e6300352f305236":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"63dc0d0ac074454ba16035252ef76a91":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"644605e0c20844e4baa9b9298c09968c":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"64ba619722a34f2183ef532677aeeb07":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"65294d58f59545a891e172385b40b7de":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"65c16eee1eb94d27913f27e95d6bafd6":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"66b052767be94bffb302f2a7c936aa16":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"6820cd5fd8154861a28f5271de3fa23e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"69537409303a4f40a56a9522306db183":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_a58ee14276f14bb88a5da2ece8f19b48","IPY_MODEL_b825c71657ab40379288da89a629def1"],"layout":"IPY_MODEL_fd228f36d47146c69e21c197412552c5"}},"69edbcef48c943f0993dc38834253bb9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"6ad5f23bfb0d4a998496c686b04872de":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_9676cfc9acb24bb7abce84754d2a2ae3","max":200,"style":"IPY_MODEL_b9cc6ca808bc49faaad07328c1e7c4d7","value":200}},"6b9add0f90054681b3649651e4c27477":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"6be987b06cd74f418111f30115cb0667":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_371c0a34975d4df6be3632a3b5689d2a","IPY_MODEL_a1537d90ea9b415bad04cc56840529da"],"layout":"IPY_MODEL_c3f601ca181a41c6a633d66aa8ad849b"}},"6d4d7ca5046e43988c3b0963efce41e1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"danger","description":"  0%","layout":"IPY_MODEL_d1b128436ddb4a2c9dcbef3b4c13f463","max":200,"style":"IPY_MODEL_30c37a13ee3c400bbe33e366fc614d7b"}},"6dea30e9d6744d1fb4873bcc0309d95d":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"6e053ade1bba42469303278ad0a8709d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_4b21bd8461824452a2124200792830d9","max":2,"style":"IPY_MODEL_77ac39df370c442fbdb75dfe327ca68d","value":2}},"6ee8e580ec0a4dbc84d20ca5cdce3659":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_9347752520f446a0ac46a244c7c90c23","style":"IPY_MODEL_041c4577cd154a11bd203ba8b9ef0814","value":100}},"6ee9907fb4e34df1adf991591c3b6ac8":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"6eef5afda12242458b9ae0f5dd6d62a0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"6f0cc152ae8a497888b014ab82d7593e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"6f2c91d254e441e8a51ccd861d5e5e25":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"6fa11001c48e4cdea94a4046fa3d7ff4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"6fb4e131d37a4198a1d9ef56f80f87df":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_5691525b552147a497f463da43c63c9f","style":"IPY_MODEL_75850f26227847abaa5d329486d25869","value":" 200/200 [01:48&lt;00:00,  1.84it/s]"}},"70671b44b73f4fab81e8e23d0d615cb7":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"70a2219d96794c6497563586ffd94141":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"70deed10e21e47fabe60f9e5d9d46aa2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"71ce339aeb8e4bb0826cd240ce32a924":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"736b343facfe4239b3719694a3db919f":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"73c21c21c4524f5f8363e90b465737d1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_0e539d49dff34320bea9d52213f3efb9","max":2,"style":"IPY_MODEL_69edbcef48c943f0993dc38834253bb9","value":2}},"74ea9583a7da4266a428df9f5aafd8d5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_3f39e1a5b59d4340aa580cc7117f5cef","style":"IPY_MODEL_2bd914e6301341aab54384c9e5d9e40c","value":" 300/300 [09:12&lt;00:00,  1.84s/it]"}},"74fd0b4380af445b8aa4987236bd7fe0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"75850f26227847abaa5d329486d25869":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"75b3e310ef8f4a54a3597cd7b372bf30":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"76abbd606f7a485dbe40ef58285f3e0e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"776d0e21db534eacb29f09a921e34b91":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"77ac39df370c442fbdb75dfe327ca68d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"77bd90f973014f17a865ccd14b15d3d3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"77cf50eb56be46f1acc17774e03d1846":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"7838d64623284d3d8a0c4b705c5705c8":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_05746fa0e0ec4b9e9809f872a2409e61","IPY_MODEL_5ba27fbd2e424400a21c37819f03e8e0"],"layout":"IPY_MODEL_4320bbdf72a348a38496d24de28574b4"}},"78391f741a51400fbfdde626a3c24ba6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_cb2b8c6b4e844840ab180e0e1ec37dfd","IPY_MODEL_2e19164d8cd34b8a803b01964f58e141"],"layout":"IPY_MODEL_f33c23a7e9b64d809207a1c01934f351"}},"786a96942a914f69b8fcead6f535a674":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"7877849a0b5f4525aa1d6ecfdf19bb81":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"78944d80568e4373939e650709180a93":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_6dea30e9d6744d1fb4873bcc0309d95d","max":300,"style":"IPY_MODEL_f5a34d7312774abfbd381767fee34b15","value":300}},"78a68dd9fc314f3794813ca48fd7bad2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_f28c596da1c34a89bd55e760aa3ba280","style":"IPY_MODEL_2142b21e4b614286b407f8b963edbed0","value":" 2/2 [00:00&lt;00:00,  4.92it/s]"}},"78d133fcdcc64f0e9c157104a07498a8":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_a1621714596f4f69b3eee003ca2ff0c2","IPY_MODEL_5d4d5c8fef9e438e9a69e8b23d7576e8"],"layout":"IPY_MODEL_cf7d571e90a248c18c36bc126415f8bb"}},"78f413bba67c4397814759d60fc7aa64":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"793f24a97a18438fa113cd308e77ab1a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"795629f364aa4de2b1fc6a673c728c94":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"7ab2d7e0745f492c95fdcc01ea526660":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"7affcb4f3c014e098d4ef3935319cca2":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"7b20403c7afc4e0f8c2e12a7b586663a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"7b5fedaeb68b4e109a132f6024c00758":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_500c6483b18c4b2080976519239678d1","max":200,"style":"IPY_MODEL_30c1af7383b34066a34b851405c9224b","value":200}},"7bb28a9ed2d2490e947bd5131ad4cfd7":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"7c622b04a67d4563af59c80b6eebe4c1":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"7cb22a856cd54450afa5386cbb319a61":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_99df768f86b647fcb0c82b2fdf07e378","max":200,"style":"IPY_MODEL_99f392a21bc24b25ae9ee17aba3cb896","value":200}},"7cf8f00f1d424d52b7311349c99bedcf":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_ae5addb52d224935935dc16673c8ed0c","style":"IPY_MODEL_fb0322676ef6410dae53a96e5a08123b","value":" 2/2 [00:59&lt;00:00, 29.94s/it]"}},"7d07aae6fd7043bc88023fa95168af3f":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"7d33a349fd384f33b124d8ab9a6a541e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"7d42b0973dc54e289cccd7fbeff101e4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_a6cd2e74e01d40ed965693b1d5a40a2b","style":"IPY_MODEL_f105b822ebfb43909ef152d96bb9ab00","value":" 0/200 [4:53:27&lt;?, ?it/s]"}},"7dcbc753322043c9bcb3da800e75db79":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"7ecfbfb9854e4fda99a5c91f66c8fbc2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_22abd941f4214f6688eacea066cd083e","style":"IPY_MODEL_258346915b2c40ecabeceec670bff283","value":" 300/300 [01:00&lt;00:00,  4.99it/s]"}},"7f16ba386f644f51b27ad7fe4c08c093":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"7f81e141eb3b4fd88a3934316bf11824":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_c6e54443904e410abb351dccc23bca5b","max":400,"style":"IPY_MODEL_75b3e310ef8f4a54a3597cd7b372bf30","value":400}},"7f865c763c074d128c345864ccb0aaa3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_54959e533bd045608c4e15e7246583d5","max":200,"style":"IPY_MODEL_77cf50eb56be46f1acc17774e03d1846","value":200}},"80e8324050854bd790bcfeb8eb647b81":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"813e680a9ae84b68bf3d3dd81f2cb483":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"8177175ee5f84a5bb4e4495284e3329d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"81eb122cc31e436fa2518540b509d67f":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"81f90fd6c6494070b2749c48b49563c4":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"8242093b80f842448baf3b0d3e1298ac":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"825638143d814b42b24892ce66d290c4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_1f13321849cb4357bdbbdf65400bd224","max":200,"style":"IPY_MODEL_040e16ebba49457f9b5d6a315c9554fb","value":200}},"8309c3cbff684aaaa1bc8a88524d6aa9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"8381899ede4f4c13a9deb3e18fa709e6":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"84455b0a64a74294bd5814ded73c05f5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"85062f5084a34c53b9070bf241f0f6eb":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"85347a3174794baaab724fb8c308f963":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_ecdb0901f6a64234afe87251cc2861d9","IPY_MODEL_22b5281f57f44b3d81b4484e1328abaf"],"layout":"IPY_MODEL_2676372c45bd45fa95569015aa8b4068"}},"864f93670a134cd9bbd23b54cbf7176e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_422bb40b54c347ef8a0ed9f5187eb8ac","style":"IPY_MODEL_bd2510ebcf1343b391e8b14656725e21","value":" 300/300 [02:41&lt;00:00,  1.85it/s]"}},"87f5c560a4b647578512d1f4883a73ed":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"8823c082c8ed493cb3178e340ab8f996":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"8829c85b980846c2b000aced08613a3a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"894399c1f4c94c3ebde809aed58fcc0d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_6d4d7ca5046e43988c3b0963efce41e1","IPY_MODEL_4332d5782bf84883a614d7dd535644d4"],"layout":"IPY_MODEL_c694c077c4ce45c79d5a89fd2150a82a"}},"8a3108b7fa6243b094d6e159f76bb71b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"8a4ec07820304bb3959a4886572e256b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_f2537df39ba74b7baf88efccc73102c5","max":300,"style":"IPY_MODEL_d7c788e387144ff58717ac2f5add29fd","value":300}},"8a539cd815044bdcb9e3effc992898aa":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_a46db3dcc2824167bdc2eea2cb1a67dd","IPY_MODEL_442a19f8ce1f4cb4b277182c54f37412"],"layout":"IPY_MODEL_16cdec1d589e4b84b9e1432f9d326da8"}},"8a574bff7bb944aa9c3f2d4f7cefd001":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"8ab38bd0c0104b04bcf7e37d5d451dca":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_41a1041df1e24f0082fd32d3e28f8289","IPY_MODEL_5aefb71b9c0e4565a3216fc3fff96d1a"],"layout":"IPY_MODEL_f33d9d8677b342a5b82d7fad684589f9"}},"8b3b79ffdd8f41bdb0cc20498532c99a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"8c46653c18cc447bac0c070b04a39364":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"8ccc549d96f74e5aa9205459c1e3b345":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"8d4b6c6c2c904fe39a50a16bf0ee1702":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"8d6012b4c1384571ad3f4902f5738572":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"8e07d701daab4f5db990c0de63b01cf1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_df585ffddcb848aea7591f62a6c563ba","max":300,"style":"IPY_MODEL_32d071ff0e994878a3ede66e8fc1a5ff","value":300}},"8e4c6a49b90344e1ad5197a5aa9df450":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"8e8af27098434fa680e7c0b40eef899f":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_6ee8e580ec0a4dbc84d20ca5cdce3659","IPY_MODEL_bab03ad950694dfb8972b97a47a2270a"],"layout":"IPY_MODEL_907d3e0a03af405cae7c2e34c33026be"}},"8e9e9e15459c4b69b13450b434a7cbab":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"8ec0ea1cfc28409591bc5c620bb3a366":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"8fabdcdea24b4b54a9bbdd56da662bcf":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"904203b941fc4bd6844520979fa109de":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"9049915c466449bf9fc9e8234f8810dd":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"907d3e0a03af405cae7c2e34c33026be":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"90af23406818480c89f896fe463aef3b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_24486927ecf1462d9deafc8721c5a67a","style":"IPY_MODEL_36285193f99749d381423663b551b044","value":" 300/300 [00:54&lt;00:00,  5.51it/s]"}},"916530172abc4fb3b7190ea2a11c7e3a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"918bb4b8c1a64800a0f68742d0a4eb03":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"91900a0131ea47018a6185433cfe384e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"92964a4277cf4e1db0ab54de566c53f1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"92df9886389744c68470d97e09e556e1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"9339a5d8fab34838a642627ec736ee7d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"9347752520f446a0ac46a244c7c90c23":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"93813a7f86d54a01bb419e5f7380577c":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"93deaf444cbe4c4f9fa4a89d899badc3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"947ccf849d4a40c9a5ae8e6633876809":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_b41cb8612e2e4cb0a7fd4051a069dafd","IPY_MODEL_f4af073fe03e408bab08676718edc754"],"layout":"IPY_MODEL_c82e33abb4ff434cbed06ce9530bb95c"}},"95107cb944314535a49dd5085f2f5947":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_b5ec8fc3592743daabebcec33adc0384","style":"IPY_MODEL_fb63e87baae64d1bb51747a09af59d0c","value":" 400/400 [58:19&lt;00:00,  8.75s/it]"}},"95744244bde34b35a990ff69d32b0a5b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"95bdc7a7fc5842cf9eeb07f811fa338e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"9637f0b173ee4e4b8666817f3c144134":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_2eac5e4754a04871ad5524b632bedfb5","style":"IPY_MODEL_aa055abd62e04115949e37da49b6a023","value":" 0/200 [3:45:23&lt;?, ?it/s]"}},"964f3401d730419a997c6932c0556d68":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"9676cfc9acb24bb7abce84754d2a2ae3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"96a94721e56e42119b08ff65b78d2269":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_cba25e88e2c54b72885b0c4e85288bab","IPY_MODEL_40cc03073d25468e91fecf6f1913cfac"],"layout":"IPY_MODEL_6171321a457b422b8131f9d1db191a56"}},"970d6f561807474e8170da3045e3220d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"9716614967bc4d30859edab335b605e3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"97283ed56c1043e0ae3331d195230a0b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"98ab0e6f34664c63b13604fcf4d54850":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"990a36284a4540c4b05288f2eb363276":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"99973e1e8c19420f94305738043628a6":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"99df768f86b647fcb0c82b2fdf07e378":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"99f392a21bc24b25ae9ee17aba3cb896":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"9a2aeb063e5141259cafc1a099cb2b09":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"9a72932bbd2f4487acabc05224f0eecd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"9b436a9d3c21459f9311bd4e5208449b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_347f9d470b754d66b877ea7932a69209","IPY_MODEL_04abc016a87541f48a7df9b6a35ba8c9"],"layout":"IPY_MODEL_77bd90f973014f17a865ccd14b15d3d3"}},"9da400dd12594b29beb1069ddf179b62":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"danger","description":"  0%","layout":"IPY_MODEL_156d230fa70f4fb2bbc7efed3ce55f50","max":200,"style":"IPY_MODEL_199f54999a9f4b818fa156890b76ef05"}},"9db566bdbaea46f2a3677493b9e227c6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_23fe8f5977a44673ad7dd704fee8dc31","IPY_MODEL_ed98421ab07449bdbfdad6f6a7125cc2"],"layout":"IPY_MODEL_8ccc549d96f74e5aa9205459c1e3b345"}},"9dcd5b65fa6c42cea1fb1cb2e41d7642":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_d28f565b70934cfda746a97502cc06f1","style":"IPY_MODEL_be0b46731a1b452299192868703f53cc","value":" 200/200 [01:54&lt;00:00,  1.75it/s]"}},"9e7e7006e76347cfb21f1453cead6cc0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"9e89d0ce3d7b415b9c0968949ee04fa4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"9fd93d0caa8b43d38438bbc8a8268516":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_1e1c21bba5bd442cabb00ed2cb4910d7","max":200,"style":"IPY_MODEL_7dcbc753322043c9bcb3da800e75db79","value":200}},"a1537d90ea9b415bad04cc56840529da":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_6f2c91d254e441e8a51ccd861d5e5e25","style":"IPY_MODEL_09871093305140deb2c16d792d5708df","value":" 0/200 [3:44:54&lt;?, ?it/s]"}},"a1621714596f4f69b3eee003ca2ff0c2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_6ee9907fb4e34df1adf991591c3b6ac8","max":300,"style":"IPY_MODEL_87f5c560a4b647578512d1f4883a73ed","value":300}},"a1a17343c6444227bfe3a68f3550e33e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a26eef444a184f29ad42e8f10f5addca":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a2a60c4849794768a5fe96ce5161b892":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_ba2e3c2689634917be04cb879a4fdebb","style":"IPY_MODEL_70a2219d96794c6497563586ffd94141","value":" 300/300 [00:56&lt;00:00,  5.30it/s]"}},"a2c373dfca9546e2b3a82023a08586b0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_c88c41d9f4ca4f60b313c112668ec5c2","IPY_MODEL_95107cb944314535a49dd5085f2f5947"],"layout":"IPY_MODEL_19bb316c90624805acb7f93073376ac4"}},"a3044cd2d6b64d789fee113ff02def66":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_6e053ade1bba42469303278ad0a8709d","IPY_MODEL_c03d1ebf8f3448e38b306b1e212b5b3d"],"layout":"IPY_MODEL_3a2eef88782e423d803d7d7a7f39802b"}},"a3ac1454cf3b4efea900927dfda0ced6":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a3c1d79d501a4402a640747905909e85":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a4364c35d04048f3902ef7c9be092fc2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"a46db3dcc2824167bdc2eea2cb1a67dd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_6f0cc152ae8a497888b014ab82d7593e","max":400,"style":"IPY_MODEL_85062f5084a34c53b9070bf241f0f6eb","value":400}},"a570aa43fec443929ccbb865f206f8c4":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a58d484527db4b7f9d001c119bef2616":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_39dcca616a374486bb6cc48db683bf55","IPY_MODEL_a2a60c4849794768a5fe96ce5161b892"],"layout":"IPY_MODEL_f6808c1d7f9d45d597266787677f1b58"}},"a58ee14276f14bb88a5da2ece8f19b48":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_4afd2928c81b4097b24f56435aa94cae","max":2,"style":"IPY_MODEL_8823c082c8ed493cb3178e340ab8f996","value":2}},"a638af5e7bc1411e9a56bd4eef41bd67":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_336c2f60995e4601bc967b86e8516d99","IPY_MODEL_5120d219c8f74d518e5e23f3746c0834"],"layout":"IPY_MODEL_66b052767be94bffb302f2a7c936aa16"}},"a6cd2e74e01d40ed965693b1d5a40a2b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a7b71edb1f18461182063ba01e2e8153":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a8195f66e3bf4b13a69a4f8c5d875b07":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"a86ffc03083f4b2096f4ae5ba888f322":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_81f90fd6c6494070b2749c48b49563c4","style":"IPY_MODEL_6eef5afda12242458b9ae0f5dd6d62a0","value":" 0/200 [3:42:48&lt;?, ?it/s]"}},"a9552da95dbd437c97870dc5b83b1617":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"a9cec229ecb34482afb6187a5dcdd6ee":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"aa055abd62e04115949e37da49b6a023":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"aa5b5994bc4b4223a0e41e85912767c1":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"aaaca3205a8d4e84a874b80e6919b9d0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_565672af0927472c8ed34f8a27e1a83a","IPY_MODEL_214edccc75174a0a9ef6b47be16bceb3"],"layout":"IPY_MODEL_e3f6e67162e84c7f98ef7e57a9bdf590"}},"ab0893c752c84b97b740a08a34c89a2a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"ab4173afdbbf4496bfd4fa490f6ff84b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"ab916a934ef043efa57547624f11ad39":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_f2550e33567d4b1da968234e96877090","style":"IPY_MODEL_d3d55d8460d84b0ea1cb32bf5b4b2222","value":" 200/200 [03:11&lt;00:00,  1.04it/s]"}},"ac30774fcfe24e3c8888cf18558a6bfc":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"ae5addb52d224935935dc16673c8ed0c":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"af1bc4e3b0274dd78db5ddf0e3b451f0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"b041420243ef426dbdc382115119e8e3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_aa5b5994bc4b4223a0e41e85912767c1","max":2,"style":"IPY_MODEL_dcb1f674394c46fbb2e695ef53cab115","value":2}},"b0478dd6470e47edad71ab2b04f29fb9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"danger","description":" 14%","layout":"IPY_MODEL_4b393b795b7d42f9968a8916cbaa56e5","max":200,"style":"IPY_MODEL_b1a566a7afba4166b132134be2081cc7","value":27}},"b09dde67a7fa4db78b8b19cf55b71ac3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_cfb1ee1152404722b268bcf4ed2e5d76","IPY_MODEL_d3b0d092bc394271b9199b4b12254184"],"layout":"IPY_MODEL_a3c1d79d501a4402a640747905909e85"}},"b1a566a7afba4166b132134be2081cc7":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"b1daf4b4191a4a80928799a74dae3829":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_05e2a4880d8c42db8432e75813854c6e","max":2,"style":"IPY_MODEL_d37c4542f4c04ca8aaac8c1d8d4f3b90","value":2}},"b22b22f3d70541108992de0120e6d5d3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"b27d7e46c5d74f459c5dae636ba0e3f0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_3bb25574b9824b02a1895a42cd36573e","IPY_MODEL_c09bfccae89c4c1893db9ee52091d294"],"layout":"IPY_MODEL_204400c2b64b4a8baba57f92ba638205"}},"b2c3096e6e9a4a6d9da80fd1ceae1b78":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_3fe6aecfd5c14291b52f3a2b9fa844b2","style":"IPY_MODEL_12d9758aab96413b843a5cb49c49cf86","value":" 2/2 [00:58&lt;00:00, 29.16s/it]"}},"b2c4d415c8cc42789e11e8d70e8af8af":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_7bb28a9ed2d2490e947bd5131ad4cfd7","style":"IPY_MODEL_8177175ee5f84a5bb4e4495284e3329d","value":" 300/300 [02:04&lt;00:00,  2.42it/s]"}},"b304c23114fc4b368054fdedebe98bec":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_a9552da95dbd437c97870dc5b83b1617","max":2,"style":"IPY_MODEL_92964a4277cf4e1db0ab54de566c53f1","value":2}},"b3f9bcfb2ac44ea8b58b9f4e2e9c98d4":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_f33f34cb34fe4013ba87ffeb6803b8ec","style":"IPY_MODEL_d604818302984961a7c16895b9889a07","value":" 300/300 [02:38&lt;00:00,  1.89it/s]"}},"b41171bb1fcd40d8b615d813192e749f":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_e9a87ce747114b1d8ab1d79318ce8d58","IPY_MODEL_4bd0762e7a8d42fa9f432c8ac4629e94"],"layout":"IPY_MODEL_2beaafc1a6584294ba6d6d51d675e9cc"}},"b41cb8612e2e4cb0a7fd4051a069dafd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_a26eef444a184f29ad42e8f10f5addca","max":2,"style":"IPY_MODEL_fe813fa0b6104b20bf37b7cf22fdb994","value":2}},"b46f9aa612b543249c5f1ba788f40d73":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"b4f2c7a1dece4fc9b36c0268d4861790":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_01746170b5794cd2942e0d5ed3c1d8ae","IPY_MODEL_12d68e6f9fae46da8dfdd6fb467efa8a"],"layout":"IPY_MODEL_108357b9ba4740da8b5c689fb34e9141"}},"b5ec8fc3592743daabebcec33adc0384":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"b71f99c19c3c402dafd7644523e5b5b1":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"b74cbed1f8d74b04a642eb4ea6020ea6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"b7650c9c6576451f995c8c5fd7259417":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_3bdb33b3c17045189cf6f154ab6f948c","IPY_MODEL_2166534bcde14ddda56d9d1c106bcdfb"],"layout":"IPY_MODEL_33c1904fd85d471c81cf350dfd006e00"}},"b825c71657ab40379288da89a629def1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_f9373a82cc314bd292402a8d8cae8748","style":"IPY_MODEL_6fa11001c48e4cdea94a4046fa3d7ff4","value":" 2/2 [00:27&lt;00:00, 13.69s/it]"}},"b88644021e7f43928d5ae08371a892fd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_5e1ed0ae189a484cb35f7ae415afd4ae","max":200,"style":"IPY_MODEL_280f8786057d40aaa970861990da1c18","value":200}},"b88f3049eeee41e8bfd65d3fea6811f5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"b9131bbb84164a8d80b478d19d0a24f0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"b9cc6ca808bc49faaad07328c1e7c4d7":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"b9f67e0cac1742e4b245b4a27f45c5d1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_005b3706ebb94e9998e51963ff182dd5","style":"IPY_MODEL_7d07aae6fd7043bc88023fa95168af3f","value":" 200/200 [00:33&lt;00:00,  6.05it/s]"}},"ba2e3c2689634917be04cb879a4fdebb":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"bab03ad950694dfb8972b97a47a2270a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_786a96942a914f69b8fcead6f535a674","style":"IPY_MODEL_28e9c87517254377a37144a64d9fab82","value":" 100/100 [02:52&lt;00:00,  1.73s/it]"}},"bc79e296edfc4f84a6a9f50f47dc9625":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_8e9e9e15459c4b69b13450b434a7cbab","style":"IPY_MODEL_71ce339aeb8e4bb0826cd240ce32a924","value":" 300/300 [00:57&lt;00:00,  5.17it/s]"}},"bd1cf0ada0ef47028539a8fa794d0120":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"bd23b9e83dde4ff3bdcf82f41950d6cc":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"bd2510ebcf1343b391e8b14656725e21":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"bdfcde8b3a0d4b60a43ce46c9cb4f490":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_52b4e3149f5243eba5136b9da6965740","style":"IPY_MODEL_ed08e5b49c174c7198b4c3c89a5ef5cc","value":100}},"be0b46731a1b452299192868703f53cc":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"be2508cb7ce64495acd34436c96ac58d":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"be819e0b4cc24b29aa88c4ce487888c2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_e07d90e463f94f0280640cfb3340ec4a","max":300,"style":"IPY_MODEL_e5e7d83c4ebc41b9a04a04ab7029d9a6","value":300}},"be8a848b0dbb47fc8f4b15a55244c94f":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"bebc345d8cf74999b6e2a25609d49062":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"bec0f0776ec14a43ab44f654baddbcd3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"bedbd74cbd0a43f088571061a48186c9":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c03d1ebf8f3448e38b306b1e212b5b3d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_b46f9aa612b543249c5f1ba788f40d73","style":"IPY_MODEL_be8a848b0dbb47fc8f4b15a55244c94f","value":" 2/2 [00:00&lt;00:00,  4.73it/s]"}},"c07fe3c2d25a4975975b15fe228884ee":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c090dc7ad12340789ba2d7d72f995729":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c09bfccae89c4c1893db9ee52091d294":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_be2508cb7ce64495acd34436c96ac58d","style":"IPY_MODEL_ee0aca055d704c9ab122e6ffaae095bc","value":" 200/200 [00:35&lt;00:00,  5.56it/s]"}},"c0e9ac1df48a481c816244727794efe8":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c1406aebf4fc4d61b1cef0cdc516ff6e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_e03b05e652664c869023fc1aa132c7cb","style":"IPY_MODEL_b74cbed1f8d74b04a642eb4ea6020ea6","value":" 300/300 [08:12&lt;00:00,  1.64s/it]"}},"c190511f14da4810aec9df4f2c3a41f5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c194d3c1193846c18c83ae2a09ec55d0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_81eb122cc31e436fa2518540b509d67f","style":"IPY_MODEL_1c4ae29996b641dbafe41bdc8a622b70","value":" 2/2 [03:16&lt;00:00, 98.37s/it]"}},"c1feae097d3d4da9921a1a7e303f0a61":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_44a22c1eaafc4672a029097656db33ff","IPY_MODEL_90af23406818480c89f896fe463aef3b"],"layout":"IPY_MODEL_d6ce30b404ce4ff997807acd0d47b554"}},"c3a0ccbb943a47b6a510887ca7b7962a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_03f0b92465f244729fd9ffe68657a746","max":300,"style":"IPY_MODEL_5a6b5846697d4ae9855f14d0629ca318","value":300}},"c3a28bf75ebc47099aa29cd99190f58d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"c3f601ca181a41c6a633d66aa8ad849b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c55539f896074ba1ab57f267bbb2ce1e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"c5cd27b908e74622b3ef9c7d705e1e76":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c655374a178c44ffa49f2d0031c31e97":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"c694c077c4ce45c79d5a89fd2150a82a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c6e54443904e410abb351dccc23bca5b":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c7c2c9410b444ca990b6a0a5ca9fac96":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"c82e33abb4ff434cbed06ce9530bb95c":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"c863855be51e48aea32d3acbed5e906b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_4f0e78a0191b422990ef3965e3388c69","style":"IPY_MODEL_a9cec229ecb34482afb6187a5dcdd6ee","value":" 2/2 [01:43&lt;00:00, 51.76s/it]"}},"c88c41d9f4ca4f60b313c112668ec5c2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_efb0d67b135840279979a935cadb1a63","max":400,"style":"IPY_MODEL_8fabdcdea24b4b54a9bbdd56da662bcf","value":400}},"ca6e65dc37c249308499fec40f7f5873":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_73c21c21c4524f5f8363e90b465737d1","IPY_MODEL_7cf8f00f1d424d52b7311349c99bedcf"],"layout":"IPY_MODEL_46509eeb153e46f1a91b8cbcae039e25"}},"cb2b8c6b4e844840ab180e0e1ec37dfd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_ee97729d91f842bab8609f16067bcdb0","max":300,"style":"IPY_MODEL_37e80dca0f3844519e288819d200a5a8","value":300}},"cba25e88e2c54b72885b0c4e85288bab":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_7c622b04a67d4563af59c80b6eebe4c1","max":2,"style":"IPY_MODEL_c55539f896074ba1ab57f267bbb2ce1e","value":2}},"cc41d24433854b0aaa2d0339cf5667fd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_36b984a22422429b926bda8dac248a67","style":"IPY_MODEL_21ce6253ddab4eb58de3a8e84e943e6d","value":" 2/2 [00:06&lt;00:00,  3.17s/it]"}},"cc46fc787dfc46e1933532edba52b4d1":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_36e1baf5fb8b43a79f4b692e5d6f2b2c","style":"IPY_MODEL_8a574bff7bb944aa9c3f2d4f7cefd001","value":" 2/2 [02:32&lt;00:00, 76.12s/it]"}},"ce06ac6e90fd4bf99b8ed1574455f7ed":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_918bb4b8c1a64800a0f68742d0a4eb03","style":"IPY_MODEL_64ba619722a34f2183ef532677aeeb07","value":" 200/200 [00:33&lt;00:00,  5.88it/s]"}},"ce9b73bc52684b538566f7f6f8964eb3":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"cf1052676d2646ec8aa389f65f52a6af":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_a570aa43fec443929ccbb865f206f8c4","max":300,"style":"IPY_MODEL_776d0e21db534eacb29f09a921e34b91","value":300}},"cf69a28284744cb19eb4962e17a61157":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"cf7d571e90a248c18c36bc126415f8bb":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"cfb1ee1152404722b268bcf4ed2e5d76":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_7877849a0b5f4525aa1d6ecfdf19bb81","max":2,"style":"IPY_MODEL_9716614967bc4d30859edab335b605e3","value":2}},"d0bb16bcc3944068b9ca2dacbee2fad6":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d0c8d1bddecf4ec7a0066a9f54bd0c6a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_a1a17343c6444227bfe3a68f3550e33e","style":"IPY_MODEL_19a8df789fd944cca9081036753dcb38","value":" 100/100 [01:13&lt;00:00,  1.37it/s]"}},"d1078b1205534916a518d952f3cc4ddc":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d117a8e1cd4f424289121c0e45aab561":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_0353ae8ac4ac41c5aa99006e56ec6b58","IPY_MODEL_d590950d75b64408b08754ce95ec9f03"],"layout":"IPY_MODEL_5883fca2f9f346469e0f2ea2e5b68bc7"}},"d1714b06f9b24fdcb4fb40ede36d9149":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d1b128436ddb4a2c9dcbef3b4c13f463":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d21c9a4c45304ff987cba8674fd6d44e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"d28f565b70934cfda746a97502cc06f1":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d2f6efe846834a8da725c90a3a84fc9d":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_60aca04caa9843e3b49884243e023069","style":"IPY_MODEL_970d6f561807474e8170da3045e3220d","value":" 2/2 [01:58&lt;00:00, 59.47s/it]"}},"d35d78bfaf124a9f92e65158e75ba90e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"d37c4542f4c04ca8aaac8c1d8d4f3b90":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"d39585ef35ae46bdb29a4fe03ea7f80e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d3b0d092bc394271b9199b4b12254184":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_35a0e7ff2c5b41569870b35ccc9f7bb2","style":"IPY_MODEL_0201f287dd224e82a7068749ccdffbfd","value":" 2/2 [02:20&lt;00:00, 70.15s/it]"}},"d3d55d8460d84b0ea1cb32bf5b4b2222":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"d590950d75b64408b08754ce95ec9f03":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_bd1cf0ada0ef47028539a8fa794d0120","style":"IPY_MODEL_289e351478334a148ce6924f6bde30b7","value":" 300/300 [00:57&lt;00:00,  5.21it/s]"}},"d604818302984961a7c16895b9889a07":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"d6c4f7c9159c4e539c747b6e2983ed2b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"d6ce30b404ce4ff997807acd0d47b554":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d6e3fa47d15946abb6215174e3dc3847":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"d778a8e621f148768319876eee2276c5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_7affcb4f3c014e098d4ef3935319cca2","max":300,"style":"IPY_MODEL_76abbd606f7a485dbe40ef58285f3e0e","value":300}},"d77e70209e394ad58c7746153dd65374":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_b041420243ef426dbdc382115119e8e3","IPY_MODEL_d2f6efe846834a8da725c90a3a84fc9d"],"layout":"IPY_MODEL_d6e3fa47d15946abb6215174e3dc3847"}},"d7b570e9ece448e7ac427b5f9a9d7aa0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"d7c788e387144ff58717ac2f5add29fd":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"d8b42274c88045ec8eeff13f5f009f17":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_95bdc7a7fc5842cf9eeb07f811fa338e","style":"IPY_MODEL_fb2a06c4e11b417bb5bcd7259d494d26","value":" 100/100 [00:26&lt;00:00,  3.80it/s]"}},"db33162ecd0d485eb50bd270b76c41cf":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_0c2b0e00bec64bd2a4feb98540a38153","IPY_MODEL_dd11d43f6c434e84ab32f114371d30c6"],"layout":"IPY_MODEL_2360820a5f7c48a3923e868e63951d53"}},"db82d5edcfdf447484a4c45a0c7b2599":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"dcb1f674394c46fbb2e695ef53cab115":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"dd11d43f6c434e84ab32f114371d30c6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_bec0f0776ec14a43ab44f654baddbcd3","style":"IPY_MODEL_92df9886389744c68470d97e09e556e1","value":" 2/2 [00:00&lt;00:00,  5.27it/s]"}},"de54816d45594d3e9ebf6a4dc355b7d9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_15fa43a5da0a4906855ceac366431d5b","IPY_MODEL_edd3b4540b1c484fb88b5e7e818751e9"],"layout":"IPY_MODEL_bebc345d8cf74999b6e2a25609d49062"}},"ded6bbcf69174565a96a3434936805aa":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"df585ffddcb848aea7591f62a6c563ba":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"e03b05e652664c869023fc1aa132c7cb":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"e07d90e463f94f0280640cfb3340ec4a":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"e1cd50413e864ede8fcee5b696107db3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_2a9591799c3544b894b018fa5e15a680","style":"IPY_MODEL_f256104b4cf14ad082651ea23bfcd2c7","value":" 200/200 [15:59&lt;00:00,  4.80s/it]"}},"e2a4e0fc30514b20adaee2fdee557be3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_97283ed56c1043e0ae3331d195230a0b","style":"IPY_MODEL_8a3108b7fa6243b094d6e159f76bb71b","value":" 200/200 [12:17&lt;00:00,  3.69s/it]"}},"e34a0470f4c4414381a0f5d8dd584c70":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_63dc0d0ac074454ba16035252ef76a91","max":2,"style":"IPY_MODEL_e4e71bff8119478888b409a852f8bac8","value":2}},"e3f6e67162e84c7f98ef7e57a9bdf590":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"e42fb4a4d5a943cc8104e4a4354bbf9b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"e4e31e648cf549dda28df8b1056041db":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_b0478dd6470e47edad71ab2b04f29fb9","IPY_MODEL_4cd41f6f88f8437698197b6ada89e1fc"],"layout":"IPY_MODEL_15a8b9830465461fa93ced228d4195f3"}},"e4e71bff8119478888b409a852f8bac8":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"e5e7d83c4ebc41b9a04a04ab7029d9a6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"e9a87ce747114b1d8ab1d79318ce8d58":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_f8f2b91deea14c9a8dd8466153b4f512","max":200,"style":"IPY_MODEL_edda0eeaca644ead8c16fb3963a5c597","value":200}},"ea52ffff5d864f0da3a799987550cf50":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_418e4881b88d4d50bef32241d7a2f663","IPY_MODEL_c194d3c1193846c18c83ae2a09ec55d0"],"layout":"IPY_MODEL_590112e86fa14ec0bd038269e1041e38"}},"eac6fbed32cd4f1b810ebf1ac66399b4":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"eb8fa51d45c04b60a7122ae24b6e445a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"ecdb0901f6a64234afe87251cc2861d9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_f77cb21b5317409d8d413902a82f66a1","style":"IPY_MODEL_70671b44b73f4fab81e8e23d0d615cb7","value":100}},"ed08e5b49c174c7198b4c3c89a5ef5cc":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"ed8b40ea79eb412096bdec71e6873bf0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_9049915c466449bf9fc9e8234f8810dd","max":200,"style":"IPY_MODEL_990a36284a4540c4b05288f2eb363276","value":200}},"ed98421ab07449bdbfdad6f6a7125cc2":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_52ded505a1e6408a8513729d78bf8e79","style":"IPY_MODEL_08ff5f75450d4e0bb0ec92a75114d641","value":" 200/200 [01:31&lt;00:00,  2.19it/s]"}},"edd3b4540b1c484fb88b5e7e818751e9":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_9a2aeb063e5141259cafc1a099cb2b09","style":"IPY_MODEL_d7b570e9ece448e7ac427b5f9a9d7aa0","value":" 100/100 [01:11&lt;00:00,  1.41it/s]"}},"edda0eeaca644ead8c16fb3963a5c597":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"ee0aca055d704c9ab122e6ffaae095bc":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"ee97729d91f842bab8609f16067bcdb0":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"efb0d67b135840279979a935cadb1a63":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f07b336518c14a58a43692379d3e77d6":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f082d1f6c2d3423f8dd5f01d75157924":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"f105b822ebfb43909ef152d96bb9ab00":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"f13c7ab103aa49c0a2e8dd0d139264f8":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f151632da6144645be164cc6702fc1a6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_7cb22a856cd54450afa5386cbb319a61","IPY_MODEL_5c79de21221744f892925486a0c51fbc"],"layout":"IPY_MODEL_633c27ebb4784f588dd25de209252431"}},"f2537df39ba74b7baf88efccc73102c5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f2550e33567d4b1da968234e96877090":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f256104b4cf14ad082651ea23bfcd2c7":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"f28c596da1c34a89bd55e760aa3ba280":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f33c23a7e9b64d809207a1c01934f351":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f33d9d8677b342a5b82d7fad684589f9":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f33f34cb34fe4013ba87ffeb6803b8ec":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f36de1e438dc4730bacab82a601fbeca":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_2390f0aec01441a598460829b3aa4e96","style":"IPY_MODEL_293c45fd969748568a082c8e78917138","value":" 300/300 [00:57&lt;00:00,  5.17it/s]"}},"f4af073fe03e408bab08676718edc754":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HTMLModel","state":{"layout":"IPY_MODEL_0dc2668c69e04a19b846b7952323ddf4","style":"IPY_MODEL_2bdaffc829654f17a037026956f08917","value":" 2/2 [00:05&lt;00:00,  2.78s/it]"}},"f52e487e3f99409498b49dbb43d9b93e":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f5a34d7312774abfbd381767fee34b15":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"f5edf0c878d04bb1b535a32bef2adb49":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_9da400dd12594b29beb1069ddf179b62","IPY_MODEL_9637f0b173ee4e4b8666817f3c144134"],"layout":"IPY_MODEL_c07fe3c2d25a4975975b15fe228884ee"}},"f6808c1d7f9d45d597266787677f1b58":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f75172c01ff34e14af386a5c891bcd88":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_99973e1e8c19420f94305738043628a6","max":200,"style":"IPY_MODEL_02bf9bf329aa4b0797fdc4a77a2e3f71","value":200}},"f7587edf77524260871b5aea2498895f":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"IntProgressModel","state":{"bar_style":"success","description":"100%","layout":"IPY_MODEL_0dbf5b71324a475cbb0a1d054390839f","style":"IPY_MODEL_9339a5d8fab34838a642627ec736ee7d","value":100}},"f77cb21b5317409d8d413902a82f66a1":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f7d17faed5744373988335dd6badeec6":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"f8f2b91deea14c9a8dd8466153b4f512":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f9373a82cc314bd292402a8d8cae8748":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"f98d2e5125f14fe99fd543e9a684bb9e":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_d778a8e621f148768319876eee2276c5","IPY_MODEL_74ea9583a7da4266a428df9f5aafd8d5"],"layout":"IPY_MODEL_5314c6a70ae348b488a4bd5397fac7f3"}},"f993f6c919fc4a499d65ea43db97928a":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_002d59f567ca46ba9eb8418ca1d20490","IPY_MODEL_51f0f7beefea4906801eb4798f190109"],"layout":"IPY_MODEL_21789dfc9274478489fdd046e7d8f445"}},"f99b9c7433c9409eac358cd028ada4f3":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_7b5fedaeb68b4e109a132f6024c00758","IPY_MODEL_12d9ec781da44d96afbce411ff6dd654"],"layout":"IPY_MODEL_b22b22f3d70541108992de0120e6d5d3"}},"fac8c672dd3a4fb09444daa2c5220741":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"fb0322676ef6410dae53a96e5a08123b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"fb2a06c4e11b417bb5bcd7259d494d26":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"fb63e87baae64d1bb51747a09af59d0c":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"DescriptionStyleModel","state":{"description_width":""}},"fba80bacb5824f839d556bbf55c1aad6":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"fc3ac025200c4c4099b4abaca46ff3d5":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"fc8e3b8d2097451697e84893872d9ef0":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_c3a0ccbb943a47b6a510887ca7b7962a","IPY_MODEL_c1406aebf4fc4d61b1cef0cdc516ff6e"],"layout":"IPY_MODEL_20eb1f87d25547a3a9936ff64fda12fb"}},"fce90cf031fc4aa8b3db5822bb26f01f":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"fd228f36d47146c69e21c197412552c5":{"model_module":"@jupyter-widgets/base","model_module_version":"1.2.0","model_name":"LayoutModel","state":{}},"fe813fa0b6104b20bf37b7cf22fdb994":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"ProgressStyleModel","state":{"description_width":"initial"}},"ffff294dbe154cb390777d4aa121847b":{"model_module":"@jupyter-widgets/controls","model_module_version":"1.5.0","model_name":"HBoxModel","state":{"children":["IPY_MODEL_54e532ae266b47b3b10e7b402bae0a26","IPY_MODEL_46a01765cf944f7f94a59406439d9b8c"],"layout":"IPY_MODEL_f52e487e3f99409498b49dbb43d9b93e"}}},"version_major":2,"version_minor":0}}},"nbformat":4,"nbformat_minor":4}
>>>>>>> 618c31a5d318c10a07d9cc3f2e8c9d06cb4c4a1d
