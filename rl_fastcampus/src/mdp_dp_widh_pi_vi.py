import matplotlib.pyplot as plt

from lib.tensorized_dp import TensorDP
from lib.gridworld import GridWorldEnv
from lib.grid_visualization import visualize_policy, visualize_value_function

# --- Making Environment
nx, ny = 5, 5
env = GridWorldEnv([nx, ny])

# --- Making DP
dp = TensorDP()
dp.set_env(env)

# --- PI Algorithm
dp.reset_policy()
info_pi = dp.policy_iteration()

figsize_mul = 10
steps = info_pi["converge"]
fig, ax = plt.subplots(
    nrows=steps, ncols=2, figsize=(steps * figsize_mul, figsize_mul * 2)
)
for i in range(steps):
    visualize_value_function(ax[i][0], info_pi["v"][i], nx, ny)
    visualize_policy(ax[i][1], info_pi["pi"][i], nx, ny)
# plt.show()

# --- VI Algorithm
dp.reset_policy()
info_vi = dp.value_iteration(compute_pi=True)

figsize_mul = 10
steps = info_pi["converge"]
fig, ax = plt.subplots(
    nrows=steps, ncols=2, figsize=(steps * figsize_mul * 0.5, figsize_mul * 3)
)
for i in range(steps):
    visualize_value_function(ax[i][0], info_pi["v"][i], nx, ny)
    visualize_policy(ax[i][1], info_pi["pi"][i], nx, ny)
# plt.show()
