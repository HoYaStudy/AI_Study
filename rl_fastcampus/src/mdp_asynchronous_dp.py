import matplotlib.pyplot as plt

from lib.gridworld import GridWorldEnv
from lib.async_dp import AsyncDP
from lib.grid_visualization import visualize_policy, visualize_value_function

# --- Making Environment
nx, ny = 5, 5
env = GridWorldEnv([nx, ny])

# --- Making DP
async_dp = AsyncDP()
async_dp.set_env(env)

# --- In-Place VI (Full-Sweeping)
info_ip_vi = async_dp.in_place_vi()

figsize_mul = 10
steps = info_ip_vi["step"]
fig, ax = plt.subplots(
    nrows=steps, ncols=2, figsize=(steps * figsize_mul * 0.5, figsize_mul * 3)
)
for i in range(steps):
    visualize_value_function(ax[i][0], info_ip_vi["v"][i], nx, ny)
    visualize_policy(ax[i][1], info_ip_vi["pi"][i], nx, ny)
# plt.show()

# --- Prioritized Sweeping VI
info_ps_vi = async_dp.prioritized_sweeping_vi()

figsize_mul = 10
steps = info_ip_vi["step"]
fig, ax = plt.subplots(
    nrows=steps, ncols=2, figsize=(steps * figsize_mul * 0.5, figsize_mul * 3)
)
for i in range(steps):
    visualize_value_function(ax[i][0], info_ip_vi["v"][i], nx, ny)
    visualize_policy(ax[i][1], info_ip_vi["pi"][i], nx, ny)
# plt.show()

# --- Partial Sweeping VI
info_ip_partial_vi = async_dp.in_place_vi_partial_update(update_prob=0.2, vi_iters=100)

steps = len(info_ip_partial_vi["v"])
viz_every = 20
n_figs = steps // viz_every
n_cols = 4
if n_figs % n_cols == 0:
    n_rows = n_figs // n_cols
else:
    n_rows = n_figs // n_cols + 1
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
ax = ax.reshape(-1)
for i in range(n_figs):
    viz_i = i * viz_every
    visualize_value_function(ax[i], info_ip_partial_vi["v"][viz_i], nx, ny)
    ax[i].set_title(f"{viz_i}th update")
for off_idx in range(n_figs, n_rows * n_cols):
    ax[off_idx].axis("off")
# plt.show()
