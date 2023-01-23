import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.tensorized_dp import TensorDP
from lib.gridworld import GridWorldEnv
from lib.grid_visualization import visualize_policy, visualize_value_function

# np.random.seed(0)

# --- Making Environment
nx, ny = 5, 5
env = GridWorldEnv([nx, ny])

# --- Making DP
dp = TensorDP()
dp.set_env(env)

policy = dp.policy
R = dp.env.R_tensor
weighted_R = policy * R
averaged_R = weighted_R.sum(axis=-1)

# --- Checking P for policy
df = pd.DataFrame(dp.get_p_pi(dp.policy))

# --- PE Algorithm
policy_state_dim = dp.policy.shape[0]
policy_action_dim = dp.policy.shape[1]

v_pi = dp.policy_evaluation()

fix, ax = plt.subplots(1, 2, figsize=(12, 6))
visualize_value_function(ax[0], v_pi, nx, ny)
_ = ax[0].set_title("Value pi")
visualize_policy(ax[1], dp.policy, nx, ny)
_ = ax[1].set_title("Policy")
# plt.show()

v_old = v_pi

# --- PI Algorithm
p_new = dp.policy_improvement()
dp.set_policy(p_new)

v_pi = dp.policy_evaluation()
fix, ax = plt.subplots(1, 2, figsize=(12, 6))
visualize_value_function(ax[0], v_pi, nx, ny)
_ = ax[0].set_title("Value pi")
visualize_policy(ax[1], dp.policy, nx, ny)
_ = ax[1].set_title("Policy")
# plt.show()

v_new = v_pi

# --- Checking PI Result
delta_v = v_new - v_old
