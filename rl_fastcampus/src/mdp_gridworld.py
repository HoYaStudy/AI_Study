import numpy as np

from lib.gridworld import GridWorldEnv

# --- Making Environment
nx, ny = 4, 4
env = GridWorldEnv(shape=[nx, ny])

observation_space = env.observation_space
action_space = env.action_space

P = env.P_tensor
R = env.R_tensor

# --- Simulation
_ = env.reset()
action_mapper = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

step_counter = 0
while True:
    env._render()

    curr_state = env.s
    action = np.random.randint(low=0, high=4)
    next_state, reward, done, info = env.step(action)

    print(f"t: {step_counter}")
    print(f"state: {curr_state}")
    print(f"action: {action_mapper[action]}")
    print(f"reward: {reward}")
    print(f"next state: {next_state}")

    step_counter += 1
    if done:
        break

# --- Multiple Simulation with Same Starting Point
def run_episode(env, s0):
    _ = env.reset()
    env.s = s0

    step_counter = 0
    while True:
        action = np.random.randint(low=0, high=4)
        _, _, done, _ = env.step(action)

        step_counter += 1
        if done:
            break
    return step_counter


n_episodes = 10
s0 = 6
for i in range(n_episodes):
    len_episode = run_episode(env, s0)
    print(f"Episode {i}: {len_episode}")
