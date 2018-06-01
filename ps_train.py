import sys
import pandas as pd
from agents.policy_search import PolicySearch_Agent
from task import Task
import numpy as np
import matplotlib.pyplot as plt

# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_pose = np.array([5., 5., 5., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results

num_episodes = 1000
target_pos = np.array([5., 5., 5.])
task = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
agent = PolicySearch_Agent(task)
rewards = []

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        if done:
            rewards.append(agent.total_reward)
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.total_reward, agent.best_reward, agent.noise_scale), end="")  # [debug]
            break
    sys.stdout.flush()

plt.plot(rewards)
plt.legend('rewards')
_ = plt.ylim()
plt.show()
