import sys
from agents.agent import *
from task import Task
import matplotlib.pyplot as plt

# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_pose = np.array([5., 5., 5., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results

num_episodes = 1500
target_pos = np.array([5., 5., 5.])
task = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
agent = DDPG(task)
rewards = []
best_ep = []

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            rewards.append(agent.total_reward)
            if agent.total_reward > agent.best_reward:
                best_ep.append(i_episode)
            print("\rEpisode = {:4d}, reward = {:7.3f} (best = {:7.3f})".format(
                i_episode, agent.total_reward, agent.best_reward), end="")  # [debug]
            break
    sys.stdout.flush()

last = rewards[-50:]

print("\rLast 50 ep mean rewards = {:7.3f}, min = {:7.3f}, max = {:7.3f}".format(
    np.mean(last), np.amin(last), np.amax(last)))

plt.plot(rewards)
plt.legend('rewards')
_ = plt.ylim()
plt.show()
