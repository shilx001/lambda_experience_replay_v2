# test the agents

import numpy as np
import gym
from Agent_Retrace import DeepQNetwork
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

env = gym.make("CartPole-v0")
print(env.action_space)

total_average_reward = []

random_seed = [
    885914478, 542024603, 19370379, 4481864, 810806, 50674, 3859, 662, 19, 6, 927243027, 685278783, 82418806, 5619756,
    925065, 86329, 5714, 948, 12, 0
]
for i in range(1):
    env.seed(random_seed[i])
    np.random.seed(random_seed[i])
    tf.set_random_seed(random_seed[i])

    tf.reset_default_graph()
    #agent = DeepQNetwork(3, 2)
    agent = DeepQNetwork(2,4)
    global_step = 0
    average_reward_track = []
    for episode in range(1000):
        observation = env.reset()
        average_reward = []
        for step in range(200):
            env.render()
            action, is_greedy = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # print("action is: "+str(action))
            #print(action)
            #if step == 199:
            #    done = 1
            agent.store_transition(observation, action, reward, observation_, done, episode, step, is_greedy)
            agent.learn()
            if done:
                print("episode " + str(episode) + " reach at top at step " + str(step))
                break
            global_step += 1
            observation = observation_
            average_reward.append(reward)
        average_reward_track.append(np.sum(average_reward))
    total_average_reward.append(average_reward_track)
    #agent.plot_cost()

total_average_reward = np.reshape(np.array(total_average_reward), [1, -1])
pickle.dump(total_average_reward,open('average_reward_retrace_mountain_car','wb'))
#pickle.dump(total_average_reward,open("average_reward_retrace_cartpole",'w'))
plt.plot(np.mean(total_average_reward, axis=0))

plt.show()
