#test the agents

import numpy as np
import gym
from Agent_Retrace import DeepQNetwork
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")
print(env.action_space)

agent=DeepQNetwork(3,2)
global_step=0
average_reward_track=[]
for episode in range(2500):
    observation = env.reset()
    average_reward=[]
    for step in range(200):
        #env.render()
        action,is_greedy=agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        #print("action is: "+str(action))
        agent.store_transition(observation,action,reward,observation_,done,episode,step,is_greedy)
        agent.learn()
        if done:
            print("episode "+str(episode)+" reach at top at step "+str(step))
            break
        global_step+=1
        observation=observation_
        average_reward.append(reward)
    average_reward_track.append(np.sum(average_reward))
#agent.plot_cost()

plt.plot(average_reward_track)

plt.show()