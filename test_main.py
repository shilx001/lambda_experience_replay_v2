#test the agents

import numpy as np
import gym
from Agent_Retrace import DeepQNetwork
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make("MountainCar-v0")
print(env.action_space)

total_average_reward=[]

random_seed=[1,100,1000,10000,100000]
for i in range(5):
    env.seed(random_seed[i])
    np.random.seed(random_seed[i])
    tf.set_random_seed(random_seed[i])

    tf.reset_default_graph()
    agent=DeepQNetwork(3,2)
    global_step=0
    average_reward_track=[]
    for episode in range(2500):
        observation = env.reset()
        average_reward=[]
        for step in range(200):
            env.render()
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
    total_average_reward.append(average_reward_track)
    #agent.plot_cost()

total_average_reward=np.reshape(np.array(total_average_reward),[5,-1])
plt.plot(np.mean(total_average_reward,axis=0))

plt.show()