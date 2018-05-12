import numpy as np
import tensorflow as tf


# Implementation of Lambda-experience-replay algorithm
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            gamma=0.9,
            e_greedy=0.1,
            replace_target_iter=300,
            memory_size=32,
            replay_start=32,
            batch_size=32,
            lambda_factor=0.8,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.replay_start = replay_start  # start training episode
        self.memory_size = memory_size  # saved episode count of LER
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.lambda_factor = lambda_factor

        self.episode_buffer = []  # 存储episode的缓存，每个episode结束后再存。

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # self.memory = np.zeros((self.memory_size, n_features * 2 + 2 + 4))  # 存入episode号与step号，方便进行后续计算
        self.memory = []  # 初试化为空

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action [batch_size, ]
        self.done = tf.placeholder(tf.float32, [None, ], name='done')  # if s_ is the end of episode
        self.coefficient = tf.placeholder(tf.float32, [None, ], name='coefficient')  # coeffecient factor
        self.lambda_return_cal=tf.placeholder(tf.float32,[None,],name='lambda_return_cal')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_') * (
                1 - self.done)  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)  # q_target只作为值，不计算梯度
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )

        with tf.variable_scope('lambda_return'):
            self.lambda_return = tf.reduce_sum(self.coefficient * (
                self.r + (1 - self.lambda_factor) * self.gamma * self.q_target * (1 - self.done)), axis=0)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval_wrt_a, self.lambda_return_cal, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_, done, episode, step, is_greedy):
        if not hasattr(self, 'episode_counter'):
            self.episode_counter = 0  # 记录存储了多少个episode
        transition = np.hstack((s, [a, r], s_, done, episode, step, is_greedy))
        # replace the old memory with new memory
        # 加上一个存储缓存，缓存存上所有的序列。
        self.episode_buffer = np.append(self.episode_buffer, transition)
        # 如果episode_buffer满了(也就是遇到了done)则转移到self.memory中。
        if done:
            episode_length = step
            self.episode_buffer = np.reshape(self.episode_buffer, [episode_length + 1, -1])  # reshape为memory的格式
            self.memory.append(self.episode_buffer)
            self.episode_counter += 1  # 记录的episode+1
            if len(self.memory) > self.memory_size:  # 如果超过了则清除最早的
                del self.memory[0]

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < 1 - self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            is_greedy = 1
        else:
            action = np.random.randint(0, self.n_actions)
            is_greedy = 0
        return action, is_greedy

    def learn(self):
        # 如果没有满则直接返回不学习
        if self.episode_counter < self.replay_start:
            return
        # 每次学习则从episode_counter中随机选取n个episode
        # 先从250个batch中随机挑选64个batch
        # 再针对每个batch随机选个七点
        selected_batch = np.random.choice(self.memory_size, self.batch_size)
        lambda_return=np.zeros([self.batch_size,])
        state_feed=np.zeros([self.batch_size,self.n_features])
        action_feed=np.zeros([self.batch_size,])
        for i in range(self.batch_size):
            current_batch = self.memory[selected_batch[i]]
            start_index=np.random.choice(current_batch.shape[0],1)
            state_feed[i]=current_batch[start_index,:self.n_features]
            action_feed[i]=current_batch[start_index,self.n_features]
            k_factor=int(np.min([np.ceil(np.log(0.01)/np.log(self.lambda_factor*self.gamma)),current_batch.shape[0]-start_index],axis=0))
            end_index=int(start_index+k_factor+1)
            current_sample_batch = [current_batch[ii] for ii in range(start_index,end_index)]
            coefficient=np.logspace(0,k_factor,num=k_factor,base=self.gamma*self.lambda_factor)
            lambda_return[i]=self.sess.run(self.lambda_return,feed_dict={self.coefficient:coefficient,
                                                                      self.s_:current_sample_batch[:,-self.n_features - 4:-4],
                                                                      self.r: current_sample_batch[:,self.n_features + 1]})
        _,cost=self.run([self._train_op,self.loss],feed_dict={self.s:state_feed,
                                                              self.a:action_feed,
                                                              self.lambda_return_cal:lambda_return})
        self.cost_his.append(cost)
        print("cost is: " + str(cost))
        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
