import numpy as np
import tensorflow as tf


# DQN algorithm
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            gamma=0.9,
            e_greedy=0.1,
            replace_target_iter=300,
            memory_size=5000,
            replay_start=5000,
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
        self.replay_start = replay_start
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.lambda_factor = lambda_factor

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2 + 4))  # 存入episode号与step号，方便进行后续计算

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
        self.s = tf.placeholder(tf.float32, [self.batch_size, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [self.batch_size, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [self.batch_size, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [self.batch_size, ], name='a')  # input Action [batch_size, ]
        self.done = tf.placeholder(tf.float32, [self.batch_size, ], name='done')  # if s_ is the end of episode
        self.c_coeficient = tf.placeholder(tf.float32, [self.batch_size, ], name='c_coefficient')

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

        loss = tf.reduce_sum(self.c_coeficient*(self.q_target-self.q_eval_wrt_a),axis=0)
        self.total_loss = tf.square(loss)
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

    def store_transition(self, s, a, r, s_, done, episode, step, is_greedy):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_, done, episode, step, is_greedy))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation * np.ones([self.batch_size, self.n_features])
        # 由于执行随机策略时可能选到这个，所以要先计算
        actions_value = np.mean(self.sess.run(self.q_eval, feed_dict={self.s: observation}))
        best_action = np.argmax(actions_value)

        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            if action == best_action:
                is_greedy = 1
            else:
                is_greedy = 0
            return action, is_greedy
        else:
            is_greedy = 1
            return best_action, is_greedy

    def learn(self):
        if self.memory_counter < self.replay_start:
            return
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        while True:
            if (self.memory > self.memory_size).any():
                start_index = np.random.choice(self.memory_size - self.batch_size - 1)
            else:
                start_index = np.random.choice(self.memory_counter - self.batch_size - 1)
            sample_index = start_index + np.arange(self.batch_size)
            if np.max(sample_index) >= self.memory_size:
                continue
            # 如果采样的全在一个episode中，则返回采样index
            batch_memory = self.memory[sample_index, :]
            if len(set(batch_memory[:, -3])) is 1:
                break

        # 采样完了如何学习？
        # loss函数要改

        # 既然所有的batch都得到了，可以算出每个batch的pi了，也是确定的，不必要在算值了。可以在输入tensorflow图之前就计算
        q_eval_s = self.sess.run(self.q_eval, feed_dict={
            self.s: batch_memory[:, :self.n_features]})  # 期望得到的值，这里是根据target还是eval?梯度更新的是eval
        expected_optimal_action = np.argmax(q_eval_s, axis=1)

        def cal_mu_p(is_greedy):  # 计算p(a|s,mu)
            return is_greedy * (1 - self.epsilon + self.epsilon / self.n_actions) + (
                                                                                    1 - is_greedy) * self.epsilon / self.n_actions

        def cal_pi_p(action, expected_action):  # 计算p(a|s,pi)
            return action == expected_action

        p_mu = cal_mu_p(batch_memory[:, -1])
        p_pi = cal_pi_p(batch_memory[:, self.n_features], expected_optimal_action)
        all_c = np.zeros([self.batch_size, ])
        c = 1
        all_c[0] = 1
        for i in range(1, self.batch_size):
            c = self.gamma * self.lambda_factor * c * np.min([1, p_pi[i] / p_mu[i]])
            all_c[i] = c

        _, cost = self.sess.run(
            [self._train_op, self.total_loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features - 4:-4],
                self.done: batch_memory[:, -4],
                self.c_coeficient: all_c
            })
        # print("learning")
        #print("cost is: " + str(cost))
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
