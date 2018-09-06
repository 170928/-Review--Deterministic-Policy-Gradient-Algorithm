# REINFORCE 로 표현되는 알고리즘을 구현하였습니다.
# 이웅원, 양혁렬, 김건우, 이영무, 이의령 님의 "파이썬과 케라스로 배우는 강화학습" 의 Keras 코드를 참조하였습니다.
# 감사합니다.

import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


EPISODES = 2500
BATCH_SIZE = 32


class REINFORCEMENT:
    def __init__(self, state_dim, action_dim):
        np.random.seed(1)
        self.action_space = [0,1,2,3,4]

        #
        self.s, self.a, self.r = [], [], []

        # Discrete Action space 일때,
        # action_dim = env.action_space.n
        # Continuous Action space 일때,
        # action_dim = env.action_space.shape[0]
        self.action_dim = action_dim

        # state_dim = env.observation_space.shape[0]
        self.state_dim = state_dim

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 선택된 action의 idx 만 1 인 tensor를 받아옵니다.
        self.actions = tf.placeholder(tf.float32, shape=[None,self.action_dim])

        # discount 된 reward 를 가져옵니다.
        # REINFORCEMENT 알고리즘은 몬테 카를로 (MC) 알고리즘이므로
        # Episode가 끝난 이후 최종 Return으로부터 discount된 reward가 필요합니다.
        self.discounted_reward = tf.placeholder(tf.float32, shape=[None,])

        self.outputs = self._build_model()

        self.params = tf.trainable_variables()

        self.action_prob = tf.reduce_sum(self.actions * self.outputs, axis = 1)
        self.cross_loss =  tf.log(self.action_prob) * self.discounted_reward
        self.loss = -tf.reduce_sum(self.cross_loss)

        self.gradients = tf.gradients(self.loss, self.params)
        self.grads_and_vars = list(zip(self.gradients, self.params))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(self.grads_and_vars)

    def get_action(self, sess, state):
        return sess.run(self.outputs, feed_dict = {self.inputs : state})


    def discount_rewards(self,rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards

    def append_sample(self, state, action, rewards):
        self.s.append(state[0])
        self.r.append(rewards)
        act = np.zeros(5)
        act[action] = 1
        self.a.append(act)

    def update(self, sess):
        disc_reward = np.float32(self.discount_rewards(self.r))
        disc_reward -= np.mean(disc_reward)
        disc_reward /= np.std(disc_reward)

        sess.run(self.optimizer, feed_dict = {self.discounted_reward : disc_reward, self.inputs : self.s, self.actions : self.a})

        self.s, self.a, self.r = [], [], []

    def _build_model(self):

        with tf.variable_scope('layer') as scope:
            w1 = weight_var([self.state_dim, 24], 'w1')
            b1 = bias_var('b1')
            w2 = weight_var([24,24], 'w2')
            b2 = bias_var('b2')
            w3 = weight_var([24,self.action_dim], 'w3')
            b3 = bias_var('b3')


            self.inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim])

            h1 = tf.matmul(self.inputs, w1) + b1
            h1 = tf.nn.relu(h1)

            h2 = tf.matmul(h1, w2) + b2
            h2 = tf.nn.relu(h2)

            h3 = tf.matmul(h2, w3) + b3
            h3 = tf.nn.relu(h3)

            outputs = tf.nn.softmax(h3)

            return outputs


def weight_var(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev=0.01), name = name)

def bias_var(name):
    return tf.Variable(tf.constant(0.0),  name = name)

if __name__=="__main__":

    env = Env()

    #action_dim = env.action_space.n
    #state_dim = env.observation_space.n

    action_dim = 5
    state_dim = 15

    print('Action :', action_dim, "//", 'State :', state_dim)

    with tf.Session() as sess:

        agent = REINFORCEMENT(state_dim, action_dim)
        sess.run(tf.global_variables_initializer())

        global_step = 0
        scores, episodes = [], []

        for e in range(2500):

            done = False
            score = 0

            state = env.reset()
            state = np.reshape(state, [1,15])

            while not done:

                env.render()

                global_step+=1

                prob = agent.get_action(sess, state)[0]
                action = np.random.choice(action_dim, 1 , p = prob)

                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1,15])
                agent.append_sample(state, action, reward)

                score += reward

                state = next_state


                if done:
                    agent.update(sess)

                    scores.append(score)
                    episodes.append(e)
                    score = round(score)
                    print("episode:", e, "  score:", score, "   time_step:", global_step )