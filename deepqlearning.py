#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
import os
import random
import numpy as np
from collections import deque

# Since this code follows the rules of the old version tf,
# it is necessary to enable TensorFlow 1. x compatibility mode so that the new version of TensorFlow can run this code.
# 启用 TensorFlow 1.x 兼容模式
tf.compat.v1.disable_eager_execution()

# Add custom module path, This is mainly to prevent the program from not finding the file path.
# 添加自定义模块路径
sys.path.append("birds/")
import wrapped_flappy_bird as game

# We need to set basic neural network parameters
# 超参数
GAME = 'bird'  # Game name, Used for log files. 游戏名称，用于日志文件
ACTIONS = 2  # Action number of bird. Stay still and fly upwards. 有效动作数量, 不动和向上飞
GAMMA = 0.99  # Attenuation rate of past observations. Generally between 0.8 and 1. 过去观测值的衰减率
OBSERVE = 10000  # Observation time steps before training, do not explore before this. 训练前的观察时间步数
EXPLORE = 2000000  # The frame rate during the exploration phase. 探索阶段的帧数
FINAL_EPSILON = 0.0001  # The final value of epsilon. epsilon 的最终值
INITIAL_EPSILON = 0.0001  # Initial value of epsilon. The initial and final values can be the same. epsilon 的初始值
REPLAY_MEMORY = 50000  # Memory replay buffer size. 记忆回放缓冲区大小
BATCH = 32  # batch size 批量大小
FRAME_PER_ACTION = 1  # Perform an action every n frames

# Create weight variables 创建权重变量
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# Create bias variables 创建偏置变量
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# Convolutional Layer 卷积层
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

# Maximum pooling layer 最大池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Create a neural network 创建神经网络
def createNetwork():
    # 网络权重
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # Input layer 输入层
    s = tf.compat.v1.placeholder(tf.float32, [None, 80, 80, 4])

    # hidden layer 隐藏层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Output layer输出层
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

# Training Network 训练网络
def trainNetwork(s, readout, h_fc1, sess):
    # Define loss function 定义损失函数
    a = tf.compat.v1.placeholder(tf.float32, [None, ACTIONS])
    y = tf.compat.v1.placeholder(tf.float32, [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    # Initialize game state 初始化游戏状态
    game_state = game.GameState()

    # Memory replay buffer 记忆回放缓冲区
    D = deque()

    # log file 日志文件
    if not os.path.exists("logs_" + GAME):
        os.makedirs("logs_" + GAME)
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # Initialize the first state 初始化第一个状态
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # Save and load network 保存和加载网络
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    checkpoint = tf.compat.v1.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # started training 开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # Select action 选择动作（epsilon-greedy）
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # Do not take any action 不执行任何动作

        # Decay epsilon 衰减 epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Execute actions and observe the next state and reward 执行动作并观察下一个状态和奖励
        x_t1_colored, r_t, terminal, score = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # Store the conversion in the memory replay buffer 将转换存储到记忆回放缓冲区
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # Start training after observation ends 观察结束后开始训练
        if t > OBSERVE:
            # Randomly sample a small batch 随机采样一个小批量
            minibatch = random.sample(D, BATCH)

            # Obtain small batch data 获取小批量数据
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # Calculate the target value 计算目标值
            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # Perform gradient descent 执行梯度下降
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch
            })

        # UPDATE state 更新状态
        s_t = s_t1
        t += 1

        if not os.path.exists('saved_networks'):
            os.makedirs('saved_networks')

        # Save the model every n iterations 每 n 次迭代保存一次模型
        if t % 50000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # Print information
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))

# Start games
def playGame():
    sess = tf.compat.v1.InteractiveSession()  # use compat.v1.InteractiveSession
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()