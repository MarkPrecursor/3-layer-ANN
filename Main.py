# -*- coding: utf-8 -*-
"""
*三层全连接神经网络的简单测试
*创建于2016.3.16
*作者：Mark
"""
import numpy
import tensorflow as tf

# 样本
Data = numpy.matrix([[1.58,2.32,-5.8],
                     [0.67,1.58,-4.78],
                     [1.04,1.01,-3.63],
                     [-1.49,2.18,-3.39],
                     [-0.41,1.21,-4.73],
                     [1.39,3.16,2.87],
                     [1.2,1.4,-3.22],
                     [-0.92,1.44,-3.22],
                     [0.45,1.33,-4.38],
                     [-0.76,0.84,-1.96],
                     [0.21,0.03,-2.21],
                     [0.37,0.28,-1.8],
                     [0.18,1.22,0.16],
                     [-0.24,0.93,-1.01],
                     [-1.18,0.39,-0.39],
                     [0.74,0.96,-1.16],
                     [-0.38,1.94,-0.48],
                     [0.02,0.72,-0.17],
                     [0.44,1.31,-0.14],
                     [0.46,1.49,0.68],
                     [-1.54,1.17,0.64],
                     [5.41,3.45,-1.33],
                     [1.55,0.99,2.69],
                     [1.86,3.19,1.51],
                     [1.68,1.79,-0.87],
                     [3.51,-0.22,-1.39],
                     [1.4,-0.44,0.92],
                     [0.44,0.83,1.97],
                     [0.25,0.68,-0.99],
                     [-0.66,-0.45,0.08]]);

# 整理标签数据
n = 10
Target = numpy.matrix([[0]*3*n, [0]*3*n, [0]*3*n]).T
Target[0:n, 0] = 1
Target[n:2*n, 1] = 1
Target[2*n:3*n, 2] = 1

sess = tf.InteractiveSession()
m_Input = tf.placeholder("float", shape=[None, 3])
m_Lable = tf.placeholder("float", shape=[None, 3])

# 为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 全连接层1
W_fc1 = weight_variable([3, 4])
b_fc1 = bias_variable([4])
h_fc1 = tf.nn.softmax(tf.matmul(m_Input, W_fc1) + b_fc1)  #激励

# 全连接层2
# W_fc2 = weight_variable([4, 4])
# b_fc2 = bias_variable([4])
# h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)  #激励

# keep_prob = tf.placeholder("float")
# h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 输出层
W_fc3 = weight_variable([4, 3])
b_fc3 = bias_variable([3])
m_Output = tf.nn.softmax(tf.matmul(h_fc1, W_fc3) + b_fc3)

# 训练和评估
cross_entropy = -tf.reduce_sum(m_Lable*tf.log(m_Output))
train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)
#correct_prediction这里返回一个布尔数组。
correct_prediction = tf.equal(tf.argmax(m_Output,1), tf.argmax(m_Lable,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(30000):
  m_Random = numpy.random.randint(0,30)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        m_Input: Data, m_Lable: Target})#, keep_prob: 1.0
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={
      m_Input: Data[m_Random], m_Lable: Target[m_Random]})#, keep_prob: 0.5

print "test accuracy %g"%accuracy.eval(feed_dict={
    m_Input: Data, m_Lable: Target})#, keep_prob: 1.0



