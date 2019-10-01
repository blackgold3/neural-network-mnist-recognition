import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# ######## 解决无法找到卷积的问题
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# ######## 解决无法找到卷积的问题

# 参数定义 #
data = input_data.read_data_sets("data", one_hot=True)
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
max_batch = 15000
batch = 100
learn_rate = 0.1
training_step = tf.Variable(0, trainable=False)
n_batch = data.train.num_examples / batch
learn_rate = tf.train.exponential_decay(learn_rate, training_step, n_batch, 0.9)
# 参数定义 #

# 网络设置 #

# ########初始化权重####
kernel1 = tf.Variable(tf.contrib.layers.xavier_initializer_conv2d(False)(shape=[3, 3, 1, 32]))
bias1 = tf.Variable(tf.constant(0.1, shape=[32]))
kernel2 = tf.Variable(tf.contrib.layers.xavier_initializer_conv2d(False)(shape=[3, 3, 32, 64]))
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
w1 = tf.Variable(tf.contrib.layers.xavier_initializer(False)(shape=[7*7*64, 1024]))
b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
w2 = tf.Variable(tf.contrib.layers.xavier_initializer(False)(shape=[1024, 10]))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))
# ########初始化权重####

# ########计算图定义####
a1 = tf.nn.relu(tf.nn.conv2d(x_image, kernel1, [1, 1, 1, 1], 'SAME') + bias1)  # output size 28*28*32
a1_pool = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # output size 14*14*32
a2 = tf.nn.relu(tf.nn.conv2d(a1_pool, kernel2, [1, 1, 1, 1], 'SAME') + bias2)  # output size 14*14*64
a2_pool = tf.nn.max_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # output size 7*7*64
full_connection = tf.reshape(a2_pool, [-1, 7 * 7 * 64])
a3 = tf.nn.relu(tf.matmul(full_connection, w1) + b1)
a3_drop = tf.nn.dropout(a3, keep_prob)
a4 = tf.matmul(a3_drop, w2) + b2
prediction = tf.nn.softmax(a4)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=a4))  # loss
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy, training_step)
# ########计算图定义####

# 网络设置 #

# 精确度计算 #
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 精确度计算 #

# 运行 #
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(max_batch+1):
    batch_xs, batch_ys = data.train.next_batch(batch)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 1000 == 0:
        acc = sess.run(accuracy, feed_dict={xs: data.validation.images, ys: data.validation.labels,
                                            keep_prob: 1.0})
        print("Iter: " + str(i) + " ,validation Accuracy " + str(acc))
acc = sess.run(accuracy, feed_dict={xs: data.test.images, ys: data.test.labels,
                                    keep_prob: 1.0})
print("test Accuracy " + str(acc))
# 运行 #
