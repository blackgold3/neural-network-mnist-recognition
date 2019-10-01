import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as tc
# #######  数据导入和初始设定 #########
data = input_data.read_data_sets("data", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
batch = 100
n_batch = data.train.num_examples / batch
max_batch = 45000
learn_rate = 0.8
training_step = tf.Variable(0, trainable=False)
learn_rate = tf.train.exponential_decay(learn_rate, training_step, n_batch, 0.9)  # 自减的学习率/只适用于梯度下降优化器
# #######  数据导入和初始设定 #########

# #######  神经网络设计  ########
is_train = tf.placeholder(tf.bool)
W1 = tf.Variable(tc.layers.xavier_initializer(False)(shape=[784, 500]))
b1 = tf.Variable(tf.constant(0.1, shape=[500]))
W2 = tf.Variable(tc.layers.xavier_initializer(False)(shape=[500, 10]))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))
a1 = tf.matmul(x, W1) + b1
a1_bn = tf.layers.batch_normalization(a1, training=is_train)            # #### 封装的BN操作
a1_ac = tf.nn.relu(a1_bn)
a2 = tf.matmul(a1_ac, W2) + b2
a2_bn = tf.layers.batch_normalization(a2, training=is_train)
a2_ac = tf.nn.relu(a2_bn)
prediction = tf.nn.softmax(a2_ac)
cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=a2_ac))
regular = tf.contrib.layers.l2_regularizer(0.001)
regular_loss = regular(W1) + regular(W2)
loss = cross_loss + regular_loss            # 总代价包括了正则化表达
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)         # #### BN的mean和σ更新不会自动完成，需要强制执行
with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, training_step)  # 训练的计算图
# #######  神经网络设计  ########


# ######  准确率计算  #######
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# ######  准确率计算  #######

# #####   计算   #########
with tf.Session() as sess:
    sess.run(init)          # 变量初始化
    for i in range(max_batch+1):          # 总的迭代步数，用了小批次梯度下降法，也就是运算的总的批次数
        batch_xs, batch_ys = data.train.next_batch(batch)         # data.train.images, data.train.labels
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, is_train: True})
        if i % 1000 == 0:
            acc = sess.run(accuracy, feed_dict={x: data.validation.images,
                                                y: data.validation.labels, is_train: False})
            print("Iter: "+str(i)+" ,validation Accuracy "+str(acc))

    acc = sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels, is_train: False})
    print("test Accuracy " + str(acc))
# #####   计算   #########
