import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # GPU使用情况
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1  # 占用GPUxx%的显存

datapath = 'D:/cifar10'

# 定义类别数，全连接隐藏层节点个数，每次训练batch数，正则化系数
NUM_CLASSES = 10
FC_SIZE = 384
batch_size = 32
max_step = 10000


# 数据加载
def load_data(datapath):
    train_data = {b'data': [], b'labels': []}
    # 加载训练数据
    for i in range(5):
        with open(datapath + "/data_batch_" + str(i + 1), mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
            train_data[b'data'] += list(data[b'data'])
            train_data[b'labels'] += data[b'labels']

    # 加载测试数据
    with open(datapath + "/test_batch", mode='rb') as file:
        test_data = pickle.load(file, encoding='bytes')

    return train_data, test_data


# 权重初始化做正则化处理：给权重增加一个L2的正则化处理，筛选出重要的特征
def variable_with_weight_loss(shape, std, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=std), dtype=tf.float32)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return var


def loss_func(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(tf.reduce_sum(cross_entropy))
    tf.add_to_collection("losses", cross_entropy_mean)
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


# 卷积层权重初始化，随机初始化均值为0，方差为0.1
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 卷积层偏置初始化为0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


# 定义卷积操作，卷积步长为1. padding = 'SAME' 表示全0填充
def conv_layer(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义最大池化操作，尺寸为2，步长为2，全0填充
def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 对输入进行占位操作，输入为BATCH*3072向量，输出为BATCH*10向量
def placeholder_inputs():
    images_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 3072])
    labels_placeholder = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
    return images_placeholder, labels_placeholder


# 网络构建
def inference():
    # 对输入进行reshape，转换成3*32*32格式，并转置成滤波器做卷积所需格式：32*32*3,32*32为其二维卷积操作维度
    image_pl = tf.reshape(images_placeholder, [-1, 3, 32, 32])
    image_pl = tf.transpose(image_pl, [0, 2, 3, 1])

    is_train = tf.placeholder_with_default(True, (), 'is_train')

    # 第一层卷积，滤波器参数5*5*3, 32个
    hidden1_w = weight_variable([5, 5, 3, 32])
    hidden1_b = bias_variable([32])
    hidden1 = conv_layer(image_pl, hidden1_w) + hidden1_b   # 卷积
    # BatchNormal
    # hidden1 = tf.contrib.layers.group_norm(hidden1)
    hidden1 = tf.layers.batch_normalization(hidden1, training=is_train, trainable=True)
    hidden1 = tf.nn.relu(hidden1)   # 卷积
    pool1 = max_pool_2x2(hidden1)  # 池化

    # 第二层卷积，滤波器参数5 * 5 * 32, 64个
    hidden2_w = weight_variable([5, 5, 32, 64])
    hidden2_b = bias_variable([64])
    hidden2 = conv_layer(pool1, hidden2_w) + hidden2_b   # 卷积
    # BatchNormal
    hidden2 = tf.layers.batch_normalization(hidden2, training=is_train, trainable=True)
    hidden2 = tf.nn.relu(hidden2)   # 卷积
    pool2 = max_pool_2x2(hidden2)

    # 将8 * 8 * 64 三维向量拉直成一行向量
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    # 第一层全连接
    dense1_w = weight_variable([8 * 8 * 64, FC_SIZE])
    dense1_b = bias_variable([FC_SIZE])
    dense1 = tf.matmul(pool2_flat, dense1_w) + dense1_b
    # BatchNormal
    dense1 = tf.layers.batch_normalization(dense1, training=is_train, trainable=True)
    dense1 = tf.nn.relu(dense1)
    # 对隐藏层使用dropout
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    dense1_drop = tf.nn.dropout(dense1, rate=1-keep_prob)

    # 第二层全连接
    dense2_w = weight_variable([FC_SIZE, NUM_CLASSES])
    dense2_b = bias_variable([NUM_CLASSES])

    # softmax
    logits = tf.matmul(dense1_drop, dense2_w) + dense2_b

    softmax = tf.nn.softmax(logits)

    # 交叉熵损失
    cross_entropy = -tf.reduce_sum(labels_placeholder * tf.math.log(softmax), reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy)

    # # 用AdamOptimizer优化器训练
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    # gvs = optimizer.compute_gradients(loss)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # train_step = optimizer.apply_gradients(capped_gvs)

    train_step = optimizer.minimize(loss)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labels_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast将数据转换成指定类型

    return accuracy, loss, train_step, keep_prob


# 网络训练
def training(accuracy, loss, train_step, keep_prob):
    train_data, test_data = load_data(datapath)

    # 对数据范围为0-255的训练数据做归一化处理使其范围为0-1，并将list转成numpy向量
    x_train = np.array(train_data[b'data']) / 255
    # 将训练输出标签变成one_hot形式并将list转成numpy向量
    y_train = np.array(pd.get_dummies(train_data[b'labels']))

    # 对数据范围为0-255的测试数据做归一化处理使其范围为0-1，并将list转成numpy向量
    x_test = test_data[b'data'] / 255
    # 将测试输出标签变成one_hot形式并将list转成numpy向量
    y_test = np.array(pd.get_dummies(test_data[b'labels']))

    # 训练
    for i in range(1000):
        # 100条数据为1个batch，轮流训练
        start = i * batch_size % 50000
        train_step.run(feed_dict={images_placeholder: x_train[start: start + batch_size],
                                  labels_placeholder: y_train[start: start + batch_size], keep_prob: 0.5})
        # 每迭代100次在前200条个测试集上测试训练效果
        if i % 100 == 0:
            # 测试准确率
            train_accuracy = accuracy.eval(feed_dict={images_placeholder: x_test[0: 200],
                                                      labels_placeholder: y_test[0: 200], keep_prob: 1.0})
            # 该次训练的损失
            loss_value = loss.eval(feed_dict={images_placeholder: x_train[start: start + batch_size],
                                              labels_placeholder: y_train[start: start + batch_size], keep_prob: 0.5})
            print("step %d, trainning accuracy: %g loss %g" % (i, train_accuracy, loss_value))

    # 测试
    test_accuracy = accuracy.eval(feed_dict={images_placeholder: x_test, labels_placeholder: y_test, keep_prob: 1.0})
    print("test accuracy %g" % test_accuracy)


if __name__ == '__main__':
    # 启动图
    sess = tf.compat.v1.InteractiveSession()

    # 构建占位符
    images_placeholder, labels_placeholder = placeholder_inputs()

    # 网络训练参数
    accuracy, loss, train_step, keep_prob = inference()

    # 开始训练
    tf.compat.v1.global_variables_initializer().run()
    training(accuracy, loss, train_step, keep_prob)

