import tensorflow as tf
import cifar10_input
import numpy as np
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU使用情况
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPUxx%的显存

# 数据存储路径
datapath = r"D:\cifar10\cifar-10-binary\cifar-10-batches-bin"


# 数据加载,这里利用tensorflow官方文档的cifar10_input.py实现
def load_data(datapath):
    # 获取数据增强后的训练集数据
    train_images, train_labels = cifar10_input.destorted_inputs(datapath, batch_size)
    # 获取裁剪后的测试数据
    test_images, test_labels = cifar10_input.inputs(eval_data=True, data_dir=datapath,
                                                    batch_size=batch_size)
    return train_images, train_labels, test_images, test_labels


# 权重初始化做正则化处理：给权重增加一个L2的正则化处理，筛选出重要的特征
def weight_variable(shape, std, w1):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=std), dtype=tf.float32)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weights), w1, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return weights


# 卷积层偏置初始化为0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


# 定义卷积操作，卷积步长为1. padding = 'SAME' 表示全0填充
def conv_layer(x, w, strides, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=strides, padding=padding)


# 对输入进行占位操作，输入为BATCH*3072向量，输出为BATCH*10向量
def placeholder_inputs():
    images_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 24, 24, 3])
    labels_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=[batch_size])
    return images_placeholder, labels_placeholder


# 网络构建
def inference(batch_noraml=False):
    # 是否使用批正则化
    if batch_noraml:
        is_train = tf.placeholder_with_default(True, (), 'is_train')
    else:
        is_train = tf.placeholder_with_default(False, (), 'is_train')

    # ######################### 第一层卷积 #########################
    hidden1_w = weight_variable(shape=[5, 5, 3, 64], std=5e-2, w1=0)
    kernel1 = conv_layer(images_placeholder, hidden1_w, [1, 1, 1, 1], padding='SAME')
    hidden1_b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]))
    hidden1 = tf.nn.bias_add(kernel1, hidden1_b)
    # BatchNormal,批正则化
    hidden1 = tf.layers.batch_normalization(hidden1, training=is_train, trainable=True)
    hidden1 = tf.nn.relu(hidden1)
    # 进行池化操作
    hidden1 = tf.nn.max_pool(hidden1, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
    hidden1 = tf.nn.lrn(hidden1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75)

    # ######################### 第二层卷积 #########################
    hidden2_w = weight_variable(shape=[5, 5, 64, 64], std=5e-2, w1=0)
    hidden2_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]))
    hidden2 = conv_layer(hidden1, hidden2_w, [1, 1, 1, 1], padding="SAME")
    hidden2 = tf.nn.bias_add(hidden2, hidden2_b)
    # BatchNormal,批正则化
    hidden2 = tf.layers.batch_normalization(hidden2, training=is_train, trainable=True)
    hidden2 = tf.nn.relu(hidden2)
    hidden2 = tf.nn.lrn(hidden2, 4, bias=1.0, alpha=0.01 / 9, beta=0.75)
    hidden2 = tf.nn.max_pool(hidden2, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

    # ######################### 第一层全连接层 #########################
    reshape = tf.reshape(hidden2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    dense1_w = weight_variable([dim, 384], std=0.04, w1=0.004)
    dense1_b = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32))
    dense1 = tf.matmul(reshape, dense1_w)+dense1_b
    # BatchNormal,批正则化
    dense1 = tf.layers.batch_normalization(dense1, training=is_train, trainable=True)
    dense1 = tf.nn.relu(dense1)

    # ######################### 第二层全连接层 #########################
    dense2_w = weight_variable([384, 192], std=0.04, w1=0.004)
    dense2_b = tf.Variable(tf.constant(0.1, shape=[192], dtype=tf.float32))
    dense2 = tf.matmul(dense1, dense2_w)+dense2_b
    # BatchNormal,批正则化
    dense2 = tf.layers.batch_normalization(dense2, training=is_train, trainable=True)
    dense2 = tf.nn.relu(dense2)

    # ######################### 第三层全连接层 #########################
    dense3_w = weight_variable([192, 10], std=1 / 192.0, w1=0)
    dense3_b = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32))
    logits = tf.add(tf.matmul(dense2, dense3_w), dense3_b)

    return logits


# 损失函数
def loss_func(logits, labels):
    labels = tf.cast(labels, tf.int32)
    # 交叉熵损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(tf.reduce_sum(cross_entropy))
    tf.add_to_collection("losses", cross_entropy_mean)
    # 损失函数
    loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
    return loss


# 计算损失,并设定是否设置梯度裁剪
def get_loss(logits, gvs_capped):
    # 获取损失函数
    loss = loss_func(logits, labels_placeholder)
    # 设置优化算法使得成本最小
    optimizer = tf.train.AdamOptimizer(1e-3)
    # 梯度裁剪
    if gvs_capped:
        gvs, var = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_step = optimizer.apply_gradients(capped_gvs)
    else:
        train_step = optimizer.minimize(loss)

    # 获取最高类的分类准确率，取top1作为衡量标准
    correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return loss, correct, train_step, accuracy


# 网络训练
def training(loss, correct, train_step, accuracy):
    # 开始训练， 获取开始时间
    start_time = time.time()
    for step in range(max_steps):
        images_batch, labels_batch = sess.run([train_images, train_labels])
        train, loss_value, train_accuracy = sess.run([train_step, loss, accuracy],
                                                     feed_dict={images_placeholder: images_batch,
                                                     labels_placeholder: labels_batch})
        # 获取计算耗费时间
        cost_time = time.time() - start_time
        a = max_steps / 10
        if step % a == 0:
            start_time = time.time()
            print("step:%d   cost time:%.3f  train accuracy:%.3f  loss:%.3f" %
                  (step, cost_time, train_accuracy, loss_value))

    # 开始测试，并计算测试集上的准确率
    num_examples = 10000
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        images_batch, labels_batch = sess.run([test_images, test_labels])
        test_correct = sess.run([correct], feed_dict={images_placeholder: images_batch,
                                                      labels_placeholder: labels_batch})
        true_count += np.sum(test_correct)
        step += 1

    test_accuracy = true_count / total_sample_count
    print("test accuracy:%.3f" % test_accuracy)


if __name__ == '__main__':
    # 定义类别数，全连接隐藏层节点个数，每次训练batch数，最大迭代次数
    NUM_CLASSES = 10
    FC_SIZE = 384
    batch_size = 128
    max_steps = 10000

    # 加载数据
    train_images, train_labels, test_images, test_labels = load_data(datapath)

    # 构建占位符
    images_placeholder, labels_placeholder = placeholder_inputs()

    # 网络训练参数
    logits = inference(batch_noraml=True)

    # 设置计算函数，loss函数，准确率计算，训练步骤,是否设置梯度裁剪
    loss, correct, train_step, accuracy = get_loss(logits, gvs_capped=False)

    # 启动图
    sess = tf.compat.v1.InteractiveSession(config=config)

    # 开始训练
    tf.compat.v1.global_variables_initializer().run()
    # 启动图片数据增强队列
    tf.train.start_queue_runners()
    training(loss, correct, train_step, accuracy)

