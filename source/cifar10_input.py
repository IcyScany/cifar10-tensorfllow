# 绝对引入，忽略目录下相同命名的包，引用系统标准的包
from __future__ import absolute_import
# 导入精确除法
from __future__ import division
# 使用python 3.x的print函数
from __future__ import print_function

import os
# xrange返回类，每次遍历返回一个值，range返回列表，一次计算返回所有值，xrange效率要高些
from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 24
# CIFAR10的数据分类数为10
NUM_CLASSES = 10
# CIFAR10的训练集有50000个图片
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# CIFAR10的测试集有10000个图片
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    # 创建空类，方便数据结构化存储
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    # 1 for cifar-10；2 for cifar-100
    label_bytes = 1
    # cifar10的图片包含32*32个像素，每个像素包含三个RGB值
    result.height = 32
    result.width = 32
    result.depth = 3
    # 计算每幅图片特征向量的字节数
    image_bytes = result.height * result.width * result.depth
    # 计算每条记录的字节数=标签字节数+每幅图片特征向量的字节数
    record_bytes = label_bytes + image_bytes
    # 读取固定长度字节数信息，可以参看文章https://blog.csdn.net/fegang2002/article/details/83046584
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    # CIFAR10数据通过Reader读取后通过Record传输变为字符串类型，既value为字符串类型
    # 但是要使用的话，需要还原为CIFAR10原始数据格式tf.uint8（8位无符号整形）类型，可以通过tf.decode_raw函数实现
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 从读取的数据记录中截取出标签值
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 从读取的数据记录中截取出图片数据，并且转换为【深，高，宽】的形状[3，32，32]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
        [result.depth, result.height, result.width],
    )
    # 转换depth_major的维度，将第一个维度放在最后，既更新为【高，宽，深】的形状[32，32，3]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(
    image, label, min_queue_examples, batch_size, shuffle
):
    # 设置入列的线程？
    num_preprocess_threads = 16
    if shuffle:
        # 把输入的图片像素数据和标签数据随机打乱后，按照批次生成输出的图片像素数据和标签数据
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            # 出列后，队列中要保持的最小元素数？
            min_after_dequeue=min_queue_examples,
        )
    else:
        # 把输入的图片像素数据和标签数据按照原顺序、按照批次生成输出的图片像素数据和标签数据
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
        )
    # 将输入的图像数据记录到缓存中，为后续展示准备
    tf.summary.image("image", image)

    return images, tf.reshape(label_batch, [batch_size])


def destorted_inputs(data_dir, batch_size):
    # 设置CIFAR10数据文件的位置和名称
    filename = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in xrange(1, 6)]
    # 如果设置的CIFAR10数据文件不存在，报错退出
    for f in filename:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)

    # 将filename中包含的文件打包生成一个先入先出队列（FIFOQueue）
    # 并且在计算图的QUEUE_RUNNER集合中添加一个QueueRunner（QueueRunner包含一个队列的一系列的入列操作）
    # 默认shuffle=True时，会对文件名进行随机打乱处理
    filename_queue = tf.train.string_input_producer(filename)

    with tf.name_scope("data_augmentation"):
        # 调用read_cifar10函数，将队列filename_queue进行处理，返回值赋予read_input
        read_input = read_cifar10(filename_queue)
        # 将图片像素数据read_input.uint8image转化为tf.float32类型，赋予reshaped_image
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # 对图片进行随机切割，转化尺寸为[24,24,3]
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        # 对切割后图片沿width方向随机翻转，有可能的结果就是从左往右，从左往左等于没有翻转
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # 对切割翻转后的图片随机调整亮度，实际上是在原图的基础上随机加上一个值(如果加上的是正值则增亮否则增暗)，
        # 此值取自[-max_delta,max_delta)，要求max_delta>=0。
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        # 对切割、翻转和随机调整亮度的图片随机调整对比度，对比度调整值取自[lower,upper]
        distorted_image = tf.image.random_contrast(
            distorted_image, lower=0.2, upper=1.8
        )
        # 对切割、翻转、随机调整亮度和对比度的图片进行标准化处理，将RGB像素的值限定在一个范围，可以加速神经网络的训练
        # 标准化处理可以使得不同的特征具有相同的尺度（Scale）。这样，在使用梯度下降法学习参数的时候，不同特征对参数的影响程度就一样了。
        float_image = tf.image.per_image_standardization(distorted_image)
        # 设置切割、翻转、随机调整亮度、对比度和标准化后的图片数据设置尺寸为[24,24,3]
        float_image.set_shape([height, width, 3])
        # 设置标签数据的形状尺寸为[1]
        read_input.label.set_shape([1])
        # 设置队列中最少样本数为每轮样本的40%？
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue
        )

        print(
            "Filling queue with %d CIFAR images before starting to train. "
            "This will take a few minutes." % min_queue_examples
        )

    return _generate_image_and_label_batch(
        float_image, read_input.label, min_queue_examples, batch_size, shuffle=True
    )


def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        # 如果不是测试数据，就从训练数据文件中读取数据
        filenames = [
            os.path.join(data_dir, "data_batch_%d.bin" % i) for i in xrange(1, 6)
        ]
        # 设置每轮样本数为训练数据每轮样本数
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        # 如果是测试数据，就从测试数据文件中读取数据
        filenames = [os.path.join(data_dir, "test_batch.bin")]
        # 设置每轮样本数为测试数据每轮样本数
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    # 检验文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file:" + f)
    with tf.name_scope("input"):
        # 将filename中包含的文件打包生成一个先入先出队列
        filename_queue = tf.train.string_input_producer(filenames)
        # 调用read_cifar10函数，将数据文件处理成结构化的类对象CIFAR10Record，并返回给read_input
        read_input = read_cifar10(filename_queue=filename_queue)
        # 将read_input中的图片像素数据转换为tf.float32类型以便后续处理
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # 将reshaped_image图片数据修剪为宽24，高24的尺寸
        resized_image = tf.image.resize_image_with_crop_or_pad(
            reshaped_image, height, width
        )
        # 标准化处理resized_image图片数据返回给float_image
        float_image = tf.image.per_image_standardization(resized_image)
        # 设置float_image尺寸为[24,24,3]
        float_image.set_shape([height, width, 3])
        # 设置标签数据尺寸为[1]
        read_input.label.set_shape([1])
        # 设置队列中最少样本数为每轮样本的40%？
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(
            num_examples_per_epoch * min_fraction_of_examples_in_queue
        )
        # 调用_generate_image_and_label_batch处理float_image数据
        return _generate_image_and_label_batch(
            float_image,
            read_input.label,
            min_queue_examples=min_queue_examples,
            batch_size=batch_size,
            shuffle=False,
        )

