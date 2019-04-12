import tensorflow as tf
import numpy as np

#利用数据集读取数据有三个基本步骤：
#1.定义数据集的构造方法
#2.定义遍历器
#3.使用get_next()方法从遍历器中读取数据张量


# #从一个数组中创建数据集
# input_data = [1, 2, 3, 5, 8]
# dataset = tf.data.Dataset.from_tensor_slices(input_data)
#
# #定义一个迭代器，用于遍历数据集。
# iterator = dataset.make_one_shot_iterator()
# x = iterator.get_next()
# y = x * x
#
# with tf.Session() as sess:
#     for i in range(len(input_data)):
#         print(sess.run(y))

# #解析一个TFRecord方法
# def parser(record):
#     features = tf.parse_single_example(
#         record,
#         features={
#             'feat1': tf.FixedLenFeature([], tf.int64),
#             'feat2': tf.FixedLenFeature([], tf.int64)
#         })
#     return features['feat1'], features['feat2']
#
# #从TFRecord中创建数据集
# input_files = ['']
# dataset = tf.data.TFRecordDataset(input_files)
#
# #map()函数表示对数据集的每一条数据进行调用相应方法。使用TFD读出来的是二进制数据，这里需要使用map调用paraser对数据解析。同样 map函数也可以用来完成其他数据的预处理
# dataset = dataset.map(parser)
#
# iterator = dataset.make_one_shot_iterator()
#
# feat1, feat2 = iterator.get_next()
#
# with tf.Session() as sess:
#     print(sess.run([feat1, feat2]))
#
#


###---------------当使用placeholder作为路径时------------------------

# input_files = tf.placeholder(tf.string)
# dataset = tf.data.TFRecordDataset(input_files)
# dataset = dataset.map(parser)
#
# #定义另一种迭代器
# iterator = dataset.make_initializable_iterator()
# feat1, feat2 = iterator.get_next()
#
# with tf.Session() as sess:
#     #首先初始化iterator 并给出具体路径的值
#     sess.run(iterator.initializer, feed_dict={input_files: ['', '']})
#
#     #遍历所有数据一个epoch
#     while True:
#         try:
#             sess.run([feat1, feat2])
#         except tf.errors.OutOfRangeError:
#             break

#####-----------------dataset高级API--------------------

#列举输入文件
train_files = tf.train.match_filenames_once('train_file-*')
test_files = tf.train.match_filenames_once('test_file-*')

#从TFRecord中解析数据。  其中的标签都是TFR中存在的（也就是说，在制作TFR时，需要将已有信息都写入TFR）
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64)
        })

    #从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(features['image'], tf.uint8)
    decoded_image.set_shape(features['height'], features['width'], features['channels'])
    label = features['label']
    return decoded_image, label

def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    distorted_image = tf.image.resize_images(distorted_image, [height, width], np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    return distorted_image



image_size = 299
batch_size = 100
shuffle_buffer = 10000  #随机打乱数据时buffer的大小



#定义数据集
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

dataset = dataset.map(lambda image, label:(
                        preprocess_for_train(image, image_size, image_size, None), label))
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

#迭代器 虽然定义数据集是没有使用placeholder  但是使用match_filename方法得到的结果和placeholder类似
iterator = dataset.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()

#定义网络结构以及优化过程
learning_rate = 0.01
