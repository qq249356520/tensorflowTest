import tensorflow as tf
import os

"""
#TFRecord的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

num_shards = 2 #定义了总共写入多少个文件（TFR）
instance_per_shard = 2 #定义了每个文件中有多少个数据

for i in range(num_shards):
    foldername = './generatefile/'
    filename = (foldername + 'data.tfrecords-%.5d-of%.5d' % (i, num_shards))
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instance_per_shard):
        #Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
"""

files = tf.train.match_filenames_once('./generatefile/data.tfrecords-*')

filename_queue = tf.train.string_input_producer(files, shuffle=False)

#读取并解析一个样本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })

with tf.Session() as sess:
    #虽然本程序没有声明变量，但使用tf.train.match_filenames_once函数时需要初始化一些变量
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)




