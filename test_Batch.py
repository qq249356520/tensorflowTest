import tensorflow as tf

files = tf.train.match_filenames_once('/home/zsy/PycharmProjects/tensorflowTest/generatefile/data.tfrecords-*')

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

example, label = features['i'], features['j']

batch_size = 3

capacity = 1000 + 3 * batch_size

#组合样例 [example, label]给出了需要组合的元素。 batch和capa分别为每个batch的个数和队列最大容量。
#当队列长度等于容量时，tensorflow将暂停入队操作，等待元素出队。当元素个数小于容量时，tf重新启动入队操作
example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #获取并打印组合后的样例。在真实问题中，这个输出一般作为神经网络的输入
    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)