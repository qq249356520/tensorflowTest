import tensorflow as tf

#读取train.tfrecord中的数据
def read_and_decode(filename):
    #创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    #创建一个队列来维护输入文件列表
    file_queue = tf.train.string_input_producer([filename], shuffle=False, num_epochs=1)
    #从文件读取一个样例，也可以使用read_up_to读取多个样例
    _, serialized_example = reader.read(file_queue)
    #解析读入的样例。如果需要解析多个样例，可以使用parse_example函数
    features = tf.parse_single_example(
        serialized_example,
        features={
            #Tensorflow提供两种不同的属性解析方法:
            # 一种是tf.train.FixedLenFeature,这种方法解析的结果是一个Tensor。
            #另一种是tf.VarLenFeature,这种方法得到的解析结果为SparseTensor，用于处理稀疏数据
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)})

    #将字符串解析成图相对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.image.per_image_standardization(img)
    labels = tf.cast(features['label'], tf.int32)
    return img, labels

#将图像打包 形成batch
def createBatch(filename, batchsize):
    images, labels = read_and_decode(filename)
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batchsize

    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                      batch_size=batchsize,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    label_batch = tf.one_hot(label_batch, depth=2)
    return image_batch, label_batch


if __name__ == "__main__":
    mapfile = '/xxxx/xxx/x.txt'
    train_filename = '/xxxxxx/xxx/xx/train.tfrecords'
    test_filename = '/xxxx/xxxx/xxx/test.tfrecords'

    image_batch, label_batch = createBatch(filename=train_filename, batchsize=2) #两幅图作为一个batch进行训练
    test_images, test_labels = createBatch(filename=test_filename, batchsize=20)
    with tf.Session as sess:
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(initop)
        #启动多线程处理数据
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while 1:
                _image_batch, _label_batch = sess.run([image_batch, label_batch])
                step += 1
                print(step)
                print(_label_batch)
        except tf.errors.OutOfRangeError:
            print("train data done!")

        try:
            step = 0
            while 1:
                _test_images, _test_labels = sess.run([test_images, test_labels])
                step += 1
                print(step)
                print(_test_labels)
        except tf.errors.OutOfRangeError:
            print("TESE done!")

        coord.request_stop()
        coord.join(thread)










