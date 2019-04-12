import tensorflow as tf
import numpy as np

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    if 'data' in dict:
        dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32 * 32 * 3) / 256
    return dict

def load_data_one(f):
    batch = unpickle(f)
    data = batch['data']
    labels = batch['labels']
    print('Loading %s: %d' % (f, len(data)))
    return data, labels

#加载数据文件
def load_data(files, data_dir, label_count):
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1 : ]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
        labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
        return data, labels

def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):
    res = [0] * len(tensors)
    batch_tensors = [(placeholder, feed_dict[placeholder]) for placeholder in batch_placeholders]
    total_size =len(batch_tensors[0][1])
    batch_count = int((total_size + batch_size - 1) / batch_size)
    for batch_idx in range(batch_count):
        current_batch_size = None
        for (placeholder, tensor) in batch_tensors:
            batch_tensor = tensor[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            current_batch_size = len(batch_tensor)
            feed_dict[placeholder] = tensor[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        tmp = session.run(tensors, feed_dict=feed_dict)
        res = [r + t * current_batch_size for (r, t) in zip(res, tmp)]
    return [ r / float(total_size) for r in res]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    """
    卷积
    Args:
    :param input:  输入的图像数据
    :param in_features: 输入通道的数量 3
    :param out_features:  输出通道的数量 16
    :param kernel_size:  卷积核的尺寸 3*3
    :param with_bias:
    :return: tensor
    """
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current

def block(input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for idx in range(layers):
        tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
        current = tf.concat((current, tmp), axis=3)
        features += growth   #通道
    return current, features

def avg_pool(inputs, s):
    return tf.nn.avg_pool(inputs, [1, s, s, 1], [1, s, s, 1], 'VALID')

def run_model(data, image_dim, label_count, depth):
    """
    DenseNet的核心函数

    :param data: 训练数据集合及测试数据集合
    :param image_dim:  图像的维度 32×32*3
    :param label_count:  要分类的分类类别
    :param depth:  网络的深度
    :return:
    """
    weight_decay = 1e-4
    layers = (depth - 4) / 3
    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, image_dim])
        ys = tf.placeholder("float", shape=[None, label_count])
        lr = tf.placeholder("float", shape=[])

        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder("bool", shape=[])

        #Data_Input_Layer1,数据输入层
        current = tf.reshape(xs, [-1, 32, 32, 3])
        current = conv2d(current, 3, 16, 3)
        #Block1_Layer,第一个block
        current, features = block(current, layers, 16, 32, is_training, keep_prob)
        current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        #Block2_Layer
        current, features = block(current, layers, features, 12, is_training, keep_prob)
        current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        #Block3_Layer
        current, features = block(current, layers, features, 12, is_training, keep_prob)
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training)
        current = tf.nn.relu(current)
        current = avg_pool(current, 8)
        #Block end
        final_dim = features
        current = tf.reshape(current, [-1, final_dim])
        Wfc = weight_variable([final_dim, label_count])
        bfc = bias_variable([label_count])
        ys_ = tf.nn.softmax(tf.matmul(current, Wfc) + bfc)
        #dense结束

        #定义计算图
        cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        #use_nesterov使用加权指数平均 每一个v都由上一个theta得到
        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2  * weight_decay)
        correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #计算图定义完成

        with tf.Session() as session:
            batch_size = 64
            learning_rate = 0.1
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_data, train_labels = data['train_data'], data['train_labels']
            batch_count = int(len(train_data) / batch_size)

            batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
            batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
            print("batch per epoch", batch_count)
            for epoch in range(1, 1 + 300):
                if epoch == 150: learning_rate = 0.01
                if epoch == 225: learning_rate = 0.001
                batch_res = None
                for batch_idx in range(batch_count):
                    xs_, ys_ = batches_data[batch_idx], batches_labels[batch_idx]
                    batch_res = session.run([train_step, cross_entropy, accuracy],
                                            feed_dict={xs : xs_, ys : ys_, lr : learning_rate, is_training : True, keep_prob : 0.8})
                    if batch_idx % 100 == 0:
                        print(epoch, batch_idx, batch_res[1:])

                save_path = saver.save(session, 'densenet_%d.ckpt' % epoch)
                test_results = run_in_batch_avg(session, [cross_entropy, accuracy], [xs, ys],
                                                feed_dict={xs: data['test_data'], ys: data['test_labels'],
                                                           is_training: False, keep_prob: 1.})
                print(epoch, batch_res[1:] , test_results)


def run():
  data_dir    = 'data'                                                                           #训练数据集合
  image_size  = 32                                                                               #图像的尺寸大小为32*32
  image_dim   = image_size * image_size * 3                                                      #图像的维度为32*32*3
  meta        = unpickle(data_dir + '/batches.meta')                                             #将图像转化为一个列
  label_names = meta['label_names']                                                              #类别标签文件
  label_count = len(label_names)                                                                 #类别标签文件的长度
  #=================================================================================================================================
  train_files = [ 'data_batch_%d' % d for d in range(1, 6) ]                                    #训练的数据集合
  train_data, train_labels = load_data(train_files, data_dir, label_count)
  pi          = np.random.permutation(len(train_data))                                            #使用程序随机打乱训练数据的排列顺序
  train_data, train_labels = train_data[pi], train_labels[pi]
  #==================================================================================================================================
  test_data, test_labels   = load_data([ 'test_batch' ], data_dir, label_count)                   #测试数据集合的加载
  print ("Train:", np.shape(train_data), np.shape(train_labels))
  print ("Test:",  np.shape(test_data),  np.shape(test_labels))
  #===================================================================================================================================
  data = { 'train_data':   train_data,
           'train_labels': train_labels,
           'test_data':    test_data,
           'test_labels': test_labels }
  #===================================================================================================================================
  #模块说明:DenseNet的核心函数
  #参数说明:[1]data----------训练数据集合及测试数据集合
  #         [2]image_dim-----图像的维度,32*32*3
  #         [3]label_count---要分类的分类类别
  #         [4]depth---------网络的深度
  #===================================================================================================================================
  run_model(data, image_dim, label_count, 40)
run()










