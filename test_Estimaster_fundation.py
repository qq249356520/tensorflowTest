#--coding:utf-8--
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)
mnist = input_data.read_data_sets()

#指定网络输入，所有这里指定的输入都会拼接起来作为整个网络的输入
feature_cloumns = [tf.feature_column.numeric_column("image", shape=[784])]


"""
#通过Tensorflow提供的封装好的Estimator定义网络模型。

  Arguments:
    features_cloumns:神经网络输入层需要的数据
    hidden_units：神经网络的结构 注意 DNNClassifier只能定义多层全连接神经网络 而hidden则给出了每一层隐藏层的节点个数
    n_classes：总共类目的数目
    optimizer:所使用的优化函数
    model_dir：将训练过程中loss的变化以及一些其他指标保存到此目录，通过TensorBoard可以可视化
"""
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_cloumns,
    hidden_units=[500],
    n_classes=10,
    optimizer=tf.train.AdamOptimizer(),
    model_dir="~~"
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True
)

#训练模型 注意 此处没有定义损失函数 ，通过DNN定义的模型会使用交叉上作为损失函数
estimator.train(input_fn=train_input_fn, steps=10000)

#定义测试时的数据输入
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=1,
    batch_size=128,
    shuffle=False
)

accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest accuracy: %g %%" %(accuracy_score * 100))
