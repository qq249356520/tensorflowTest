# --coding:utf-8--
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)


# 通过tf.layers来定义模型结构。可以使用原生态tf api或者其他高层封装。
def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    net = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.conv2d(net, 64, 3, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    net = tf.layers.dropout(net, rate=0.4, training=is_training)
    return tf.layers.dense(net, 10)


"""
#自定义estimator中使用的模型。

  Arguments:
    features:输入函数中会提供的输入层张亮。这是一个字典，字典里的内容是通过tf.estimator.inputs.numpy_input_fn中x参数的内容指定的。
    label：正确分类标签,这个字段的内容是通过numpy_input_fn中y参数给出，
    mode:train/evaluate/predict
    params:字典  超参数
"""
def model_fn(featuers, labels, mode, params):
    predict = lenet(featuers["image"], mode == tf.estimator.ModeKeys.TRAIN)
    #如果在预测模式 只需要将结果返回
    if mode == tf.estimator.ModeKeys.PREDICT:
        #使用EstimatorSpec传递返回值，并通过predictions参数指定返回的结果
            return tf.estimator.EstimatorSpec(mode = mode, predictions={"result":tf.argmax(predict, 1)})
    #定义损失
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])

    #定义训练过程
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    #定义评测标准
    eval_metric_ops = {"my_metric": tf.metrics.accuracy(tf.argmax(predict, 1), labels)}

    #返回模型训练过程需要使用的损失函数、训练过程和评测方法
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=False)

#通过自定义的方式生成Esttimator类，这里需要提供模型定义的函数并通过params参数指定模型定义时使用的超参数
model_params = {"learning_rate": 0.01}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

#训练
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True
)
estimator.train(input_fn=train_input_fn, steps=30000)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.test.images},
    y=mnist.test.labels.astype(np.int32),
    num_epochs=1,
    batch_size=128,
    shuffle=False
)
test_results = estimator.evaluate(input_fn=test_input_fn)

#这里的my_metric中的内容就是model_fn中eval_metric_ops定义的评测指标
accuracy_score = test_results["my_metric"]
print("\nTest accuracy: %g %%" % (accuracy_score * 100))

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.test.images[:10]},
    num_epochs=1,
    shuffle=False
)
predictions = estimator.predict(input_fn=predict_input_fn)

for i, p in enumerate(predictions):
    print("Prediction %s: %s" % (i + 1, p["result"]))
