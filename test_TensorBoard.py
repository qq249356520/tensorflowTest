#coding:utf-8
import tensorflow as tf

#定义一个简单的计算图 实现向量加法
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='add')

#生成一个写日志的writer，并且将当前的TF计算图写入日志。
writer = tf.summary.FileWriter('/home/zsy/PycharmProjects/tensorflowTest/log', tf.get_default_graph())
writer.close()
