#-*- coding:utf-8 -*-

import tensorflow as tf
#加载slim库
slim = tf.contrib.slim

#slim.arg_scope()函数可以用于设置默认的参数取值。此函数的第一个参数是一个函数列表，在这个列表中的函数将使用默认的参数设置。
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
    """
    ……
    此处省略Inception-v3模型中其他的网络结构而直接实现最后一个。
    我们假设输入图片经过之前的神经网络钱箱传播的结果保存在变量net中
    net = 上一层的输出节点矩阵
    """
    net = 0
    #为一个Inception模块声明一个统一的变量命名空间
    with tf.variable_scope('Mixed_7c'):
        #给Inception模块中每一条路径声明一个命名空间
        with tf.variable_scope('Branch_0'):
            #实现一个过滤器边长为1,深度为320的卷积层
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1*1')

        #Inception模块中的第二条路径。这条路径上的结构本身也是一个Inception结构
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net,  384, [1, 1], scope='Conv2d_0a_1*1')
            #tf.concat函数可以将多个矩阵拼接起来。其中第一个参数制定了拼接的维度，这里给出的3指的是 矩阵是在深度这个维度上进行拼接
            branch_1 = tf.concat(3, [
                slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1*3'),
                slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3*1')])

        #第三条路径,也是个inception结构
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1*1')
            branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3*3')
            branch_2 = tf.concat(3, [
                slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1*3'),
                slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3*1')])

        # Inception模块中的第4条
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], 'AvgPool_0a_3*3')
            branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1*1')

        #当前inception模块的最后输出是由上面四个结果拼接而来
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])



