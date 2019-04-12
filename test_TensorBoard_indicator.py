# -*- coding:utf-8 -*-

import tensorflow as tf

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        #记录张量元素中的取值分布。对于给出的图表名称和张量，此函数会生成一个Summary protocol buffer。
        #将summary写入TensorBoard日志文件后，在HISTOGRAMS栏和DISTRIBUTION栏下都会出现对应名称的图表。
        # 和tensorflow中其他操作类似，此函数不会被立刻执行，只有当sess, run函数明确调用这个操作时，tensorflow才会真正生成并输出这个buffer。
        tf.summary.histogram(name, var)