import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
from PIL import Image
#生成整数型的属性
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#制作TFRecord格式
def createTFRecord(filename, mapfile):
    class_map = {}
    #数据源地址
    data_dir = '/home/zsy/PycharmProjects/testTFR/'
    #数据文件夹
    classes = {'class1', 'class2'}

    #输出tf格式的文件地址,filename就是传入的地址
    writer = tf.python_io.TFRecordWriter(filename)

    for index, name in enumerate(classes):
        class_path = data_dir + name + '/'
        class_map[index] = name

        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_row = img.tobytes()  #将图片转化为二进制格式
            #将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
            example = tf.train.Example(features=tf.train.Feature(feature={
                'label':_int64_features(index),
                'image_raw':_bytes_feature(img_row)}))

            writer.write(example.SerializeToString())
    writer.close()


    textfile = open(mapfile, 'w+')
    for key in class_map.keys():
        textfile.writelines(str(key) + '.' + class_map[key] + '\n') #构造一个txt 存放所有图片名字
    textfile.close()


