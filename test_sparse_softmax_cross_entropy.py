import tensorflow as tf
import numpy as np

labels = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.float32)

#logits代表预测值 wx+b的输出，准备进行softmax
logits = np.array([[1, 2, 7],
                   [3, 5, 2],
                   [6, 1, 3],
                   [8, 2, 0],
                   [3, 6, 1]], dtype=np.float32)

#公式计算  -np.log(y * softmax_out)
#y = n * c, softmax_out is n * c ,相当于将每个样本softmax的c个特征中最大的取出来，再取-就是求最小
softmax_out = tf.nn.softmax(logits)
cross_entropy1 = -tf.reduce_sum(labels * tf.log(softmax_out), axis=1)  #对应元素相乘

cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

classes = tf.argmax(labels, axis=1) #[2, 1, 0, 0, 1]
cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)

cross_entropy4 = tf.losses.sparse_softmax_cross_entropy()

cross_entropy5 = tf.losses.sparse_softmax_cross_entropy()





sess = tf.Session()
print(sess.run(cross_entropy1))

print(sess.run(cross_entropy2))

print(sess.run(cross_entropy3))


#使用一维label计算，对每个样本取第k个元素，k代表实际类别
