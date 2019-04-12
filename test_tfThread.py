import tensorflow as tf
import numpy as np
import threading
import  time

# def MyLoop(coord, worker_id):
#     while not coord.should_stop():
#         if np.random.rand() * 1000 % 5 > 3:
#             print("Stopping from id : %d" % worker_id)
#             coord.request_stop()
#         else:
#             print("On id: %d" % worker_id)
#         time.sleep(1)
#
# coord = tf.train.Coordinator()
#
# threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)]
#
# for t in threads: t.start()
#
# coord.join(threads)


#tf.QueueRunner主要用于启动多个线程来操作同一个队列，启动这些线程可以通过tf.Coordinator来进行统一管理
#定义一个tf先进先出队列，队列中最多可以有100个实数
queue = tf.FIFOQueue(100, 'float')
#这是一个入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

#使用tf.train.QueueRunner来创建多个线程来进行这个入队操作
#其中第二个参数便表示了需要启动5个线程，每个线程运行的都是op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

#将QueueRunner加入TF计算图上制定的集合。
#若是add函数没有指定集合，那么加入默认集合tf.GraphKeys.QUEUE_RUNNERS.
tf.train.add_queue_runner(qr) #加入默认集合

out_tensor = queue.dequeue()  #这是一个出队操作

with tf.Session() as sess:
    #定义一个coord来协同已经启动的线程
    coord = tf.train.Coordinator()
    #在使用QueueRunner时，需要明确调用start_queue_runner来启动所有线程。否则因为没有线程进行入队操作，当进行出队操作时，程序就会一直等待入队操作
    #tf.train.start函数会默认启动tf.GraphKeys.QUEUE_RUNNERS集合中所有的QueueRunner。
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #获取队列中的值
    for _ in range(3) :
        print(sess.run(out_tensor))

    #使用coord停止线程
    coord.request_stop()
    coord.join(threads)

