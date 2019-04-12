
#method1 sys
import sys
gpus = sys.argv[1]
batch_size = sys.argv[2]
print(gpus)
print(batch_size)

#method2 argparse  输入参数时要输入 -gpu=1
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32,
                    help='xxxxxxxx')
args = parser.parse_args()
print(args.gpus)  #自动解析gpu和batchsize
print(args.batch_size)


"""
tensorflow只提供以下几种方法： 
tf.app.flags.DEFINE_string， 
tf.app.flags.DEFINE_integer, 
tf.app.flags.DEFINE_boolean, 
tf.app.flags.DEFINE_float 四种方法，分别对应str, int,bool,float类型的参数。这里对bool的解析比较严格，传入1会被解析成True，其余任何值都会被解析成False。
脚本中需要定义一个接收一个参数的main方法：def main(_):，这个传入的参数是脚本名，一般用不到， 所以用下划线接收。
以batch_size参数为例，传入这个参数时使用的名称为--batch_size，也就是说，中划线不会像在argparse 中一样被解析成下划线。
tf.app.run()会寻找并执行入口脚本的main方法。也只有在执行了tf.app.run()之后才能从FLAGS中取出参数。 
从它的签名来看，它也是可以自己指定需要执行的方法的，不一定非得叫main
"""
#method3 tf.app.run  -gpus=1
import tensorflow as tf
tf.app.flags.DEFINE_string('gpus', None, 'gpus to use')
tf.app.flags.DEFINE_integer('batch_size', 5, 'batch_size')

FLAGS = tf.app.flags.FLAGS

def main(_):
    print(FLAGS.gpus)
    print(FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()



