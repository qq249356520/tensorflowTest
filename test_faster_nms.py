# -*- coding:utf-8 -*-
import numpy as np

def py_cpu_nms(dets, thresh):
    """

    :param dets:N*M 二维数组, N是BBOX的个数， M的前四位对应的是（x1, y1, x2, y2） 第5位是对应的分数  x y为坐标
    :param thresh:0.3 0.5....
    :return: box after nms
    """

    x1 = dets[:, 0] #意思是取一个二维数组中所有行的第0列  是numpy数组中的一种写法
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    sorces = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1) #求每个box的面积
    #首先进行分数排序
    order = sorces.argsort()[::-1] #对分数进行倒序排序  order存的就是排序后的下标
    # argsort()函数用法：对待操作数组元素进行从小到大排序(-1从大到小)，并将排序后对应原数组元素的下标输出到生成数组中
    keep = []#用来保存最后留下的box
    
    while order.size > 0:
        i = order[0] #无条件保留每次置信度最高的box  i代表的是下标，是sorces中分数最高的下标
        keep.append(i)  #第i + 1个box
        #置信度最高的box和其他剩下bbox的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])  # np.maximum 两个数字逐位比，取其较大值。返回一个数组
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算置信度最高的bbox和其他剩下的bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersect = w * h  #inter是数组

        #求交叉区域的面积占两者（置信度最高的bbox和其他bbox）面积和的比例  iou值
        iou = intersect / (areas[i] + areas[order[1:]] - intersect)  #iou也是按照倒序排序排列的 iou由大到小
        #保留小于thresh的框，进入下一次迭代
        inds = np.where(iou <= thresh)[0]  #idx保存的是满足iou<=thresh的第一个iou值的下标
        #因为order[0]是我们的areas[i] 所以得到的inds还要+1才是下一个order[0]
        #上头的iou是从1：开始算的 所以得到的inds的个数并没有去除当前的order[0]
        order = order[inds + 1]

     return keep

