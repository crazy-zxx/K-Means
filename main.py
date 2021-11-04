import math

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


# K-Means算法实现类
class KMeans(object):

    def __init__(self, data, k):
        """
        类的构造器
        :param data: 待划分数据集
        :param k: 划分的类数
        """
        self.data = np.array(data)
        self.k = k
        self.iteration = 0
        # k个质心
        self.centroids = None
        # 每个数据所属的质心id
        self.data_centroid = np.zeros(self.data.shape[0], dtype=np.int32)

    def train(self, iteration):
        """
        训练划分数据
        :param iteration: 指定迭代次数
        """
        self.iteration = iteration
        # 随机初始化质心
        self.get_random_centroid()
        # 迭代
        while self.iteration > 0:
            # 遍历所有点分配其到最近的质心
            self.traversal_divide()
            # 更新质心
            self.update_centroid()
            self.iteration -= 1

    def get_random_centroid(self):
        """
        获取初始化的随机质心
        """
        # 数据数量
        num_samples = self.data.shape[0]
        # 从数据中任取k个做初始质心
        # 获取打乱的id序列
        centroids_random_ids = np.random.permutation(num_samples)
        # 取打乱后的前k个id对应的数据作为质心
        self.centroids = self.data[centroids_random_ids[:self.k]]

    def traversal_divide(self):
        """
        遍历数据集并划分其到最近的质心簇中
        """
        # 数据数量
        num_samples = self.data.shape[0]
        # 遍历所有数据
        for i in range(num_samples):
            # 记录每个数据到k个质心的距离
            dist = np.zeros(self.k)
            for j in range(self.k):
                dist[j] = KMeans.get_euclidean(self.data[i], self.centroids[j])
            # 确定每个数据所属的质心id，距离最小的
            self.data_centroid[i] = np.argmin(dist)

    def update_centroid(self):
        """
        更新质心
        """
        # 数据数量
        num_samples = self.data.shape[0]
        # 统计k个类中的数据量
        count = np.zeros(self.k)
        # 质心清零，用来重新累计类内元素和
        self.centroids[:] = 0.0
        # 累计每个类内数据
        for i in range(num_samples):
            self.centroids[self.data_centroid[i]] += self.data[i]
            count[self.data_centroid[i]] += 1
        for i in range(self.k):
            self.centroids[i] /= count[i]

    @staticmethod
    def get_euclidean(point1, point2):
        """
        获取两向量的欧拉距离
        :param point1: 向量1
        :param point2: 向量2
        :return: 两向量的欧拉距离
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))


if __name__ == '__main__':
    # 从excel加载测试数据
    data = pd.read_excel('test.xls')

    # # 测试数据
    # dataset = """
    # 1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    # 6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    # 11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    # 16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    # 21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    # 26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
    # # 数据处理 dataset是30个样本（密度，含糖量）的列表
    # a = dataset.split(',')
    # data = [[float(a[i]), float(a[i + 1])] for i in range(1, len(a) - 1, 3)]

    # 分类数目
    k = 4
    # 迭代次数
    iteration = 100
    # 创建对象
    km = KMeans(data, k)
    # 训练
    km.train(iteration)
    # 输出质心
    print(km.centroids)

    # 绘图
    plt.figure(1)
    # 子图1
    plt.subplot(211)
    # 散点图
    plt.plot(km.data[:, 0], km.data[:, 1], '.')
    # 图像标题
    plt.title('original')
    # 子图2
    plt.subplot(212)
    # 颜色
    color_val = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    # 按类别划分x、y坐标值到k个list
    x = [list() for i in range(k)]
    y = [list() for i in range(k)]
    for i in range(km.data.shape[0]):
        x[km.data_centroid[i]].append(km.data[i, 0])
        y[km.data_centroid[i]].append(km.data[i, 1])
    # 绘制散点图，不同类别用不同颜色
    for i in range(k):
        plt.plot(x[i], y[i], color_val[i % len(color_val)] + '.')
        plt.plot(km.centroids[i, 0], km.centroids[i, 1], color_val[i % len(color_val)] + '*')
    # 图像标题
    plt.title('result')
    # 调整子图间距
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # 显示图像
    plt.show()
