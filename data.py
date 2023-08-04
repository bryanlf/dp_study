import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')
# df = df.loc[:10316, :]
df = df.drop(columns=['time'])
df1 = pd.read_csv('data.csv')
df1 = df1.drop(columns=['time'])


class Data_util(object):
    def __init__(self, train_size, valid_size, window, horizon, output_dim, cuda):
        self.cuda = cuda
        self.data = df.to_numpy()  # 将DataFrame转换为Numpy数组
        self.data1 = df1.to_numpy()  # 将DataFrame转换为Numpy数组
        self.window = window
        self.horizon = horizon
        self.n, self.m = self.data.shape
        self.output_dim = output_dim  # 输出维度
        self._normalized()
        self._split(int(train_size * self.n), int((train_size + valid_size) * self.n))

    def _normalized(self):  # 对数据进行标准化，
        for i in range(self.m):
            mu = np.mean(self.data[:, i])
            sigma = np.std(self.data[:, i])
            self.data[:, i] = (self.data[:, i] - mu) / sigma

    def _split(self, train, valid):
        train_set = range(self.window + self.horizon - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):  # 按样本自然顺序，这里相当于将训练样本根据时间步长构建
        n = len(idx_set)  # n是总时序数据集数量，
        X = torch.zeros((n, self.window, self.m))  # 先做个容器，大小为[window，时序数据的维度]
        Y = torch.zeros((n, self.output_dim))  # y是求损失函数时用到的ground truth 大小为【总时序数据集数量，时序数据的维度】
        # 将数据按滑动窗口【window*时序数据的维度】的形式一组组的装起来
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])  # torch.from_numpy将nmupy数组转化为张量形式
            Y[i, :] = torch.from_numpy(self.data1[idx_set[i], :self.output_dim])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):  # 生成batch数据
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)  # randperm将0~length-1随机打乱后获得的数字序列
        else:
            index = torch.tensor(range(length))
            index = index.long()  # locng()长整型
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size


Data = Data_util(0.6, 0.2, 10, 1, 41, False)
print(Data.train[0].shape, Data.train[1].shape)
#print(Data.train[0], Data.train[1])
