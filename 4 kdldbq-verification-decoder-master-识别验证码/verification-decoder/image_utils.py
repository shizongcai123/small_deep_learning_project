import numpy as np
import glob
import code_utils as cu
from PIL import Image


class ImageUtils():
    def __init__(self):
        self.test_data = glob.glob('test-images/*.jpeg')#这里图片的形式是：前边为图片像素/text_1w.jpg
        self.test_label = np.array(
            [cu.in_transition(self.test_data[index].split('/')[-1].split('.')[0].split('_')[0]) for index in
             range(len(self.test_data))])#得到所有图片代表的验证码的one-hot编码矩阵

        self.train_data = glob.glob('train-images/*.jpeg')
        self.train_label = np.array(
            [cu.in_transition(self.train_data[index].split('/')[-1].split('.')[0].split('_')[0]) for index in
             range(len(self.train_data))])#shape是【？，4，35】35指0-9，a-z的one-hot编码数

    @staticmethod
    def sample(capacity, batch_size, datas, labels):
        sample_index = np.random.choice(capacity, batch_size)#在[0,capacity)内生成batch_size个数，即形状是【1，batch_size】
        _datas = np.array([np.array(Image.open(datas[index]).convert("1"))[:, :, np.newaxis] for index in sample_index])
        #.convert("1")将图片转换为二值图
        #np.newaxis为新加一个维度，即是深度。
        #最后获得一个随机batch_size的图片矩阵_datas形状为【batch_size, 26,70,1】
        _labels = labels[sample_index, :]#形状是【batch_size, 4,35】
        return _datas, _labels#获得对应的batch_size个图片与对应的标签

    @staticmethod
    def trainstion_data(datas, sample_index=None, start=0, end=1000):
        if sample_index != None:
            return np.array(
                [np.array(Image.open(datas[index]).convert("1"))[:, :, np.newaxis] for index in sample_index])
        return np.array(
            [np.array(Image.open(datas[index]).convert("1"))[:, :, np.newaxis] for index in range(start, end)])
