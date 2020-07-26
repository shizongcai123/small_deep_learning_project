import tensorflow as tf
import numpy as np
class Cifar10DataReader():
    import os
    import random
    import numpy as np
    import pickle
    def __init__(self, cifar_file, one_hot=False, file_number=1):
        self.batch_index = 0  # 第i批次
        self.file_number = file_number  # 第i个文件数
        self.cifar_file = cifar_file  # 数据集所在dir
        self.one_hot = one_hot
        self.train_data = self.read_train_file()  # 一个数据文件的训练集数据，得到的是一个10000大小的list，
        self.test_data = self.read_test_data()  # 得到10000个测试集数据

    # 读取数据函数，返回dict
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            try:
                dicts = self.pickle.load(fo, encoding='bytes')
            except Exception as e:
                print('load error', e)
            return dicts
    # 读取一个训练集文件,返回数据list
    def read_train_file(self, files=''):
        if files:
            files = self.os.path.join(self.cifar_file, files)
        else:
            files = self.os.path.join(self.cifar_file, 'data_batch_%d' % self.file_number)
        dict_train = self.unpickle(files)
        train_data = list(zip(dict_train[b'data'], dict_train[b'labels']))  # 将数据和对应标签打包
        #得到的是个字典，注意字典的索引是字节型的，比如要读取data,那么应该是dic[b'data'],字符串前面加b才是字节
        self.np.random.shuffle(train_data)#对第一维进行打乱。
        print('成功读取到训练集数据：data_batch_%d' % self.file_number)
        print('train的大小',len(train_data))
        return train_data  #train_data的长度是10000，即10000个图片

    # 读取测试集数据
    def read_test_data(self):
        files = self.os.path.join(self.cifar_file, 'test_batch')
        dict_test = self.unpickle(files)
        test_data = list(zip(dict_test[b'data'], dict_test[b'labels']))  # 将数据和对应标签打包
        print('成功读取测试集数据')
        print('test的大小：', len(test_data))
        return test_data

    # 编码得到的数据，变成张量，并分别得到数据和标签
    def encodedata(self, detum):
        rdatas = list()
        rlabels = list()
        for d, l in detum:
            rdatas.append(self.np.reshape(self.np.reshape(d, [3, 1024]).T, [32, 32, 3]))#【重点】也可以先将[3,1024]转成[3,32,32]再转成[32,32,3]
            if self.one_hot:
                hot = self.np.zeros(10)
                hot[int(l)] = 1
                rlabels.append(hot)
            else:
                rlabels.append(l)
        return rdatas, rlabels

    # 得到batch_size大小的数据和标签
    def nex_train_data(self, batch_size=100):
        assert 10000 % batch_size == 0, 'erro batch_size can not divied!'  # 判断批次大小是否能被整除

        # 获得一个batch_size的数据
        if self.batch_index < len(self.train_data) // batch_size:  # 是否超出一个文件的数据量
            detum = self.train_data[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            datas, labels = self.encodedata(detum)
            self.batch_index += 1
        else:  # 超出了就加载下一个文件
            self.batch_index = 0
            if self.file_number == 5:
                self.file_number = 1
            else:
                self.file_number += 1
            #self.read_train_file()
            self.train_data = self.read_train_file()
            return self.nex_train_data(batch_size=batch_size)
        return datas, labels#这里读出来的label形状是【100，】而不是【100,1】

    # 随机抽取batch_size大小的训练集
    def next_test_data(self, batch_size=100):
        detum = self.random.sample(self.test_data, batch_size)  # 随机抽取
        datas, labels = self.encodedata(detum)
        return datas, labels


def cnn(inputs, filters, kernel_size):
    layer = tf.layers.conv2d(inputs, filters=filters, kernel_size = kernel_size, padding='SAME')
    norm = tf.layers.batch_normalization(layer)
    re = tf.nn.relu(norm)
    pool = tf.layers.max_pooling2d(re, pool_size=[1,1],strides=2,padding='SAME')
    dropout = tf.layers.dropout(pool,rate= 0.9)
    return dropout




inputs = tf.placeholder(tf.float32, [None, 32,32,3])
label = tf.placeholder(tf.int32, [None, 10])
with tf.variable_scope('CNN'):
    with tf.variable_scope('cnn1'):
        dropout1 = cnn(inputs,32,kernel_size = [3,3])
    with tf.variable_scope('cnn2'):
        dropout2 = cnn(dropout1, 64, kernel_size = [3,3])
    with tf.variable_scope('cnn3'):
        dropout3 = cnn(dropout2, 64,kernel_size = [3,3])

    fallent = tf.reshape(dropout3,[-1,4 * 4 * 64])
    with tf.variable_scope('hidden4'):
        dense1 = tf.layers.dense(fallent, units=1024,activation=tf.nn.relu)
        hidden1 = tf.layers.dropout(dense1, rate=0.9)
    with tf.variable_scope('hidden5'):
        dense2 = tf.layers.dense(hidden1, units=1024, activation=tf.nn.relu)
        hidden2 = tf.layers.dropout(dense2, rate= 0.9)
    with tf.variable_scope('out'):
        dense = tf.layers.dense(hidden2,units=10)
        result = dense#[124,10]
with tf.variable_scope('loss'):
    #loss = tf.losses.sparse_softmax_cross_entropy(labels= label, logits=result)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=result)
with tf.variable_scope('train'):
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    optimmizer = tf.train.AdadeltaOptimizer(learning_rate = 0.1 )
    train_op = optimmizer.minimize(loss, global_step = global_step)
#tf.summary.scalar('loss', loss)
with tf.Session() as sess:
    Cifar = Cifar10DataReader(r'E:\python\ML_learning\tf_learning\Cifar_10_text\cifar-10-batches-bin',one_hot=True)
    sess.run(tf.global_variables_initializer())
    for episode in range(100):
        example_batch , label_batch= Cifar.nex_train_data()

        # example_batch = example_batch.eval()
        # label_batch = label_batch.eval()#这里出了问题
        outloss, outresult, _, outglobal_step = sess.run([loss, result, train_op, global_step],feed_dict={inputs: example_batch, label: label_batch})
        # tf.summary.scalar('loss', loss)
        # merged = tf.summary.merge_all()  # tensorflow >= 0.12
        # writer = tf.summary.FileWriter(r'E:\python\ML_learning\tf_learning\Cifar_10_text\cifar-10-batches-bin',sess.graph)  # tensorflow >=0.12
        # result222 = sess.run(merged, feed_dict={inputs: example_batch, label: label_batch})
        # writer.add_summary(result222, global_step)
        print('loss=',outloss)
# sess.close()