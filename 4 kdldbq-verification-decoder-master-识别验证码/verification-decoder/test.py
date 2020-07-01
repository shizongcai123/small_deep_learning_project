import tensorflow as tf
import numpy as np
import code_utils
from model import Model
from image_utils import ImageUtils
import os

image = tf.placeholder(tf.float32, [None, 26, 70, 1])  # 定义图片的大小
# 定义每个预测值得维度
labels = dict(
    digit1=tf.placeholder(tf.float32, [None, 36]),
    digit2=tf.placeholder(tf.float32, [None, 36]),
    digit3=tf.placeholder(tf.float32, [None, 36]),
    digit4=tf.placeholder(tf.float32, [None, 36])
)
training_options = dict(
    drop_rate=0.2,
    learning_rate=1e-6,  # 学习率 
    decay_steps=10000,  # 多少步降低学习率 在7w步的时候，从1000调成了1w
    decay_rate=1,
    # 每次降低 1 - decay_rate， 其实是不需要做这个操作的，因为我用的是RMSPropOptimizer，在了解这个优化器之前，我并不知道，所以还是让lr逐步下降了。。。。。。如果你觉得训练速度过于缓慢，设置为1就好
    batch_size=32,  # 每次训练多少张图片
    show_loss=20,  # 貌似没用到
    total_episode=2000001,  # 总训练回合
    show_test=10,  # 多少步展示一下测试数据的预测率以及预测值、真实值
    output_board=True,  # 是否输出到tensorboard
    logs_step=100,  # 多少步往tensorboard里写入 同时存放model
    log_path="logs/",  # tensorboard的log文件存放在哪里
    model_path='model/',  # model保存在那个文件夹
    model_name='model.ckpt'  # model文件名
)
model = Model()  # 初始化model类

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    # 保证存放model的文件夹存在
    if not os.path.exists(training_options['model_path']):
        os.mkdir(training_options['model_path'])
    # 定义神经网络
    net, train = model.build_network(training_options=training_options, image=image,
                                     drop_rate=training_options['drop_rate'], labels=labels)
    # 获得saver对象，可以保存model以及读取model
    saver = tf.train.Saver()
    saver.restore(sess, training_options['model_path'] + training_options['model_name'])
    # 初始化 imageUtils类，获得所有训练，测试数据

    imageUtils = ImageUtils()
    test_data = imageUtils.test_data
    test_label = imageUtils.test_label
    total_result = np.zeros((200, 2))
    for i in range(200):
        result = sess.run([net['digit1'], net['digit2'], net['digit3'], net['digit4']],
                          feed_dict={image: imageUtils.trainstion_data(test_data, start=i * 1000,end=i * 1000 + 1000)})
        result = code_utils.batch_out_transition(result)
        predicted = [result[0][index] + result[1][index] + result[2][index] + result[3][index]
                     for index in range(len(result[0]))]
        label = code_utils.batch_out_transition(test_label[i * 1000:i * 1000 + 1000])
        four_right_count = np.count_nonzero([predicted[index] == label[index] for index in range(len(predicted))])
        one_right_count = np.count_nonzero(
            [predicted[index][s_index] == label[index][s_index] for index in range(len(predicted)) for s_index in
             range(len(predicted[index]))])
        total_result[i, 0] = four_right_count / 1000 * 100
        total_result[i, 1] = one_right_count / 4000 * 100
        print('一千个测试数据： 四个字符同时正确率: {0:.2f}%\t\t单个字符正确率: {1:.2f}%'.format(total_result[i, 0],
                                                              total_result[i, 1]))
    print('总结果： 四个字符同时正确率: {0:.2f}%\t\t单个字符正确率: {1:.2f}%'.format(np.mean(total_result[:, 0]),
                                                                              np.mean(total_result[:, 1])))
