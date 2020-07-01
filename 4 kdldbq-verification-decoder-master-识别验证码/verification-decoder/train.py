import tensorflow as tf
import numpy as np
import code_utils
from model import Model
from image_utils import ImageUtils
import os

# tf.set_random_seed(1)
image = tf.placeholder(tf.float32, [None, 26, 70, 1])  # 定义图片的大小
# 定义每个预测值得维度
labels = dict(
    digit1=tf.placeholder(tf.float32, [None, 36]),
    digit2=tf.placeholder(tf.float32, [None, 36]),
    digit3=tf.placeholder(tf.float32, [None, 36]),
    digit4=tf.placeholder(tf.float32, [None, 36])
)
training_options = dict(
    drop_rate=0.9,
    learning_rate=1e-3,  # 学习率
    decay_steps=10000,  # 多少步降低学习率
    decay_rate=1,  # 每次降低 1 - decay_rate
    batch_size=32,  # 每次训练多少张图片
    show_loss=20,  # 貌似没用到
    total_episode=9999999999,  # 总训练回合
    show_test=1000,  # 多少步展示一下测试数据的预测率以及预测值、真实值
    output_board=True,  # 是否输出到tensorboard
    logs_step=10000,  # 多少步往tensorboard里写入 同时存放model
    save_step=10000,
    log_path="logs/",  # tensorboard的log文件存放在哪里
    model_path='net_model/',  # model保存在那个文件夹
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
    # 如果写入tensorboard
    if training_options['output_board']:
        merged = tf.summary.merge_all()  # tensorflow >= 0.12
        writer = tf.summary.FileWriter(training_options['log_path'], sess.graph)  # tensorflow >=0.12
    # 获得saver对象，可以保存model以及读取model
    saver = tf.train.Saver()
    # 如果已经存在model副本就直接读取，否则就初始化神经网络参数
    if os.path.exists(training_options['model_path'] + '/checkpoint'):
        saver.restore(sess, training_options['model_path'] + training_options['model_name'])
    else:
        sess.run(tf.global_variables_initializer())
    # 初始化 imageUtils类，获得所有训练，测试数据
    imageUtils = ImageUtils()
    train_data = imageUtils.train_data
    train_label = imageUtils.train_label
    test_data = imageUtils.test_data
    test_label = imageUtils.test_label
    # 开始训练啦
    for episode in range(training_options['total_episode']):
        # 随机获得定义好数量的训练数据
        sample_datas, sample_labels = imageUtils.sample(len(train_data), training_options['batch_size'],
                                                        train_data, train_label)#sample——labels形状是【batch_size, 4,35】，
        # 训练神经网络
        global_step, _, loss = sess.run(
            [train['global_step'], train['train'], train['loss']],
            feed_dict={image: sample_datas, labels['digit1']: sample_labels[:, 0],#所有batch_szie的第0行，即所有验证码的第一个词形状为【batch_szie,35】
                       labels['digit2']: sample_labels[:, 1],
                       labels['digit3']: sample_labels[:, 2],
                       labels['digit4']: sample_labels[:, 3]})
        # 打印 [总共训练了多少回合， loss值]
        print('total episode: {0:}\t\tloss: {1:.4f}'.format(global_step, loss))
        # 指定的回合数时保存tensorboard log文件，保存model（神经网络参数）
        if episode % training_options['logs_step'] == 0 and training_options['output_board']:
            result = sess.run(merged, feed_dict={image: sample_datas, labels['digit1']: sample_labels[:, 0],
                                                 labels['digit2']: sample_labels[:, 1],
                                                 labels['digit3']: sample_labels[:, 2],
                                                 labels['digit4']: sample_labels[:, 3]})
            writer.add_summary(result, global_step)
        if episode != 0 and episode % training_options['save_step'] == 0:
            saver.save(sess, training_options['model_path'] + training_options['model_name'])
        # 指定回合数时 打印预测值、真实值、正确率
        if episode % training_options['show_test'] == 0:
            t_sample_datas, t_sample_labels = imageUtils.sample(len(test_data), 10, test_data, test_label)
            result = sess.run([net['digit1'], net['digit2'], net['digit3'], net['digit4']],
                              feed_dict={image: t_sample_datas})
            result = code_utils.batch_out_transition(result)
            predicted = [result[0][index] + result[1][index] + result[2][index] + result[3][index]
                         for index in range(len(result[0]))]
            label = code_utils.batch_out_transition(t_sample_labels)
            four_right_count = np.count_nonzero([predicted[index] == label[index] for index in range(len(predicted))])
            one_right_count = np.count_nonzero(
                [predicted[index][s_index] == label[index][s_index] for index in range(len(predicted)) for s_index in
                 range(len(predicted[index]))])
            print('predicted:\t{} \nlabel:\t\t{}'.format(predicted,
                                                         label))
            print('4 match: {0:.2f}%\t\t1 match: {1:.2f}%'.format(four_right_count / 10 * 100,
                                                                  one_right_count / 40 * 100))