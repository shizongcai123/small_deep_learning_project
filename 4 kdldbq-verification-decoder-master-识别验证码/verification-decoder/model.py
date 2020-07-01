import tensorflow as tf


class Model():
    def build_network(self, training_options, image, drop_rate, labels):
        with tf.variable_scope('CNN'):
            # 卷积卷积卷积卷积
            with tf.variable_scope('hidden1'):
                hidden1 = self.cnn(image, 32, kernel_size=[3, 3])
            with tf.variable_scope('hidden2'):
                hidden2 = self.cnn(hidden1, 64, kernel_size=[3, 3])
            with tf.variable_scope('hidden3'):
                hidden3 = self.cnn(hidden2, 64, kernel_size=[3, 3])
            # with tf.variable_scope('hidden4'):
            #     hidden4 = self.cnn(hidden3, 100, kernel_size=[1, 1])
            # 更改一下形状，因为全连接神经网络需要2维数据的input
            flatten = tf.reshape(hidden3, [-1, 4 * 9 * 64])  # --> 5 * 2 * 160 = 1600
            # 第一层全连接
            with tf.variable_scope('hidden5'):
                dense = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
                hidden5 = tf.layers.dropout(dense, rate=drop_rate)
            # 第二层全连接
            with tf.variable_scope('hidden6'):
                dense = tf.layers.dense(hidden5, 1024, activation=tf.nn.relu)
                hidden6 = tf.layers.dropout(dense, rate=drop_rate)
            # 第一个字符预测输出
            with tf.variable_scope('digit1'):
                dense = tf.layers.dense(hidden6, units=36)#36个类别
                self.digit1 = dense
                tf.summary.histogram('digit1', self.digit1)
            # 第二个
            with tf.variable_scope('digit2'):
                dense = tf.layers.dense(hidden6, units=36)
                self.digit2 = dense
                tf.summary.histogram('digit2', self.digit2)
            # 第三个
            with tf.variable_scope('digit3'):
                dense = tf.layers.dense(hidden6, units=36)
                self.digit3 = dense
                tf.summary.histogram('digit3', self.digit3)
            # 第四个
            with tf.variable_scope('digit4'):
                dense = tf.layers.dense(hidden6, units=36)
                self.digit4 = dense
                tf.summary.histogram('digit4', self.digit4)

            layer = dict(
                digit1=self.digit1,#形状为【batch_size, 36】
                digit2=self.digit2,
                digit3=self.digit3,
                digit4=self.digit4
            )
            train = self.train(training_options, labels)
        return layer, train

    def train(self, training_options, labels):
        # 计算loss
        with tf.variable_scope('loss'):
            digit1_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels['digit1'], logits=self.digit1)
            digit2_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels['digit2'], logits=self.digit2)
            digit3_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels['digit3'], logits=self.digit3)
            digit4_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels['digit4'], logits=self.digit4)
            loss = digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy
        # 梯度下降
        with tf.variable_scope('train'):
            # 定义总训练步数，和学习率的 更新方式
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(training_options['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=global_step)
        tf.summary.scalar('loss', loss)
        return dict(loss=loss, train=train_op, global_step=global_step)

    @staticmethod
    def cnn(input, filters, kernel_size):
        conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, padding='SAME')#(？, 26, 70, 32)
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[1, 1], strides=2, padding='SAME')
        dropout = tf.layers.dropout(pool, rate=0.9)
        return dropout

