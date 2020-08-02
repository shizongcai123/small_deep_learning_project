import parameter_setting as set
import VGGmodle
import tensorflow as tf
import numpy as np
import cv2
def content_loss(sess, model):
    sess.run(tf.assign(model.net['input'], model.content))
    content_layers = set.CONTENT_LOSS_LAYERS
    loss = 0.0
    for layer, w in content_layers:
        p = sess.run(model.net[layer])
        x = model.net[layer]
        M = p.shape[1]* p.shape[2]
        N = p.shape[3]
        loss += (1.0 / (2 * M * N)) * tf.reduce_sum(tf.pow(p - x, 2)) * 2
    loss /= len(content_layers)
    return loss

def style_loss(sess, model):
    sess.run(tf.assign(model.net['input'], model.style))
    style_layers = set.STYLE_LOSS_LAYERS
    loss = 0.0
    for layer, w in style_layers:
        a = sess.run(model.net[layer])
        x = model.net[layer]
        M = a.shape[1]* a.shape[2]
        N = a.shape[3]
        A = gram(a,M,N)
        G = gram(x,M,N)
        loss += (1.0 / (4 * M * M * N * N)) * tf.reduce_sum(tf.pow(G - A, 2)) * w
    loss /= len(style_layers)
    return loss

def gram(x,size,deep):
    x = tf.reshape(x, (size,deep))
    g = tf.matmul(tf.transpose(x), x)
    return g

def total_loss(sess,model):
    content = content_loss(sess, model)
    style = style_loss(sess, model)
    loss = set.ALPHA * content+ set.BETA * style
    return loss

def train():
    model = VGGmodle.Model(set.CONTENT_IMAGE,set.STYLE_IMAGE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss = total_loss(sess, model)
        optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(model.net['input'], model.random_image))
        for step in range(set.TRAIN_STEPS):
            sess.run(optimizer)
            if step % 500 == 0:
                print('step {} is down.'.format(step))
                # 取出input的内容，这是生成的图片
                img = sess.run(model.net['input'])
                # 训练过程是减去均值的，这里要加上
                img += set.IMAGE_MEAN_VALUE
                # 这里是一个batch_size=1的batch，所以img[0]才是图片内容
                img = img[0]
                # 将像素值限定在0-255，并转为整型
                img = np.clip(img, 0, 255).astype(np.uint8)
                # 保存图片
                cv2.imwrite('{}-{}.jpg'.format(set.OUTPUT_IMAGE,step),img)
        img = sess.run(model.net['input'])
        img += set.IMAGE_MEAN_VALUE
        img = img[0]
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite('{}.jpg'.format(set.OUTPUT_IMAGE), img)
if __name__ == '__main__':
    train()