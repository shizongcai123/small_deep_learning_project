import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
import parameter_setting as set
import cv2
class Model(object):
    def __init__(self, content_path, style_path):
        self.content = self.loadimage(content_path)
        self.style = self.loadimage(style_path)
        self.random_image = self.get_random_image()
        self.net = self.vggnet()

    def vggnet(self):
        vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
        vgg_layers = vgg['layers'][0]
        net={}
        net['input'] = tf.Variable(np.zeros([1,set.IMAGE_HEIGHT,set.IMAGE_WIDTH,3]),dtype=tf.float32)
        # 参数对应的层数可以参考vgg模型图
        net['conv1_1'] = self.conv_relu(net['input'], self.get_wb(vgg_layers, 0))
        net['conv1_2'] = self.conv_relu(net['conv1_1'], self.get_wb(vgg_layers, 2))
        net['pool1'] = self.pool(net['conv1_2'])
        net['conv2_1'] = self.conv_relu(net['pool1'], self.get_wb(vgg_layers, 5))
        net['conv2_2'] = self.conv_relu(net['conv2_1'], self.get_wb(vgg_layers, 7))
        net['pool2'] = self.pool(net['conv2_2'])
        net['conv3_1'] = self.conv_relu(net['pool2'], self.get_wb(vgg_layers, 10))
        net['conv3_2'] = self.conv_relu(net['conv3_1'], self.get_wb(vgg_layers, 12))
        net['conv3_3'] = self.conv_relu(net['conv3_2'], self.get_wb(vgg_layers, 14))
        net['conv3_4'] = self.conv_relu(net['conv3_3'], self.get_wb(vgg_layers, 16))
        net['pool3'] = self.pool(net['conv3_4'])
        net['conv4_1'] = self.conv_relu(net['pool3'], self.get_wb(vgg_layers, 19))
        net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
        net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
        net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
        net['pool4'] = self.pool(net['conv4_4'])
        net['conv5_1'] = self.conv_relu(net['pool4'], self.get_wb(vgg_layers, 28))
        net['conv5_2'] = self.conv_relu(net['conv5_1'], self.get_wb(vgg_layers, 30))
        net['conv5_3'] = self.conv_relu(net['conv5_2'], self.get_wb(vgg_layers, 32))
        net['conv5_4'] = self.conv_relu(net['conv5_3'], self.get_wb(vgg_layers, 34))
        net['pool5'] = self.pool(net['conv5_4'])
        return net
    def conv_relu(self, input, w):
        conv = tf.nn.conv2d(input, w[0], strides=[1,1,1,1],padding='SAME')
        re = tf.nn.relu(conv+w[1])
        return re
    def get_wb(self,layers,i):
        w = tf.constant(layers[i][0][0][0][0][0])
        bias = layers[i][0][0][0][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return w, b
    def pool(self,input):
        pool = tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        return pool
    def loadimage(self,path):
        image = cv2.imread(path)#(2304, 3359, 3)=[高，宽，通道]
        image = cv2.resize(image,(set.IMAGE_WIDTH,set.IMAGE_HEIGHT ))#这里cv2.resize的参数是(宽，高)
        image = np.reshape(image, (1,set.IMAGE_HEIGHT, set.IMAGE_WIDTH,3))#这里的第二维是高，第三维是宽
        image = image -set.IMAGE_MEAN_VALUE
        return image
    def get_random_image(self):
        noise_image = np.random.uniform(-20,20,size=[1,set.IMAGE_HEIGHT, set.IMAGE_WIDTH,3])
        random_img = noise_image * set.NOISE + self.content * (1 - set.NOISE)
        return random_img
