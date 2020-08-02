import imageio
import numpy as np
import tensorflow as tf
import parameter_setting as set
import scipy.io
import scipy.misc
from PIL import Image
import cv2
def loadimage(path):
    image = cv2.imread(path)
    print(image.shape)
    image = cv2.resize(image,(set.IMAGE_WIDTH, set.IMAGE_HEIGHT))
    print(image.shape)
    image = np.array(image)
    image = np.reshape(image, (1,set.IMAGE_HEIGHT, set.IMAGE_WIDTH,3))
    image = image -set.IMAGE_MEAN_VALUE
    print(image.shape)
    return image

if __name__ == '__main__':
    img = loadimage('images/style.jpg')
    img += set.IMAGE_MEAN_VALUE
    img = img[0]
    print(img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite('output/output2.jpg', img)