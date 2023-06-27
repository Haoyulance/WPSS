
import tensorflow as tf
from tensorflow.contrib.framework import argsort

def divide_nonzero(array1, array2):
    denominator = tf.clip_by_value(array2, tf.keras.backend.epsilon(), 1e10 - tf.keras.backend.epsilon())
    return tf.math.divide(array1, denominator)

def sortbyabs(a, axis=-1):
    sorted = argsort(tf.abs(a), axis=axis, stable=True)
    print('after sorted%s'%get_shape(sorted))
    result = tf.gather_nd(a, sorted)
    print(get_shape(result))
    return result

def gaussian_filter(name, img, kernel_size=3):
    with tf.variable_scope('gauss_kernel_%s' % name):
        def gauss_kernel(kernel_size):
            sigma = tf.get_variable('sigma', [], initializer=tf.constant_initializer(0.1), trainable=True)
            ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            xx, yy, zz = tf.meshgrid(ax, ax, ax)
            kernel = tf.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2.0 * sigma ** 2))
            kernel = kernel / tf.reduce_sum(kernel)
            kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, 1, 1])
            return kernel,sigma
        gaussian_kernel, sigma = gauss_kernel(kernel_size)
        gaussian_kernel = gaussian_kernel[..., tf.newaxis]
        return tf.nn.conv3d(img, gaussian_kernel, strides=[1, 1, 1, 1, 1], padding='SAME'), sigma

def image_gradient(image, axis=0):
    height, width, depth = get_shape(image)
    if axis == 0:
        dx = image[1:, :, :] - image[:-1, :, :]
        last = tf.zeros([1, width, depth])
        dx = tf.concat([dx, last], axis=0)
    if axis == 1:
        dx = image[:, 1:, :] - image[:, :-1, :]
        last = tf.zeros([height, 1, depth])
        dx = tf.concat([dx, last], axis=1)
    if axis == 2:
        dx = image[:, :, 1:] - image[:, :, :-1]
        last = tf.zeros([height, width, 1])
        dx = tf.concat([dx, last], axis=2)
    return dx

def get_shape(tensor):
    return tensor.get_shape().as_list()