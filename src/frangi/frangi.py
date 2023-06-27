
from frangi.utils import *

def frangi(nd_array, black_vessels=True):

    with tf.variable_scope('frangi_filter'):
        eigen0 = nd_array[0]
        eigen1 = nd_array[1]
        eigen2 = nd_array[2]
        shape = get_shape(nd_array[0])
        height, width, depth, channels = shape
        num = 0
        c = tf.get_variable('norm_%s' % num, shape=[1, 1, 1, channels],
                            initializer=tf.constant_initializer(0.1), trainable=True)
        C_param = list()
        for i in range(channels):
            C_param.append(tf.summary.scalar('C_%s'%i, tf.reduce_mean(c[:,:,:,i])))
        alpha = tf.get_variable('alpha%s' % num, shape=[1, 1, 1, channels],
                                initializer=tf.constant_initializer(0.5), trainable=True)
        alpha_param = list()
        for i in range(channels):
            alpha_param.append(tf.summary.scalar('alpha_%s' % i, tf.reduce_mean(alpha[:, :, :, i])))
        beta = tf.get_variable('beta%s' % num, shape=[1, 1, 1, channels],
                               initializer=tf.constant_initializer(0.5), trainable=True)
        beta_param = list()
        for i in range(channels):
            beta_param.append(tf.summary.scalar('beta_%s' % i, tf.reduce_mean(beta[:, :, :, i])))
        result = compute_vesselness(eigen0, eigen1, eigen2, alpha=alpha, beta=beta, c=c,
                                               black_white=black_vessels)
        result = tf.squeeze(result)
        result = tf.expand_dims(result, axis=0)
        return result

def compute_vesselness(eigen1, eigen2, eigen3, alpha, beta, c, black_white):
    Ra = divide_nonzero(tf.abs(eigen2), tf.abs(eigen3))
    Rb = divide_nonzero(tf.abs(eigen1), tf.sqrt(tf.abs(tf.multiply(eigen2, eigen3))))
    S = tf.sqrt(tf.square(eigen1) + tf.square(eigen2) + tf.square(eigen3))
    plate = 1.0 - tf.math.exp(-Ra**2 / (2 * alpha**2))
    blob = tf.math.exp(-Rb**2 / (2 * beta**2))
    background = 1 - tf.math.exp(-S**2 / (2 * c**2))
    filtered = plate * blob * background
    filtered = tf.squeeze(filtered)

    return filtered


