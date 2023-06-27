
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from network.utils import *

def _diagonal_initializer(shape, *ignored, **ignored_too):
    return np.eye(shape[0], shape[1], dtype=np.float32)

def _potts_model_initializer(shape, *ignored, **ignored_too):
    return -1 * _diagonal_initializer(shape)


class CrfRnnLayer(Layer):

    def __init__(self, image_dims, num_classes, num_iterations, p_attn, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.p_attn = p_attn
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)
        sk = tf.summary.scalar('spatial_ker_weights', tf.reduce_mean(self.spatial_ker_weights))

        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)
        bk = tf.summary.scalar('bilateral_ker_weights', tf.reduce_mean(self.bilateral_ker_weights))

        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)
        cm = tf.summary.scalar('compatibility_matrix', tf.reduce_mean(self.compatibility_matrix))
        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries = inputs[0]
        rgb = inputs[1]
        c, h, w, d = self.num_classes, self.image_dims[0], self.image_dims[1], self.image_dims[2]
        q_values = unaries
        x, y, z = tf.meshgrid(tf.range(1, h+1), tf.range(1, w+1), tf.range(1, d+1))
        position = tf.stack([y, x, z], axis=-1)
        position = tf.cast(position, dtype=tf.float32)
        position = tf.expand_dims(position, axis=0)

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, -1)
            spatial_out = spatial_attn('spatial_attn', position, rgb, softmax_out)
            spatial_out = tf.squeeze(spatial_out)
            spatial_out = tf.transpose(spatial_out, perm=[3, 0, 1, 2])

            bilateral_out = position_attn('position_attn', position, rgb, softmax_out, times=i)
            bilateral_out = tf.squeeze(bilateral_out)
            bilateral_out = tf.transpose(bilateral_out, perm=[3, 0, 1, 2])

            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)
            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w, d))
            pairwise = tf.expand_dims(pairwise, axis=0)
            pairwise = tf.transpose(pairwise, perm=[0, 2, 3, 4, 1])
            q_values = unaries + pairwise

        return q_values

    def compute_output_shape(self, input_shape):
        return input_shape
