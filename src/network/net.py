
import tensorflow as tf
from network.crfrnn_layer import CrfRnnLayer
from frangi.frangi import *

def get_shape(tensor):
    return tensor.get_shape().as_list()

def batch_norm(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)  # for training with batch size larger than 1
        # bn = tf.contrib.layers.instance_norm(*args, **kwargs)  # for training with batch size equals to 1
    return bn

def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)

class u_net(object):
    def __init__(self, width, height, depth, inputs, is_training, ochan, F_filter, mask, position_attn=None, stddev=0.02, center=True, scale=True, reuse=None):
        self._is_training = is_training
        self._stddev = stddev
        self._ochan = ochan
        with tf.variable_scope('unet', initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._center = center
            self._scale = scale
            self._prob = 0.5
            self._inputs = inputs
            self._mask = mask
            self._encoder = self._build_encoder(inputs, '_0')
            self._decoder = self._build_decoder(self._encoder, '_0', ochan=self._ochan)
            self.filter = frangi(F_filter, black_vessels=True)
            self.filter *= self._mask
            self.activated_filter_origin = self.filter
            self._encoder_filter = self._build_encoder(self.activated_filter_origin, '_1')
            self._decoder_filter = self._build_decoder(self._encoder_filter, '_1', ochan=self._ochan)
            self.activated = self._decoder_filter['final']['fmap']
            self.activated_filter = tf.concat([self.activated, self.activated_filter_origin], axis=-1)
            self.position_attn = position_attn
            self._output = self._crf(width, height, depth, self._decoder['final']['conv'], self.activated_filter,
                                     self.position_attn, iter=5)

    def _build_encoder_layer(self, name, size, inputs, k, bn=True, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name, size):
            layer['filter'] = tf.get_variable('filter', [size, size, size, get_shape(inputs)[-1], k])
            layer['conv'] = tf.nn.conv3d(inputs, layer['filter'], strides=[1, 1, 1, 1, 1], padding='SAME')
            layer['bn'] = batch_norm(layer['conv'], center=self._center, scale=self._scale, training=self._is_training) if bn else layer['conv']
            layer['dropout'] = tf.layers.dropout(layer['bn'], self._prob, training=self._is_training) if use_dropout else layer['bn']
            layer['fmap'] = lkrelu(layer['dropout'], slope=0.2)
            return layer

    def _build_encoder(self, inputs, name):
        encoder = dict()
        with tf.variable_scope('encoder'+name):
            encoder['l5'] = self._build_encoder_layer('l5', 3, inputs, 64, bn=False)
            encoder['l6'] = self._build_encoder_layer('l6', 3, encoder['l5']['fmap'], 128)
            encoder['l7'] = self._build_encoder_layer('l7', 2, encoder['l6']['fmap'], 256)
            encoder['l8'] = self._build_encoder_layer('l8', 2, encoder['l7']['fmap'], 256)
        return encoder

    def _build_decoder_layer(self, name, inputs, size, stride, output_shape_from, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name):
            output_shape = tf.shape(output_shape_from)
            layer['filter'] = tf.get_variable('filter', [size, size, size, get_shape(output_shape_from)[-1],
                                                         get_shape(inputs)[-1]])
            layer['conv'] = tf.nn.conv3d_transpose(inputs, layer['filter'], output_shape=output_shape,
                                                   strides=[1, stride, stride, stride, 1],
                                                   padding='SAME')
            layer['bn'] = batch_norm(tf.reshape(layer['conv'], output_shape), center=self._center, scale=self._scale, training=self._is_training)
            layer['dropout'] = tf.layers.dropout(layer['bn'], self._prob, training=self._is_training) if use_dropout else layer['bn']
            layer['fmap'] = tf.nn.relu(layer['dropout'])
        return layer

    def _build_decoder(self, encoder, name, ochan):
        decoder = dict()
        with tf.variable_scope('decoder'+name):
            decoder['dl1'] = self._build_decoder_layer('dl1', encoder['l8']['fmap'], 2, 1,
                                                       output_shape_from=encoder['l7']['fmap'],
                                                       use_dropout=True)

            fmap_concat = tf.concat([decoder['dl1']['fmap'], encoder['l7']['fmap']], axis=4)
            decoder['dl2'] = self._build_decoder_layer('dl2', fmap_concat, 2, 1,
                                                       output_shape_from=encoder['l6']['fmap'],
                                                       use_dropout=True)

            fmap_concat = tf.concat([decoder['dl2']['fmap'], encoder['l6']['fmap']], axis=4)
            decoder['dl3'] = self._build_decoder_layer('dl3', fmap_concat, 3, 1,
                                                       output_shape_from=encoder['l5']['fmap'],
                                                       use_dropout=True)

            fmap_concat = tf.concat([decoder['dl3']['fmap'], encoder['l5']['fmap']], axis=4)
            decoder['dl4'] = self._build_decoder_layer('dl4', fmap_concat, 3, 1,
                                                       output_shape_from=self._inputs)

            with tf.variable_scope('final'):
                cl9 = dict()
                cl9['filter'] = tf.get_variable('filter', [3, 3, 3, get_shape(decoder['dl4']['fmap'])[-1], ochan])
                cl9['conv'] = tf.nn.conv3d(decoder['dl4']['fmap'], cl9['filter'], strides=[1, 1, 1, 1, 1],
                                           padding='SAME')
                cl9['fmap'] = tf.nn.softmax(cl9['conv'])
                decoder['final'] = cl9
        return decoder

    def _crf(self, width, height, depth, unary, img, position_attn, iter=2):

        output = CrfRnnLayer(image_dims=(width, height, depth),
                             num_classes=2,
                             num_iterations=iter,
                             p_attn=position_attn,
                             name='crfrnn')([unary, img])
        output = tf.nn.softmax(output)
        return output