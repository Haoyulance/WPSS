
import tensorflow as tf
import numpy as np

def get_shape(tensor):
    return tensor.get_shape().as_list()

def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5 + eps)

def FP(qc, image, mask, use_mask=True):
    if use_mask:
        negative = np.sum(mask[qc==0])
    else:
        negative = len(qc[qc == 0])
    fp = np.sum(image[image != qc])
    return fp/negative

def dsc(pred, target, term=1):
    total_positive = np.sum(target)
    error = target - pred
    FN = np.sum(error[error > 0])
    FP = -np.sum(error[error < 0])
    recall = 1. - FN / total_positive
    precision = 1. - FP / (FP + total_positive - FN)
    dsc = (1. + term**2) * recall * precision / ((term**2 * precision) + recall)
    return dsc

def p_attn(height=32, width=32, depth=32):
    xx, yy, zz = np.meshgrid(np.arange(height), np.arange(width), np.arange(depth))
    location_num = height * width * depth
    position = np.stack([yy, xx, zz], axis=-1)
    position = position.astype(float)
    norm = np.linalg.norm(position, axis=-1)
    norm = np.expand_dims(norm, axis=-1)
    position = position / (np.sqrt(norm) + 1e-20)
    theta = np.reshape(position, [location_num, 3])
    attn = np.matmul(theta, theta.transpose())
    zeros = np.ones(attn.shape) - np.diag(np.ones(location_num))
    attn *= zeros
    attn = np.expand_dims(attn, axis=0)
    return attn

def cosine_decay(learning_rate, global_step, decay_steps, alpha=0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    return decayed_learning_rate

def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
    w_shape = get_shape(weights)
    w_mat = tf.reshape(weights, [-1, w_shape[-1]]) # convert to 2 dimension but total dimension still the same
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /=sigma

    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def snconv3d(input_, output_dim, size, stride, sn_iters=1, update_collection=None):
    w = tf.get_variable('filter', [size, size, size, get_shape(input_)[-1], output_dim])
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv = tf.nn.conv3d(input_, w_bar, strides=[1, stride, stride, stride, 1], padding='SAME')
    return conv

def snconv3d_tranpose(input_, output_dim_from, size, stride, sn_iters=1, update_collection=None):
    w = tf.get_variable('filter', [size, size, size, get_shape(output_dim_from)[-1], get_shape(input_)[-1]])
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv = tf.nn.conv3d_transpose(input_, w_bar, strides=[1, stride, stride, stride, 1], padding='SAME',
                                  output_shape=tf.shape(output_dim_from))
    return conv

def snconv3d_1x1(input_, output_dim, sn_iters=1, sn=True, update_collection=None, init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
    with tf.variable_scope(name):
        w = tf.get_variable('filter', [1, 1, 1, get_shape(input_)[-1], output_dim], initializer=init)
        if sn:
            w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        else:
            w_bar = w
        conv = tf.nn.conv3d(input_, w_bar, strides=[1, 1, 1, 1, 1], padding='SAME')
        return conv

def spatial_attn(name, spatal, x, uniary, sn=True, update_collection=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        batch_size, height, width, depth, num_channels = x.get_shape().as_list()
        output_channels = get_shape(uniary)[-1]

        if not batch_size:
            batch_size = 1
        location_num = height * width * depth

        downsampled = location_num
        theta = snconv3d_1x1(x, sn=sn, output_dim=num_channels, update_collection=update_collection, name='theta')
        theta = tf.reshape(theta, [batch_size, location_num, num_channels])

        phi = snconv3d_1x1(x, sn=sn, output_dim=num_channels, update_collection=update_collection, name='phi')
        phi = tf.reshape(phi, [batch_size, downsampled, num_channels])

        attn = tf.matmul(theta, phi, transpose_b=True)
        zeros = tf.ones(get_shape(attn)) - tf.matrix_diag(tf.ones(location_num))
        attn *= zeros
        attn = tf.nn.softmax(attn)

        g = snconv3d_1x1(uniary, sn=sn, output_dim=output_channels, update_collection=update_collection, name='g')
        g = tf.reshape(g, [batch_size, downsampled, output_channels])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, height, width, depth, output_channels])
        attn_g = snconv3d_1x1(attn_g, sn=sn, output_dim=output_channels, update_collection=update_collection, name='attn')
        return attn_g

def position_attn(name, x, rgb, uniary, sn=True, update_collection=None, times=0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        batch_size, height, width, depth, num_channels = x.get_shape().as_list()
        output_channels = get_shape(uniary)[-1]
        if not batch_size:
            batch_size = 1

        location_num = height * width * depth

        position = x
        theta = tf.reshape(position, [1, location_num, 3])
        theta = tf.tile(theta, [location_num, 1, 1])
        attn = tf.transpose(theta, perm=[1,0,2]) - theta
        attn = tf.math.square(tf.norm(attn, axis=-1))
        position_term = tf.get_variable('position_term', [], initializer=tf.constant_initializer(1), trainable=True)
        if times == 0:
            a = tf.summary.scalar('position_term', position_term)
        attn *= position_term
        attn = tf.math.exp(-attn)

        zeros = tf.ones(get_shape(attn)) - tf.matrix_diag(tf.ones(location_num))
        attn *= zeros
        attn = tf.nn.softmax(attn)

        attn = tf.expand_dims(attn, axis=0)
        uniary =  tf.reshape(uniary, [batch_size, location_num, output_channels])
        attn_p = tf.matmul(attn, uniary)
        attn_p = tf.reshape(attn_p, [batch_size, height, width, depth, output_channels])
        return attn_p

def channel_attn(name, x, uniary, sn=True, update_collection=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        batch_size, height, width, depth, num_channels = x.get_shape().as_list()
        output_channels = get_shape(uniary)[-1]

        if not batch_size:
            batch_size = 1
        location_num = height * width * depth

        key = snconv3d_1x1(x, sn=sn, output_dim=output_channels, update_collection=update_collection, name='key')
        key = tf.transpose(key, perm=[0, 4, 1, 2, 3], name='transpose1')
        key = tf.reshape(key, [batch_size, output_channels, location_num])

        header = snconv3d_1x1(x, sn=sn, output_dim=output_channels, update_collection=update_collection, name='header')
        header = tf.transpose(header, perm=[0, 4, 1, 2, 3], name='transpose2')
        header = tf.reshape(header, [batch_size, output_channels, location_num])

        mat = tf.matmul(key, header, transpose_b=True)
        mat = tf.nn.softmax(mat)

        g = snconv3d_1x1(uniary, sn=sn, output_dim=output_channels, update_collection=update_collection, name='g')
        g = tf.transpose(g, perm=[0, 4, 1, 2, 3], name='transpose3')
        g = tf.reshape(g, [batch_size, output_channels, location_num])

        attn_c = tf.matmul(mat, g)
        attn_c = tf.transpose(attn_c, perm=[0, 2, 1], name='transpose4')
        attn_c = tf.reshape(attn_c, [batch_size, height, width, depth, output_channels])
        attn_c = snconv3d_1x1(attn_c, sn=sn, output_dim=output_channels, update_collection=update_collection, name='attn_channel')

        return attn_c