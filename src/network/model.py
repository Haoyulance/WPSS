
import tensorflow as tf
from network.net import u_net

def weighted_CE(beta, output, target):
    output = tf.clip_by_value(output, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    output = tf.log(output / (1 - output))
    loss = target[:,:,:,:,0] * (-tf.log(tf.sigmoid(output[:,:,:,:,0]))) * (1 - beta) + (1 - target[:,:,:,:,0]) * (-tf.log(1 - tf.sigmoid(output[:,:,:,:,0]))) * beta + \
           target[:,:,:,:,1] * (-tf.log(tf.sigmoid(output[:,:,:,:,1]))) * beta + (1 - target[:,:,:,:,1]) * (-tf.log(1 - tf.sigmoid(output[:,:,:,:,1]))) * (1 - beta)
    return tf.reduce_sum(loss, axis=-1)


class model(object):
    def __init__(self, width, height, depth, ichan, ochan, scale, lr=0.01, beta1=0.5):
        self._is_training = tf.placeholder(tf.bool)
        self._inputs = tf.placeholder(tf.float32, [1, width, height, depth, ichan])
        self._target = tf.placeholder(tf.float32, [1, width, height, depth, ochan])
        self._F_filter = tf.placeholder(tf.float32, [3, width, height, depth, scale])
        self._mask = tf.placeholder(tf.float32, [1, width, height, depth, scale])
        self.loss_val = tf.placeholder(tf.float32, None)
        self.loss_train = tf.placeholder(tf.float32, None)
        self.fp_val = tf.placeholder(tf.float32, None)
        self.dice_val = tf.placeholder(tf.float32, None)
        self.lr = lr
        self.net = u_net(width, height, depth, self._inputs, self._is_training, ochan, self._F_filter, self._mask)
        self.loss = weighted_CE(0.1, self.net._output, self._target)
        self.loss_without_crf = weighted_CE(0.1, self.net._decoder['final']['fmap'], self._target)
        self.loss2 = weighted_CE(0.1, self.net.activated, self._target)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        loss = tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        loss2 = tf.summary.scalar('loss_wo_crf', tf.reduce_mean(self.loss_without_crf))

        with tf.control_dependencies(update_ops):
            op = tf.train.AdamOptimizer(self.lr, beta1=beta1)
            op2 = tf.train.AdamOptimizer(self.lr, beta1=beta1)
            op3 = tf.train.AdamOptimizer(self.lr, beta1=beta1)
            his = list()
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/frangi_filter') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/encoder_1') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/decoder_1')
            grads = op.compute_gradients(self.loss2, var_list=variables)
            grads = [(tf.clip_by_value(grad, -10e4, 10e4), var) for grad, var in grads]
            for grad, var in grads:
                if grad is not None:
                    his.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            self._trian_step = op.apply_gradients(grads)

            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/encoder_0') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/decoder_0')
            grads = op2.compute_gradients(self.loss_without_crf, var_list=variables)
            grads = [(tf.clip_by_value(grad, -10e4, 10e4), var) for grad, var in grads]
            for grad, var in grads:
                if grad is not None:
                    his.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            self._trian_step2 = op2.apply_gradients(grads)

            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/crfrnn') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/spatial_attn') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='unet/position_attn')
            grads = op3.compute_gradients(self.loss, var_list=variables)
            grads = [(tf.clip_by_value(grad, -10e4, 10e4), var) for grad, var in grads]
            for grad, var in grads:
                if grad is not None:
                    his.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            self._trian_step3 = op3.apply_gradients(grads)

        self.sum_merge = tf.summary.merge_all()
        self.val_loss = tf.summary.scalar('validation_loss', self.loss_val)
        self.train_loss = tf.summary.scalar('train_loss', self.loss_train)
        self.FP = tf.summary.scalar('False positive on validation', self.fp_val)
        self.Dice = tf.summary.scalar('Dice coefficient on validation', self.dice_val)

    def train_step(self, sess, inputs, targets, F_filter, mask, is_training=True):
        _, _, _, loss, loss2 = sess.run([self._trian_step, self._trian_step2, self._trian_step3, self.loss, self.loss_without_crf],
                           feed_dict={self._inputs: inputs, self._target: targets,
                                      self._is_training:is_training, self._F_filter: F_filter, self._mask:mask})
        summary_temp = sess.run(self.sum_merge, feed_dict={self._inputs: inputs, self._target: targets,
                                      self._is_training:is_training, self._F_filter: F_filter, self._mask:mask})
        return loss, loss2, summary_temp

    def val_step(self, sess, inputs, targets, F_filter, mask, is_training=False):
        output, loss, loss2 = sess.run([self.net._output, self.loss, self.loss_without_crf],
                                feed_dict={self._inputs: inputs, self._target: targets,
                                           self._is_training: is_training, self._F_filter: F_filter, self._mask: mask})
        return output, output, output, loss, loss2

    def test_step(self):

        return


