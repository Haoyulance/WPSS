
import tensorflow as tf
import os
import shutil
import nibabel as nib
import copy
from network.model import model
from frangi.hessian import *
from network.utils import dsc, FP

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_size', 112, 'training data size')
flags.DEFINE_integer('patch_size', 16, 'size of the pathch at three dimensions')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('val_portion', 0.25, 'percentage of the dataset used as validation')
flags.DEFINE_string('train_dr', None, 'training directory')
flags.DEFINE_integer('batch_size', 1, 'batch_size')
flags.DEFINE_integer('num_iter', 3000, 'number of iteration')
flags.DEFINE_string('logdir', None, 'path of tensorboard log')
flags.DEFINE_string('modality', 'epc_frangi', 'input and target modality names')
flags.DEFINE_string('dataset', None, 'path of dataset')
flags.DEFINE_string('output', None, 'output directory')
flags.DEFINE_integer('gpu', None, '# of gpu to use')
flags.DEFINE_boolean('mask', False, 'whether use the mask for evaluation')

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
config.gpu_options.allow_growth = True

def main(object):
    root = FLAGS.dataset
    file_root = FLAGS.output
    board = FLAGS.logdir

    if os.path.isdir(file_root):
        shutil.rmtree(file_root)
    if not os.path.isdir(file_root):
        os.makedirs(file_root)
    if os.path.isdir(board):
        shutil.rmtree(board)
    if not os.path.isdir(board):
        os.makedirs(board)

    subjects = os.listdir(root)
    [suffix1, suffix2] = FLAGS.modality.split('_')
    suffix1 += '.nii.gz'
    suffix2 += '.nii.gz'

    scale_range = np.arange(0.1, 1.5, 1)

    shape = FLAGS.patch_size
    val_step = 1 // FLAGS.val_portion
    with tf.device('/gpu:%s' %FLAGS.gpu):
        net = model(shape, shape, shape, ichan=1, ochan=2, scale=len(scale_range), lr=FLAGS.lr)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        previous = 100
        f_summary = tf.summary.FileWriter(logdir=board, graph=sess.graph)
        cou = 0
        dice_crf = list()

        for step in range(FLAGS.num_iter):
            pos = step % FLAGS.training_size
            sub = subjects[pos]
            target = nib.load(os.path.join(root, sub, suffix2))
            affine = target.affine
            target = target.get_fdata()
            input = nib.load(os.path.join(root, sub, suffix1))
            input = input.get_fdata()
            [w, h, d] = input.shape
            if FLAGS.mask:
                wmmask = nib.load(os.path.join(root, sub, "mask.nii.gz")).get_fdata()
            else:
                wmmask = 1

            frangied = list()
            mask = list()
            for sigma in scale_range:
                f_filtered = hessian_filter(input, sigma=sigma)
                mask.append(filter_out_background(True, f_filtered[1], f_filtered[2]))
                frangied.append(np.array([f_filtered[0], f_filtered[1], f_filtered[2]]))
            f_filtered = np.stack(frangied, axis=-1)
            mask = np.stack(mask, axis=-1)
            mask = np.expand_dims(mask, axis=0)

            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=-1)
            target = np.expand_dims(target, axis=0)
            target = np.expand_dims(target, axis=-1)
            target2 = np.ones(np.shape(target)) - target
            target = np.concatenate([target, target2], axis=-1)
            gt = copy.deepcopy(target)

            width_range = np.arange(0, w, shape // 2)
            height_range = np.arange(0, h, shape // 2)
            depth_range = np.arange(0, d, shape // 2)

            total_loss = list()
            total_loss2 = list()
            np.random.shuffle(width_range)
            np.random.shuffle(height_range)
            np.random.shuffle(depth_range)

            if pos % val_step == 0 and step != 0:
                for w in width_range:
                    for h in height_range:
                        for d in depth_range:
                            input_img = input[:, w:w + shape, h:h + shape, d:d + shape, :]
                            target_img = target[:, w:w + shape, h:h + shape, d:d + shape, :]
                            f_filtered_img = f_filtered[:, w:w + shape, h:h + shape, d:d + shape, :]
                            mask_img = mask[:, w:w + shape, h:h + shape, d:d + shape, :]

                            if np.count_nonzero(input_img) == 0:
                                continue

                            input_img = input_img / np.max(input_img)
                            output, loss, loss2 = net.val_step(sess, input_img, target_img, f_filtered_img, mask_img)
                            target[:, w:w + shape, h:h + shape, d:d + shape, :] = output
                            if np.sum(target_img[:, :, :, :, 0]) / (np.product(target_img.shape) / 2) < 0.01:
                                continue

                            loss = np.average(loss)
                            total_loss.append(loss)
                            loss2 = np.average(loss2)
                            total_loss2.append(loss2)

                map_1 = target[0, :, :, :, 0]
                map_0 = target[0, :, :, :, 1]
                final = copy.deepcopy(target[0, :, :, :, 0])
                final[map_1 >= map_0] = 1
                final[map_1 < map_0] = 0

                fpp = FP(gt[0, :, :, :, 0], final, wmmask, use_mask=FLAGS.mask)
                print('False Positive rate: %s' % (fpp))

                if fpp < previous:
                    saver.save(sess, os.path.join(board, 'model_step_%s' % step))
                    previous = fpp
                    print("saved step %s model" % step)
                    print("current best FP %s" % fpp)

                dice = dsc(final, gt[0, :, :, :, 0])
                print('dice score is: %s ' % (dice))
                dice_crf.append(dice)

                final = nib.Nifti1Image(final, affine)
                nib.save(final, file_root + '/%s_%s.nii.gz' % (step, subjects[pos]))
                np.save(file_root + '/dice_crf', dice_crf)


            else:
                limit = 0
                for w in width_range:
                    for h in height_range:
                        for d in depth_range:
                            input_img = input[:, w:w + shape, h:h + shape, d:d + shape, :]
                            target_img = target[:, w:w + shape, h:h + shape, d:d + shape, :]
                            f_filtered_img = f_filtered[:, w:w + shape, h:h + shape, d:d + shape, :]
                            mask_img = mask[:, w:w + shape, h:h + shape, d:d + shape, :]

                            if np.count_nonzero(input_img) == 0:
                                continue
                            if np.sum(target_img[:, :, :, :, 0]) / (np.product(target_img.shape) / 2) < 0.01:
                                continue

                            input_img = input_img / np.max(input_img)
                            loss, loss2, summary_temp = net.train_step(sess, input_img, target_img, f_filtered_img, mask_img)
                            f_summary.add_summary(summary=summary_temp, global_step=cou)
                            loss = np.average(loss)
                            total_loss.append(loss)
                            loss2 = np.average(loss2)
                            total_loss2.append(loss2)
                            cou += 1
                            limit += 1

            loss = np.average(total_loss)
            loss2 = np.average(total_loss2)
            if pos % val_step == 0 and step != 0:
                merge = tf.summary.merge([net.val_loss, net.FP, net.Dice])
                summary = sess.run(merge,
                                   feed_dict={net.loss_val: loss, net.fp_val: fpp, net.dice_val: dice})
                f_summary.add_summary(summary=summary, global_step=step)
            else:
                summary = sess.run(net.train_loss,
                                   feed_dict={net.loss_train: loss})
                f_summary.add_summary(summary=summary, global_step=step)
            print('step %s loss is %s | loss without crf is %s' % (step, loss, loss2))


if __name__ == "__main__":
    tf.app.run()
