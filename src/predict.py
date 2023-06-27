
import os
import nibabel as nib
import copy
import tensorflow as tf
from frangi.hessian import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('modality', 'epc', 'input and target modality names')
flags.DEFINE_integer('patch_size', 16, 'size of the pathch at three dimensions')
flags.DEFINE_string('dataset', None, 'path of test dataset')
flags.DEFINE_string('weights', None, 'path of saved weights')
flags.DEFINE_integer('gpu', None, '# of gpu to use')

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
meta_graph = FLAGS.weights + '.meta'
meta_weight = FLAGS.weights

subjects = os.listdir(FLAGS.dataset)

with tf.device('/gpu:%s'%FLAGS.gpu):
    saver = tf.train.import_meta_graph(meta_graph)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, meta_weight)
    graph = tf.get_default_graph()
    op = graph.get_tensor_by_name('unet/Softmax:0')
    unet_op = graph.get_tensor_by_name('unet/decoder_0/final/Softmax:0')
    Input = graph.get_tensor_by_name('Placeholder_1:0')
    F_filter = graph.get_tensor_by_name('Placeholder_3:0')
    Mask = graph.get_tensor_by_name('Placeholder_4:0')
    is_train = graph.get_tensor_by_name('Placeholder:0')

    for sub_name in subjects:
        path = os.path.join(FLAGS.dataset, sub_name, '%s.nii.gz'%FLAGS.modality)
        input = nib.load(path)
        affine = input.affine
        input = input.get_fdata()

        shape = FLAGS.patch_size
        [w, h, d] = input.shape
        result = np.zeros([1, w, h, d, 2])

        frangied = list()
        mask = list()
        scale_range = np.arange(0.1, 1.5, 1)
        for sigma in scale_range:
            f_filtered = hessian_filter(input, sigma=sigma)
            mask.append(filter_out_background(True, f_filtered[1], f_filtered[2]))
            frangied.append(np.array([f_filtered[0], f_filtered[1], f_filtered[2]]))
        f_filtered = np.stack(frangied, axis=-1)
        mask = np.stack(mask, axis=-1)
        mask = np.expand_dims(mask, axis=0)
        input = np.expand_dims(input, axis=0)
        input = np.expand_dims(input, axis=-1)

        width_range = np.arange(0, w, shape // 2)
        height_range = np.arange(0, h, shape // 2)
        depth_range = np.arange(0, d, shape // 2)

        for w in width_range:
            for h in height_range:
                for d in depth_range:
                    input_img = input[:, w:w + shape, h:h + shape, d:d + shape, :]
                    f_filtered_img = f_filtered[:, w:w + shape, h:h + shape, d:d + shape, :]
                    mask_img = mask[:, w:w + shape, h:h + shape, d:d + shape, :]
                    if np.count_nonzero(input_img) == 0:
                        continue
                    input_img = input_img / np.max(input_img)
                    [output] = sess.run([op], feed_dict={Input:input_img, is_train:False, F_filter:f_filtered_img, Mask:mask_img})
                    result[:, w:w + shape, h:h + shape, d:d + shape, :] = output + result[:, w:w + shape, h:h + shape, d:d + shape, :]

        map_1 = result[0, :, :, :, 0]
        map_0 = result[0, :, :, :, 1]
        final = copy.deepcopy(result[0, :, :, :, 0])
        final[map_1 > map_0] = 1
        final[map_1 <= map_0] = 0
        final = nib.Nifti1Image(final, affine)
        nib.save(final, os.path.join(FLAGS.dataset, sub_name, 'pvs_seg_%s.nii.gz'% sub_name))
        print('Finished PVS segmentation for subject %s'% sub_name)
