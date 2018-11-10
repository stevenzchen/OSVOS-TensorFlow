from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""This script is adapted from osvos_demo.py.

This script assumes that it is being run from the JITNet root folder.
"""
import os
import sys
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos

# import functions from online_scene_distillation main file
sys.path.append(os.path.realpath('./src'))
sys.path.append(os.path.realpath('./datasets'))
sys.path.append(os.path.realpath('./utils'))

# create flags not present in online_scene_distillation
tf.app.flags.DEFINE_integer('start_frame', 0, 'Start frame')
tf.app.flags.DEFINE_integer('class_index', 1, 'Class index')

from online_scene_distillation import update_stats
import osvos_dataset
from mask_rcnn_tfrecords import visualize_masks


FLAGS = tf.app.flags.FLAGS
# set variables using flags
max_frames = FLAGS.max_frames
training_stride = FLAGS.training_stride
stats_path = FLAGS.stats_path
start_frame = FLAGS.start_frame
# this height and width should be the network's 480p input resolution
# default: (480, 854)
height = FLAGS.height
width = FLAGS.width
sequence = FLAGS.sequence
dataset_dir = FLAGS.dataset_dir
sequence_limit = FLAGS.sequence_limit
class_index = FLAGS.class_index

gpu_id = 0

# initialize metrics
per_frame_stats = {}
num_classes = 2 # foreground object and background

class_correct = np.zeros(num_classes, np.float32)
class_total = np.zeros(num_classes, np.float32)
class_tp = np.zeros(num_classes, np.float32)
class_fp = np.zeros(num_classes, np.float32)
class_fn = np.zeros(num_classes, np.float32)
class_iou = np.zeros(num_classes, np.float32)

# training parameters
parent_path = os.path.join(os.getcwd(), 'OSVOS-TensorFlow', 'models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = os.path.join(os.getcwd(), 'OSVOS-TensorFlow', 'models', sequence)

ckpt_path = parent_path

# NOTE: this defaults to 500 (30 sec runtime on GCE V100)
max_training_iters = 500

# initialize dataset
dataset = osvos_dataset.OSVOS_Dataset(sequence, dataset_dir, sequence_limit, training_stride, height, width, class_index, start_frame)

curr_frame = 0
train_iter = 1

while curr_frame < max_frames:
    # train phase
    print('------ training at frame', curr_frame, '---------')
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable((train_iter - 1) * max_training_iters, name='global_step', trainable=False)
            osvos.train_finetune(dataset, ckpt_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=sequence)

    # test phase
    print('------ testing at frame', curr_frame, '---------')
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            ckpt_path = os.path.join(os.getcwd(), 'OSVOS-TensorFlow', 'models', sequence, sequence + '.ckpt-' + str(train_iter * max_training_iters))
            imgs, preds, labels = osvos.test(dataset, ckpt_path, stats_path)
            for i in range(len(imgs)):
                img = imgs[i]
                img = img[0]
                pred = preds[i]
                label = labels[i]
                label = label[0]
                label[label > 0] = 1

                labels_vals = np.reshape(label, (1, height, width))
                pred_ext = np.reshape(pred, (1, height, width))
                update_stats(labels_vals, pred_ext, class_tp, class_fp, class_fn,
                             class_total, class_correct, 
                             np.ones(labels_vals.shape, dtype=np.bool),
                             None, curr_frame, True, None, per_frame_stats)
                
                curr_frame += 1
    
                # generate visualization
#                 vis_shape = (height, width, 3)
#                 vis_labels = visualize_masks(pred_ext, 1, vis_shape,
#                                              num_classes=num_classes)
#                 vis_labels = vis_labels[0]
#                 labels_image = cv2.addWeighted(img, 0.5, vis_labels, 0.5, 0)
#                 pil_image = Image.fromarray(labels_image)
#                 pil_image.save('/home/stevenzc3/osvos/{:03d}.png'.format(curr_frame))
    
    # reset dataset
    dataset.reset_cycle()
    
    train_iter += 1
                

np.save(stats_path, [per_frame_stats])
