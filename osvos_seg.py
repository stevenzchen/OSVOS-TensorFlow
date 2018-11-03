from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""This script is adapted from osvos_demo.py.

This script assumes that it is being run from the JITNet root folder.
"""
import os
import sys
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
height = FLAGS.height
width = FLAGS.width
sequence = FLAGS.sequence
dataset_dir = FLAGS.dataset_dir
sequence_limit = FLAGS.sequence_limit
class_index = FLAGS.class_index

gpu_id = 0

# Train parameters
parent_path = os.path.join(os.getcwd(), 'OSVOS-TensorFlow', 'models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = os.path.join(os.getcwd(), 'OSVOS-TensorFlow', 'models', sequence)
max_training_iters = 500 # TODO: should this be higher?

# initialize dataset
dataset = osvos_dataset.OSVOS_Dataset(sequence, dataset_dir, sequence_limit, training_stride, height, width, class_index, start_frame)

# Train the network
learning_rate = 1e-8
save_step = max_training_iters
side_supervision = 3
display_step = 10
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                             save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=sequence)
        
# TODO: we should also update stats for the teacher frame

# TODO: make sure it uses a new model for every parent frame

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join('models', sequence, sequence + '.ckpt-' + str(max_training_iters))
        preds, labels = osvos.test(dataset, checkpoint_path, stats_path)

# TODO: save + display predictions and labels

# TODO: run this in a loop until we hit max_frames