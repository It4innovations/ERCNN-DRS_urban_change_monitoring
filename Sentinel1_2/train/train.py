#
# Author: Georg Zitzlsberger (georg.zitzlsberger<ad>vsb.cz)
# Copyright (C) 2020-2021 Georg Zitzlsberger, IT4Innovations,
#                         VSB-Technical University of Ostrava, Czech Republic
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import os
os.environ["XDG_CACHE_HOME"]="/tmp"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Get rid of the pesky TF messages

import re
import sys
import math
import numpy as np
from datetime import datetime
from datetime import timezone
import dateutil.parser
import tensorflow as tf
from eolearn.core import EOPatch
import matplotlib.pyplot as plt
import skimage.transform
import warnings
warnings.filterwarnings("ignore")
import horovod.tensorflow.keras as hvd
from tensorflow.python.keras.utils import losses_utils

import sys
sys.path.append('../model/')
from model import ERCNN_DRS

# Initialize Horovod
hvd.init()

# PARAMETERS FOR TRAINING #####################################################
sites=["Rotterdam","Limassol"]
base_dir = "/data/data_temp/training_s12/"
tf_record_file_out_train_path = base_dir + "/train/"
tf_record_file_out_val_path = base_dir + "/val/"

tile_size_x = 32
tile_size_y = 32

time_start = datetime(2017, 1, 1, 0, 0, 0) # When to start first window
time_range = 60*60*24*182 # Window observation period (in seconds) (S2: 182 days)
time_step = 60*60*24*2 # Every 2nd day

buffer_size = math.ceil(time_range / time_step) + 1

num_cycles = 5
parallel_calls = tf.data.experimental.AUTOTUNE

# Should be sufficient considering that all sites have different # of windows
# and are interleaved. After some warm up, there should be more entropy,
# later in the epoch.
shuffle_size = 200

nb_epochs = 500
batch_size = 32

no_channels = 17

sar = False
combined = True

best_weights_file = "./snapshots/best_weights_ercnn_drs_new.hdf5"
best_weights_train_file = "./snapshots/best_weights_{epoch:04d}.hdf5"
#best_weights_continue_with = ""

log_dir = "./logs/"
###############################################################################

if hvd.rank() == 0:
    verbose = True
else:
    verbose = False

gpus = tf.config.experimental.list_physical_devices('GPU')
if verbose:
    print("Num GPUs Available: ", len(gpus))
local_gpus = hvd.local_rank()
print(local_gpus)
if gpus:
    try:
        tf.config.set_visible_devices(gpus[local_gpus], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def _parse_tfr_element_train(element):
    parse_dic = {
        "Feature": tf.io.FixedLenFeature([], tf.string),
        "Label": tf.io.FixedLenFeature([], tf.string)
    }
    example_message = tf.io.parse_single_example(element, parse_dic)

    feature_f = example_message["Feature"]
    feature = tf.ensure_shape(tf.io.parse_tensor(
                                feature_f,
                                out_type=tf.float32),
                              [None, tile_size_y, tile_size_x, no_channels])

    label_f = example_message["Label"]
    label = tf.ensure_shape(tf.io.parse_tensor(
                                label_f,
                                out_type=tf.float32),
                            [tile_size_y, tile_size_x])
    return feature, label

def read_TFRecords(file):
    dataset = tf.data.TFRecordDataset(file,
                                      compression_type="GZIP",
                                      num_parallel_reads=4) \
                     .map(_parse_tfr_element_train, num_parallel_calls=4)
    return dataset

def build_dataset(directory):
    tile_files = []
    for site in sites:
        if verbose:
            print("site: {}".format(site))

        tiles = [os.path.join(directory, f) for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and
                        f.startswith("{}_".format(site)) and
                        f.endswith(".tfrecords")]
        if verbose:
            print("\ttotal tiles:", len(tiles))
        tile_files += tiles

    dataset = tf.data.Dataset.from_tensor_slices(tile_files) \
                             .shuffle(len(tile_files))
    dataset = dataset.interleave(lambda x: read_TFRecords(x),
                                 num_parallel_calls=parallel_calls,
                                 cycle_length=num_cycles, block_length=1)

    return dataset

if verbose:
    print("---Training Dataset:")
train_ds = build_dataset(tf_record_file_out_train_path)
if verbose:
    print("---Validation Dataset:")
val_ds = build_dataset(tf_record_file_out_val_path)

# TFRecords are streams w/o length information; count them once
def count_elements(ds):
    count = 0
    for item in ds:
        count += 1
        if count % 1000 == 0:
            print(".", end = "")
    return count

this_count = count_elements(train_ds)
if verbose:
    print("# of training windows: ", this_count)

# Drop the end so we got multiple of #GPUs * batch size
this_rest = this_count % (hvd.size() * batch_size)
train_cache_dir = "/tmp/train_ds_{}".format(hvd.rank()) #"/mnt/nvmeof/train_ds_{}".format(hvd.rank())
os.makedirs(train_cache_dir)
train_ds = train_ds.take(this_count - this_rest) \
                   .shard(num_shards=hvd.size(), index=hvd.rank()) \
                   .cache(filename=train_cache_dir) \
                   .shuffle(shuffle_size)
train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(
                            (this_count - this_rest) // (hvd.size())))
print(hvd.rank(),
     " training samples: #",
      tf.data.experimental.cardinality(train_ds))

if combined:
    train_ds = train_ds.map(lambda x, y: (x[:, :, :, :], y))
else:
    if sar:
        train_ds = train_ds.map(lambda x, y: (x[:, :, :, 0:4], y))
    else:
        train_ds = train_ds.map(lambda x, y: (x[:, :, :, 4:], y))

this_count = count_elements(val_ds)
if verbose:
    print("# of validation windows: ", this_count)

this_rest = this_count % hvd.size()
val_cache_dir = "/tmp/val_ds_{}".format(hvd.rank()) #"/mnt/nvmeof/val_ds_{}".format(hvd.rank())
os.makedirs(val_cache_dir)
val_ds = val_ds.take(this_count - this_rest) \
               .shard(num_shards=hvd.size(), index=hvd.rank()) \
               .cache(filename=val_cache_dir)
val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(
                        (this_count - this_rest) // hvd.size()))

print(hvd.rank(),
      " validation samples: #",
      tf.data.experimental.cardinality(val_ds))

if combined:
    val_ds = val_ds.map(lambda x, y: (x[:, :, :, :], y))
else:
    if sar:
        val_ds = val_ds.map(lambda x, y: (x[:, :, :, 0:4], y))
    else:
        val_ds = val_ds.map(lambda x, y: (x[:, :, :, 4:], y))

training_ds = train_ds.padded_batch(batch_size,
                                    padded_shapes=([buffer_size,
                                                    tile_size_y,
                                                    tile_size_x,
                                                    no_channels],
                                                   [tile_size_y,
                                                    tile_size_x]),
                                    padding_values=-1.0).prefetch(32)

validation_ds = val_ds.shard(hvd.size(), hvd.local_rank()) \
                      .padded_batch(batch_size,
                                    padded_shapes=([buffer_size,
                                                    tile_size_y,
                                                    tile_size_x,
                                                    no_channels],
                                                   [tile_size_y,
                                                    tile_size_x]),
                                    padding_values=-1.0).prefetch(32)

# Define the custom loss function for Tanimoto loss with complement
class TanimotoCompl(tf.keras.losses.Loss):
    def __init__(self,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='TanimotoCompl'):
        super(TanimotoCompl, self).__init__(reduction=reduction, name=name)
        self._axis = axis

    def tanimoto_compl(self, y_true, y_pred):
        # ATTENTION: Only compute the loss w/o the border pixels!
        y_true_f = tf.keras.backend.flatten(y_true[:,1:-1,1:-1])
        y_pred_f = tf.keras.backend.flatten(y_pred[:,1:-1,1:-1])

        true_pos = tf.keras.backend.sum(y_true_f * y_pred_f)
        true_neg = tf.keras.backend.sum((1 - y_true_f) * (1 - y_pred_f))

        tanimoto = (true_pos)/(tf.keras.backend.sum(             \
                                    y_true_f**2 + y_pred_f**2) - \
                               true_pos + 0.0001)
        tanimoto_comp = (true_neg)/(tf.keras.backend.sum((1-y_true_f)**2 + \
                                                         (1-y_pred_f)**2) - \
                                    true_neg + 0.0001)
        return 1 - (tanimoto + tanimoto_comp)/2

    def call(self, y_true, y_pred):
        return self.tanimoto_compl(y_true, y_pred)


# Load the model
ercnn_drs = ERCNN_DRS(combined, sar)
model = ercnn_drs.build_model(buffer_size, tile_size_y, tile_size_x, no_channels)

opt = tf.keras.optimizers.SGD(lr=.001 * hvd.size(), momentum=0.8)
opt = hvd.DistributedOptimizer(opt)
model.compile(optimizer=opt,
              loss=TanimotoCompl(),
              metrics=[
                  tf.keras.metrics.MeanAbsoluteError(),
                  tf.keras.metrics.RootMeanSquaredError(),
                  tf.keras.metrics.MeanSquaredLogarithmicError()
              ],
              experimental_run_tf_function=False)

# Use existing best weights if available...
if "best_weights_continue_with" in globals():
    if os.path.isfile(best_weights_continue_with):
        if verbose:
            print("ATTENTION: loaded training weights {}".format(
                                        best_weights_continue_with))
        model.load_weights(best_weights_continue_with)

os.makedirs(log_dir, exist_ok=True)
log_dir_s = log_dir + datetime.now().strftime("%Y%m%d-%H%M%S") + \
                      "_" + str(hvd.local_rank())
os.makedirs(log_dir_s, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_s,
                                                      histogram_freq=1,
                                                      update_freq='batch',
                                                      profile_batch=0)

callbacks = []
callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
callbacks.append(hvd.callbacks.MetricAverageCallback())
if verbose:
    # Checkpointing only on rank 0 (verbose == True)
    save_best = tf.keras.callbacks.ModelCheckpoint(filepath=best_weights_file,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True)
    save_best_train = tf.keras.callbacks.ModelCheckpoint(filepath=best_weights_train_file,
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=False)

    callbacks.append(save_best)
    callbacks.append(save_best_train)

# We collect logs for every GPU intentionally
callbacks.append(tensorboard_callback)

model.fit(
    training_ds,
    validation_data = validation_ds,
    epochs = nb_epochs,
    callbacks = callbacks,
    verbose=2 if hvd.rank() == 0 else 0)
