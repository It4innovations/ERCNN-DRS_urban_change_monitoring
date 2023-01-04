#
# Author: Georg Zitzlsberger (georg.zitzlsberger<ad>vsb.cz)
# Copyright (C) 2020-2023 Georg Zitzlsberger, IT4Innovations,
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
import math
import numpy as np
from datetime import datetime
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import horovod.tensorflow as hvd
from tensorflow.python.keras.utils import losses_utils
from model import ERCNN_DRS

# Initialize Horovod
hvd.init()

# PARAMETERS FOR TRAINING #####################################################
sites=["Liege"]
sites_params=[(20, 45, 56)] # random windows (beg., low, high)

exp = "V1" # The version to use V1-3

base_dir = "./{}/data/".format(exp)
tf_record_file_out_train_path = base_dir + "/train/"
tf_record_file_out_val_path = base_dir + "/val/"

npy_gt = "./numpy_ground_truth/"

tile_size_x = 32
tile_size_y = 32

buffer_size = 92
parallel_calls = tf.data.experimental.AUTOTUNE

nb_epochs = 500
batch_size = 8
val_batch_size = 8

no_channels = 17

sar = False
combined = True

snapshot_dir = "./{}/snapshots/".format(exp)
best_weights_train_file = snapshot_dir + "/best_weights_{epoch:04d}.h5"
best_weights_continue_with = "../models/baseline.hdf5"
log_dir = "./{}/logs/".format(exp)

###############################################################################

if hvd.rank() == 0:
    verbose = True
else:
    verbose = False

gpus = tf.config.experimental.list_physical_devices('GPU')
if verbose:
    print("Num GPUs Available: ", len(gpus))
local_gpus = hvd.local_rank()
if verbose:
    print("Local GPU#: ", local_gpus)
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
    return feature #, label # Label is not used; loaded separately

@tf.function
def random_vals(fle):
    i = tf.constant(0)
    vals = (-1, -1, -1)

    for i in range(0, len(sites)):
        if tf.strings.regex_full_match(
                              fle,
                              ".*/{}_[0-9]+_[0-9]+.tfrecords".format(sites[i])):
            vals = sites_params[i]
    return tf.cast(vals, dtype=tf.int64)

@tf.function
def read_TFRecords(file):
    vals = random_vals(file)

    dataset = tf.data.TFRecordDataset(file,
                                      compression_type="GZIP",
                                      num_parallel_reads=4)                    \
                     .map(_parse_tfr_element_train, num_parallel_calls=4)

    # Randomly select only a subset of windows...
    rand_offs = tf.concat([tf.random.uniform(shape=[1],
                                             dtype=tf.int64,
                                             minval=vals[0],
                                             maxval=vals[0]+1),
                           tf.random.uniform(shape=[9],
                                             dtype=tf.int64,
                                             minval=vals[1],
                                             maxval=vals[2]),
                           # 600 for moving past the end
                           tf.constant([600], dtype=tf.int64)], axis=0)
    sel_idxs = tf.math.cumsum(rand_offs) # Get abs. indices
    filter_idx = tf.autograph.experimental.do_not_convert(
                        lambda idxf, x:
                            tf.reduce_any(
                                tf.equal(sel_idxs,
                                    tf.broadcast_to(idxf, tf.shape(sel_idxs)))))
    dataset = dataset.enumerate().filter(filter_idx).map(lambda idxf, x: x)
    return dataset

def build_dataset(directory):
    tile_files = []
    for site in sites:
        if verbose:
            print("site: {}".format(site))

        tiles = [os.path.join(directory, f) for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and
                        f.startswith("{}_".format(site)) and
                        f[len(site)+1].isdigit() and
                        f.endswith(".tfrecords")]

        if verbose:
            print("\ttotal tiles:", len(tiles))
        tile_files += tiles

    label_tiles = []
    file_ids = []
    augment_codes = []
    ext_tile_files = []
    for tle in tile_files:
        # masking (2) times horiz. flipping (2) times four rotations
        for ac in range(0, 2*2*4):
            found_ids = re.findall(
                    r"(.*)/([a-zA-Z_]+)_([0-9]+)_([0-9]+).tfrecords", tle)
            label_tiles.append("{}/{}_{}_{}.npy".format(#found_ids[0][0],
                                                        npy_gt,
                                                        found_ids[0][1],
                                                        found_ids[0][2],
                                                        found_ids[0][3]))

            # ID is defined as 1000000*sites + 10000*y + 100*x + ac
            # sites are # of sites, y and x are tile coordinates, ac is
            # augmentation code; sites < 100, values y and x < 100, ac in [0,7]
            file_ids.append(1000000*sites.index(found_ids[0][1]) +             \
                            10000*int(found_ids[0][2])           +             \
                            100*int(found_ids[0][3])             +             \
                            ac)

            augment_codes.append(ac)

            ext_tile_files.append(tle)

    ds_ids = tf.data.Dataset.from_tensor_slices([float(f) for f in file_ids])
    file_ds = tf.data.Dataset.from_tensor_slices(ext_tile_files)
    label_data = np.stack([np.load(f) for f in label_tiles])

    # Threshold for ground truth > 0.0 -> 1.0
    label_data = np.where(label_data > 0.0, np.float32(1.0), np.float32(0.0))

    label_ds = tf.data.Dataset.from_tensor_slices(label_data)
    augment_code_ds = tf.data.Dataset.from_tensor_slices(augment_codes)
    dataset = tf.data.Dataset.zip((ds_ids, file_ds, label_ds, augment_code_ds))\
                             .shuffle(len(ext_tile_files), seed=123)

    dataset = dataset.apply(tf.data.experimental.assert_cardinality(
                                                           len(ext_tile_files)))
    return dataset


# TFRecords are streams w/o length information; count them once
def count_elements(ds):
    count = 0
    for item in ds:
        count += 1
        if count % 1000 == 0:
            print(".", end = "")
    return count

# Training data...
if verbose:
    print("---Training Dataset:")
train_ds = build_dataset(tf_record_file_out_train_path)
print("  overall training samples[{}]: #".format(
                                    hvd.rank()),
                                    tf.data.experimental.cardinality(train_ds))

this_count = count_elements(train_ds)

def print_vals(val_ds):
    for item in val_ds:
        print("{}: {}, ac: {}".format(hvd.rank(), item[1], item[3]))

# Drop the end so we got multiple of #GPUs * batch size
this_rest = this_count % (hvd.size() * batch_size)
assert this_rest == 0,                                                         \
       "Training DS is not multiple of hvd.size() * batch_size"
if this_rest != 0:
    print("ATTENTION: removed {} samples [train]".format(this_rest))

train_ds = train_ds.take(this_count - this_rest)                               \
                   .shard(num_shards=hvd.size(), index=hvd.rank())             \
                   .cache()

train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(
                            (this_count - this_rest) // (hvd.size())))

print(hvd.rank(),
      "    training samples: #",
      tf.data.experimental.cardinality(train_ds))

training_ds = train_ds.shuffle(tf.data.experimental.cardinality(train_ds))     \
                      .batch(batch_size)

# Validation data...
if verbose:
    print("---Validation Dataset:")
val_ds = build_dataset(tf_record_file_out_val_path)
print("  overall validation samples[{}]: #".format(
                                    hvd.rank()),
                                    tf.data.experimental.cardinality(val_ds))

this_count = count_elements(val_ds)

this_rest = this_count % (hvd.size() * val_batch_size)
assert this_rest == 0,                                                         \
       "Validation DS is not multiple of hvd.size() * val_batch_size"
if this_rest != 0:
    print("ATTENTION: removed {} samples [val]".format(this_rest))

val_ds = val_ds.take(this_count - this_rest)                                   \
               .shard(num_shards=hvd.size(), index=hvd.rank())                 \
               .cache()

val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(
                        (this_count - this_rest) // hvd.size()))

print(hvd.rank(),
      "    validation samples: #",
      tf.data.experimental.cardinality(val_ds))
validation_ds = val_ds.batch(val_batch_size)


session_data = {}
session_data["best_loss"] = 1.0 # Max. to start with
session_data["global_steps"] = 0

# Define the custom loss function for Tanimoto loss with complement
class TanimotoCompl(tf.keras.losses.Loss):
    def __init__(self,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='TanimotoCompl'):
        super(TanimotoCompl, self).__init__(reduction=reduction, name=name)
        self._axis = axis

    def tanimoto_compl(self, y_true, y_pred):
        @tf.function
        def red_sum(val):
            # ATTENTION: Only compute the loss w/o the border pixels!
            return tf.math.reduce_sum(
                            tf.math.reduce_sum(val[:,1:-1,1:-1], 1), 1)

        @tf.function
        def compl(val):
            return tf.math.subtract(np.float32(1.0), val)

        true_pos = tf.math.multiply(y_true, y_pred)
        true_neg = tf.math.multiply(compl(y_true), compl(y_pred))

        t1 = tf.math.add(tf.math.pow(y_true, 2), tf.math.pow(y_pred, 2))

        tanimoto = tf.math.divide(red_sum(true_pos), tf.math.add(
                                                tf.math.subtract(
                                                    red_sum(t1),
                                                    red_sum(true_pos)),
                                                0.0001))

        t2 = tf.math.add(tf.math.pow(compl(y_true), 2),
                         tf.math.pow(compl(y_pred), 2))

        tanimoto_comp = tf.math.divide(red_sum(true_neg), tf.math.add(
                                                        tf.math.subtract(
                                                            red_sum(t2),
                                                            red_sum(true_neg)),
                                                        0.0001))

        return tf.math.subtract(np.float32(1.0), tf.math.divide(
                                                    tf.math.add(tanimoto,
                                                                tanimoto_comp),
                                                    2.0))

    def call(self, y_true, y_pred):
        return self.tanimoto_compl(y_true, y_pred)

# Load the model
ercnn_drs = ERCNN_DRS(combined, sar)
model = ercnn_drs.build_model(buffer_size,
                              tile_size_y,
                              tile_size_x,
                              no_channels)

opt = tf.keras.optimizers.SGD(lr=(.001 * hvd.size()), momentum=0.8)

# Use existing best weights if available...
if "best_weights_continue_with" in globals():
    if os.path.isfile(best_weights_continue_with):
        if verbose:
            print("ATTENTION: loaded training weights {}".format(
                                        best_weights_continue_with))
        model.load_weights(best_weights_continue_with)

os.makedirs(log_dir, exist_ok=True)
log_dir_s = log_dir + datetime.now().strftime("%Y%m%d-%H%M%S") +               \
                      "_" + str(hvd.local_rank())
os.makedirs(log_dir_s, exist_ok=True)
log_dir_s_val = log_dir_s + "/val"
os.makedirs(log_dir_s_val, exist_ok=True)

loss_fn=TanimotoCompl(reduction=losses_utils.ReductionV2.NONE)
writer = tf.summary.create_file_writer(log_dir_s)
writer_val = tf.summary.create_file_writer(log_dir_s_val)
os.makedirs(snapshot_dir, exist_ok=True)

def save_state(epoch):
    if hvd.rank() == 0: # Always save
        tf.keras.models.save_model(
                            model,
                            best_weights_train_file.format(epoch=epoch+1),
                            overwrite=True,
                            include_optimizer=True)

@tf.function
def train_step(windows_ds, labels, ids_mask):
    this_iter = iter(windows_ds)
    optional = this_iter.get_next_as_optional()

    with tf.GradientTape() as tape:
        max_out = tf.zeros_like(labels, dtype=tf.float32)
        max_out_th = tf.zeros_like(labels, dtype=tf.float32)
        count_wins = tf.zeros_like([1], dtype=tf.float32)

        while optional.has_value():
            x = optional.get_value()
            x = tf.ensure_shape(x,
                                [batch_size,
                                 buffer_size,
                                 tile_size_y,
                                 tile_size_x,
                                 no_channels])
            outp = model(x, training=True)
            max_out = tf.math.maximum(max_out, outp)
            optional = this_iter.get_next_as_optional()
            count_wins = tf.math.add(count_wins, 1)

        loss_values = loss_fn(labels, max_out)
        masked_losses = tf.boolean_mask(loss_values,
                                        tf.where(ids_mask == -1.0, False, True))
        loss_value = tf.math.reduce_mean(masked_losses)

    tape = hvd.DistributedGradientTape(tape) # Sync. gradients
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss_values, max_out, loss_value

@tf.function
def val_step(windows_ds, labels, batch_size):
    this_iter = iter(windows_ds)
    optional = this_iter.get_next_as_optional()

    max_out = tf.zeros_like(labels, dtype=tf.float32)
    count_wins = tf.zeros_like([1], dtype=tf.float32)
    while optional.has_value():
        x = optional.get_value()
        x = tf.ensure_shape(x,
                            [batch_size,
                             buffer_size,
                             tile_size_y,
                             tile_size_x,
                             no_channels])
        val_outp = model(x, training=False)
        max_out = tf.math.maximum(max_out, val_outp)
        optional = this_iter.get_next_as_optional()
        count_wins = tf.math.add(count_wins, 1)
    loss_values = loss_fn(labels, max_out)

    return tf.math.reduce_mean(loss_values)

@tf.function
def augment(x, augment_code, is_label=False):
    # This is a 5D augmentation (batch, no_window, y, x, c)

    # Encoding is as follows:
    # augment_code = this_rand_fh*2^0 + this_rand_fv*2^1 + this_rand_rot*2^3
    this_rand_mask = int(augment_code%2)
    this_rand_fh = int(augment_code/2%2)
    this_rand_rot = int(augment_code/4)

    # Create the mask
    # Even columns contain original, odd columns only receive first time step
    # (frozen/no change)
    mask = tf.equal(tf.math.mod(tf.range(0, tf.shape(x)[-2]), 2), 0)
    x_img = tf.broadcast_to(mask, [tf.shape(x)[-3], tf.shape(x)[-2]])
    x_ch = tf.broadcast_to(tf.expand_dims(x_img, -1),
                           [tf.shape(x)[-3],
                           tf.shape(x)[-2],
                           tf.shape(x)[-1]])

    if is_label == False:
        x = tf.map_fn(fn=lambda args:
                        tf.cond(tf.equal(args[1], 1),
                                lambda: tf.where(
                                    x_ch,
                                    args[0],
                                    tf.broadcast_to(
                                        args[0][0, :, :, :],
                                        tf.shape(args[0]))),
                                lambda: args[0]),
                      elems=(x, this_rand_mask), fn_output_signature=tf.float32)
    else:
        x = tf.map_fn(fn=lambda args:
                        tf.cond(tf.equal(args[1], 1),
                                lambda: tf.where(x_ch, args[0], 0),
                                lambda: args[0]),
                      elems=(x, this_rand_mask), fn_output_signature=tf.float32)

    x = tf.map_fn(fn=lambda args:
                    tf.cond(tf.equal(args[1], 1),
                        lambda: tf.reverse(args[0], [2]),
                        lambda: args[0]), # flip horizontal
                  elems=(x, this_rand_fh), fn_output_signature=tf.float32)

    x = tf.map_fn(fn=lambda args:
                      tf.switch_case(args[1],
                           branch_fns={
                               0: lambda: args[0], # no rot
                               1: lambda: tf.transpose(
                                       tf.reverse(args[0], [1]),
                                       [0, 2, 1, 3]), # rot 90 clockwise
                               2: lambda: tf.transpose(
                                       tf.reverse(args[0], [2]),
                                       [0, 2, 1, 3]), # rot 90 counter-clockwise
                               3: lambda: tf.reverse(args[0], [1,2])}, # rot 180
                           default = lambda: args[0]),
                  elems=(x, this_rand_rot), fn_output_signature=tf.float32)
    return x

###########################################################################
# Main...
epoch = 0
while True:
    print("\nStart of epoch {} [{}]".format(epoch+1, hvd.rank()))

    #######################################################################
    # Training...
    train_loss = tf.metrics.Mean()
    for (step, (ids, file_ds, y, augment_code)) in enumerate(training_ds):
        # Get local IDs invovled in this batch

        ids_mask = ids

        y_reshaped_aug = augment(
                            tf.reshape(y,
                                       [batch_size,
                                        1,
                                        tile_size_y,
                                        tile_size_x,
                                        1]),
                            augment_code,
                            is_label=True)
        y = tf.reshape(y_reshaped_aug, [batch_size, tile_size_y, tile_size_x])

        #######################################################################
        # Load dataset (i.e. list of windows) for tiles
        tiles_ds = tf.data.Dataset.from_tensor_slices(file_ds)
        windows_ds = tiles_ds.interleave(lambda x: read_TFRecords(x),
                                     num_parallel_calls=parallel_calls,
                                     cycle_length=batch_size, block_length=1)

        windows_ds = windows_ds.padded_batch(batch_size,
                                    padded_shapes=[buffer_size,
                                                    tile_size_y,
                                                    tile_size_x,
                                                    no_channels],
                                    padding_values=-1.0).prefetch(32)

        windows_ds = windows_ds.map(lambda x: augment(x, augment_code))
        #######################################################################

        loss_values, pred, loss_value = train_step(windows_ds, y, ids_mask)

        if not np.isnan(loss_value):
            train_loss(loss_value)
        with writer.as_default():
            session_data["global_steps"] += 1
            tf.summary.scalar('trianing_step_loss',
                              train_loss.result(),
                              step=session_data["global_steps"])

        if epoch == 0 and step == 0:
            print("Initialize... {}".format(hvd.rank()))
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

    #######################################################################
    # Validation...
    val_loss = tf.metrics.Mean()
    # Repeat validation data three times to consider also different windows.
    # This also reduces noise in validation loss.
    for (step, (ids, file_ds, y, augment_code)) in enumerate(
                                                    validation_ds.repeat(3)):

        y_reshaped_aug = augment(
                            tf.reshape(y,
                                       [val_batch_size,
                                        1,
                                        tile_size_y,
                                        tile_size_x,
                                        1]),
                            augment_code,
                            is_label=True)
        y = tf.reshape(y_reshaped_aug,
                       [val_batch_size, tile_size_y, tile_size_x])

        #######################################################################
        # Load dataset (i.e. list of windows) for tiles
        tiles_ds = tf.data.Dataset.from_tensor_slices(file_ds)
        windows_ds = tiles_ds.interleave(lambda x: read_TFRecords(x),
                                     num_parallel_calls=parallel_calls,
                                     cycle_length=val_batch_size, block_length=1)

        windows_ds = windows_ds.padded_batch(val_batch_size,
                                    padded_shapes=[buffer_size,
                                                    tile_size_y,
                                                    tile_size_x,
                                                    no_channels],
                                    padding_values=-1.0).prefetch(32)

        windows_ds = windows_ds.map(lambda x: augment(x, augment_code))
        #######################################################################

        loss_value = val_step(windows_ds, y, val_batch_size)

        val_loss(loss_value)
    with writer.as_default():
        with tf.name_scope("training"):
            tf.summary.scalar('epoch_loss', train_loss.result(), step = epoch)
    with writer_val.as_default():
        with tf.name_scope("training"):
            tf.summary.scalar('epoch_loss', val_loss.result(), step = epoch)

    # Get average of all worker train losses
    all_train_loss = hvd.allreduce(train_loss.result())
    train_loss.reset_states()
    # Get average of all worker val. losses
    all_val_loss = hvd.allreduce(val_loss.result())
    val_loss.reset_states()
    if all_val_loss < session_data["best_loss"]:
        if hvd.rank() == 0: # Only save best
            print("Better validation loss - saving")
        session_data["best_loss"] = all_val_loss
        save_state(epoch) # Only save snapshots for improvements

    if epoch == nb_epochs:
        break
    epoch += 1
