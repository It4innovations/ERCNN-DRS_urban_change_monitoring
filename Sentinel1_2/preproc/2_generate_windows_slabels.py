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
from multiprocessing import get_context

import os
os.environ["XDG_CACHE_HOME"]="/tmp"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Get rid of the pesky TF messages

import re
import math
import numpy as np
from datetime import datetime, timedelta, date
from datetime import timezone
import dateutil.parser
import tensorflow as tf
import pickle
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
import time
import sys
sys.path.append('../label/')
from label import Synthetic_Label


# PARAMETERS FOR TRAINING #####################################################
save_csv = True
sites={"Limassol":[0.5], "Rotterdam":[0.25], "Liege":[0.25]}
tf_record_file_path = "/data/data_temp/tstack_s12/"
tf_record_out_path = "/data/data_temp/training_s12/"
tf_record_file_out_train_path = tf_record_out_path + "/train/"
tf_record_file_out_val_path = tf_record_out_path + "/val/"
beta_coeffs_file = tf_record_out_path + "/beta_coeffs"

tile_size_x = 32
tile_size_y = 32

time_start = datetime(2017, 1, 1, 0, 0, 0) # When to start first window
time_range = 60*60*24*182 # Window observation period (in seconds) (S2: 182 days)
time_step = 60*60*24*2 # Every 2nd day

buffer_size = math.ceil(time_range / time_step) + 1

# ATTENTION:
# It can happen that duplicate windows are created with ==1! This is due to the
# period before does not have the same resolution as the period of interest.
window_shift = 1

min_obs = 35 # how many observations per window we need at least

num_processors = 2#36

tfr_options = tf.io.TFRecordOptions(compression_type="GZIP")
###############################################################################

def get_num_tiles(site):
    print("site: {}".format(site))

    tiles = [f for f in os.listdir(tf_record_file_path)
                 if os.path.isfile(os.path.join(tf_record_file_path, f)) and
                    f.startswith("{}_".format(site)) and
                    f.endswith(".tfrecords")]
    print("\ttotal tiles:", len(tiles))
    num_tiles_y = 0
    num_tiles_x = 0
    tile_pattern = re.compile("^{}_([0-9]+)_([0-9]+).tfrecords$".format(site))
    for tle in tiles:
        match = tile_pattern.match(tle)
        assert match, "TFRecord file does not have expected pattern: " + tle
        this_y = int(match[1])
        this_x = int(match[2])
        num_tiles_y = this_y if this_y > num_tiles_y else num_tiles_y
        num_tiles_x = this_x if this_x > num_tiles_x else num_tiles_x

    # Tiles are 0-based!
    num_tiles_y += 1
    num_tiles_x += 1
    print("\tnum tiles:", num_tiles_y,  num_tiles_x)
    return num_tiles_y, num_tiles_x

def _parse_tfr_element(element):
    parse_dic = {
        "Timestamp": tf.io.FixedLenFeature([], tf.string),
        "S1_ascending": tf.io.FixedLenFeature([], tf.string),
        "S1_descending": tf.io.FixedLenFeature([], tf.string),
        "S2": tf.io.FixedLenFeature([], tf.string)
    }
    example_message = tf.io.parse_single_example(element, parse_dic)

    timestamp_f = example_message["Timestamp"]
    timestamp = tf.ensure_shape(tf.io.parse_tensor(timestamp_f,
                                                   out_type=tf.float64), None)

    s1_ascending_f = example_message["S1_ascending"]
    s1_ascending = tf.ensure_shape(tf.io.parse_tensor(s1_ascending_f,
                                                      out_type=tf.float32),
                                   [tile_size_y, tile_size_x, 2])

    s1_descending_f = example_message["S1_descending"]
    s1_descending = tf.ensure_shape(tf.io.parse_tensor(s1_descending_f,
                                                       out_type=tf.float32),
                                    [tile_size_y, tile_size_x, 2])

    s2_f = example_message["S2"]
    s2 = tf.ensure_shape(tf.io.parse_tensor(s2_f,
                                            out_type=tf.float32),
                         [tile_size_y, tile_size_x, 13])

    return timestamp, s1_ascending, s1_descending, s2

def sub_to_batch(timestamp, s1_ascending, s1_descending, s2):
    def first_only(batch):
        return tf.broadcast_to(batch[0], [tf.size(batch)])

    sub_s2 = zip(timestamp, s2)
    ref = timestamp.batch(buffer_size * 3).map(first_only)

    # 1st period (S2 only)
    batch_1st = sub_s2.batch(buffer_size * 3)
    final_1st = tf.data.Dataset.zip((batch_1st, ref))
    filtered_1st = final_1st.unbatch() \
                    .filter(lambda x, y: tf.math.logical_and(
                                          tf.math.greater_equal(x[0], y),
                                          tf.math.less(x[0], y + time_range))) \
                    .map(lambda x, y: x[1]) \
                    .batch(buffer_size)

    # 3rd period (S2 only)
    batch_3rd = sub_s2.batch(buffer_size * 3)
    final_3rd = tf.data.Dataset.zip((batch_3rd, ref))
    filtered_3rd = final_3rd.unbatch() \
                    .filter(lambda x, y: tf.math.logical_and(
                                     tf.math.greater_equal(x[0],
                                                           y + time_range * 2),
                                     tf.math.less(x[0], y + time_range * 3))) \
                    .map(lambda x, y: x[1]) \
                    .batch(buffer_size)

    # Get 2nd period (all sources)
    sub = zip(timestamp, s1_ascending, s1_descending, s2)
    batch = sub.batch(buffer_size * 3)
    final = tf.data.Dataset.zip((batch, ref))
    filtered = final.unbatch() \
                    .filter(lambda x, y: tf.math.logical_and(
                                     tf.math.greater_equal(x[0],
                                                           y + time_range * 1),
                                     tf.math.less(x[0], y + time_range * 2))) \
                    .map(lambda x, y: x) \
                    .batch(buffer_size)

    return zip(filtered, filtered_1st, filtered_3rd)

def stack_layers(stack, s2_1, s2_3):
    return tf.concat([
                stack[1],
                stack[2],
                stack[3]], axis=-1)

def annotate_ds(file, shift, seed, just_timestamps = False,
                just_beta_coeffs = False, window_betas = None):
    def generate_label(first_stack, betas):
        stack = first_stack[0]
        s2_1 = first_stack[1]
        s2_3 = first_stack[2]
        beta1 = betas[0]
        beta3 = betas[1]
        return tf.ensure_shape(tf.numpy_function(
                                    Synthetic_Label.compute_label_S2_S1_ENDISI,
                                    [stack[1], stack[2], stack[3], s2_1, s2_3,
                                    shift, beta1, beta3], tf.float32),
                               [tile_size_y, tile_size_x])

    def generate_beta_coeffs(stack, s2_1, s2_3):
        return tf.ensure_shape(tf.numpy_function(
                          Synthetic_Label.compute_label_S2_S1_ENDISI_beta_coeefs,
                          [s2_1, s2_3], tf.float32),
                               [2, 3])

    def just_timestamps_map(stack, s2_1, s2_3):
        return stack[0]

    # Workaraound to get rid of warning reg. AUTOGRAPH
    filter_obs = tf.autograph.experimental.do_not_convert(
                        lambda stack, s2_1, s2_3: \
                        tf.math.logical_and( \
                            tf.math.logical_and( \
                                tf.shape(stack[3])[0] >= min_obs, \
                                tf.shape(s2_1)[0] >= min_obs), \
                                tf.shape(s2_3)[0] >= min_obs))

    # Workaraound to get rid of warning reg. AUTOGRAPH
    filter_rand = tf.autograph.experimental.do_not_convert(
                        lambda *args: \
                        tf.random.uniform([],
                                          0,
                                          10,
                                          dtype=tf.dtypes.int32,
                                          seed=seed) == 0)

    source = tf.data.TFRecordDataset(file,
                                     compression_type="GZIP",
                                     num_parallel_reads=4) \
            .map(_parse_tfr_element, num_parallel_calls=4) \
            .window(buffer_size * 3, shift=window_shift, stride=1) \
            .flat_map(sub_to_batch) \
            .filter(filter_obs) # Windows have to have #min_obs observations!

    if just_beta_coeffs:
        coeffs_ds = source.map(generate_beta_coeffs, num_parallel_calls=1)
        return coeffs_ds

    if just_timestamps == False:
        window_betas_ds = tf.data.Dataset.from_tensor_slices(window_betas)
        new_source = tf.data.Dataset.zip((source, window_betas_ds))
        # ATTENTION: It is mandatory that both filter_rand have the same seed!
        # Otherwise they will combine the wrong label with the data.
        filtered_label_source = new_source.filter(filter_rand)
        filtered_source = source.filter(filter_rand)
        label_ds = filtered_label_source.map(generate_label,
                                             num_parallel_calls=1)
        return tf.data.Dataset.zip((filtered_source.map(stack_layers,
                                                        num_parallel_calls=4),
                                    label_ds))
    else:
        return source.map(just_timestamps_map, num_parallel_calls=4)

def write_training_data(file, dataset):
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value]))

    tfr_options = tf.io.TFRecordOptions(compression_type="GZIP")
    this_tfr = tf.io.TFRecordWriter(file, options=tfr_options)

    for elem in dataset:
        feature = tf.io.serialize_tensor(elem[0][:, :, :, :])
        label = tf.io.serialize_tensor(elem[1][:, :])

        sample_record = {
            "Feature": _bytes_feature(feature),
            "Label": _bytes_feature(label)
        }

        sample = tf.train.Example(
                        features=tf.train.Features(feature=sample_record))
        this_tfr.write(sample.SerializeToString())

    this_tfr.close()


def write_train(args):
    import time
    fle = args[0]
    param = args[1]
    train_ds = annotate_ds(fle[0],
                           param,
                           fle[3],
                           just_timestamps = False,
                           just_beta_coeffs = False,
                           window_betas = fle[2])
    write_training_data(fle[1], train_ds)
    return fle

def write_val(args):
    import time
    fle = args[0]
    param = args[1]
    val_ds = annotate_ds(fle[0],
                         param,
                         fle[3],
                         just_timestamps = False,
                         just_beta_coeffs = False,
                         window_betas = fle[2])
    # only use 1/4th of the temporal data for validation
    val_ds = val_ds.shard(num_shards=4, index=0)
    write_training_data(fle[1], val_ds)
    return fle

def get_coeffs(args):
    import os
    import time

    coeffs_ds = annotate_ds(args[0][0],
                            args[1],
                            0,
                            just_timestamps = False,
                            just_beta_coeffs = True)

    window_coeff = []
    no_elem = 0
    for elem in coeffs_ds:
        window_coeff.append(elem)

    return_coeff = np.array(window_coeff)
    return (os.path.basename(args[0][0]), return_coeff)

if __name__ == '__main__':
    start_time = time.time()

    os.makedirs(tf_record_out_path, exist_ok=True)

    ########################################################
    # Save tiles to CSV
    if save_csv:
        import csv
        import datetime

        for site,params in sites.items():
            # Just one tile as representative...
            j = 0
            i = 0
            sample_file = tf_record_file_path + \
                          "{}_{}_{}.tfrecords".format(site, j, i)
            get_windows_ds = annotate_ds(sample_file,
                                         params[0],
                                         0,
                                         just_timestamps = True)

            with open(tf_record_out_path + \
                      "{}_windows.csv".format(site), mode = "w") as csv_file:
                csv_writer = csv.writer(csv_file,
                                        delimiter=',',
                                        quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)

                this_id = 0
                for item in get_windows_ds:
                    window_start = datetime.datetime.utcfromtimestamp(
                                                        item[0].numpy())
                    window_end = datetime.datetime.utcfromtimestamp(
                                                        item[-1].numpy())
                    window_amount = len(item.numpy())
                    csv_writer.writerow([this_id,
                                         window_start,
                                         window_end,
                                         window_amount])
                    this_id += 1
    ########################################################

    # Write partial beta coeffs...

    def get_betas(bcfile, filenames, param):
        import pickle
        with get_context("spawn").Pool(processes = num_processors) as p:
            output = p.map(get_coeffs, [(i, param) for i in filenames])

        print("Writing Pickle...", end="")
        outdict = dict((x, y.tolist()) for x, y in output)
        with open(bcfile, "wb") as handle:
            pickle.dump(outdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("...done")
    # ...end beta coeffs

    os.makedirs(tf_record_file_out_train_path, exist_ok=True)
    os.makedirs(tf_record_file_out_val_path, exist_ok=True)

    for site,params in sites.items():
        train_filenames = []
        val_filenames = []
        num_tiles_y, num_tiles_x = get_num_tiles(site)

        # Compute beta coeffs (only if not yet done)
        beta_coeffs_file_this = beta_coeffs_file + "_{}.pkl".format(site)
        if not os.path.isfile(beta_coeffs_file_this):
            coeff_filenames = []
            for j in range(0, num_tiles_y):
                for i in range(0, num_tiles_x):
                    sample_file = tf_record_file_path + \
                                  "{}_{}_{}.tfrecords".format(site, j, i)
                    if os.path.isfile(sample_file):
                        coeff_filenames.append(
                                (sample_file,
                                 tf_record_file_out_train_path + \
                                 "{}_{}_{}.tfrecords".format(site, j, i)))

            get_betas(beta_coeffs_file_this, coeff_filenames, params[0])

        # Load beta coeffs and compute full betas for each window
        # (a pair for obs. before and after)
        with open(beta_coeffs_file_this, "rb") as handle:
            beta_coeffs = pickle.load(handle)
        np_coeff = np.array(list(beta_coeffs.values()))
        window_betas_tmp = []
        for window_no in range(0, np_coeff.shape[1]):
            beta1 = Synthetic_Label.compute_label_S2_S1_ENDISI_comp_betas(
                                        np_coeff[:, window_no, 0, :]) # before
            beta2 = Synthetic_Label.compute_label_S2_S1_ENDISI_comp_betas(
                                        np_coeff[:, window_no, 1, :]) # after
            window_betas_tmp.append((beta1, beta2))
        window_betas = np.array(window_betas_tmp)
        print(window_betas)

        for j in range(0, num_tiles_y):
            for i in range(0, num_tiles_x):
                this_seed = 1000*j + i # every tile has it's own seed
                sample_file = tf_record_file_path + \
                              "{}_{}_{}.tfrecords".format(site, j, i)

                if (j + i/2) % 2 == 0:
                    if os.path.isfile(sample_file):
                        train_filenames.append(
                                    (sample_file,
                                     tf_record_file_out_train_path + \
                                     "{}_{}_{}.tfrecords".format(site, j, i),
                                     window_betas, this_seed))
                elif (j + (i+1)/2 + 1) % 4 == 0:
                    if os.path.isfile(sample_file):
                        val_filenames.append(
                                    (sample_file,
                                     tf_record_file_out_val_path + \
                                     "{}_{}_{}.tfrecords".format(site, j, i),
                                     window_betas,
                                     this_seed))

        print("{}: train: {}, val: {}".format(site,
                                              len(train_filenames),
                                              len(val_filenames)))

        with get_context("spawn").Pool(processes = num_processors) as p:
            output = p.map(write_train, [(i, params[0]) for i in train_filenames])
            output = p.map(write_val, [(i, params[0]) for i in val_filenames])

    print("--- %s seconds ---" % (time.time() - start_time))
