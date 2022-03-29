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

import sys
import math
import multiprocessing
import time
import numpy as np
from datetime import datetime, timedelta, date
from datetime import timezone
import dateutil.parser
import tensorflow as tf
from eolearn.core import EOPatch
import matplotlib.pyplot as plt
import skimage.transform
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

# PARAMETERS FOR TRAINING #####################################################
no_workers = 10 # for each site

sites=["Liege", "Rotterdam", "Limassol"]

data_dir_root_LS5 = "/data/eo_data/LS5/"
data_dir_root_ERS12 = "/data/eo_data/ERS12/"
tf_record_file_path = "/data/data_temp/tstack_ls5ers12/"

tile_size_x = 32
tile_size_y = 32

time_start = datetime(1991, 1, 1, 0, 0, 0) # When to start first window
time_range = 60*60*24*365 # Window observation period (in seconds) (LS5: 356 days)
time_step = 1 # Every observation

buffer_size = 110
###############################################################################

def getEOPatches(sites, data_dir_root, postfix = ""):
    all_files = {}
    for site in sites:
        all_files["{}".format(site)] = []
        root = data_dir_root + \
               "{}/".format(site) + \
               "eopatches{}/".format(postfix)

        list_patches = [dateutil.parser.parse(name) \
                        for name in os.listdir(root) \
                        if os.path.isdir('{}{}'.format(root, name))]
        list_patches.sort()

        time_patches = []
        for patch_dt in list_patches:
            new_datetime = patch_dt.strftime("%Y%m%dT%H%M%S")
            if os.path.exists(root + new_datetime):
                all_files["{}".format(site)].append(
                                (patch_dt, [root + new_datetime]))
    return all_files

def writer_proc(site, num_tiles_y, num_tiles_x, queues):
    def _floats_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    import numpy as np
    import collections

    tfr_options = tf.io.TFRecordOptions(compression_type="GZIP")
    tfr_tile_files = []
    for j in range(0, num_tiles_y):
        tfr_tile_files.append([])
        for i in range(0, num_tiles_x):
            tfr_tile_files[j].append(
                    tf.io.TFRecordWriter(tf_record_file_path + \
                        "{}_{}_{}.tfrecords".format(site, j, i),
                    options=tfr_options))

    cur_idx = 0
    buffer = np.full(len(queues), -1)
    while True:
        for idx, queue in enumerate(queues):
            if buffer[idx] == -1:
                if (queue.full()):
                    buffer[idx] = queue.get()

        if np.all(buffer == -2):
            break

        while np.where(buffer==cur_idx)[0].size != 0:
            assert np.where(buffer==cur_idx)[0].size == 1, \
                   "Same ID more than once!"

            print("Writing: ", cur_idx, " ({})".format(site))
            this_idx = int(np.where(buffer==cur_idx)[0])
            (ts, \
             prev_frame_ERS12_ascending_t, \
             prev_frame_ERS12_descending_t, \
             prev_frame_LS5_t) = queues[this_idx].get()

            # Write all tiles
            tf_ts = ts.replace(tzinfo=timezone.utc).timestamp()
            ts_serial=tf.io.serialize_tensor(tf.convert_to_tensor(tf_ts, tf.float64))

            for j in range(0, num_tiles_y):
                for i in range(0, num_tiles_x):
                    ers12_ascending_serial=tf.io.serialize_tensor(
                            prev_frame_ERS12_ascending_t[j, i, :, :, :])
                    ers12_descending_serial=tf.io.serialize_tensor(
                            prev_frame_ERS12_descending_t[j, i, :, :, :])
                    ls5_serial=tf.io.serialize_tensor(
                            prev_frame_LS5_t[j, i, :, :, :])

                    sample_record = {
                        "Timestamp": _bytes_feature(ts_serial),
                        "ERS12_ascending": _bytes_feature(ers12_ascending_serial),
                        "ERS12_descending": _bytes_feature(ers12_descending_serial),
                        "LS5": _bytes_feature(ls5_serial)
                    }

                    sample = tf.train.Example(
                            features=tf.train.Features(feature=sample_record))
                    tfr_tile_files[j][i].write(sample.SerializeToString())

            buffer[this_idx] = -1
            cur_idx += 1

    print("\t\t Wrote: {} samples ({})".format(cur_idx, site))
    # Close all TFRecords
    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            tfr_tile_files[j][i].close()

def worker_proc(queue, writer_queue):
    # Tile all streams
    def tile_stream(stream, height, width, tile_size_y, tile_size_x, channels):
        stream_t = tf.image.extract_patches(
                tf.reshape(stream, [1, height, width, channels]), \
                           sizes=[1, tile_size_y, tile_size_x, 1], \
                           strides=[1, tile_size_y, tile_size_x, 1], \
                           rates=[1, 1, 1, 1], \
                           padding='VALID')
        cor_stream_t = tf.reshape(stream_t, \
                                  [stream_t.shape[1], \
                                   stream_t.shape[2], \
                                   tile_size_y, \
                                   tile_size_x, \
                                   -1])
        return cor_stream_t

    while True:
        import time
        import random

        (order,
         ts,
         prev_frame_ERS12_ascending,
         prev_frame_ERS12_descending,
         prev_frame_LS5,
         new_ers12_asc,
         new_ers12_dsc,
         new_ls5,
         min_height_ERS12,
         min_width_ERS12,
         min_height_LS5,
         min_width_LS5) = queue.get()

        if (order == -2):
            writer_queue.put(-2)
            break

        # Try to only compute what's really changed to speed up processing
        if new_ers12_asc:
            prev_frame_ERS12_ascending_t = tile_stream(
                    prev_frame_ERS12_ascending,
                    min_height_ERS12,
                    min_width_ERS12,
                    tile_size_y,
                    tile_size_x,
                    1)

        if new_ers12_dsc:
            prev_frame_ERS12_descending_t = tile_stream(
                    prev_frame_ERS12_descending,
                    min_height_ERS12,
                    min_width_ERS12,
                    tile_size_y,
                    tile_size_x,
                    1)

        if new_ls5:
            # NN interpolation to ERS-1/2 resolution
            prev_frame_LS5_scaled = skimage.transform.resize(
                                      prev_frame_LS5[:, :, :],
                                      (min_height_ERS12, min_width_ERS12, 7),
                                      order=0)
            prev_frame_LS5_t = tile_stream(
                    prev_frame_LS5_scaled,
                    min_height_ERS12,
                    min_width_ERS12,
                    tile_size_y,
                    tile_size_x,
                    7)

        writer_queue.put(order)
        writer_queue.put((ts,
                          prev_frame_ERS12_ascending_t,
                          prev_frame_ERS12_descending_t,
                          prev_frame_LS5_t))

def process_site(site):
    def get_min_resolution(site):
        min_height_LS5 = 9999
        min_width_LS5 = 9999
        min_height_ERS12_ascending = 9999
        min_width_ERS12_ascending = 9999
        min_height_ERS12_descending = 9999
        min_width_ERS12_descending = 9999

        for fle in all_files_LS5["{}".format(site)]:
            ptch = EOPatch.load(fle[1][0])
            height = ptch.data['Bands'].shape[1]
            width = ptch.data['Bands'].shape[2]
            if min_height_LS5 > height:
                min_height_LS5 = height
            if min_width_LS5 > width:
                min_width_LS5 = width

        for fle in all_files_ERS12_ascending["{}".format(site)]:
            ptch = EOPatch.load(fle[1][0])
            height = ptch.data['intensity'].shape[1]
            width = ptch.data['intensity'].shape[2]
            if min_height_ERS12_ascending > height:
                min_height_ERS12_ascending = height
            if min_width_ERS12_ascending > width:
                min_width_ERS12_ascending = width

        for fle in all_files_ERS12_descending["{}".format(site)]:
            ptch = EOPatch.load(fle[1][0])
            height = ptch.data['intensity'].shape[1]
            width = ptch.data['intensity'].shape[2]
            if min_height_ERS12_descending > height:
                min_height_ERS12_descending = height
            if min_width_ERS12_descending > width:
                min_width_ERS12_descending = width

        min_height_ERS12 = min(min_height_ERS12_ascending, min_height_ERS12_descending)
        min_width_ERS12 = min(min_width_ERS12_ascending, min_width_ERS12_descending)
        return min_height_LS5, min_width_LS5, min_height_ERS12, min_width_ERS12

    # vvv Initialize vvv
    print("Site: {}".format(site))

    all_files_LS5 = getEOPatches(sites, data_dir_root_LS5, postfix = "_toa")
    all_files_ERS12_ascending = getEOPatches(sites,
                                          data_dir_root_ERS12,
                                          postfix = "_ascending")
    all_files_ERS12_descending = getEOPatches(sites,
                                           data_dir_root_ERS12,
                                           postfix = "_descending")
    list_time_stamps = []
    for typ in all_files_LS5["{}".format(site)] +           \
               all_files_ERS12_ascending["{}".format(site)] + \
               all_files_ERS12_descending["{}".format(site)]:

        match_idx = [idx for idx, x in enumerate(list_time_stamps) \
                     if x[0] == typ[0]]

        if match_idx:
            assert len(match_idx) < 2, "Too many matches (>1)???"
            list_time_stamps[match_idx[0]][1].append(typ[1])
        else:
            list_time_stamps.append(typ)

    print("\tTotal time stamps: {}".format(len(list_time_stamps)))
    list_time_stamps.sort(key=lambda tup: tup[0])

    min_height_LS5, min_width_LS5, \
        min_height_ERS12, min_width_ERS12 = get_min_resolution(site)

    # Just for testing (smaller subset)
    #min_height_LS5=min_width_LS5=min_height_ERS12=min_width_ERS12 = 200

    print("\tResolutions:")
    print("\t\tLS5: {}, {}".format(min_height_LS5, min_width_LS5))
    print("\t\tERS12: {}, {}".format(min_height_ERS12, min_width_ERS12))

    list_time_stamps = []
    for typ in all_files_LS5["{}".format(site)] +             \
               all_files_ERS12_ascending["{}".format(site)] + \
               all_files_ERS12_descending["{}".format(site)]:

        match_idx = [idx for idx, x in enumerate(list_time_stamps) \
                     if x[0] == typ[0]]

        if match_idx:
            assert False, "Should never be used!"
            assert len(match_idx) < 2, "Too many matches (>1)???"
            list_time_stamps[match_idx[0]][1].append(typ[1])
        else:
            list_time_stamps.append(typ)

    print("\tTotal time stamps (ignoring time_start): {}".format(
                                                len(list_time_stamps)))
    list_time_stamps.sort(key=lambda tup: tup[0])

    # Run through once to identify where to set the steps
    start_cur_window = time_start
    prev_ts = None
    eff_windows = 0
    window_list = []
    for ts in list_time_stamps:
        if ts[0] < start_cur_window: # only for the beginning (filt. prev. obs.)
            continue
        if ts[0] - start_cur_window >= timedelta(seconds=time_step):
            eff_windows += 1
            off_by = (ts[0] - start_cur_window)//timedelta(seconds=time_step)
            start_cur_window += off_by * timedelta(seconds=time_step)
            if prev_ts != None: # only for the beginning (no prev. observation)
                window_list.append(prev_ts)
        prev_ts = ts[0]
    if prev_ts != None and not prev_ts in window_list:
        eff_windows += 1
        window_list.append(prev_ts)

    print("\t\tEffective time stamps " +
          "(from time_start with steps of time_step): {}".format(eff_windows))

    prev_frame_LS5 = np.zeros((min_height_LS5,
                               min_width_LS5,
                               7), dtype=np.float32)
    prev_frame_ERS12_ascending = np.zeros((min_height_ERS12,
                                           min_width_ERS12,
                                           1), dtype=np.float32)
    prev_frame_ERS12_descending = np.zeros((min_height_ERS12,
                                            min_width_ERS12,
                                            1), dtype=np.float32)

    num_tiles_y = min_height_ERS12//tile_size_y
    num_tiles_x = min_width_ERS12//tile_size_x
    # ^^^ Initialize ^^^

    # vvv Setup multiprocessing vvv
    m = multiprocessing.Manager()
    writer_queues = [m.Queue(maxsize=1) for _ in range(no_workers)]

    workerqueue = multiprocessing.Queue(maxsize=no_workers)
    workerprocesses = []
    for i in range(no_workers): workerprocesses.append( \
            multiprocessing.Process(target=worker_proc,
                                    args=(workerqueue, writer_queues[i],)))
    for wproc in workerprocesses:
        wproc.start()

    writer_p = multiprocessing.Process(target=writer_proc,
                                       args=((site,
                                              num_tiles_y,
                                              num_tiles_x,
                                              writer_queues,)))
    writer_p.daemon = True
    writer_p.start()
    # ^^^ Setup multiprocessing ^^^

    # vvv Iterate over samples vvv
    idx = 0
    new_ers12_asc = True
    new_ers12_dsc = True
    new_ls5 = True
    for item in list_time_stamps:
        #print(".", end="")
        for ptch in item[1]:
            if ptch.find("/ERS12/") != -1:
                #print("ERS12")
                if ptch.find("/eopatches_ascending/") != -1:
                    #print("\tascending")
                    new_patch = EOPatch.load(ptch)
                    new_frame = np.where( \
                        new_patch.mask['IS_DATA'] \
                                      [0, :min_height_ERS12, :min_width_ERS12, 0:1],
                        new_patch.data['intensity'] \
                                      [0, :min_height_ERS12, :min_width_ERS12, 0:1],
                        prev_frame_ERS12_ascending[:min_height_ERS12, \
                                                :min_width_ERS12, 0:1])
                    prev_frame_ERS12_ascending = new_frame
                    new_ers12_asc = True
                elif ptch.find("/eopatches_descending/") != -1:
                    #print("\tdescending")
                    new_patch = EOPatch.load(ptch)
                    new_frame = np.where(
                        new_patch.mask['IS_DATA'] \
                                      [0, :min_height_ERS12, :min_width_ERS12, 0:1],
                        new_patch.data['intensity'] \
                                      [0, :min_height_ERS12, :min_width_ERS12, 0:1],
                        prev_frame_ERS12_descending[:min_height_ERS12, \
                                                 :min_width_ERS12, 0:1])
                    prev_frame_ERS12_descending = new_frame
                    new_ers12_dsc = True
                else:
                    assert False, "Unknown ERS12 type"
            elif ptch.find("/LS5/") != -1:
                #print("LS5")
                new_patch = EOPatch.load(ptch)
                new_frame = np.where(
                        np.logical_not(
                            new_patch.mask['QA'] \
                                          [0, \
                                           :min_height_LS5, \
                                           :min_width_LS5, \
                                           0:7] & 0b10101011100),
                            new_patch.data['Bands'] \
                                          [0,
                                           :min_height_LS5, \
                                           :min_width_LS5, \
                                           0:7],
                prev_frame_LS5[:min_height_LS5, :min_width_LS5, 0:7])
                prev_frame_LS5 = new_frame
                new_ls5 = True
            else:
                assert False, "Unknown type"

        # Only write end of time_step
        if item[0] not in window_list:
            continue

        workerqueue.put((idx,
                         item[0],
                         prev_frame_ERS12_ascending,
                         prev_frame_ERS12_descending,
                         prev_frame_LS5,
                         new_ers12_asc,
                         new_ers12_dsc,
                         new_ls5,
                         min_height_ERS12,
                         min_width_ERS12,
                         min_height_LS5,
                         min_width_LS5))
        idx += 1
        new_ers12_asc = True
        new_ers12_dsc = True
        new_ls5 = True
    # ^^^ Iterate over samples ^^^

    # Close down workers/writer
    print("Total writes: ", idx)
    for _ in range(no_workers): workerqueue.put((-2,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None,
                                                 None))
    while any(wproc.is_alive() for wproc in workerprocesses):
            time.sleep(1)
    print("Finished")
    writer_p.join()

if __name__ == '__main__':
    startt = time.time()

    os.makedirs(tf_record_file_path, exist_ok=True)

    my_site_processes = []
    for site in sites: my_site_processes.append(
                    multiprocessing.Process(target=process_site, args=(site,)))
    for site_proc in my_site_processes:
        site_proc.start()
    while any(site_proc.is_alive() for site_proc in my_site_processes):
            time.sleep(1)
    print("Processing took {} seconds".format(time.time() - startt))
