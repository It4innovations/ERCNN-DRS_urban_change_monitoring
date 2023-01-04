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
import tensorflow as tf
from tensorflow.keras import layers, activations

class ERCNN_DRS():
    def __init__(self, combined, sar):
        self.combined = combined
        self.sar = sar

    def reconf_mask(self, mask, ydim, xdim, filters):
        # Adjust mask to new last dimension (filters)
        mask = mask[:,:,0,0,0]
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)
        new_mask = tf.tile(mask, [1, 1, ydim, xdim, filters])
        return new_mask

    def conv_block(self, inp, filters, bias, timedist, mask, name):
        sub_model = tf.keras.Sequential(name=name+"_seq")
        sub_model.add(layers.Conv2D(filters=filters,
                                    kernel_size=(3,3),
                                    strides=(1,1),
                                    padding="SAME",
                                    use_bias=bias))
        sub_model.add(layers.BatchNormalization())
        sub_model.add(layers.Activation(activations.relu))

        output = layers.TimeDistributed(sub_model,
                                        name=name)(inputs=inp, mask=mask) \
                        if timedist else sub_model(inp)
        return output

    def conv_lstm(self, inp, filters, mask, name):
        convlstm = layers.ConvLSTM2D(
            filters=filters,
            kernel_size=(3,3),
            strides=(1,1),
            padding="SAME",
            use_bias=True,
            return_state=False,
            recurrent_dropout=0.4,
            name=name,
            return_sequences=False
            )(inputs=inp, mask=mask[:,:,0,0,0])
        return convlstm

    def path_opt(self, buffer_size, tile_size_y, tile_size_x, opt, mask, c):
        mask_opt = self.reconf_mask(mask, tile_size_y, tile_size_x, c)
        conv1 = self.conv_block(opt,
                                filters=c,
                                bias=True,
                                timedist=True,
                                mask=mask_opt,
                                name="conv_opt_1_" + str(c))
        conv2 = self.conv_block(conv1,
                                filters=c,
                                bias=True,
                                timedist=True,
                                mask=mask_opt,
                                name="conv_opt_2_" + str(c))

        mask_opt_2 = self.reconf_mask(mask, tile_size_y, tile_size_x, c)
        output = self.conv_lstm(conv2,
                                filters=c,
                                mask=mask_opt_2,
                                name="convlstm_opt_" + str(c))
        return output

    def path_sar(self, buffer_size, tile_size_y, tile_size_x, sar, mask, c):
        mask_sar = self.reconf_mask(mask, tile_size_y, tile_size_x, c)
        conv = self.conv_block(sar,
                               filters=c,
                               bias=True,
                               timedist=True,
                               mask=mask_sar,
                               name="conv_sar_" + str(c))

        mask_sar_2 = self.reconf_mask(mask, tile_size_y, tile_size_x, c)
        output = self.conv_lstm(conv,
                                filters=c,
                                mask=mask_sar_2,
                                name="convlstm_sar_" + str(c))
        return output

    def dispatch(self, buffer_size, tile_size_y, tile_size_x, sar, opt, mask):
        # Compose the sub-networks
        if sar != None:
            output_sar = self.path_sar(buffer_size,
                                       tile_size_y,
                                       tile_size_x,
                                       sar,
                                       mask,
                                       10)
        if opt != None:
            output_opt = self.path_opt(buffer_size,
                                       tile_size_y,
                                       tile_size_x,
                                       opt,
                                       mask,
                                       26)

        if not (sar is None) and not (opt is None):
            concat = layers.Concatenate(name="concat_sar_opt")([output_sar,
                                                                output_opt])
        elif not (sar is None):
            concat = output_sar
        elif not (opt is None):
            concat = output_opt

        conv1 = self.conv_block(concat,
                                filters=8,
                                bias=True,
                                timedist=False,
                                mask=None,
                                name="conv_concat_1_8")
        conv2 = self.conv_block(conv1,
                                filters=8,
                                bias=True,
                                timedist=False,
                                mask=None,
                                name="conv_concat_2_8")
        output = layers.Conv2D(filters=1,
                               kernel_size=(1,1),
                               strides=(1,1),
                               padding="VALID",
                               activation="sigmoid",
                               use_bias=True,
                               name="conv_sigmoid")(conv2)
        return layers.Reshape([tile_size_y, tile_size_x],
                              name="reshape_ebbi_sar_opt")(output)

    def build_model(self, buffer_size, tile_size_y, tile_size_x, channels):
        inp = tf.keras.Input(shape=[buffer_size,
                                    tile_size_y,
                                    tile_size_x,
                                    channels], dtype=tf.float32)

        mask = layers.Masking(mask_value=-1.0)(inp)

        if self.combined: # Create ensemble of SAR/optical sub-networks
            # Split: first 2 bands (SAR) + last 7 bands (optical)
            sar, opt = tf.split(inp, num_or_size_splits=[4, 13], axis=-1)
            output = self.dispatch(buffer_size,
                                   tile_size_y,
                                   tile_size_x,
                                   sar,
                                   opt,
                                   mask)
        else: # Create only SAR or optical sub-network
            if self.sar: # SAR only
                output = self.dispatch(buffer_size,
                                       tile_size_y,
                                       tile_size_x,
                                       inp,
                                       None,
                                       mask)
            else: # Optical only
                output = self.dispatch(buffer_size,
                                       tile_size_y,
                                       tile_size_x,
                                       None,
                                       inp,
                                       mask)

        return tf.keras.Model(inp, output)
