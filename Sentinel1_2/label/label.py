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
sys.path.append('../../external/')
from label import OMNIBUS

class Synthetic_Label():
    @staticmethod
    def omnibus_chmap_dual(window, norm_by, sigma):
        import numpy as np

        chmap = np.zeros((window.shape[1], window.shape[2]), dtype=np.int64)

        omni = OMNIBUS(window.shape[1], window.shape[2], 4, sigma, window[0,:,:,0], None)

        for idx in range(1, len(window)):
            chmap_tmp, omni.PV(window[idx, :, :, 0], window[idx, :, :, 1])
            chmap = np.where(chmap_tmp == True, chmap + 1, chmap)

        return chmap/norm_by

    @staticmethod
    def ENDISI_S2_beta_coeff(window):
        import numpy as np
        import earthpy.spatial as es

        blue = np.nan_to_num(window[:,:,1], nan = 0.0)
        green = np.nan_to_num(window[:,:,2], nan = 0.0)
        mir = np.nan_to_num(window[:,:,11], nan = 0.0)
        mir2 = np.nan_to_num(window[:,:,12], nan = 0.0)

        mndwi = np.nan_to_num(es.normalized_diff(green, mir), nan = 0.0)
        mir_ratio = np.nan_to_num(mir/mir2, nan = 0.0)

        return [np.float32(np.mean(blue)), np.float32(np.mean(mir_ratio)), np.float32(np.mean(mndwi**2))]

    @staticmethod
    def ENDISI_S2(window, beta, A):
        import numpy as np
        import earthpy.spatial as es

        blue = np.nan_to_num(window[:,:,1], nan = 0.0)
        green = np.nan_to_num(window[:,:,2], nan = 0.0)
        red = np.nan_to_num(window[:,:,3], nan = 0.0)
        nir = np.nan_to_num(window[:,:,7], nan = 0.0)
        mir = np.nan_to_num(window[:,:,11], nan = 0.0)
        mir2 = np.nan_to_num(window[:,:,12], nan = 0.0)

        mndwi = np.nan_to_num(es.normalized_diff(green, mir), nan = 0.0)
        mir_ratio = np.nan_to_num(mir/mir2, nan = 0.0)
        res = np.nan_to_num(es.normalized_diff(blue, beta * (mir_ratio + mndwi**2)), nan = 0.0)
        res = np.where(res + A> 0.0, res + A, 0.0)

        mndbi = np.nan_to_num(es.normalized_diff(mir, blue), nan = 0.0)
        mndbi = np.where(mndbi > 0.0, mndbi, 0.0)
        mndwi = np.where(mndwi > 0.0, mndwi, 0.0)
        res = res - 2 * mndwi - 1*mndbi

        res = np.clip(res, 0.0, 1.0)
        return res

    @staticmethod
    def compute_label_S2_S1_ENDISI(s1_ascending, s1_descending, s2, s2_before, s2_after, shift, beta1, beta3):
        import math
        import numpy as np

        assert ((s1_ascending.shape[1] == s1_descending.shape[1] == s2.shape[1]) and \
               (s1_ascending.shape[2] == s1_descending.shape[2] == s2.shape[2])) ,     \
               "Tiles of the sources have different sizes ({}, {}, {}/{}, {}, {})".format( \
                   s1_ascending.shape[1], s1_descending.shape[1], s2.shape[1],      \
                   s1_ascending.shape[2], s1_descending.shape[2], s2.shape[2])

        s2_before_avg = np.mean(s2_before, axis=0)
        s2_before_ebbi = Generate_Label.ENDISI_S2(s2_before_avg, beta1, shift)

        s2_after_avg = np.mean(s2_after, axis=0)
        s2_after_ebbi = Generate_Label.ENDISI_S2(s2_after_avg, beta3, shift)

        s2_diff = np.abs(s2_after_ebbi - s2_before_ebbi)

        chmap_asc = Generate_Label.omnibus_chmap_dual(s1_ascending, s1_ascending.shape[0], 0.001)
        chmap_dsc = Generate_Label.omnibus_chmap_dual(s1_descending, s1_descending.shape[0], 0.001)

        chmap = (chmap_asc+chmap_dsc)/2
        return np.float32(np.clip(chmap * s2_diff * 10, 0.0, 1.0))

    @staticmethod
    def compute_label_S2_S1_ENDISI_beta_coeefs(s2_before, s2_after):
        import math
        import numpy as np

        s2_before_avg = np.mean(s2_before, axis=0)
        s2_before_coeffs = Generate_Label.ENDISI_S2_beta_coeff(s2_before_avg)
        s2_after_avg = np.mean(s2_after, axis=0)
        s2_after_coeffs = Generate_Label.ENDISI_S2_beta_coeff(s2_after_avg)
        return np.array([s2_before_coeffs, s2_after_coeffs])

    @staticmethod
    def compute_label_S2_S1_ENDISI_comp_betas(coeffs):
        blue_avg = 0.0
        swir_ratio = 0.0
        mndwi_avg = 0.0

        for tile in range(0, coeffs.shape[0]):
            blue_avg += coeffs[tile, 0]
            swir_ratio += coeffs[tile, 1]
            mndwi_avg += coeffs[tile, 2]

        beta = 2*blue_avg/(swir_ratio + mndwi_avg)
        return beta
