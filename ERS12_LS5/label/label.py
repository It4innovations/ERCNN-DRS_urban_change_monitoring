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
    def omnibus_chmap(window, norm_by, sigma):
        import numpy as np

        chmap = np.zeros((window.shape[1], window.shape[2]), dtype=np.int64)

        omni = OMNIBUS(window.shape[1], window.shape[2], 3, sigma, window[0,:,:,0], None)

        for idx in range(1, len(window)):
            chmap_tmp, omni.PV(window[idx, :, :, 0], None)
            chmap = np.where(chmap_tmp == True, chmap + 1, chmap)

        return chmap/norm_by

    @staticmethod
    def ENDISI_LS5_beta_coeff(window):
        import numpy as np
        import earthpy.spatial as es

        blue = np.nan_to_num(window[:,:,0], nan = 0.0)
        green = np.nan_to_num(window[:,:,1], nan = 0.0)
        mir = np.nan_to_num(window[:,:,4], nan = 0.0)
        mir2 = np.nan_to_num(window[:,:,6], nan = 0.0)

        mndwi = np.nan_to_num(es.normalized_diff(green, mir), nan = 0.0)
        mir_ratio = np.nan_to_num(mir/mir2, nan = 0.0)

        return [np.float32(np.mean(blue)), np.float32(np.mean(mir_ratio)), np.float32(np.mean(mndwi**2))]

    @staticmethod
    def ENDISI_LS5(window, beta, A):
        import numpy as np
        import earthpy.spatial as es

        blue = np.nan_to_num(window[:,:,0], nan = 0.0)
        green = np.nan_to_num(window[:,:,1], nan = 0.0)
        red = np.nan_to_num(window[:,:,2], nan = 0.0)
        nir = np.nan_to_num(window[:,:,3], nan = 0.0)
        mir = np.nan_to_num(window[:,:,4], nan = 0.0)
        mir2 = np.nan_to_num(window[:,:,6], nan = 0.0)

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
    def compute_label_LS5_ERS12_ENDISI(ers12_ascending, ers12_descending, ls5, ls5_before, ls5_after, shift, beta1, beta3):
        import math
        import numpy as np

        assert ((ers12_ascending.shape[1] == ers12_descending.shape[1] == ls5.shape[1]) and \
               (ers12_ascending.shape[2] == ers12_descending.shape[2] == ls5.shape[2])) ,     \
               "Tiles of the sources have different sizes ({}, {}, {}/{}, {}, {})".format( \
                   ers12_ascending.shape[1], ers12_descending.shape[1], ls5.shape[1],      \
                   ers12_ascending.shape[2], ers12_descending.shape[2], ls5.shape[2])

        ls5_before_avg = np.mean(ls5_before, axis=0)
        ls5_before_ebbi = Generate_Label.ENDISI_LS5(ls5_before_avg, beta1, shift)

        ls5_after_avg = np.mean(ls5_after, axis=0)
        ls5_after_ebbi = Generate_Label.ENDISI_LS5(ls5_after_avg, beta3, shift)

        ls5_diff = np.abs(ls5_after_ebbi - ls5_before_ebbi)

        chmap_asc = Generate_Label.omnibus_chmap(ers12_ascending, ers12_ascending.shape[0], 0.1)
        chmap_dsc = Generate_Label.omnibus_chmap(ers12_descending, ers12_descending.shape[0], 0.1)

        chmap = (chmap_asc+chmap_dsc)/2
        return np.float32(np.clip(chmap * ls5_diff * 30, 0.0, 1.0))

    @staticmethod
    def compute_label_LS5_ERS12_ENDISI_beta_coeefs(ls5_before, ls5_after):
        import math
        import numpy as np

        ls5_before_avg = np.mean(ls5_before, axis=0)
        ls5_before_coeffs = Generate_Label.ENDISI_LS5_beta_coeff(ls5_before_avg)

        ls5_after_avg = np.mean(ls5_after, axis=0)
        ls5_after_coeffs = Generate_Label.ENDISI_LS5_beta_coeff(ls5_after_avg)

        return np.array([ls5_before_coeffs, ls5_after_coeffs])

    @staticmethod
    def compute_label_LS5_ERS12_ENDISI_comp_betas(coeffs):
        blue_avg = 0.0
        swir_ratio = 0.0
        mndwi_avg = 0.0

        for tile in range(0, coeffs.shape[0]):
            blue_avg += coeffs[tile, 0]
            swir_ratio += coeffs[tile, 1]
            mndwi_avg += coeffs[tile, 2]

        beta = 2*blue_avg/(swir_ratio + mndwi_avg)
        return beta
