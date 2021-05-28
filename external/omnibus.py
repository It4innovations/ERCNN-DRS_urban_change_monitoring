# MIT License
#
# This implementation was taken from https://github.com/mortcanty/SARDocker
#
# Original Author:
# Copyright (C) 2016 Mort Canty
#
# Modification:
# Copyright (C) 2020-2021 Georg Zitzlsberger, IT4Innovations,
#                         VSB-Technical University of Ostrava, Czech Republic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
class OMNIBUS():
    def __init__(self, sizex, sizey, enl, sigma, b1, b2):
        self.enl = enl
        self.sigma = sigma
        self.k = np.zeros((sizex, sizey), dtype=np.float64)
        self.xsi = np.zeros((sizex, sizey), dtype=np.float64)
        self.xsi1 = np.zeros((sizex, sizey), dtype=np.float64)
        self.cnt = np.zeros((sizex, sizey), dtype=np.float64)

        self.PV_init(b1, b2)

    def PV_init(self, b1, b2):
        import sys
        import numpy as np
        from scipy import stats

        self.cnt += 1.

        if b2 is not None:
            detsumj1 = self.k*self.xsi
            k1 = self.enl*np.float64(b1)
            self.xsi1 = self.enl*np.float64(b2)
            self.k += k1; self.xsi += self.xsi1
        else:
            detsumj1 = self.k.copy()
            k1 = self.enl*np.float64(b1)
            self.k += k1

    def PV(self, b1, b2):
        import sys
        import numpy as np
        from scipy import stats

        self.cnt += 1.
        eps = sys.float_info.min

        if b2 is not None:
            p = 1
            f = 2
            detsumj1 = self.k*self.xsi
            k1 = self.enl*np.float64(b1)
            self.xsi1 = self.enl*np.float64(b2)
            detj = k1*self.xsi1
            self.k += k1; self.xsi += self.xsi1
            detsumj = self.k*self.xsi
        else:
            p = 1
            f = 1
            detsumj1 = self.k.copy()
            k1 = self.enl*np.float64(b1)
            detj = k1
            self.k += k1
            detsumj = self.k.copy()

        detsumj = np.nan_to_num(detsumj)
        detsumj = np.where(detsumj <= eps,eps,detsumj)
        logdetsumj = np.log(detsumj)
        detsumj1 = np.nan_to_num(detsumj1)
        detsumj1 = np.where(detsumj1 <= eps,eps,detsumj1)
        logdetsumj1 = np.log(detsumj1)
        detj = np.nan_to_num(detj)
        detj = np.where(detj <= eps,eps,detj)
        logdetj = np.log(detj)

        lnRj = self.enl*(f*(self.cnt*np.log(self.cnt)-
                            (self.cnt-1)*np.log(self.cnt-1.)) +
                         (self.cnt-1)*logdetsumj1 + logdetj -
                         self.cnt*logdetsumj)
        rhoj = 1 - (2.*p**2 - 1)*
                   (1. + 1./(self.cnt*(self.cnt-1)))/(6.*p*self.enl)
        omega2j = -(f/4.)*(1.-1./rhoj)**2 +
                   (1./(24.*self.enl*self.enl))*p*p*(p*p-1)*
                     (1+(2.*self.cnt-1)/(self.cnt*(self.cnt-1))**2)/rhoj**2
        Z = -2*rhoj*lnRj

        res = 1.0 - ((1.-omega2j)*stats.chi2.cdf(Z,[f])+
                     omega2j*stats.chi2.cdf(Z,[f+4]))
        cmap = res <= self.sigma
        self.cnt = np.where(cmap, 1, self.cnt)
        self.k = np.where(cmap, k1, self.k)
        if b2 is not None:
            self.xsi1 = np.where(cmap, self.xsi1, self.xsi)
        return cmap
