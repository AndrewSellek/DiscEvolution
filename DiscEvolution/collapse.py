# collapse.py
#
# Author: A. Sellek
# Date: 27 - Jun- 2024
#
# Implementation of Collapse Models
################################################################################
import numpy as np
from DiscEvolution.constants import *

class CollapseBase(object):

    def __init__(self, disc, Menv=0.0, tenv=5e5, omega_cd=1e-14, T_cd=10):
        self._Menv = Menv
        self._tenv = tenv
        self._Mtot = self._Menv + disc.star.M + disc.Mtot()/Msun
        self._omega_cd = omega_cd
        self._T_cd = T_cd
        self._mu = disc.mu

        self._Mdot = self._Menv/self._tenv

    def source_profile(self):
        raise NotImplementedError("Must be defined in subclass")

    def __call__(self, disc, t, dt, dry=False):
        # Applies mass accretion and returns mass accretion rate
        if t>=self._tenv*yr:
            return 0
        else:
            if not dry:
                dSigma = self.source_profile(disc) * dt
                norm = np.nansum(dSigma/dt*2*np.pi*disc.R*np.gradient(disc.R))
                disc._Sigma[~np.isnan(dSigma)] += dSigma[~np.isnan(dSigma)] * self._Mdot * Msun/AU**2/yr / norm
            return self._Mdot*dt * Msun/yr

    def ASCII_header(self):
        return ("# Collapse, Menv: {}, tenv: {}, omega_cd: {}, T_cd: {}"
                "".format(self._Menv,self._tenv,self._omega_cd,self._T_cd))

    def HDF5_attributes(self):
        header = {}
        header['Menv'] = '{}'.format(self._Menv)
        header['tenv'] = '{}'.format(self._tenv)
        header['omega_cd'] = '{}'.format(self._omega_cd)
        header['T_cd'] = '{}'.format(self._T_cd)
        return self.__class__.__name__, header


class HuesoGuillot05(CollapseBase):

    def __init__(self, disc, **kwargs):
        super().__init__(disc, **kwargs)

    def Rcent(self, disc):
        # According to Shu 1977 prescription (Eq. 8)
        Macc = disc.star.M + disc.Mtot()/Msun
        return 53 * (self._omega_cd/1e-14)**2 * (self._T_cd/10)**-4 * (self._mu/2.2)**4 * Macc**3

    def source_profile(self, disc):
        # Eq. 6
        x = disc.R/self.Rcent(disc)
        return 1 / (8*np.pi) * 1/self.Rcent(disc)**2 * x**-1.5 * (1-x**0.5)**-0.5

