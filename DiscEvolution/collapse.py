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

        if disc.chem:
            self.chem = {k: v[0] for k, v in zip(disc.chem.ice.names, disc.chem.ice.data+disc.chem.gas.data)}
        if hasattr(disc, "integ_dust_frac"):
            self._dust_frac = disc.integ_dust_frac[0]

    def source_profile(self):
        raise NotImplementedError("Must be defined in subclass")

    def max_timestep(self, disc, t):
        raise NotImplementedError("Must be defined in subclass")

    def __call__(self, disc, t, dt, dry=False):
        # Applies mass accretion and returns mass accretion rate
        if t>=self._tenv*yr:
            Macc_return = 0
        else:
            if not dry:
                dSigma = self.source_profile(disc) * dt
                print(np.argwhere(np.isnan(np.log10(disc.Sigma_G))), disc.Sigma[np.argwhere(np.isnan(np.log10(disc.Sigma_G)))], disc._eps[:,np.argwhere(np.isnan(np.log10(disc.Sigma_G)))])
                norm = np.sum(dSigma/dt*2*np.pi*disc.R*np.gradient(disc.R))
                dSS = dSigma * self._Mdot * Msun/AU**2/yr / norm / disc._Sigma
                if disc.chem:
                    for k in self.chem.keys():
                        disc.chem.ice[k] += (self.chem[k]-disc.chem.ice[k])*dSS/(1+dSS)
                elif hasattr(disc, "integ_dust_frac"):
                    disc._eps[0] += (self._dust_frac*(1-disc._fm)-disc._eps[0])*dSS/(1+dSS)
                    disc._eps[1] += (self._dust_frac*disc._fm-disc._eps[1])*dSS/(1+dSS)
                disc._Sigma += dSigma * self._Mdot * Msun/AU**2/yr / norm
                print(np.argwhere(np.isnan(np.log10(disc.Sigma_G))), disc.Sigma[np.argwhere(np.isnan(np.log10(disc.Sigma_G)))], disc._eps[:,np.argwhere(np.isnan(np.log10(disc.Sigma_G)))])
                if len(np.argwhere(np.isnan(np.log10(disc.Sigma_G))))>0:
                    raise Exception
            Macc_return = self._Mdot*dt * Msun/yr
        return Macc_return

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

    def max_timestep(self, disc, t):
        if t>=self._tenv*yr:
            return np.inf
        else:
            Macc = disc.star.M + disc.Mtot()/Msun
            return 1/2*np.nanmin(1/6 * disc.grid.dRe/(53 * (self._omega_cd/1e-14)**2 * (self._T_cd/10)**-4 * (self._mu/2.2)**4 * Macc**2)/(self._Mdot/yr))

    def source_profile(self, disc):
        # Eq. 6
        x = disc.R/self.Rcent(disc)
        profile = 1 / (8*np.pi) * 1/self.Rcent(disc)**2 * x**-1.5 * (1-x**0.5)**-0.5
        profile[np.isnan(profile)] = 0.0
        return profile

