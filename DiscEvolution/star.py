# star.py
#
# Author: R. Booth
# Date: 8 - Nov - 2016
#  
# Contains stellar properties classes
################################################################################
import numpy as np
from .constants import Msun, Rsun, AU, hbar, k_B, yr, Omega0
from scipy.integrate import quad

# Base class for all stars, implements general properties that should be
# common to all stars
class StarBase(object):
    """Wrapper class for stellar properties.

    args:
        M     : mass, Msun.                  default = 1
        R     : radius, Rsun.                default = 1
        T_eff : effective temperature, K.    default = 5770
        age   : Stellar age, yr.             default = 0
    """
    def __init__(self, M=1,R=1,T_eff=5770, age=0):
        self._M    = M
        self._Rs   = R
        self._Rau  = R*Rsun/AU
        self._Teff = T_eff
        self._age = age

    def Omega_k(self,  r):
        """Keplerian angular speed of a test particle.

        args:
            r : distance, AU
        returns:
           Omega : 2 Pi AU / yr
        """
        return np.sqrt(self._M / (r*r*r))

    def v_k(self,  r):
        """Keplerian velocity of a test particle.

        args:
            r : distance, AU
        returns:
           Omega : 2 Pi AU / yr
        """
        return np.sqrt(self._M / r)

    def r_Hill(self, R, M):
        """Compute the hill radius of a planet

        args:
            R : radius, AU
            M : planet mass
        """
        return R * (M / (3*self._M))**(1/3.)

    def evolve(self, age, M=None):
        """Update the stellar properties based on current age and mass

        args:
           age : stellar age in Omega(1)^-1
           M   : mass, Msun
        """
        raise AttributeError("StarBase::Evolve must be implemented in "
                             "class")
        
    @property
    def Rs(self):
        """Radius is Rsun"""
        return self._Rs

    @property
    def Rau(self):
        """Radius in AU"""
        return self._Rau

    @property
    def M(self):
        """Mass in Msun"""
        return self._M

    @property
    def T_eff(self):
        """Effective temperature in K"""
        return self._Teff

    @property
    def age(self):
        """Stellar age"""
        return self._age

    def ASCII_header(self):
        """Print stellar type header"""
        head = '# {} M: {}Msun, R: {}Rsun, T: {}K, age: {}yr'
        return head.format(self.__class__.__name__,
                           self.M, self.Rs, self.T_eff, self.age)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "M"   : "{} Msun".format(self.M),
                                          "R"   : "{} Rsun".format(self._Rs),
                                           "T"   : "{} K".format(self.T_eff),
                                           "age" : "{} yr".format(self.age),
                                        }


# A simple non-evolving star class
class SimpleStar(StarBase):
    """Simple non-evolving star."""
    def __init__(self, **kwargs):
        super(SimpleStar, self).__init__(**kwargs)

    def evolve(self, age, M=None):
        """Update the stellar properties based on current age and mass

        args:
           age : stellar age in yr
           M   : mass, Msun
        """
        self._age = age
        if M is not None: self._M = M

    @staticmethod
    def from_string(string):
        """Read a simple star from a string"""
        string = string.replace('# SimpleStar', '').strip()
        kwargs = {}
        for item in string.split(','):
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'M':
                kwargs[key] = float(val.replace('Msun','').strip())
            elif key == 'R':
                kwargs[key] = float(val.replace('Rsun','').strip())
            elif key == 'T' or key == 'T_eff':
                kwargs['T_eff'] = float(val.replace('K','').strip())
            elif key == 'age':
                kwargs[key] = float(val.replace('yr', '').strip())
            else:
                raise AttributeError("Error: Attribute {} for SimpleStar not "
                                     "known".format(key))
        return SimpleStar(**kwargs)
                                     
# A star with a photoevaporating luminosity
class PhotoStar(SimpleStar):
    def __init__(self, LX=0, Phi=0, Tacc=12000, **kwargs):
        super().__init__(**kwargs)
        self._L_X = LX
        self._Phi = Phi
        self._LFUV = 0
        
        hc_kT = 2*np.pi*hbar*2.9979e10/k_B/Tacc
        self._facc_FUV = quad(self.Planck, hc_kT/2400e-8, hc_kT/912e-8)[0]/quad(self.Planck, 0, np.inf)[0]

    @property
    def L_X(self):
        """X-ray Luminosity"""
        return self._L_X

    @property
    def Phi(self):
        """EUV Photon Luminosity"""
        return self._Phi

    @property
    def LFUV(self):
        """FUV Luminosity from Accretion"""
        return self._LFUV
        
    def Planck(self, nu):
        return nu**3/(np.exp(nu)-1)
        
    def Mdot_to_FUV(self, Mdot):
        Lacc = 0.8*self.M*Msun/(self.Rs*Rsun)*6.674e-8*(Mdot*Msun/(yr/Omega0))
        return self._facc_FUV*Lacc
        
    def evolve(self, age, Mdot, M=None):
        """Update the stellar properties based on current age and mass

        args:
           age : stellar age in yr
           M   : mass, Msun
        """
        self._age = age
        if M is not None: self._M = M
        self._LFUV = self.Mdot_to_FUV(Mdot)


def from_file(filename):
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                raise AttributeError("Error: Star type not found in header")
            elif "SimpleStar" in line:
                return SimpleStar.from_string(line)
            else:
                continue
            
