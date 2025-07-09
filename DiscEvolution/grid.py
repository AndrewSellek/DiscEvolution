# grid.py
#
# Author: R.Booth
# Date: 8 - Nov - 2016
#
# A simple 1D grid with log or power-law spacings. 
#
################################################################################
from __future__ import print_function
import numpy as np

class Grid(object):
    """Construct a simple 1D grid with different spacings"""
    def __init__(self, R0, R1, N, spacing='log', refine=None):

        if spacing == 'log':
            self._setup_log(R0, R1, N)
        elif spacing == 'linear':
            self._setup_powerlaw(R0, R1, N, 1.0)
        elif spacing == 'natural':
            self._setup_powerlaw(R0, R1, N, 0.5)
        elif spacing == 'naturalRefine':
            self._refine_powerlaw(R0, R1, N, 0.5, refine)
        else:
            try:
                self._setup_powerlaw(R0, R1, N, float(spacing))
            except ValueError:
                raise AttributeError("Spacing must be a power law index, or "
                                     "one of 'log', 'linear' and 'natural'")
        self._setup_aux()
        self._N = len(self._Rc)

        self._R0 = R0
        self._R1 = R1
        self._spacing = spacing
            
    def _setup_log(self, R0, R1, N):
        """Setup a grid in log-spacing"""
        lgR0 = np.log10(R0)
        lgR1 = np.log10(R1)
        
        dlogR = (lgR1 - lgR0) / N
        
        Ree = np.power(10, lgR0 + np.arange(-2, N+3, dtype='f8') * dlogR)
        
        self._Re  = Ree[2:-2]
        self._Rce = np.sqrt(Ree[2:-1] * Ree[1:-2])
        self._Rc  = self._Rce[1:-1]

        self._Ree = Ree

    def _setup_powerlaw(self, R0, R1, N, alpha):
        """Setup a power law grid"""
        alpha = float(alpha)
        alpha1 = 1/alpha

        R0a = R0**alpha
        R1a = R1**alpha

        dRa = (R1a - R0a) / N

        Ree_a = R0a + np.arange(-2, N+3, dtype='f8') * dRa
        Rce_a = 0.5*(Ree_a[2:-1] + Ree_a[1:-2])

        Ree = Ree_a**alpha1
        Rce = Rce_a**alpha1

        self._Re  = Ree[2:-2]
        self._Rce = Rce
        self._Rc  = Rce[1:-1]

        self._Ree = Ree

    def _refine_powerlaw(self, R0, R1, N, alpha, refine):
        """Setup a power law grid with refinement region"""
        # Refine = (inner radius, outer radius, increase in resolution)
        alpha = float(alpha)
        alpha1 = 1/alpha
        R0a = R0**alpha
        R1a = R1**alpha
        dRa = (R1a - R0a) / N

        # Coarse grid
        Ree_a = R0a + np.arange(-2, N+3, dtype='f8') * dRa  # Length N+5, edges inc two ghost on each end
        #Rce_a = 0.5*(Ree_a[2:-1] + Ree_a[1:-2]) # Length N+2, centres inc one ghost on each end
        Ree = Ree_a**alpha1
        #Rce = Rce_a**alpha1

        # Refining region
        RR0 = Ree[(Ree<refine[0])][-1]
        RR1 = Ree[(Ree>refine[1])][0]
        RR0a = RR0**alpha
        RR1a = RR1**alpha
        NR = int((RR1a - RR0a) / dRa * refine[2])
        dRRa = (RR1a - RR0a) / NR
        RRee_a = RR0a + np.arange(1, NR, dtype='f8') * dRRa # Length NR-1, edges

        # Combine into final grid
        Ree_a = np.concatenate((Ree_a[(Ree<refine[0])], RRee_a, Ree_a[(Ree>refine[1])]))
        Rce_a = 0.5*(Ree_a[2:-1] + Ree_a[1:-2]) # Length N+2, centres inc one ghost on each end
        Ree = Ree_a**alpha1
        Rce = Rce_a**alpha1

        # Store
        self._Re  = Ree[2:-2]   # Edges only in grid; length N+1
        self._Rce = Rce         # All centres
        self._Rc  = Rce[1:-1]   # Centres only in grid; length N
        self._Ree = Ree         # All edges

    def _setup_aux(self):
        self._dRe  = np.diff(self._Re)
        self._dRc  = np.diff(self._Rc)
        self._dRce = np.diff(self._Rce)

        self._dRe2  = np.diff(self._Re**2)
        self._dRc2  = np.diff(self._Rc**2)
        self._dRce2 = np.diff(self._Rce**2)

    def ASCII_header(self):
        """Write grid info header"""
        head = '# {} R0: {}, R1: {}, N: {}, spacing: {}'
        return head.format(self.__class__.__name__,
                           self._R0, self._R1, self.Ncells, self._spacing)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        def fmt(x):  return "{}".format(x)
        return self.__class__.__name__, { "R0" : fmt(self._R0),
                                          "R1" : fmt(self._R1),
                                          "N"  : fmt(self.Ncells),
                                          "spacing" : fmt(self._spacing),
                                          }
        
    @property
    def Rc(self):
        return self._Rc
    @property
    def Re(self):
        return self._Re
    @property
    def Rce(self):
        return self._Rce
    @property
    def Ree(self):
        return self._Ree

    @property
    def dRc(self):
        return self._dRc
    @property
    def dRe(self):
        return self._dRe
    @property
    def dRce(self):
        return self._dRce

    @property
    def dRc2(self):
        return self._dRc2
    @property
    def dRe2(self):
        return self._dRe2
    @property
    def dRce2(self):
        return self._dRce2

    @property
    def Ncells(self):
        return self._N

    def interp_centre(self, R, data):
        """Interpolate data to new radii

        args:
            R    : new radii
            data : data defined at grid centres
        """
        return np.interp(R, self.Rc, data)


    def interp_edges(self, R, data):
        """Interpolate data to new radii

        args:
            R    : new radii
            data : data defined at grid edges
        """
        return np.interp(R, self.Re, data)


    @staticmethod
    def from_string(string):
        """Read a Grid from a string"""
        string = string.replace('# Grid', '').strip()
        kwargs = {}
        args = [None, None, None]
        for item in string.split(','):
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'R0':
                args[0] = float(val)
            elif key == 'R1':
                args[1] = float(val)
            elif key == 'N':
                args[2] = int(val)
            elif key == 'spacing':
                kwargs[key] = val
            else:
                raise AttributeError("Error: Attribute {} for Grid not "
                                     "known".format(key))
        return Grid(*args, **kwargs)

def from_file(filename):
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                raise AttributeError("Error: Grid type not found in header")
            elif "Grid" in line:
                return Grid.from_string(line)
            else:
                continue
