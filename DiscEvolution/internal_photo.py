# internal_photo.py
#
# Author: A. Sellek
# Date: 12 - Aug - 2020
#
# Implementation of Photoevaporation Models
################################################################################
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from DiscEvolution.constants import *
from DiscEvolution.star import PhotoStar
from scipy.signal import argrelmin

class NotHoleError(Exception):
    """Raised if finds an outer edge, not a hole"""
    pass

class PhotoBase():
    def __init__(self, disc, Regime=None, Type=None, Mdot=None):
        # Basic mass loss properties
        self._regime = Regime   # EUV or X-ray
        self._type   = Type     # 'Primordial' or 'InnerHole'
        self._Sigmadot = np.zeros_like(disc.R)
        self.mdot_XE(disc.star, Mdot=Mdot)

        # Evolutionary state flags
        self._Hole  = False     # Has the hole started to open?
        self._reset = False     # Have we needed to reset a decoy hole?
        self._empty = False     # When no longer a valid hole radius or all below density threshold
        self._Thin  = False     # Is the hole exposed (ie low column density to star)? 

        # Parameters of hole
        self._R_hole = None
        self._N_hole = None
        # The column density threshold below which the inner disc is "Thin"
        if self._regime=='X-ray':
            self._N_crit = 1e22
        elif self._regime=='EUV':
            self._N_crit = 1e18
        elif self._regime=='FUV':
            self._N_crit = 0.0      # No TD regime implemented for FUV
        else:
            self._N_crit = 0.0      # (if 0, can never switch)

        # Outer radius
        self._R_out = max(disc.R_edge)
        
    def mdot_XE(self, star, Mdot=0):
        # Generic wrapper for initiating X-ray or EUV mass loss 
        # Without prescription, mass loss is 0
        self._Mdot      = Mdot
        self._Mdot_true = Mdot

    def Sigma_dot(self, R, star):
        if self._type=='Primordial':
            self.Sigma_dot_Primordial(R, star)
        elif self._type=='InnerHole':
            self.Sigma_dot_InnerHole(R, star)

    def Sigma_dot_Primordial(self, R, star, ret=False):
        # Without prescription, mass loss is 0
        if ret:
            return np.zeros(len(R)+1)
        else:
            self._Sigmadot = np.zeros_like(R)

    def Sigma_dot_InnerHole(self, R, star, ret=False):
        # Without prescription, mass loss is 0
        if ret:
            return np.zeros(len(R)+1)
        else:
            self._Sigmadot = np.zeros_like(R)

    def scaled_R(self, R, star):
        # Prescriptions may rescale the radius variable 
        # Without prescription, radius is unscaled
        return R

    def R_inner(self, star):
        # Innermost mass loss
        return 0

    def check_dt(self, disc, dt):
        # Work out the timescale to clear cell
        where_photoevap = (self.dSigmadt > 0)
        t_w = np.full_like(disc.R,np.inf)
        t_w[where_photoevap] = disc.Sigma_G[where_photoevap] / self.dSigmadt[where_photoevap]

        # Return minimum value for cells inside outer edge        
        indisc = (disc.R < self._R_out) * where_photoevap   # Prohibit hole outside of mass loss region.
        try:
            imin = argrelmin(t_w[indisc])[0][0] # Find local minima in clearing time, neglecting outer edge where tails off. Take first to avoid solutions due to noise in dusty outskirts            
        except IndexError:                      # If no local minimum, try to find hole as wherever the min is.          
            imin = np.argmin(t_w[indisc])           

        # Check against timestep and report
        if (dt > t_w[where_photoevap][imin]):         # If an entire cell can deplete
            #if not self._Hole:
            #    print("Alert - hole can open after this timestep at {:.2f} AU".format(disc.R[imin]))
            #    print("Outer radius is currently {:.2f} AU".format(self._R_out))
            self._Hole = True       # Set hole flag
        return t_w[where_photoevap][imin]

    def remove_mass(self, disc, dt, external_photo=None):
        # Find disc "outer edge" so we can apply mass loss only inside
        if external_photo:
            self._R_out = external_photo._Rot      # If external photoevaporation is present, only consider radii inside its influence
        else:
            self._R_out = disc.Rout(thresh=1e-10)
        if disc.Rout()==0.0:
            print("Disc everywhere below density threshold. Declare Empty.")
            self._empty = True

        # Check whether hole can open
        if not self._Hole: #self._type=='Primordial':
            self.check_dt(disc, dt)

        # Determine mass loss
        dSigma = np.minimum(self.dSigmadt * dt, disc.Sigma_G)   # Limit mass loss to density of cell
        dSigma *= (disc.R < self._R_out)                        # Only apply mass loss inside disc outer edge

        # Apply, preserving the dust mass
        if hasattr(disc, 'Sigma_D'):
            Sigma_D = disc.Sigma_D                              # Save the dust density
        disc._Sigma -= dSigma
        if hasattr(disc, 'Sigma_D'):
            dusty = Sigma_D.sum(0)>0
            disc.dust_frac[:,dusty] = np.fmin(Sigma_D[:,dusty]/disc.Sigma[dusty],disc.dust_frac[:,dusty]/disc.dust_frac.sum(0)[dusty])
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)           # Renormalise to 1 if it exceeds

        # Calculate actual mass loss given limit
        if dt>0:
            dM = 2*np.pi * disc.R * dSigma
            self._Mdot_true = np.trapz(dM,disc.R) / dt * AU**2 / Msun

    def get_Rhole(self, disc, external_photo=None):
        """Deal with calls when there is no hole"""
        if not self._Hole:
            print("No hole for which to get radius. Ignoring command and returning nans.")
            return np.nan, np.nan

        """Otherwise continue on to find hole
           First find outer edge of disc - hole must be inside this"""
        if external_photo:
            self._R_out = external_photo._Rot      # If external photoevaporation is present, only consider radii inside its influence
        else:
            self._R_out = disc.Rout(thresh=1e-10)
        where_photoevap = (self.dSigmadt > 0)
        indisc = (disc.R < self._R_out) * where_photoevap   # Prohibit hole outside of mass loss region.
        empty_indisc = (disc.Sigma_G <= 1e-10) * indisc     # Consider empty if below 10^-10 g/cm^2

        try:
            if np.sum(empty_indisc) == 0:                   # If none in disc are empty
                minima = argrelmin(disc.Sigma_G)
                if len(minima[0]) > 0:
                    i_hole_out = minima[0][0]  # Position of hole is minimum density
                else:   # No empty cells anymore - disc has cleared to outside
                    raise NotHoleError
            else:
                # First find the inner edge of the innermost hole
                i_hole_in  = np.nonzero(empty_indisc)[0][0]
                # The hole cell is defined as the one inside the first non-empty cell outside the inner edge of the hole
                outer_disc = ~empty_indisc * (disc.R>disc.R_edge[i_hole_in])
                if np.sum(outer_disc) > 0:
                    i_hole_out = np.nonzero(outer_disc)[0][0] - 1  
                else:   # No non-empty cells outside this - this is not a hole, but an outer edge.
                    raise NotHoleError

            if i_hole_out == np.nonzero(indisc)[0][-1]: # This is not a hole, but the outermost photoevaporating cell
                raise NotHoleError

            """If hole position drops by an order of magnitude, it is likely that the previous was really the clearing of low surface density material in the outer disc, so reset"""
            if self._R_hole:
                R_old = self._R_hole
                if disc.R_edge[i_hole_out+1]/R_old<0.1:
                    self._reset = True
                    
            """If everything worked, update hole properties"""
            if not self._R_hole:
                print("Hole opened at {:.2f} AU".format(disc.R_edge[i_hole_out+1]))
            self._R_hole = disc.R_edge[i_hole_out+1]
            self._N_hole = disc.column_density[i_hole_out]

            # Test whether Thin
            if (self._N_hole < self._N_crit):
                self._Thin = True

        except NotHoleError:
            """Potential hole isn't a hole but an outer edge"""
            if self._type == 'Primordial':
                self._Hole = False
                self._reset = True
                if self._R_hole:
                    print("No hole found")
                    print("Last known location {} AU".format(self._R_hole))
                return 0, 0
            elif self._type == 'InnerHole':
                if not self._empty:
                    print("Transition Disc has cleared to outside")
                    self._empty = True
                    # Proceed as usual to report but without update

        # Save state if tracking
        return self._R_hole, self._N_hole
        
    @property
    def Mdot(self):
        return self._Mdot

    @property
    def dSigmadt(self):
        return self._Sigmadot

    def __call__(self, disc, dt, external_photo=None):
        # For inner hole discs, need to update the hole radius and then the mass-loss as the normalisation changes based on R, not just x~R-Rhole.
        if self._type=='InnerHole':
            self.get_Rhole(disc)
            self.Sigma_dot(disc.R_edge, disc.star)

        # Remove the mass
        self.remove_mass(disc,dt, external_photo)   

        # Check for new holes
        if self._Hole and not self._Thin:   #  If there is a hole but the inner disc is not already optically thin, update its properties
            R_hole, N_hole = self.get_Rhole(disc, external_photo)

            # Check if hole is now large enough that inner disc optically thin, switch internal photoevaporation to direct field if so
            if self._Thin:
                print("Column density to hole has fallen to N = {} < {} g cm^-2".format(N_hole,self._N_crit))
                self._type = 'InnerHole'

                # Run the mass loss rates to update the table
                self.mdot_XE(disc.star)
                self.Sigma_dot(disc.R_edge, disc.star)

                # Report
                print("At initiation of InnerHole Type, M_D = {} M_J, Mdot = {}, t_clear ~ {} yr".format(disc.Mtot()/Mjup, self._Mdot, disc.Mtot()/Msun/self._Mdot))

    def ASCII_header(self):
        return ("# InternalEvaporation, Type: {}, Mdot: {}"
                "".format(self._type+self.__class__.__name__,self._Mdot))

    def HDF5_attributes(self):
        header = {}
        header['Type'] = self._type+"/"+self._regime
        header['Mdot'] = '{}'.format(self._Mdot)
        return self.__class__.__name__, header

#################################################################################
"""""""""
X-ray dominated photoevaporation
-Following prescription of Owen, Ercolano and Clarke (2012)
-Following prescription of Picogna, Ercolano, Owen and Weber (2019)
-Following prescription of Picogna, Ercolano, Owen and Weber (2021)
-Following prescription of Picogna, Ercolano and Espallait (2021)
"""""""""
#################################################################################
"""Owen, Ercolano and Clarke (2012)"""
class XrayDiscOwen(PhotoBase):
    def __init__(self, disc, Type='Primordial', R_hole=None, Mdot=None):
        super().__init__(disc, Regime='X-ray', Type=Type, Mdot=Mdot)

        # Parameters for Primordial mass loss profile
        self._a1 = 0.15138
        self._b1 = -1.2182
        self._c1 = 3.4046
        self._d1 = -3.5717
        self._e1 = -0.32762
        self._f1 = 3.6064
        self._g1 = -2.4918
        # Parameters for Inner Hole mass loss profile
        self._a2 = -0.438226
        self._b2 = -0.10658387
        self._c2 = 0.5699464
        self._d2 = 0.010732277
        self._e2 = -0.131809597
        self._f2 = -1.32285709

        # If initiating with an Inner Hole disc, need to update properties
        if self._type == 'InnerHole':
            self._Hole = True
            self._R_hole = R_hole
            #self.get_Rhole(disc)

        # Run the mass loss rates to update the table
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_XE(self, star, Mdot=None):
        # In Msun/yr
        if Mdot is not None:
            self._Mdot = Mdot
        elif self._type=='Primordial':
            self._Mdot = 6.25e-9 * star.M**(-0.068) * (star.L_X / 1e30)**(1.14) # Equation B1
        elif self._type=='InnerHole':
            self._Mdot = 4.8e-9 * star.M**(-0.148) * (star.L_X / 1e30)**(1.14)  # Equation B4
        else:
            raise NotImplementedError("Disc is of unrecognised type, and no mass-loss rate has been manually specified")
        self._Mdot_true = self._Mdot

    def scaled_R(self, R, star):
        # Where R in AU
        x = 0.85 * R / star.M                       # Equation B3
        if self._Hole:
            y = 0.95 * (R-self._R_hole) / star.M    # Equation B6
        else:
            y = R
        return x, y

    def R_inner(self, star):
        # Innermost mass loss
        return 0.7 / 0.85 * star.M

    def Sigma_dot_Primordial(self, R, star, ret=False):
        # Equation B2
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (x >= 0.7) * (x<=99) # No mass loss close to star, mass loss prescription becomes negative at log10(x)=1.996
        logx = np.log(x[where_photoevap])
        log10 = np.log(10)
        log10x = logx/log10

        # First term
        exponent = self._a1 * log10x**6 + self._b1 * log10x**5 + self._c1 * log10x**4 + self._d1 * log10x**3 + self._e1 * log10x**2 + self._f1 * log10x + self._g1
        t1 = 10**exponent

        # Second term
        terms = 6*self._a1*logx**5/log10**7 + 5*self._b1*logx**4/log10**6 + 4*self._c1*logx**3/log10**5 + 3*self._d1*logx**2/log10**4 + 2*self._e1*logx/log10**3 + self._f1/log10**2
        t2 = terms/x[where_photoevap]**2

        # Third term
        t3 = np.exp(-(x[where_photoevap]/100)**10)

        # Combine terms
        Sigmadot[where_photoevap] = t1 * t2 * t3

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot = np.maximum(Sigmadot,0)
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def Sigma_dot_InnerHole(self, R, star, ret=False):
        # Equation B5
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (y >= 0.0) # No mass loss inside hole
        use_y = y[where_photoevap]

        # Exponent of second term
        exp2 = -(use_y/57)**10

        # Numerator
        terms = self._a2*self._b2 * np.exp(self._b2*use_y+exp2) + self._c2*self._d2 * np.exp(self._d2*use_y+exp2) + self._e2*self._f2 * np.exp(self._f2*use_y+exp2)

        # Divide by Denominator
        Sigmadot[where_photoevap] = terms/R[where_photoevap]

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot = np.maximum(Sigmadot,0)
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        # Mopping up in the gap
        mop_up = (x >= 0.7) * (y < 0.0)
        Sigmadot[mop_up] = np.inf

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

"""Picogna, Ercolano, Owen and Weber (2019)"""
class XrayDiscPicogna(PhotoBase):
    def __init__(self, disc, Type='Primordial', R_hole=None, Mdot=None):
        super().__init__(disc, Regime='X-ray', Type=Type, Mdot=Mdot)

        # Parameters for Primordial mass loss profile
        self._a1 = -0.5885
        self._b1 = 4.3130
        self._c1 = -12.1214
        self._d1 = 16.3587
        self._e1 = -11.4721
        self._f1 = 5.7248
        self._g1 = -2.8562
        # Parameters for Inner Hole mass loss profile
        self._a2 = 0.11843
        self._b2 = 0.99695
        self._c2 = 0.48835

        # If initiating with an Inner Hole disc, need to update properties
        if self._type == 'InnerHole':
            self._Hole = True
            self._R_hole = R_hole
            #self.get_Rhole(disc)

        # Run the mass loss rates to update the table
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_XE(self, star, Mdot=None):
        # In Msun/yr
        if Mdot is not None:
            self._Mdot = Mdot
        elif self._type=='Primordial':
            logMd = -2.7326 * np.exp((np.log(np.log(star.L_X)/np.log(10))-3.3307)**2/-2.9868e-3) - 7.2580  # Equation 5
            self._Mdot = 10**logMd
        elif self._type=='InnerHole':
            logMd = -2.7326 * np.exp((np.log(np.log(star.L_X)/np.log(10))-3.3307)**2/-2.9868e-3) - 7.2580  # 1.12 * Equation 5
            self._Mdot = 1.12 * (10**logMd)
        else:
            raise NotImplementedError("Disc is of unrecognised type, and no mass-loss rate has been manually specified")
        self._Mdot_true = self._Mdot

    def scaled_R(self, R, star):
        # Where R in AU
        # All are divided by stellar mass normalised to 0.7 Msun (value used by Picogna+19) to represent rescaling by gravitational radius 
        x = R / (star.M/0.7)
        if self._Hole:
            y = (R-self._R_hole) / (star.M/0.7)    # Equation B6
        else:
            y = R / (star.M/0.7)
        return x, y

    def R_inner(self, star):
        # Innermost mass loss
        if self._type=='Primordial':
            return 0                # Mass loss possible throughout
        elif self._type=='InnerHole':
            return self._R_hole     # Mass loss profile applies outside hole
        else:
            return 0                # If unspecified, assume mass loss possible throughout 

    def Sigma_dot_Primordial(self, R, star, ret=False):
        # Equation 2
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (x<=np.inf) # Mass loss prescription becomes negative at x=1.3785
        logx = np.log(x[where_photoevap])
        log10 = np.log(10)
        log10x = logx/log10

        # First term
        exponent = self._a1 * log10x**6 + self._b1 * log10x**5 + self._c1 * log10x**4 + self._d1 * log10x**3 + self._e1 * log10x**2 + self._f1 * log10x + self._g1
        t1 = 10**exponent

        # Second term
        terms = 6*self._a1*log10x**5 + 5*self._b1*log10x**4 + 4*self._c1*log10x**3 + 3*self._d1*log10x**2 + 2*self._e1*log10x + self._f1
        t2 = terms/(2*np.pi*x[where_photoevap]**2)

        # Combine terms
        Sigmadot[where_photoevap] = t1 * t2
        Sigmadot = np.maximum(Sigmadot,0)

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def Sigma_dot_InnerHole(self, R, star, ret=False):
        # Equation 4
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (y > 0.0) * (y < -self._c2/np.log(self._b2)) # No mass loss inside hole, becomes negative at x=-c/ln(b)
        use_y = y[where_photoevap]

        # Numerator
        terms = self._a2 * np.power(self._b2,use_y) * np.power(use_y,self._c2-1) * (use_y * np.log(self._b2) + self._c2)

        # Divide by Denominator
        Sigmadot[where_photoevap] = terms/(2*np.pi*R[where_photoevap])
        Sigmadot = np.maximum(Sigmadot,0)

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        # Mopping up in the gap - assume usual primordial rates there.
        Sigmadot[(y<=0.0) * (x<=137)] = self.Sigma_dot_Primordial(R, star, ret=True)[(y<=0.0)*(x<=137)]/1.12 # divide by 1.12 so that normalise to correct mass loss rate
        mop_up = (x > 137) * (y < 0.0)
        Sigmadot[mop_up] = np.inf   # Avoid having discontinuous mass-loss by filling in the rest

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

"""Ercolano, Picogna, Monsch, Drake and Preibisch (2021)"""
"""Includes parameters to recreate their tests following Picogna et al. 2019"""
class XrayDiscErcolano(PhotoBase):
    def __init__(self, disc, Type='Primordial', R_hole=None, P19=False, Mdot=None):
        super().__init__(disc, Regime='X-ray', Type=Type, Mdot=Mdot)
        # Luminosity parameters
        if P19:
            self._spec = 'P19'
        elif np.log10(disc.star.L_X)<29.5:
            self._spec = 29
        elif np.log10(disc.star.L_X)>30.5:
            self._spec = 31
        else:
            self._spec = 30
        self._softFrac = {'P19': 0.2, 29: 0.53, 30: 0.47, 31: 0.42}

        # Parameters for Primordial mass loss profile
        self._a1 = {'P19': -0.5885, 29: -0.9152, 30:  0.3034, 31: -1.2845}
        self._b1 = {'P19':  4.3130, 29:  8.5032, 30: -1.5323, 31:  9.3601}
        self._c1 = {'P19':-12.1214, 29:-32.0623, 30:  1.5766, 31:-27.7371}
        self._d1 = {'P19': 16.3587, 29: 62.8336, 30:  4.0211, 31: 42.9367}
        self._e1 = {'P19':-11.4721, 29: 67.9150, 30:-11.1311, 31:-37.3244}
        self._f1 = {'P19':  5.7248, 29: 39.2652, 30: 10.6550, 31:  8.8216}
        self._g1 = {'P19': -2.8562, 29:-10.1113, 30: -4.5769, 31: -5.3780}
        self._Rcut = {'P19': 120., 29: 206., 30: 105., 31: 69.}

        # If initiating with an Inner Hole disc, need to update properties
        if self._type == 'InnerHole':
            raise NotImplementedError
            self._Hole = True
            self._R_hole = R_hole
            #self.get_Rhole(disc)

        # Run the mass loss rates to update the table
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_XE(self, star, Mdot=None):
        # In Msun/yr
        if Mdot is not None:
            self._Mdot = Mdot
        elif self._type is not None:
            logMd = -1.947e17 * np.exp((np.log(np.log(star.L_X*0.5)/np.log(10))+1.572e-4)**2/-2.866e-1) - 6.694  # Equation 5
            # self._softFrac[self._spec]
            self._Mdot = 10**logMd
        else:
            raise NotImplementedError("Disc is of unrecognised type, and no mass-loss rate has been manually specified")
        self._Mdot_true = self._Mdot

    def scaled_R(self, R, star):
        # Where R in AU
        # All are divided by stellar mass normalised to 0.7 Msun (value used by Ercolano+21) to represent rescaling by gravitational radius 
        x = R / (star.M/0.7)
        if self._Hole:
            y = (R-self._R_hole) / (star.M/0.7)    # Equation B6
        else:
            y = R / (star.M/0.7)
        return x, y

    def R_inner(self, star):
        # Innermost mass loss
        if self._type=='Primordial':
            return 0                # Mass loss possible throughout
        elif self._type=='InnerHole':
            return self._R_hole     # Mass loss profile applies outside hole
        else:
            return 0                # If unspecified, assume mass loss possible throughout 

    def Sigma_dot_Primordial(self, R, star, ret=False):
        # Equation 3
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (x<=np.inf)
        logx = np.log(x[where_photoevap])
        log10 = np.log(10)
        log10x = logx/log10

        # First term
        exponent = self._a1[self._spec] * log10x**6 + self._b1[self._spec] * log10x**5 + self._c1[self._spec] * log10x**4 + self._d1[self._spec] * log10x**3 + self._e1[self._spec] * log10x**2 + self._f1[self._spec] * log10x + self._g1[self._spec]
        t1 = 10**exponent

        # Second term
        terms = 6*self._a1[self._spec]*log10x**5 + 5*self._b1[self._spec]*log10x**4 + 4*self._c1[self._spec]*log10x**3 + 3*self._d1[self._spec]*log10x**2 + 2*self._e1[self._spec]*log10x + self._f1[self._spec]
        t2 = terms/(2*np.pi*x[where_photoevap]**2)

        # Cut-off (Equation 6)
        t3 = np.exp(-np.power(x[where_photoevap]/self._Rcut[self._spec],10))

        # Combine terms
        Sigmadot[where_photoevap] = t1 * t2 * t3
        Sigmadot = np.maximum(Sigmadot,0)

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def Sigma_dot_InnerHole(self, R, star, ret=False):
        # Not provided
        # Keep using the primordial one
        return self.Sigma_dot_Primordial(R, star, ret=ret)

"""Picogna, Ercolano and Espaillat (2021)"""
class XrayDiscPicogna21(PhotoBase):
    def __init__(self, disc, Type='Primordial', R_hole=None, P19=False, Mdot=None):
        super().__init__(disc, Regime='X-ray', Type=Type, Mdot=Mdot)

        # Parameters for Primordial mass loss profile
        self._a1 = {0.1: -3.8337, 0.3:  -1.3206, 0.5: -1.2320, 1.0: -0.6344}
        self._b1 = {0.1: 22.9100, 0.3:  13.0475, 0.5: 10.8505, 1.0:  6.3587}
        self._c1 = {0.1:-55.1282, 0.3: -53.6990, 0.5:-38.6939, 1.0:-26.1445}
        self._d1 = {0.1: 67.8919, 0.3: 117.6027, 0.5: 71.2489, 1.0: 56.4477}
        self._e1 = {0.1:-45.0138, 0.3:-144.3769, 0.5:-71.4279, 1.0:-67.7403}
        self._f1 = {0.1: 16.2977, 0.3 : 94.7854, 0.5: 37.8707, 1.0: 43.9212}
        self._g1 = {0.1: -3.5426, 0.3 :-26.7363, 0.5: -9.3508, 1.0:-13.2316}
        print("WARNING, haven't implemented a cut-off")

        # If initiating with an Inner Hole disc, need to update properties
        if self._type == 'InnerHole':
            raise NotImplementedError
            self._Hole = True
            self._R_hole = R_hole
            #self.get_Rhole(disc)

        # Run the mass loss rates to update the table
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_XE(self, star, Mdot=None):
        # In Msun/yr
        if Mdot is not None:
            self._Mdot = Mdot
        elif self._type is not None:
            self._Mdot = 3.93e-8 * star.M   # Equation 5
        else:
            raise NotImplementedError("Disc is of unrecognised type, and no mass-loss rate has been manually specified")
        self._Mdot_true = self._Mdot

    def scaled_R(self, R, star):
        # Where R in AU
        # No need for rescaling as profiles given separately for each mass
        x = R
        if self._Hole:
            y = (R-self._R_hole)
        else:
            y = R
        return x, y

    def R_inner(self, star):
        # Innermost mass loss
        if self._type=='Primordial':
            return 0                # Mass loss possible throughout
        elif self._type=='InnerHole':
            return self._R_hole     # Mass loss profile applies outside hole
        else:
            return 0                # If unspecified, assume mass loss possible throughout 

    def Sigma_dot_Primordial(self, R, star, ret=False):
        # Equation 6
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (x<=np.inf)       # No outer cut off given
        logx = np.log(x[where_photoevap])
        log10 = np.log(10)
        log10x = logx/log10

        # First term
        exponent = self._a1[star.M] * log10x**6 + self._b1[star.M] * log10x**5 + self._c1[star.M] * log10x**4 + self._d1[star.M] * log10x**3 + self._e1[star.M] * log10x**2 + self._f1[star.M] * log10x + self._g1[star.M]
        t1 = 10**exponent

        # Second term
        terms = 6*self._a1[star.M]*log10x**5 + 5*self._b1[star.M]*log10x**4 + 4*self._c1[star.M]*log10x**3 + 3*self._d1[star.M]*log10x**2 + 2*self._e1[star.M]*log10x + self._f1[star.M]
        t2 = terms/(2*np.pi*x[where_photoevap]**2)

        # Cut-off
        #t3 = np.exp(-np.power(x[where_photoevap]/self._Rcut[star.M],10))

        # Combine terms
        Sigmadot[where_photoevap] = t1 * t2 #* t3
        Sigmadot = np.maximum(Sigmadot,0)

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def Sigma_dot_InnerHole(self, R, star, ret=False):
        # Not provided
        # Keep using the primordial one
        return self.Sigma_dot_Primordial(R, star, ret=ret)

"""Sellek et al. (2024b)"""
class XrayDiscSellek24(PhotoBase):
    def __init__(self, disc, Type='Primordial', R_hole=None, P19=False, Mdot=None):
        super().__init__(disc, Regime='X-ray', Type=Type, Mdot=Mdot)

        # Parameters for Primordial mass loss profile
        self._a1 = {1.0: -1.2108}
        self._b1 = {1.0:  9.6815}
        self._c1 = {1.0:-31.226}
        self._d1 = {1.0: 52.293}
        self._e1 = {1.0:-48.475}
        self._f1 = {1.0: 25.093}
        self._g1 = {1.0:-7.6443}
        print("WARNING, haven't implemented a cut-off")

        # If initiating with an Inner Hole disc, need to update properties
        if self._type == 'InnerHole':
            raise NotImplementedError
            self._Hole = True
            self._R_hole = R_hole
            #self.get_Rhole(disc)

        # Run the mass loss rates to update the table
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_XE(self, star, Mdot=None):
        # In Msun/yr
        if Mdot is not None:
            self._Mdot = Mdot
        elif self._type is not None:
            self._Mdot = 4.3178e-9 * star.M   # Equation 5
        else:
            raise NotImplementedError("Disc is of unrecognised type, and no mass-loss rate has been manually specified")
        self._Mdot_true = self._Mdot

    def scaled_R(self, R, star):
        # Where R in AU
        # All are divided by stellar mass normalised to 1.0 Msun (value used by Sellek+24) to represent rescaling by gravitational radius 
        x = R / (star.M/1.0)
        if self._Hole:
            y = (R-self._R_hole) / (star.M/1.0)    # Equation B6
        else:
            y = R / (star.M/1.0)
        return x, y

    def R_inner(self, star):
        # Innermost mass loss
        if self._type=='Primordial':
            return 0                # Mass loss possible throughout
        elif self._type=='InnerHole':
            return self._R_hole     # Mass loss profile applies outside hole
        else:
            return 0                # If unspecified, assume mass loss possible throughout 

    def Sigma_dot_Primordial(self, R, star, ret=False):
        # Equation 6
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (x<=np.inf)       # No outer cut off given
        logx = np.log(x[where_photoevap])
        log10 = np.log(10)
        log10x = logx/log10

        # First term
        try:
            exponent = self._a1[star.M] * log10x**6 + self._b1[star.M] * log10x**5 + self._c1[star.M] * log10x**4 + self._d1[star.M] * log10x**3 + self._e1[star.M] * log10x**2 + self._f1[star.M] * log10x + self._g1[star.M]
        except:
            exponent = self._a1[1.0] * log10x**6 + self._b1[1.0] * log10x**5 + self._c1[1.0] * log10x**4 + self._d1[1.0] * log10x**3 + self._e1[1.0] * log10x**2 + self._f1[1.0] * log10x + self._g1[1.0]
        t1 = 10**exponent

        # Second term
        try:
            terms = 6*self._a1[star.M]*log10x**5 + 5*self._b1[star.M]*log10x**4 + 4*self._c1[star.M]*log10x**3 + 3*self._d1[star.M]*log10x**2 + 2*self._e1[star.M]*log10x + self._f1[star.M]
        except:
            terms = 6*self._a1[1.0]*log10x**5 + 5*self._b1[1.0]*log10x**4 + 4*self._c1[1.0]*log10x**3 + 3*self._d1[1.0]*log10x**2 + 2*self._e1[1.0]*log10x + self._f1[1.0]
        t2 = terms/(2*np.pi*x[where_photoevap]**2)

        # Cut-off
        #t3 = np.exp(-np.power(x[where_photoevap]/self._Rcut[star.M],10))

        # Combine terms
        Sigmadot[where_photoevap] = t1 * t2 #* t3
        Sigmadot = np.maximum(Sigmadot,0)

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def Sigma_dot_InnerHole(self, R, star, ret=False):
        # Not provided
        # Keep using the primordial one
        return self.Sigma_dot_Primordial(R, star, ret=ret)

#################################################################################
"""""""""
FUV dominated photoevaporation
-Following prescription of Komaki, Nakatani and Yoshida (2021)
"""""""""
#################################################################################
class FUVDiscKomaki21(PhotoBase):
    def __init__(self, disc, Type='Primordial', R_hole=None, P19=False):
        super().__init__(disc, Regime='FUV', Type=Type)

        # Parameters for Primordial mass loss profile
        self._b1 = {0.5:  1.060, 0.7:  0.693, 1.0:  0.131, 1.7:  1.370, 3.0:  0.033, 7.0:  0.594}
        self._c1 = {0.5: -1.050, 0.7: -0.950, 1.0: -0.465, 1.7: -1.410, 3.0: -0.786, 7.0: -1.000}
        self._d1 = {0.5: -0.236, 0.7: -0.038, 1.0:  0.451, 1.7: -1.420, 3.0:  0.786, 7.0:  0.234}
        self._e1 = {0.5:  0.570, 0.7:  0.678, 1.0:  0.376, 1.7:  1.300, 3.0:  0.557, 7.0:  0.513}
        self._f1 = {0.5: -1.620, 0.7: -1.670, 1.0: -1.670, 1.7: -1.060, 3.0: -1.580, 7.0: -1.850}
        self._g1 = {0.5:-12.700, 0.7:-12.600, 1.0:-12.600, 1.7:-12.600, 3.0:-13.100, 7.0:-12.100}

        # If initiating with an Inner Hole disc, need to update properties
        if self._type == 'InnerHole':
            raise NotImplementedError
            self._Hole = True
            self._R_hole = R_hole
            #self.get_Rhole(disc)

        # Run the mass loss rates to update the table
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_XE(self, star, Mdot=None):
        # In Msun/yr
        if Mdot is not None:
            self._Mdot = Mdot
        elif self._type is not None:
            self._Mdot = 3.93e-8 * star.M   # Equation 5
        else:
            raise NotImplementedError("Disc is of unrecognised type, and no mass-loss rate has been manually specified")
        self._Mdot_true = self._Mdot

    def scaled_R(self, R, star):
        # Where R in AU
        # No need for rescaling as profiles given separately for each mass
        x = R / (8.87*star.M)
        if self._Hole:
            y = (R-self._R_hole)
        else:
            y = R
        return x, y

    def R_inner(self, star):
        # Innermost mass loss
        if self._type=='Primordial':
            return 0                # Mass loss possible throughout
        elif self._type=='InnerHole':
            return self._R_hole     # Mass loss profile applies outside hole
        else:
            return 0                # If unspecified, assume mass loss possible throughout 

    def Sigma_dot_Primordial(self, R, star, ret=False):
        # Equation 6
        Sigmadot = np.zeros_like(R)
        x, y = self.scaled_R(R,star)
        where_photoevap = (x<=20)       # No outer cut off given
        logx = np.log(x[where_photoevap])
        log10 = np.log(10)
        log10x = logx/log10

        # First term
        exponent = self._b1[star.M] * log10x**5 + self._c1[star.M] * log10x**4 + self._d1[star.M] * log10x**3 + self._e1[star.M] * log10x**2 + self._f1[star.M] * log10x + self._g1[star.M]
        t1 = 10**exponent

        # Combine terms
        Sigmadot[where_photoevap] = t1
        Sigmadot = np.maximum(Sigmadot,0)

        # Normalise, convert to cgs
        Sigmadot *= yr/Omega0 #Msun / AU**2 #* self.Mdot / total # in g cm^-2 / yr

        # Work out total mass loss rate for normalisation
        #M_dot = 2*np.pi * R * Sigmadot
        #total = np.trapz(M_dot,R)

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def Sigma_dot_InnerHole(self, R, star, ret=False):
        # Not provided
        # Keep using the primordial one
        return self.Sigma_dot_Primordial(R, star, ret=ret)

#################################################################################
"""""""""
EUV dominated photoevaporation
-Following prescription given in Alexander and Armitage (2007)
 and based on Font, McCarthy, Johnstone and Ballantyne (2004) for Primordial Discs
 and based on Alexander, Clarke and Pringle (2006) for Inner Hole Discs
"""""""""
#################################################################################
class EUVDiscAlexander(PhotoBase):
    def __init__(self, disc, Type='Primordial', R_hole=None):
        super().__init__(disc, Regime='EUV', Type=Type)
        
        # Parameters for mass loss profiles
        self._cs = 10                                           # Sound speed in km s^-1
        self._RG = disc.star.M / (self._cs*1e5 /Omega0/AU)**2   # Gravitational Radius in AU
        self._mu = 1.35
        self._aB = 2.6e-13                                      # Case B Recombination coeff. in cm^3 s^-1
        self._C1 = 0.14
        self._A  = 0.3423
        self._B  = -0.3612
        self._D  = 0.2457
        self._C2 = 0.235
        self._a  = 2.42

        h = disc.H/disc.R
        he = np.empty_like(disc.R_edge)
        he[1:-1] = 0.5*(h[1:] + h[-1:])
        he[0] = 1.5*h[0] - 0.5*h[1] 
        he[-1] = 1.5*h[-1] - 0.5*h[-2]
        self._h = he

        # If initiating with an Inner Hole disc, need to update properties
        if self._type == 'InnerHole':
            self._Hole = True
            self._R_hole = R_hole
            #self.get_Rhole(disc)

        # Run the mass loss rates to update the table
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_XE(self, star, Mdot=0):
        # Store Mdot calculated from profile
        self._Mdot = Mdot  # In Msun/yr
        self._Mdot_true = self._Mdot

    def scaled_R(self, R, star):
        if self._type=='Primordial':
            return R / self._RG         # Normalise to RG
        elif self._type=='InnerHole':
            return R / self.R_inner()   # Normalise to inner edge
        else:
            return R                    # If unspecified, don't modify

    def R_inner(self):
        # Innermost mass loss
        if self._type=='Primordial':
            return 0.1 * self._RG   # Mass loss profile is only positive for >0.1 RG
        elif self._type=='InnerHole':
            return self._R_hole     # Mass loss profile applies outside hole
        else:
            return 0                # If unspecified, assume mass-loss possible throughout 

    def Sigma_dot_Primordial(self, R, star, ret=False):
        Sigmadot = np.zeros_like(R)
        x = self.scaled_R(R,star)
        where_photoevap = (x >= 0.1)    # No mass loss close to star

        # Equation A3
        nG = self._C1 * (3 * star.Phi / (4*np.pi * (self._RG*AU)**3 * self._aB))**(1/2)        # cm^-3
        # Equation A2
        n0 = nG * (2 / (x**7.5 + x**12.5))**(1/5)

        # Equation A4
        u1 = self._cs*1e5*yr/Omega0 * self._A * np.exp(self._B * (x-0.1)) * (x-0.1)**self._D  # cm yr^-1

        # Combine terms (Equation A1)
        Sigmadot[where_photoevap] = 2 * self._mu * m_H * (n0 * u1)[where_photoevap]  # g cm^-2 /yr
        Sigmadot = np.maximum(Sigmadot,0)

        # Work out total mass loss rate
        dMdot = 2*np.pi * R * Sigmadot
        Mdot  = np.trapz(dMdot,R)  # g yr^-1 (AU/cm)^2

        # Normalise, convert to cgs
        Mdot  = Mdot * AU**2/Msun   # g yr^-1

        # Store result
        self.mdot_XE(star, Mdot=Mdot)

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2


    def Sigma_dot_InnerHole(self, R, star, ret=False):
        Sigmadot = np.zeros_like(R)
        x = self.scaled_R(R,star)
        where_photoevap = (x > 1)    # No mass loss inside hole

        # Combine terms (Equation A5)
        Sigmadot[where_photoevap] = (2 * self._mu * m_H * self._C2 * self._cs*1e5*yr/Omega0 * (star.Phi / (4*np.pi * (self.R_inner()*AU)**3 * self._aB * self._h))**(1/2) * x**(-self._a))[where_photoevap]  # g cm^-2 /yr
        Sigmadot = np.maximum(Sigmadot,0)

        # Work out total mass loss rate
        dMdot = 2*np.pi * R * Sigmadot
        Mdot  = np.trapz(dMdot,R)  # g yr^-1 (AU/cm)^2

        # Normalise, convert to cgs
        Mdot  = Mdot * AU**2/Msun   # g yr^-1

        # Store result
        self.mdot_XE(star, Mdot=Mdot)

        # Mopping up in the gap
        mop_up = (R >= 0.1 * self._RG) * (x <= 1.0)
        Sigmadot[mop_up] = np.inf

        if ret:
            # Return unaveraged values at cell edges
            return Sigmadot
        else:
            # Store values as average of mass loss rate at cell edges
            self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

#################################################################################
"""""""""
Functions for running as main
Designed for plotting to test things out
"""""""""
#################################################################################
        
class DummyDisc(object):
    def __init__(self, R, star, MD=10, RC=100):
        self._M = MD * Mjup
        self.Rc = RC
        self.R_edge = R
        self.R = 0.5*(self.R_edge[1:]+self.R_edge[:-1])
        self._Sigma = self._M / (2 * np.pi * self.Rc * self.R * AU**2) * np.exp(-self.R/self.Rc)
        self.star = star

    def Rout(self, thresh=None):
        return max(self.R_edge)

    @property
    def Sigma(self):
        return self._Sigma

    @property
    def Sigma_G(self):
        return self._Sigma

def main():
    Sigma_dot_plot()
    #Test_Removal()

def Test_Removal():
    """Removes gas fom a power law disc in regular timesteps without viscous evolution etc"""
    star1 = PhotoStar(LX=1e30, M=1.0, R=2.5, T_eff=4000)
    R = np.linspace(0.1,200,2000)
    disc1 = DummyDisc(R, star1, RC=10)

    internal_photo = XrayDiscPicogna(disc1)

    plt.figure()
    for t in np.linspace(0,2e3,6):
        internal_photo(disc1, 2e3)
        plt.loglog(0.5*(R[1:]+R[:-1]), disc1.Sigma, label='{}'.format(t))
    plt.xlabel("R / AU")
    plt.ylabel("$\Sigma_G~/~\mathrm{g~cm^{-2}}$")
    plt.legend(title='Time / yr')
    plt.show()

def Sigma_dot_plot():
    """Plot a comparison of the mass loss rate prescriptions"""
    from control_scripts import run_model
    # Set up dummy model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))

    fig, ax = plt.subplots(1,1, figsize=(15*cm, 9*cm))
    fig.subplots_adjust(top=0.9, left=0.15, right=0.975, bottom=0.15)

    starX = PhotoStar(LX=1e30,  M=model['star']['mass'], R=model['star']['radius'], T_eff=model['star']['T_eff'])
    starE = PhotoStar(Phi=1e42, M=model['star']['mass'], R=model['star']['radius'], T_eff=model['star']['T_eff'])

    disc = run_model.setup_disc(model)
    R = disc.R

    # Calculate EUV rates
    disc._star = starE
    internal_photo_E = EUVDiscAlexander(disc)    
    Sigma_dot_E = internal_photo_E.dSigmadt
    photoevaporating_E = (Sigma_dot_E>0)
    t_w_E = disc.Sigma[photoevaporating_E] / Sigma_dot_E[photoevaporating_E]
    print("Mdot maximum at R = {} AU".format(R[np.argmax(Sigma_dot_E)]))    
    #print("Time minimum at R = {} AU".format(R[photoevaporating_E][np.argmin(t_w_E)]))
    plt.loglog(R, Sigma_dot_E, color='moccasin', linestyle=':', label='EUV (AA07)')#, $\Phi={}~\mathrm{{s^{{-1}}}}$'.format(1e42), linestyle='--')

    # Calculate X-ray rates
    disc._star = starX
    internal_photo_X = XrayDiscOwen(disc)    
    Sigma_dot_X = internal_photo_X.dSigmadt
    photoevaporating_X = (Sigma_dot_X>0)
    t_w_X = disc.Sigma[photoevaporating_X] / Sigma_dot_X[photoevaporating_X]
    print("Mdot maximum at R = {} AU".format(R[np.argmax(Sigma_dot_X)]))    
    #print("Time minimum at R = {} AU".format(R[photoevaporating_X][np.argmin(t_w_X)]))
    plt.loglog(R, Sigma_dot_X, color='peachpuff', label='X-ray (OEC12)')#, $L_X={}~\mathrm{{erg~s^{{-1}}}}, M_*={}~M_\odot$'.format(1e30, 1))

    # Calculate X-ray rates
    disc._star = starX
    internal_photo_X2 = XrayDiscPicogna(disc)    
    Sigma_dot_X2 = internal_photo_X2.dSigmadt
    photoevaporating_X2 = (Sigma_dot_X2>0)
    t_w_X2 = disc.Sigma[photoevaporating_X2] / Sigma_dot_X2[photoevaporating_X2]
    print("Mdot maximum at R = {} AU".format(R[np.argmax(Sigma_dot_X2)]))    
    #print("Time minimum at R = {} AU".format(R[photoevaporating_X2][np.argmin(t_w_X2)]))
    plt.loglog(R, Sigma_dot_X2, color='salmon', label='X-ray (PEOW19)')#, $L_X={}~\mathrm{{erg~s^{{-1}}}}$'.format(1e30))

    # Calculate X-ray rates
    disc._star = starX
    internal_photo_X3 = XrayDiscErcolano(disc)    
    Sigma_dot_X3 = internal_photo_X3.dSigmadt
    photoevaporating_X3 = (Sigma_dot_X3>0)
    t_w_X3 = disc.Sigma[photoevaporating_X3] / Sigma_dot_X3[photoevaporating_X3]
    print("Mdot maximum at R = {} AU".format(R[np.argmax(Sigma_dot_X3)]))    
    #print("Time minimum at R = {} AU".format(R[photoevaporating_X3][np.argmin(t_w_X3)]))
    plt.loglog(R, Sigma_dot_X3, color='orangered', label='X-ray (EPMDP21)')#, $L_X={}~\mathrm{{erg~s^{{-1}}}}$'.format(1e30))

    # Calculate X-ray rates
    disc._star = starX
    internal_photo_X4 = XrayDiscPicogna21(disc)    
    Sigma_dot_X4 = internal_photo_X4.dSigmadt
    photoevaporating_X4 = (Sigma_dot_X4>0)
    t_w_X4 = disc.Sigma[photoevaporating_X4] / Sigma_dot_X4[photoevaporating_X4]
    print("Mdot maximum at R = {} AU".format(R[np.argmax(Sigma_dot_X4)]))    
    #print("Time minimum at R = {} AU".format(R[photoevaporating_X4][np.argmin(t_w_X4)]))
    plt.loglog(R, Sigma_dot_X4, color='firebrick', label='X-ray (PEE21)')#, $M_*={}~M_\odot$'.format(1))

    # Calculate FUV rates
    disc._star = starX
    internal_photo_FUV = FUVDiscKomaki21(disc)    
    Sigma_dot_FUV = internal_photo_FUV.dSigmadt
    photoevaporating_FUV = (Sigma_dot_FUV>0)
    t_w_FUV = disc.Sigma[photoevaporating_FUV] / Sigma_dot_FUV[photoevaporating_FUV]
    print("Mdot maximum at R = {} AU".format(R[np.argmax(Sigma_dot_FUV)]))    
    #print("Time minimum at R = {} AU".format(R[photoevaporating_X4][np.argmin(t_w_X4)]))
    plt.loglog(R, Sigma_dot_FUV, color='turquoise', linestyle='-.', label='FUV (KNY21)')#, $M_*={}~M_\odot$'.format(1))

    # Plot mass loss rates
    plt.xlabel("R / AU", fontsize=12)
    plt.ylabel("$\dot{\Sigma}_{\\rm w}$ / g cm$^{-2}$ yr$^{-1}$", fontsize=12)
    plt.xlim([0.1,3000])
    plt.ylim([1e-7,1e-3])
    ax.tick_params(axis='x', which='both', labelsize=11)
    ax.tick_params(axis='y', which='both', labelsize=11)
    plt.suptitle('Comparison of Photoevaporation Mass Loss Profiles', fontsize=14)
    plt.legend(fontsize=12, handletextpad=0.4, handlelength=1.0, borderpad=0.4, borderaxespad=0.4, labelspacing=0.4, loc=1)
    plt.savefig('/home/ads79/thesisPlots/Mdotprofiles.pdf')
    #plt.show()

    # Plot depletion time
    plt.figure(figsize=(6,6))
    plt.loglog(R[photoevaporating_E], t_w_E, label='EUV (AA07), $\Phi={}~\mathrm{{s^{{-1}}}}$'.format(1e42), linestyle='--')
    plt.loglog(R[photoevaporating_X], t_w_X, label='X-ray (OEC12), $L_X={}~\mathrm{{erg~s^{{-1}}}}$'.format(1e30))
    plt.loglog(R[photoevaporating_X2], t_w_X2, label='X-ray (PEOW19), $L_X={}~\mathrm{{erg~s^{{-1}}}}$'.format(1e30))
    plt.loglog(R[photoevaporating_X3], t_w_X3, label='X-ray (EPMDP21), $L_X={}~\mathrm{{erg~s^{{-1}}}}$'.format(1e30))
    plt.loglog(R[photoevaporating_X4], t_w_X4, label='X-ray (PEE21), $L_X={}~\mathrm{{erg~s^{{-1}}}}$'.format(1e30))
    plt.xlabel("R / AU")
    plt.ylabel("$t_w / \mathrm{yr}$")
    plt.xlim([0.1,1000])
    plt.ylim([1e5,1e9])
    #plt.legend()
    plt.show()

if __name__ == "__main__":
    # Set extra things
    DefaultModel = "../control_scripts/DiscConfig_default.json"

    import matplotlib.patches as pat
    cm=1/2.54

    from matplotlib import rcParams
    rcParams['savefig.dpi'] = 600
    rcParams['figure.dpi']  = 600
    plt.rcParams['text.usetex'] = "True"
    plt.rcParams['font.family'] = "serif"

    main()

