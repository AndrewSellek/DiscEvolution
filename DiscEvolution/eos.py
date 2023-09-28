from __future__ import print_function
import numpy as np
from scipy.integrate import quad
from .brent import brentq
from .constants import GasConst, sig_SB, AU, Omega0, Msun, Mjup, Mearth
from . import opacity
################################################################################
# Thermodynamics classes
################################################################################
class EOS_Table(object):
    """Base class for equation of state evaluated at certain locations.

    Stores pre-computed temperatures, viscosities etc. Derived classes need to
    provide the funcitons called by set_grid.
    """
    def __init__(self, ptbn_kwargs={"Type":"None"}):
        self._gamma = 1.0
        self._mu    = 2.4
        
        # Set perturbation properties
        if ptbn_kwargs["Type"]=='Gaussian':
            print("Initialising {} perturbation".format(ptbn_kwargs["Type"]))
            self._ptbn  = self._f_Gaussian_ptbn
            self._ptbn_amplitude = ptbn_kwargs['Amplitude']
            self._ptbn_radius = ptbn_kwargs['Radius']
            self._ptbn_width = ptbn_kwargs['Width']
        elif ptbn_kwargs["Type"]=='Kanagawa':
            print("Initialising {} perturbation".format(ptbn_kwargs["Type"]))
            self._ptbn  = self._f_Kanagawa_ptbn
            self._Mp = ptbn_kwargs['Planet Mass']   # Earth units
            self._Ms = ptbn_kwargs['Star Mass']     # Solar units
            self._Rp = ptbn_kwargs['Radius']
        elif ptbn_kwargs["Type"]=='Duffell':
            print("Initialising {} perturbation".format(ptbn_kwargs["Type"]))
            raise NotImplementedError
            self._ptbn  = self._f_Duffell_ptbn
            self._Mp = ptbn_kwargs['Planet Mass']   # Earth units
            self._Ms = ptbn_kwargs['Star Mass']     # Solar units
            self._Rp = ptbn_kwargs['Radius']
        elif ptbn_kwargs["Type"]!='None':
            print(ptbn_kwargs["Type"], "perturbation not recognised, default to None")
            self._ptbn  = self._f_no_ptbn
            self._enhance_diffusion = False
        else:
            print("Initialising with no perturbation")
            self._ptbn  = self._f_no_ptbn
            self._enhance_diffusion = False
            
        # Set perturbation growth
        if self._ptbn!=self._f_no_ptbn:
            try:
                self._growth_scaling = ptbn_kwargs['Growth Scaling']    # 1/2 = 2D Bondi Pebble; 2/3 = 2D Hill Pebble/Oligarchic Planetestimal; 1 = 3D Pebble/Hill Gas; 4/3 = Runaway Planetesimal; 3 = Bondi Gas
                self._growth_timescale = ptbn_kwargs['Growth Timescale']  
                self._growth_endtime = ptbn_kwargs['Growth Endtime']  
            except KeyError:
                self._growth_scaling = None
            try:
                self._enhance_diffusion = ptbn_kwargs['Enhance Diffusion']
            except KeyError:
                self._enhance_diffusion = False
    
    def set_grid(self, grid):
        self._R      = grid.Rc
        self._set_arrays()

    def _set_arrays(self, t=0):
        R  = self._R
        self._cs     = self._f_cs(R)
        self._H      = self._f_H(R)
        self._alpha  = self._f_alpha(R)
        self._nu_diff, self._nu_eff = self._f_nu(R, t=t)
        
    def _f_growth(self, t=0):
        if self._growth_scaling is None:
            growth_mass = 1.0
        elif self._growth_scaling==1:
            growth_mass = np.minimum(1.0, np.exp((t-self._growth_endtime)/self._growth_timescale))
        else:
            growth_mass = np.minimum(1.0, np.power(np.maximum(1 + (1-self._growth_scaling) * (t-self._growth_endtime)/self._growth_timescale,0.0), 1/(1-self._growth_scaling)))
        growth_torque = growth_mass**2 # Lin & Papaloizou 1979; Kanagawa et al. 2015
        return growth_torque

    def _f_no_ptbn(self, R, t=0):
        return np.ones_like(R)
        
    def _f_Gaussian_ptbn(self, R, t=0):
        return 1. + (self._ptbn_amplitude-1.) * self._f_growth(t) * np.exp(-0.5*np.power(R-self._ptbn_radius,2)/np.power(self._ptbn_width*self._f_H(self._ptbn_radius),2))

    # Define auxiliary functions for Kanagawa prescription
    def K_Kanagawa(self, R,t):
        K_R = np.power((self._Mp*Mearth)/(self._Ms*Msun), 2) * np.power(self._H/R, -5) / self._f_alpha(R) * self._f_growth(t) 
        return K_R[np.argmin(np.abs(R-self._Rp))]
    def Kprime_Kanagawa(self, R,t):
        K_R = np.power((self._Mp*Mearth)/(self._Ms*Msun), 2) * np.power(self._H/R, -3) / self._f_alpha(R) * self._f_growth(t) 
        return K_R[np.argmin(np.abs(R-self._Rp))]
    def DR1_Kanagawa(self, R,t):
        return (self.amplitude_Kanagawa(R,t)/4+0.08)*self.Kprime_Kanagawa(R,t)**0.25*self._Rp
    def DR2_Kanagawa(self, R,t):
        return 0.33*self.Kprime_Kanagawa(R,t)**0.25*self._Rp
    def amplitude_Kanagawa(self, R,t):
        # This is for Sigma, so the viscosity profile needs 1/(this value)
        return 1/(1+0.04*self.K_Kanagawa(R,t))

    def _f_Kanagawa_ptbn(self, R, t=0):
        # Kanagawa et al. 2017 based on their 2015 and 2016 studies of depth and width respectively
        profile = np.ones_like(R)
        profile[(np.abs(R-self._Rp)<self.DR2_Kanagawa(R,t))] = 1/(4.0*np.power(self.Kprime_Kanagawa(R,t),-1/4)*np.abs(R-self._Rp)/self._Rp-0.32)[(np.abs(R-self._Rp)<self.DR2_Kanagawa(R,t))]
        profile[(np.abs(R-self._Rp)<self.DR1_Kanagawa(R,t))] = 1/self.amplitude_Kanagawa(R,t)
        return profile
        
    def _f_Duffell_ptbn(self, R, t=0):
        # https://ui.adsabs.harvard.edu/abs/2020ApJ...889...16D/abstract
        raise NotImplementedError

    @property
    def cs(self):
        return self._cs

    @property
    def H(self):
        return self._H

    @property
    def alpha(self):
        return self._alpha

    @property
    def nu(self):
        return self._nu_eff
        
    @property
    def nu_eff(self):
        return self._nu_eff
        
    @property
    def nu_diff(self):
        return self._nu_diff
        
    @property
    def perturbation(self):
        return self._ptbn

    @property
    def gamma(self):
        return self._gamma

    @property
    def mu(self):
        return self._mu

    def update(self, dt, Sigma, star=None, amax=None, G_0=None):
        """Update the eos"""
        if star:
            t = star.age
        else:
            t = 0
        if dt>0:
            self._nu_diff, self._nu_eff = self._f_nu(self._R, t=t)
        # Return Sigma divided by pertubation - only used in initial conditions
        return Sigma/self._ptbn(self._R, t=t)

    def ASCII_header(self):
        """Print eos header"""
        head = '# {} gamma: {}, mu: {}, ptbn: {}'
        return head.format(self.__class__.__name__,
                           self.gamma, self.mu, self._ptbn)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        def fmt(x):  return "{}".format(x)
        return self.__class__.__name__, { "gamma" : fmt(self.gamma),
                                          "mu" : fmt(self.mu) }
    
class LocallyIsothermalEOS(EOS_Table):
    """Simple locally isothermal power law equation of state:

    args:
        h0      : aspect ratio at 1AU
        q       : power-law index of sound-speed
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.4
    """
    def __init__(self, star, h0, q, alpha_t, mu=2.4, ptbn_kwargs={"Type":"None"}):
        super(LocallyIsothermalEOS, self).__init__(ptbn_kwargs=ptbn_kwargs)
        
        self._h0 = h0                               # Dimensionless
        self._cs0 = h0 * star.M**0.5                # Earth orbital velocity = 2pi AU/yr = AU/tdyn
        self._q = q
        self._H0 = h0                               # AU
        self._T0 = (AU*Omega0)**2 * mu / GasConst   # cm^2/AU^2 tdyn^2/s^2 g/mol molK/erg = AU^-2 tdyn^2 K
        self._mu = mu
        self._alpha_t = alpha_t
        
    def _f_cs(self, R):
        return self._cs0 * R**self._q

    def _f_H(self, R):
        return self._H0 * R**(1.5+self._q)
    
    def _f_nu(self, R, t=0):
        # Returns diffusive and total effective viscosities
        return self._f_alpha(R) * self._f_cs(R) * self._f_H(R), self._f_alpha(R) * self._f_cs(R) * self._f_H(R) * self._ptbn(R, t=t)

    def _f_alpha(self, R):
        """Handle being passed a list of alphas
        Since no change in alpha return single value"""
        if type(self._alpha_t) is list:
            if len(self._alpha_t)>1:
                print("Only one value of alpha needed. Using {} and ignoring {}.".format(self._alpha_t[0], self._alpha_t[1:]))
            return self._alpha_t[0]
        else:
            return self._alpha_t

    @property
    def T(self):
        return self._T0 * self.cs**2

    @property
    def Pr(self):
        return np.zeros_like(self._R)

    def ASCII_header(self):
        """LocallyIsothermalEOS header string"""
        head = super(LocallyIsothermalEOS, self).ASCII_header()
        head += ', h0: {}, q: {}, alpha: {}'
        return head.format(self._h0, self._q, self._alpha_t)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(LocallyIsothermalEOS, self).HDF5_attributes()
        head["h0"]   = "{}".format(self._h0)
        head["q"]     = "{}".format(self._q)
        head["alpha"] = "{}".format(self._alpha_t)
        return name, head

    @staticmethod
    def from_file(filename):
        raise NotImplementedError('')

    @property
    def star(self):
        return self._star

class TanhAlphaEOS(LocallyIsothermalEOS):
    """Variant that allows for smooth (tanh) changes in alpha"""
    def __init__(self, star, h0, q, alpha_t, mu=2.4, R_alpha=None, w=1.0, t_trap=-np.inf, ptbn_kwargs={"Type":"None"}):
        super(TanhAlphaEOS, self).__init__(star, h0, q, alpha_t, mu=mu,ptbn_kwargs=ptbn_kwargs)
        
        self._R_alpha = R_alpha
        self._R_alpha_w = w
        self._t_trap = t_trap

    def _f_nu(self, R, t=0):
        # Return Sigma divided by pertubation - only used in initial conditions
        return self._f_alpha(R, t=t) * self._f_cs(R) * self._f_H(R), self._f_alpha(R, t=t) * self._f_cs(R) * self._f_H(R) * self._ptbn(R, t=t)

    def _f_alpha(self, R, t=0):
        if self._R_alpha:
            # Break into regions of different alpha
            logalpha = np.full_like(R, np.log(self._alpha_t[0]))
            for r in range(0, len(self._R_alpha)):
                ka = np.log(self._alpha_t[r+1]/self._alpha_t[r])
                Ha = self.H[np.argmin(np.abs(R-self._R_alpha[r]))]
                logalpha += ka * self.smooth_step(R, self._R_alpha[r], self._R_alpha_w*Ha) * self.smooth_step(t, self._t_trap, max(1e-300,self._t_trap/10))
            return np.exp(logalpha)
        else:
            super(VariableAlphaEOS, self)._f_alpha(R)

    def update(self, dt, Sigma, star=None, amax=None, G_0=None):
        """Update the eos"""
        if star:
            t = star.age
        else:
            t = 0
        if dt>0 and self._t_trap>0:
            self._alpha  = self._f_alpha(self._R, t=t)
            self._nu_diff, self._nu_eff = self._f_nu(self._R, t=t)
        # Return Sigma divided by pertubation - only used in initial conditions
        return Sigma/self._ptbn(self._R, t=t)

    def smooth_step(self, x, center=0, width=1):
        return 0.5 * (1 + np.tanh((x-center)/width))

class ExternalHeatEOS(LocallyIsothermalEOS):
    """Variant following Haworth (2021) as parametrized by Sellek et al. (in prep)
    Uses 2400 A as limit for consistency with Haworth papers. 2070 A is also common 

    args:
        h0      : aspect ratio due to Star at 1AU
        q       : power-law index of sound-speed
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.4
        T_ext   : blackbody temperature of external FUV source (K), default=39000 (Theta1 Orionis C)
        #f_FUV   : fraction of external luminosity emitted in FUV, default=0.506    # Now calculated from T_ext
        G_0     : external FUV field in units of G_0 (1.63*10^-3 erg/s/cm^2)
    """

    def Planck(self, nu):
        return nu**3/(np.exp(nu)-1)

    def __init__(self, star, h0, q, alpha_t, mu=2.4, G_0=0.0, T_ext=39000, ptbn_kwargs={"Type":"None"}):
        super(ExternalHeatEOS, self).__init__(star, h0, q, alpha_t, mu=mu, ptbn_kwargs=ptbn_kwargs)

        self._hc_kT = 6.63e-27*3.00e10/1.38e-16/T_ext
        self._f_FUV = quad(self.Planck, self._hc_kT/2400e-8, self._hc_kT/912e-8)[0]/quad(self.Planck, 0, np.inf)[0]
        print("For T_ext={}, f_FUV={}".format(T_ext, self._f_FUV))
        self._G_0   = G_0
        self._Mstar = star.M

    def _f_T(self, R):
        return (self._T0**4 * self._h0**8 * self._Mstar**0.6 * R**(8*self._q) + 1.63e-3/sig_SB/self._f_FUV * self._G_0)**(1/4)

    def _f_cs(self, R):
        return np.sqrt(self._f_T(R)/self._T0)

    def _f_H(self, R):
        return self._f_cs(R) / self._Mstar**0.5 * R**1.5

    def update(self, dt, Sigma, star=None, amax=None, G_0=None, T_ext=None):
        if star:
            t = star.age
        else:
            t = 0
        # Update G_0, f_FUV and M* if necessary
        if G_0 is not None:
            self._G_0   = G_0
        if T_ext is not None:
            self._f_FUV = quad(self.Planck, self._hc_kT/2400e-8, self._hc_kT/912e-8)[0]/quad(self.Planck, 0, np.inf)[0]
        if star is not None:
            self._Mstar = star.M
        # Recalculate sound speed, scale height and viscosity profiles
        self._set_arrays(t=t)
        # Return Sigma divided by pertubation - only used in initial conditions
        return Sigma/self._ptbn(self._R, t=t)

_sqrt2pi = np.sqrt(2*np.pi)
class IrradiatedEOS(EOS_Table):
    """Model for an active irradiated disc.

    From Nakamoto & Nakagawa (1994), Hueso & Guillot (2005).

    args:
        star    : Stellar properties
        alpha_t : Viscous alpha parameter
        Tc      : External irradiation temperature (nebular), default=10
        Tmax    : Maximum temperature allowed in the disc, default=1500
        mu      : Mean molecular weight, default = 2.4
        gamma   : Ratio of specific heats
        kappa   : Opacity, default=Zhu2012
        accrete : Whether to include heating due to accretion,
                  default=True
    """
    def __init__(self, star, alpha_t, Tc=10, Tmax=1500., mu=2.4, gamma=1.4,
                 kappa=None,
                 accrete=True, tol=None, ptbn_kwargs={"Type":"None"}): # tol is no longer used
        super(IrradiatedEOS, self).__init__(ptbn_kwargs=ptbn_kwargs)

        self._star = star
        
        self._dlogHdlogRm1 = 2/7.

        self._alpha_t = alpha_t
        
        self._Tc = Tc
        self._Tmax = Tmax
        self._mu = mu

        self._accrete = accrete
        self._gamma = gamma

        if kappa is None:
            self._kappa = opacity.Zhu2012
        else:
            self._kappa = kappa
        
        self._T = None

        self._compute_constants()

    def _compute_constants(self):
        self._sigTc4 = sig_SB*self._Tc**4
        self._cs0 = (Omega0**-1/AU) * (GasConst / self._mu)**0.5
        self._H0  = (Omega0**-1/AU) * (GasConst / (self._mu*self._star.M))**0.5


    def update(self, dt, Sigma, star=None, amax=1e-5, G_0=None):
        if star:
            self._star = star
            self._compute_constants()
        star = self._star
            
        # Temperature/gensity independent quantities:
        R = self._R
        Om_k = Omega0 * star.Omega_k(R)

        X = star.Rau/R
        f_flat  = (2/(3*np.pi)) * X**3
        f_flare = 0.5 * self._dlogHdlogRm1 * X**2
        
        # Heat capacity
        mu = self._mu
        #C_V = (k_B / (self._gamma - 1)) * (1 / (mu * m_H))
        
        alpha = self._alpha_t
        if not self._accrete:
            alpha = 0.

        # Local references 
        max_heat = sig_SB * (self._Tmax*self._Tmax)*(self._Tmax*self._Tmax)
        star_heat = sig_SB * star.T_eff**4
        sqrt2pi = np.sqrt(2*np.pi)            
        def balance(Tm):
            """Thermal balance"""
            cs = np.sqrt(GasConst * Tm / mu)
            H = cs / Om_k

            kappa = self._kappa(Sigma / (sqrt2pi * H), Tm, amax)
            tau = 0.5 * Sigma * kappa
            H /= AU

            # External irradiation
            dEdt = self._sigTc4
            
            # Compute the heating from stellar irradiation
            dEdt += star_heat * (f_flat + f_flare * (H/R))

            # Viscous Heating
            visc_heat = 1.125*alpha*cs*cs * Om_k
            dEdt += visc_heat*(0.375*tau*Sigma + 1./kappa)
            
            # Prevent heating above the temperature cap:
            dEdt = np.minimum(dEdt, max_heat)

            # Cooling
            Tm2 = Tm*Tm
            dEdt -= sig_SB * Tm2*Tm2

            # Change in temperature
            return (dEdt/Omega0) # / (C_V*Sigma)

        # Solve the balance using brent's method (needs ~ 20 iterations)
        T0 = self._Tc
        T1 = self._Tmax
        if self._T is not None:
            dedt = balance(self._T)
            T0 = np.where(dedt > 0, self._T, T0)
            T1 = np.where(dedt < 0, self._T, T1)

        self._T =  brentq(balance, T0, T1)
        self._Sigma = Sigma

        # Save the opacity:
        cs = np.sqrt(GasConst * self._T / mu)
        H = cs / Om_k
        self._kappa_arr = self._kappa(Sigma / (sqrt2pi * H), self._T, amax)
        
        if star:
            t = star.age
        else:
            t = 0
        self._set_arrays(t=t)

        # Return Sigma divided by pertubation - only used in initial conditions
        return Sigma/self._ptbn(self._R, t=t)

    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = None

    def _set_arrays(self, t=0):
        super(IrradiatedEOS,self)._set_arrays(t=t)
        self._Pr = self._f_Pr()
    
    def __H(self, R, T):
        return self._H0 * np.sqrt(T * R*R*R)

    def _f_cs(self, R):
        return self._cs0 * self._T**0.5

    def _f_H(self, R):
        return self.__H(R, self._T)
    
    def _f_nu(self, R):
        # Returns diffusive and total effective viscosities
        return self._f_alpha(R) * self._f_cs(R) * self._f_H(R), self._f_alpha(R) * self._f_cs(R) * self._f_H(R) * self._ptbn(R)

    def _f_alpha(self, R):
        """Handle being passed a list of alphas
        Since no change in alpha return single value"""
        if type(self._alpha_t) is list:
            if len(self._alpha_t)>1:
                print("Only one value of alpha needed. Using {} and ignoring {}.".format(self._alpha_t[0], self._alpha_t[1:]))
            return self._alpha_t[0]
        else:
            return self._alpha_t

    def _f_Pr(self):
        kappa = self._kappa_arr
        tau = 0.5 * self._Sigma * kappa
        f_esc = 1 + 2/(3*tau*tau)
        Pr_1 =  2.25 * self._gamma * (self._gamma - 1) * f_esc
        return 1. / Pr_1

    @property
    def T(self):
        return self._T

    @property
    def Pr(self):
        return self._Pr

    @property
    def star(self):
        return self._star

    def ASCII_header(self):
        """IrradiatedEOS header"""
        head = super(IrradiatedEOS, self).ASCII_header()
        head += ', opacity: {}, T_extern: {}K, accrete: {}, alpha: {}'
        head += ', Tmax: {}K'
        return head.format(self._kappa.__class__.__name__,
                           self._Tc, self._accrete, self._alpha_t,
                           self._Tmax)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(IrradiatedEOS, self).HDF5_attributes()

        head["opacity"]  = self._kappa.__class__.__name__
        head["T_extern"] = "{} K".format(self._Tc)
        head["accrete"]  = "{}".format(bool(self._accrete))
        head["alpha"]    = "{}".format(self._alpha_t)
        head["Tmax"]     = "{} K".format(self._Tmax)

        return name, head

    @staticmethod
    def from_file(filename):
        import star

        star = star.from_file(filename)
        alpha = None

        with open(filename) as f:
            for line in f:
                if not line.startswith('#'):
                    raise AttributeError("Error: EOS type not found in header")
                elif "IrradiatedEOS" in line:
                    string = line 
                    break 
                else:
                    continue

        kwargs = {}
        for item in string.split(','):    
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'gamma' or key == 'mu':
                kwargs[key] = float(val.strip())
            elif key == 'alpha':
                alpha = float(val.strip())
            elif key == 'accrete':
                kwargs[key] = bool(val.strip())
            elif key == 'T_extern':
                kwargs['Tc'] = float(val.replace('K','').strip())

        return IrradiatedEOS(star, alpha, **kwargs)

def from_file(filename):
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                raise AttributeError("Error: EOS type not found in header")
            elif "IrradiatedEOS" in line:
                return IrradiatedEOS.from_file(filename)      
            elif "LocallyIsothermalEOS" in line:
                return LocallyIsothermalEOS.from_file(filename)
            else:
                continue

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .star import SimpleStar
    from .grid import Grid

    alpha = 1e-3
    star = SimpleStar(M=1.0, R=3.0, T_eff=4280.)

    active  = IrradiatedEOS(star, alpha)
    passive = IrradiatedEOS(star, alpha, accrete=False)
    marco   = IrradiatedEOS(star, alpha, kappa=opacity.Tazzari2016())

    powerlaw = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)

    grid = Grid(0.1, 500, 1000, spacing='log')
    
    Sigma = 2.2e3 / grid.Rc**1.5

    amax = 10 / grid.Rc**1.5
    
    c  = { 'active' : 'r', 'passive' : 'b', 'marco' : 'm',
           'isothermal' : 'g' }
    ls = { 0 : '-', 1 : '--' }
    for i in range(2):
        for eos, name in [[active, 'active'],
                          [marco, 'marco'],
                          [passive, 'passive'],
                          [powerlaw, 'isothermal']]:
            eos.set_grid(grid)
            eos.update(0, Sigma, amax=amax)

            label = None
            if ls[i] == '-':
                label = name
                
            plt.loglog(grid.Rc, eos.T, c[name] + ls[i], label=label)
        Sigma /= 10
    plt.legend()
    plt.show()
    
                    
