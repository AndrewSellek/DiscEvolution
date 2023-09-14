# photoevaportation.py
#
# Author: R. Booth
# Date: 24 - Feb - 2017
#
# Models for photo-evaporation of a disc
###############################################################################
import numpy as np
from .constants import AU, Msun, yr, Mjup
from .dust import DustyDisc
from .FRIED import photorate

class ExternalPhotoevaporationBase(object):
    """Base class for handling the external photo-evaporation of discs

    Implementations of ExternalPhotoevaporation classes must provide the
    following methods:
        mass_loss_rate(disc)     : returns mass-loss rate from outer edge
                                   (Msun/yr).
        max_size_entrained(dist) : returns the maximum size of dust entrained in
                                   the flow (cm).
    """

    def unweighted_rates(self, disc):
        """Calculates the raw mass loss rates for each annulus in code units"""
        # Locate and select cells that aren't empty OF GAS
        Sigma_G = disc.Sigma_G
        not_empty = (disc.Sigma_G > 0)

        # Annulus GAS masses
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_gas = disc.Sigma_G * dA

        # Get the photo-evaporation rates at each cell as if it were the edge USING GAS SIGMA
        Mdot = self.mass_loss_rate(disc,not_empty)

        # Convert Msun / yr to g / dynamical time
        dM_dot = Mdot * Msun / (2 * np.pi)

        return (dM_dot, dM_gas)

    def get_timescale(self, disc):
        """From mass loss rates, calculate mass loss timescales for each cell"""
        # Retrieve unweighted rates
        (dM_evap, dM_gas) = self.unweighted_rates(disc)     
        not_empty = (disc.Sigma_G > 0)

        # Work out which cells we need to empty of gas & entrained dust
        M_tot = np.cumsum(dM_gas[::-1])[::-1]
        Dt_R = np.zeros_like(dM_evap)
        Dt_R[not_empty] = dM_gas[not_empty] / dM_evap[not_empty] # Dynamical time for each annulus to be depleted of mass
        Dt = np.cumsum(Dt_R[::-1])[::-1] # Dynamical time to deplete each annulus and those exterior

        # Record mass loss
        if sum(not_empty)!=0:
            self._Rot  = disc.R[not_empty][-1]
            self._Mdot = dM_evap[not_empty][-1]*(yr/Msun)
        else:
            self._Rot  = 0
            self._Mdot = 0
            self._empty = True
        
        # Return mass loss rate, annulus mass, cumulative mass and cumulative timescale
        return (dM_evap, dM_gas, M_tot, Dt)

    def timescale_remove(self, disc, dt):
        """Remove gas from cells according to timescale implementation"""
        """Only implemented correctly for case with no dust"""
        # Retrieve timescales
        (dM_evap, dM_tot, M_tot, Dt) = self.get_timescale(disc)

        # Calculate which cells have times shorter than the timestep and empty
        excess_t = dt - Dt
        empty = (excess_t > 0)
        disc._Sigma[empty] = 0

        # Deal with marginal cell (first for which dt<Dt) as long as entire disc isn't removed
        if (np.sum(empty)<np.size(disc.Sigma_G)):
            half_empty = -(np.sum(empty) + 1) # ID (from end) of half depleted cell
            disc._Sigma[half_empty] *= -1.0*excess_t[half_empty] / (Dt[half_empty]-Dt[half_empty+1]) # Work out fraction left in cell 

        self._Mcum_gas  += dt*dM_evap[empty].sum()+dM_tot[half_empty]*-1.0*excess_t[half_empty] / (Dt[half_empty]-Dt[half_empty+1])  # Record mass loss
        
    def optically_thin_weighting(self, disc):
        """Identify optical thickness transition and weight raw mass loss rates exterior"""
        # Retrieve unweighted rates
        (Mdot, dM_gas) = self.unweighted_rates(disc)

        # Find the maximum, corresponding to optically thin/thick boundary
        i_max = np.size(Mdot) - np.argmax(Mdot[::-1]) - 1
        self._Rot = disc.R[i_max]

        # Weighting function USING GAS MASS
        ot_radii = (disc.R >= self._Rot)
        s_tot = np.sum(dM_gas[ot_radii])
        s_weight = dM_gas/s_tot
        s_weight *= ot_radii # Set weight of all cells inside the maximum to zero.
        M_dot_tot = np.sum(Mdot * s_weight) # Contribution of all cells to mass loss rate
        M_dot_eff = M_dot_tot * s_weight # Effective mass loss rate

        # Record mass loss
        self._Mdot = M_dot_tot*(yr/Msun)

        return (M_dot_eff, dM_gas)

    def weighted_removal(self, disc, dt):
        """Remove gas according to the weighted prescription"""
        # Annulus Areas
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        # Mass loss rates
        (dM_dot, dM_gas) = self.optically_thin_weighting(disc)

        # Account for dust entrainment
        # First get initial dust conditions
        if (isinstance(disc,DustyDisc)):
            # Record state before removal for calculating abundances/dust fractions 
            Sigma_G0 = disc.Sigma_G
            Sigma_D0 = disc.Sigma_D
            Sigma_0  = Sigma_G0 + Sigma_D0.sum(0)
            not_dustless = (Sigma_D0.sum(0) > 0)
            f_m = np.zeros_like(disc.Sigma)
            f_m[not_dustless] = disc.dust_frac[1,:].flatten()[not_dustless]/disc.integ_dust_frac[not_dustless]  # Large fraction

            # Update the maximum entrained size
            self.max_size_entrained(disc)
            # Work out the total mass in entrainable dust
            M_ent = self.dust_entrainment(disc)

        # Apply gas loss to Sigma
        dM_evap = dM_dot * dt
        disc._Sigma = np.maximum(disc._Sigma - dM_evap / dA, 0)
        self._Mcum_gas  += dM_evap.sum()  # Record mass loss
        
        if (isinstance(disc,DustyDisc)):
            new_Sigma_G = np.maximum(Sigma_G0 - dM_evap / dA, 0)
            # Apply dust loss (proportionally to gas loss) to Sigma
            M_ent_w = np.zeros_like(M_ent)
            M_ent_w[(dM_gas > 0)] = M_ent[(dM_gas > 0)] * dM_evap[(dM_gas > 0)] / dM_gas[(dM_gas > 0)]
            disc._Sigma = np.maximum(disc._Sigma - M_ent_w / dA, 0)
            new_Sigma_D = np.maximum(Sigma_D0.sum(0) - M_ent_w / dA, 0)
            self._Mcum_dust += M_ent_w.sum() # Record mass loss

            # Work out composition of wind
            if disc.chem:
                for atom in disc.chem.gas.atomic_abundance().atom_ids:
                    M_ent_ice = np.zeros_like(M_ent)
                    M_ent_gas = np.zeros_like(M_ent)
                    atom_ice = disc.chem.ice.atomic_abundance()[atom]/disc.chem.ice.total_abund         # Mass abundances of atoms / ice mass = mass fraction in ice
                    atom_gas = disc.chem.gas.atomic_abundance()[atom]/(1-disc.chem.ice.total_abund)     # Mass abundances of atoms / gas mass = mass fraction in gas
                    M_ent_ice[(dM_gas > 0)] = M_ent_w[(dM_gas > 0)]*atom_ice[(dM_gas > 0)]              # Mass of atom in removed gas
                    M_ent_gas[(dM_gas > 0)] = dM_evap[(dM_gas > 0)]*atom_gas[(dM_gas > 0)]              # Mass of atom in removed ice
                    self._Mcum_chem[atom] += np.nansum(M_ent_ice+M_ent_gas)                             # Cumulative mass lost
                    self._wind_abun[atom]  = np.nansum(M_ent_ice+M_ent_gas)/np.nansum(M_ent_w+dM_evap)  # Current composition
                self._wind_abun['d'] = M_ent_w.sum()/np.nansum(M_ent_w+dM_evap)                         # Dust mass fraction in wind

            not_empty    = (disc.Sigma > 0)
            not_gasless  = not_empty * (disc.Sigma_G > 0)
            not_dustless = not_empty * (Sigma_D0.sum(0) > 0)
            if disc.chem:
                # Update gas abundances
                for spec in disc.chem.gas.species:
                    disc.chem.gas[spec][not_gasless] *= new_Sigma_G[not_gasless] / Sigma_G0[not_gasless] * Sigma_0[not_gasless] / disc.Sigma[not_gasless]
                    disc.chem.gas[spec][~not_gasless] = 0.
                # Update ice abundances
                for spec in disc.chem.ice.species:
                    disc.chem.ice[spec][not_dustless] *= new_Sigma_D[not_dustless] / Sigma_D0.sum(0)[not_dustless] * Sigma_0[not_dustless] / disc.Sigma[not_dustless]
                    disc.chem.ice[spec][~not_dustless] = 0.
                disc.update_ices(disc.chem.ice)
            else:
                # Update the dust mass fractions directly
                disc._eps[0][not_dustless] *= new_Sigma_D[not_dustless] / Sigma_D0.sum(0)[not_dustless] * Sigma_0[not_dustless] / disc.Sigma[not_dustless]
                disc._eps[1][not_dustless] *= new_Sigma_D[not_dustless] / Sigma_D0.sum(0)[not_dustless] * Sigma_0[not_dustless] / disc.Sigma[not_dustless]
                disc._eps[:,~not_dustless] = 0.

    def dust_entrainment(self, disc):
        # Representative sizes
        a_ent = self._amax
        a = disc.grain_size
        amax = a[1,:].flatten()

        # Annulus DUST masses
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_dust = disc.Sigma_D * dA

        # Select cells with gas
        not_empty = (disc.Sigma_G>0)
        M_ent = np.zeros_like(disc.Sigma)

        # Calculate total that is entrained
        f_ent = np.minimum(np.ones_like(amax)[not_empty],[(a_ent[not_empty]/amax[not_empty])**(4-disc._p)]).flatten() # Take as entrained lower of all dust mass, or the fraction from MRN
        M_ent[not_empty] = dM_dust.sum(0)[not_empty] * f_ent
        return M_ent

    def __call__(self, disc, dt, age):
        """Remove gas and dust from the edge of a disc"""
        raise NotImplementedError("Derived class must implement the choice of mass loss proceedure")

def Facchini_limit(disc, Mdot):
    """
    Equation 35 of Facchini et al (2016)
    Note following definitions:
    F = H / sqrt(H^2+R^2) (dimensionless)
    v_th = \sqrt(8/pi) C_S in AU / t_dyn
    Mdot is in units of Msun yr^-1
    G=1 in units AU^3 Msun^-1 t_dyn^-2
    """
    
    F = disc.H / np.sqrt(disc.H**2+disc.R**2)
    rho = disc._rho_s
    Mstar = disc.star.M # In Msun
    v_th = np.sqrt(8/np.pi) * disc.cs
    
    a_entr = (v_th * Mdot) / (Mstar * 4 * np.pi * F * rho)
    a_entr *= Msun / AU**2 / yr
    return a_entr

## Available instances of the photoevaporation module ##
## - Fixed (user defined rate)
## - FRIED_S (uses the surface density at the edge directly)
## - FRIED_MS (extrapolates from surface density at the edge to M400) 
## - FRIED_M  (extrapolates from disc mass to M400)

class FixedExternalEvaporation(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a constant mass loss rate, which
    entrains dust below a fixed size.

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-8
        amax : maximum grain size entrained, default = 10 micron
    """

    def __init__(self, disc, Mdot=1e-8, tshield=0, amax=10):
        self._Mdot = Mdot
        self._tshield = tshield * yr
        self._amax = amax * np.ones_like(disc.R)
        self._empty = False

        self._Mcum_gas  = 0.0
        self._Mcum_dust = 0.0
        self._Mcum_chem = {atom: 0.0 for atom in disc.chem.gas.atomic_abundance().atom_ids}
        self._wind_abun = {atom: 0.0 for atom in disc.chem.gas.atomic_abundance().atom_ids}

    def __call__(self, disc, dt):
        if (self._Mdot > 0):
            self.timescale_remove(disc, dt)
    
    def mass_loss_rate(self, disc, not_empty):
        return self._Mdot * np.ones_like(disc.Sigma)

    def max_size_entrained(self, disc):
        return self._amax * np.ones_like(disc.Sigma)

    def ASCII_header(self):
        return ("# FixedExternalEvaportation, Mdot: {}, amax: {}"
                "".format(self._Mdot, self._amax))

    def HDF5_attributes(self):
        header = {}
        header['Mdot'] = '{}'.format(self._Mdot)
        header['amax'] = '{}'.format(self._amax)
        return self.__class__.__name__, header
    
class TimeExternalEvaporation(ExternalPhotoevaporationBase):
    """Mass loss via external evaporation at a constant mass-loss timescale,
    Almost certainly deprecated in this branch
        Mdot = pi R^2 Sigma / t_loss.

    args:
        time-scale : mass loss time-scale in years
        amax : maximum grain size entrained, default = 10 micron
    """

    def __init__(self, time=1e6, amax=1e-3):
        self._time = time
        self._amax = amax

        self._Mcum_gas  = 0.0
        self._Mcum_dust = 0.0
        self._Mcum_chem = {atom: 0.0 for atom in disc.chem.gas.atomic_abundance().atom_ids}
        self._wind_abun = {atom: 0.0 for atom in disc.chem.gas.atomic_abundance().atom_ids}

    def mass_loss_rate(self, disc):
        k = np.pi * AU**2 / Msun
        return k * disc.R**2 * disc.Sigma / self._time

    def max_size_entrained(self, disc):
        return self._amax * np.ones_like(disc.Sigma)

    def ASCII_header(self):
        return ("# TimeExternalEvaportation, time: {}, amax: {}"
                "".format(self._time, self._amax))

    def HDF5_attributes(self):
        header = {}
        header['time'] = '{}'.format(self._time)
        header['amax'] = '{}'.format(self._amax)
        return self.__class__.__name__, header

###### FRIED Variants
class FRIEDExternalEvaporationBase(ExternalPhotoevaporationBase):
    """
    General FRIED Implementation
    Implementations of this must specify which function from photorate to use.
    They must also set self._density depending on whether the interpolation over the FRIED grid is over a function of the surface densty (True) or total mass (False).

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-10
        amax : maximum grain size entrained, default = 0
        FUV  : The irradiating FUV field in Habing unit (G0), default = 0
        tshield : A delay to the onset of photoevaporation in yr (in development)
    """

    def __init__(self, disc, tshield=0, amax=0, Mcum_gas = 0.0, Mcum_dust = 0.0, Mcum_chem=None, evolvedDust=True):
        self._Mdot = 0.0
        self._tshield = tshield * yr
        self._amax = amax * np.zeros_like(disc.R)
        self._FUV = disc.FUV
        self._Rot = max(disc.R)
        self._evolvedDust = evolvedDust

        self._Mcum_gas  = Mcum_gas
        self._Mcum_dust = Mcum_dust
        if Mcum_chem:
            self._Mcum_chem = {atom: Mcum_chem[atom] for atom in disc.chem.gas.atomic_abundance().atom_ids}
            self._wind_abun = {atom: Mcum_chem[atom] for atom in disc.chem.gas.atomic_abundance().atom_ids}
            self._wind_abun['d'] = Mcum_chem['d']
        elif disc.chem:
            self._Mcum_chem = {atom: 0.0 for atom in disc.chem.gas.atomic_abundance().atom_ids}
            self._wind_abun = {atom: 0.0 for atom in disc.chem.gas.atomic_abundance().atom_ids} 
            self._wind_abun['d'] = 0.0   

    def __call__(self, disc, dt):
        if self._density:    # For FRIED mass loss rates calculated with density, need to use optical depth method
            self.weighted_removal(disc, dt)
        else:                # For FRIED mass loss rates calculated with total mass, can use timescale method (doesn't account for dust)
            self.timescale_remove(disc, dt)

    def mass_loss_rate(self, disc, not_empty):
        calc_rates = np.zeros_like(disc.R)
        if self._density:    # For FRIED mass loss rates calculated with density, calculate directly
            calc_rates[not_empty] = self.FRIED_Rates.PE_rate(( disc.Sigma_G[not_empty], disc.R[not_empty] ))
        else:                # For FRIED mass loss rates calculated with total mass, first calculate the cumulative mass
            Re = disc.R_edge * AU
            dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
            dM_gas = disc.Sigma_G * dA
            integ_mass = np.cumsum(dM_gas)/ Mjup
            calc_rates[not_empty] = self.FRIED_Rates.PE_rate(( integ_mass[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        calc_rates[norate] = 1e-10
        if not self._evolvedDust:
            calc_rates[(calc_rates>1e-10)] = calc_rates[(calc_rates>1e-10)]/10
        return calc_rates

    def max_size_entrained(self, disc):
        # Update maximum entrained size
        self._amax = Facchini_limit(disc,self._Mdot)
        return self._amax

class FRIEDExternalEvaporationS(FRIEDExternalEvaporationBase):
    """External photoevaporation flow with a mass loss rate which is dependent on radius and surface density.
    """

    def __init__(self, disc, **kwargs):
        super().__init__(disc, **kwargs)
        self.FRIED_Rates = photorate.FRIED_2DS(photorate.grid_parameters,photorate.grid_rate,disc.star.M,self._FUV)
        self._density = True

    def ASCII_header(self):
        return ("# FRIEDExternalEvaporationS: {} G0".format(self._FUV))

    def HDF5_attributes(self):
        header = {}
        return self.__class__.__name__, header

class FRIEDExternalEvaporationMS(FRIEDExternalEvaporationBase):
    """
    External photoevaporation flow with a mass loss rate which is dependent on radius and surface density.
    Calculated by converting to the mass within 400 AU (M400 ~ R Sigma)
    """

    def __init__(self, disc, **kwargs):
        super().__init__(disc, **kwargs)
        self.FRIED_Rates = photorate.FRIED_2DM400S(photorate.grid_parameters,photorate.grid_rate,disc.star.M,self._FUV)
        self._density = True

    def ASCII_header(self):
        return ("# FRIEDExternalEvaporationMS: {} G0".format(self._FUV))

    def HDF5_attributes(self):
        header = {}
        return self.__class__.__name__, header

class FRIEDExternalEvaporationfMS(FRIEDExternalEvaporationBase):
    """
    External photoevaporation flow with a mass loss rate which is dependent on radius and surface density.
    Calculated by converting to the mass within 400 AU (M400 ~ R Sigma)
    """

    def __init__(self, disc, **kwargs):
        super().__init__(disc, **kwargs)
        self.FRIED_Rates = photorate.FRIED_2DfM400S(photorate.grid_parameters,photorate.grid_rate,disc.star.M,self._FUV)
        self._density = True

    def ASCII_header(self):
        return ("# FRIEDExternalEvaporationfMS: {} G0".format(self._FUV))

    def HDF5_attributes(self):
        header = {}
        return self.__class__.__name__, header

class FRIEDExternalEvaporationM(FRIEDExternalEvaporationBase):
    """
    External photoevaporation flow with a mass loss rate which is dependent on radius and integrated mass interior.
    Calculated by converting to the mass within 400 AU (M400 ~ M / R)
    """

    def __init__(self, disc, **kwargs):
        super().__init__(disc, **kwargs)
        self.FRIED_Rates = photorate.FRIED_2DM400M(photorate.grid_parameters,photorate.grid_rate,disc.star.M,self._FUV)
        self._density = False

    def ASCII_header(self):
        return ("# FRIEDExternalEvaporationM: {} G0".format(self._FUV))

    def HDF5_attributes(self):
        header = {}
        return self.__class__.__name__, header

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .grid import Grid
    from .eos import LocallyIsothermalEOS
    from .star import SimpleStar
    from .dust import FixedSizeDust

    # Set up accretion disc properties
    Mdot = 1e-8
    alpha = 1e-3

    Mdot *= Msun / (2 * np.pi)
    Mdot /= AU ** 2
    Rd = 100.

    grid = Grid(0.1, 1000, 1000, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-grid.Rc / Rd)

    # Setup a dusty disc model
    disc = FixedSizeDust(grid, star, eos, 0.01, [1e-4, 0.1], Sigma=Sigma)

    # Setup the photo-evaporation
    photoEvap = FixedExternalEvaporation()

    times = np.linspace(0, 1e7, 6) * 2 * np.pi

    dA = np.pi * np.diff((disc.R_edge * AU) ** 2) / Msun

    # Test the removal of gas / dust
    t, M = [], []
    tc = 0
    for ti in times:
        photoEvap(disc, ti - tc)

        tc = ti
        t.append(tc / (2 * np.pi))
        M.append((disc.Sigma * dA).sum())

        c = plt.plot(disc.R, disc.Sigma_G)[0].get_color()
        plt.loglog(disc.R, disc.Sigma_D[0], c + ':')
        plt.loglog(disc.R, 0.1 * disc.Sigma_D[1], c + '--')

    plt.xlabel('R')
    plt.ylabel('Sigma')

    t = np.array(t)
    plt.figure()
    plt.plot(t, M)
    plt.plot(t, M[0] - 1e-8 * t, 'k')
    plt.xlabel('t [yr]')
    plt.ylabel('M [Msun]')
    plt.show()
