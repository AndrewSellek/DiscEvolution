from __future__ import print_function
from operator import xor
import numpy as np
from ..constants import *
from .base_chem import ChemicalAbund, MolecularIceAbund
from .base_chem import TimeDependentChem, EquilibriumChem

################################################################################
# Simple Chemistry wrappers
################################################################################
class SimpleAtomAbund(ChemicalAbund):
    """Class to hold the raw atomic abundaces of C/N/O/Si/S for the CNO chemistry"""

    def __init__(self, *sizes):
        self.atom_ids = ['H', 'He', 'C', 'N', 'O', 'Si', 'S']
        masses = [1., 4., 12., 14., 16., 28., 32.]

        super(SimpleAtomAbund, self).__init__(self.atom_ids, masses, *sizes)

    def set_solar_abundances(self, muH=1.41):
        """Solar mass fractions of C, N, O, Si and S (Asplund 2021; Table 2)

        args:
            muH : mean atomic mass, default = 1.41
        """
        log12 = np.array([12.00, 8.46, 7.83, 8.69, 7.51, 7.12])
        A_abund = np.power(10,log12-12.0)
        m_abund = np.array(self.masses)*A_abund
        self._AH = np.sum(A_abund)
        self._muH = np.sum(m_abund)
        self._mu = self._muH/self._AH
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / self._muH

    def set_protosolar_abundances(self, muH=1.41):
        """Protoolar mass fractions of C, N, O, Si and S (Asplund 2021; Table B.1)
        Main difference, besides listing isotopes separately, is that all are adjsuted up by ~0.06 to account for atomic diffusion

        args:
            muH : mean atomic mass, default = 1.41
        """
        log12 = np.array([12.00, 8.52, 7.89, 8.75, 7.57, 7.16])
        A_abund = np.power(10,log12-12.0)
        m_abund = np.array(self.masses)*A_abund
        self._AH = np.sum(A_abund)
        self._muH = np.sum(m_abund)
        self._mu = self._muH/self._AH
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / self._muH

    def set_adopted_abundances(self, muH=1.41):
        """Adopted mass fractions of C, N, O, Si and S (Sellek et al. in prep)
        Si is solar (Asplund 2021) = cosmic (Nieva & Przybilla 2012)
        C is nebular (Simon-Diaz & Stasinska 2011)
        O is nebular + assumed silicates = intermediate between solar & cosmic

        args:
            muH : mean atomic mass, default = 1.41
        """
        A_abund = np.array([1.0, 9.8e-2, 2.3e-4, 0.0, 5.3e-4, 3.2e-5, 0.0])
        m_abund = np.array(self.masses)*A_abund
        self._AH = np.sum(A_abund)
        self._muH = np.sum(m_abund)
        self._mu = self._muH/self._AH
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / self._muH

    def set_mass_abundances(self, muH=1.41):
        """Mass abundance fractions of C, N, O, Si and S.

        args:
            muH : mean atomic mass, default = 1.41
        """
        m_abund = np.array(self.masses)
        self._data[:] = np.outer(m_abund, np.ones(self.size))

    def set_unit_abundances(self, muH=1.41):
        """Unit abundance fractions of C, N, O and Si.

        args:
            muH : mean atomic mass, default = 1.41
        """
        m_abund = np.ones_like(self.masses)
        self._data[:] = np.outer(m_abund, np.ones(self.size))
     
    """   
    @property
    def mu(self):
        return self._mu

    @property
    def muH(self):
        return self._muH
        
    @property
    def AH(self):
        return self._AH
    """

class SimpleMolAbund(ChemicalAbund):
    """Class that holds the abundances of molecules needed for chemistry"""

    def __init__(self, *sizes):
        mol_ids = ['Si-grain', 'C-grain',
                   'H2O', 'O2',
                   'CO2', 'CO', 'CH3OH', 'CH4', 'C2H2', 'C2H4', 'C2H6',
                   'H','H2','He']
        mol_mass = [76., 12.,
                    18., 32.,
                    44., 28., 32., 16., 26., 28., 30.,
                    1., 2., 4.]

        super(SimpleMolAbund, self).__init__(mol_ids, mol_mass, *sizes)

        # Atomic make up of the molecules:
        self._n_spec = {
                        'Si-grain': {'O': 3, 'Si': 1, },
                        'C-grain': {'C': 1, },
                        'H2O': {'O': 1, 'H': 2, },
                        'O2': {'O': 2, },
                        'CO2': {'C': 1, 'O': 2, },
                        'CO': {'C': 1, 'O': 1, },
                        'CH3OH': {'C': 1, 'O': 1, 'H': 4, },
                        'CH4': {'C': 1, 'H': 4, },
                        'C2H2': {'C': 2, 'H': 2, },
                        'C2H4': {'C': 2, 'H': 4, },
                        'C2H6': {'C': 2, 'H': 6, },
                        'H': {'H': 1, },
                        'H2': {'H': 2, },
                        'He': {'He': 1, },
                        }

    def atomic_abundance(self):
        """Compute the mass abundances of atomic species in the molecules"""

        atomic_abund = SimpleAtomAbund(self.data.shape[1])
        for mol in self.species:
            nspec = self._n_spec[mol]
            for atom in nspec:
                n_atom = (self[mol] / self.mass(mol)) * nspec[atom]
                atomic_abund[atom] += n_atom * atomic_abund.mass(atom)

        return atomic_abund
        
        
###############################################################################
# Specific Chemical models
###############################################################################
class ChemExtended(object):
    """Chemical ratios from Sellek+ (in prep.)

    args:
        fNH3 : Fraction of nitrogen in NH_3
        fix_grains : Whether to fix the dust grain abundances when recomputing
                     the molecular abundances
        fix_NH3    : Whether to fix the nitrogen abundance when recomputing the
                     molecular abundances
    """
    def __init__(self, ratesFile=None, zetaCR=1.30e-17):
        """Initialisation of reactions""" 
        self._zetaCR = zetaCR
        self._atunnel = 1.0e-8
        self._gas_reactions = []
        self._gas_rates     = []
        self._ice_reactions = []
        self._ice_rates     = []

        if ratesFile is not None:
            for row in open(ratesFile):
                if row[0]=='#':
                    continue
                reaction, *params = row.split("#")[0].replace("\n","").split("\t")
                if "(s)" in reaction:
                    spec1, spec2, Ebar = params
                    self._ice_reactions.append(reaction.replace("(s)",""))
                    self._ice_rates.append((spec1,spec2,float(Ebar)))
                else:
                    alpha, beta, gamma = params
                    self._gas_reactions.append(reaction)
                    self._gas_rates.append((float(alpha),float(beta),float(gamma)))
                    
        self._Nreacts = len(self._gas_reactions+self._ice_reactions)
        print("Included {} reactions:\n".format(self._Nreacts), '\n'.join(self._gas_reactions), '\n'.join(self._ice_reactions))

    def ASCII_header(self):
        """Extended chem header"""
        return (super(ChemExtended, self).ASCII_header())

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        __, header = super(ChemExtended, self).HDF5_attributes()
        return self.__class__.__name__, header
        
    def UMISTformat(self, T, alpha, beta, gamma):
        return alpha * (T/300)**beta * np.exp(-gamma/T)

    def UMISTformat_CR(self, T, alpha, beta, gamma):
        return self._zetaCR * (T/300)**beta * gamma

    def grain_surface(self, T, n_d, n_ice, spec1, spec2, Ebar):
        Nsites_tot = self._mu * self._etaNbind * n_d
        Cgr = np.minimum(1, Nsites_tot**2/n_ice**2) / Nsites_tot
        fdiff = 0.3
        mu = self.reduced_mass(spec1,spec2)
        nu1, nu2 = self._nu_pre['pure']['spec1'], self._nu_pre['pure']['spec2']
        T1,  T2  = self._Tbind['pure']['spec1'],  self._Tbind['pure']['spec2']
        Prob_spec1_spec2 = np.exp(-2.*self._atunnel/hbar * np.sqrt(2.*mu*m_H*k_B*Ebar))
        return Cgr * Prob_spec1_spec2 * (nu1 * np.exp(-fdiff*T1/T) + nu2 * np.exp(-fdiff*T2/T))

    def grain_surface_H(self, T, n_d, n_ice, Ebar = 860):
        """Reaction rate with H on grain surfaces"""
        Nsites_tot = self._mu * self._etaNbind * n_d
        Cgr = np.minimum(1, Nsites_tot**2/n_ice**2) / Nsites_tot
        fdiff = 0.3
        mu = 34/35
        nu_H  = 1.54e11
        Hbind = 450
        hop_H = np.maximum( np.exp(-fdiff*Hbind/T), np.exp(-2.*self._atunnel/hbar * np.sqrt(2.*m_H*k_B*fdiff*Hbind )) )
        Prob_spec1_H = np.exp(-2.*self._atunnel/hbar * np.sqrt(2.*mu*m_H*k_B*Ebar))
        return Cgr * Prob_spec1_H * nu_H * hop_H

    def initial_molecular_abundance(self, atomic_abund, T, rho_dust):
        """Compute the initial fractions of species present given total abundances

        args:
             atomic_abund : atomic abundaces, SimpleAtomAbund object

        returns:
            nmol : array(3, N) molecular mass-densities
        """

        # Get total element budget - number abundances with respect to total number of atoms
        H  = atomic_abund.number_abund('H')
        He = atomic_abund.number_abund('He')
        C  = atomic_abund.number_abund('C')
        O  = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')

        # Set up the number abundances for molecules
        mol_abund = SimpleMolAbund(atomic_abund.size)

        # Set the grain abundances
        mol_abund['C-grain']  = 0.39 * C
        mol_abund['Si-grain'] = Si

        # Assign C budget
        mol_abund['CO2']      = 0.09 * C
        mol_abund['CO']       = 0.50 * C
        mol_abund['CH3OH']    = 0.01 * C
        mol_abund['CH4']      = 0.01 * C
        mol_abund['C2H2']     = 0.00 * C
        mol_abund['C2H4']     = 0.00 * C
        mol_abund['C2H6']     = 0.00 * C
        
        # Assign O budget; water is 20%, any remainder goes into O2
        mol_abund['H2O']      = 0.20 * O
        for spec in ['Si-grain','H2O','CO2','CO','CH3OH']:
            O -= mol_abund[spec]*mol_abund._n_spec[spec]['O']
        mol_abund['O2']       = np.maximum(O / mol_abund._n_spec['O2']['O'], 0.)
        
        # Set the volatile abundances
        mol_abund['He']       = He
        for spec in ['H2O','CH3OH','CH4','C2H2','C2H4','C2H6']:
            H -= mol_abund[spec]*mol_abund._n_spec[spec]['H']
        mol_abund['H2']       = np.maximum(H, 0.) / mol_abund._n_spec['H2']['H']
        """
        k_CR  = self.UMISTformat_CR(T, 1.30e-17, 0, 2.4) / Omega0                       # Assuming 2.4 H created per CR ionization (Krijt et al. 2020)
        k_ads = self._f_ads * self._v_therm(T, mol_abund.mass(spec)) * rho_dust / m_H
        fatom = k_CR/(k_CR+k_ads)
        mol_abund['H']        = np.maximum(H, 0.) * fatom / mol_abund._n_spec['H']['H']
        mol_abund['H2']       = np.maximum(H, 0.) * (1-fatom) / mol_abund._n_spec['H2']['H']
        """

        #  Convert number abundances with respect to total number of atoms to mass fractions
        for spec in mol_abund.species:
            mol_abund[spec] *= mol_abund.mass(spec)/atomic_abund.mu()
            
        return mol_abund
        
    def convert_molecular_abundance(self, T, rho, ice_abund, gas_abund, dt):
        """Compute the fractions of species present given total abundances

        args:
             T            : array(N)   temperature (K)
             rho          : array(N)   gas density (g/cm^3)
             ice_abund    : SimpleMolAbund object: ice abundances
             gas_abund    : SimpleMolAbund object: gas abundances

        returns:
            nmol : array(3, N) molecular mass-densities
        """
        calc_dt = np.inf
        
        # Define 'densities' of refractories and ices
        n_d   = 0.
        n_ice = 0.
        He_sinks = rho/m_H * (gas_abund['H2']/gas_abund.mass('H2')) *  self.UMISTformat(T, 4e-14, 0, 0) # Base level is H2 ionization
        for spec in ice_abund.species:
            if spec=='H' or spec=='He':
                continue
            elif spec=='H2':
                n_H2   = rho*gas_abund[spec]/(gas_abund.mass(spec)*m_H)
            elif 'grain' in spec:
                n_d   += rho*ice_abund[spec]/(self._mu*m_H)
            else:
                n_ice += rho*ice_abund[spec]/(ice_abund.mass(spec)*m_H)
                
        ## Store all rates before doing reactions
        norm_rates = {}
        # Grain surface
        for react, rate in zip(self._ice_reactions,self._ice_rates):
            reactants, products = react.replace(' ','').split('->')
            reactants = reactants.split('+')
            weights_r = [float(reactant.split('*')[0]) if '*' in reactant else 1.0 for reactant in reactants]
            reactants = [reactant.split('*')[-1] for reactant in reactants]
            products  = products.split('+')
            weights_p = [float(product.split('*')[0]) if '*' in product else 1.0 for product in products]
            products  = [product.split('*')[-1] for product in products]
            if 'CR' in reactants:
                raise NotImplementedError
            elif "H" in reactants:
                krate = self.grain_surface_H(T, n_d, n_ice, rate[-1])
            else:
                krate = self.grain_surface(T, n_d, n_ice, *rate)
            if "H" in reactants:
                S_chem_fudge = 3
                n_S_ice = 2e-8 * rho / (self._mu*m_H)
                k_CR = self.UMISTformat_CR(T, self._zetaCR, 0, 2.4)
                k_react = self.grain_surface_H(T, n_d, n_ice) * S_chem_fudge * n_S_ice
                ice_abund_H = k_CR/k_react * n_H2
                weights_r.pop(reactants.index('H'))
                reactants.pop(reactants.index('H'))
                norm_rates[react] = ice_abund_H * (ice_abund[reactants[0]]/ice_abund.mass(reactants[0])) * krate/Omega0
            else:
                norm_rates[react] = rho/m_H * (ice_abund[reactants[0]]/ice_abund.mass(reactants[0])) * (ice_abund[reactants[1]]/ice_abund.mass(reactants[1])) * krate/Omega0
        # Gas phase
        for react, rate in zip(self._gas_reactions,self._gas_rates):
            reactants, products = react.replace(' ','').split('->')
            reactants = reactants.split('+')
            weights_r = [float(reactant.split('*')[0]) if '*' in reactant else 1.0 for reactant in reactants]
            reactants = [reactant.split('*')[-1] for reactant in reactants]
            products  = products.split('+')
            weights_p = [float(product.split('*')[0]) if '*' in product else 1.0 for product in products]
            products  = [product.split('*')[-1] for product in products]
            
            if 'CR' in reactants:
                krate = self.UMISTformat_CR(T, *rate)
                weights_r.pop(reactants.index('CR'))
                reactants.pop(reactants.index('CR'))
            elif 'Hej' in reactants:
                krate = self.UMISTformat(T, *rate)
                reactants[reactants.index('Hej')]='He'
                He_sinks += rho/m_H * (gas_abund[reactants[0]]/gas_abund.mass(reactants[0])) * krate
            else:
                krate = self.UMISTformat(T, *rate)
            if len(reactants)==2:            
                norm_rates[react] = rho/m_H * (gas_abund[reactants[0]]/gas_abund.mass(reactants[0])) * (gas_abund[reactants[1]]/gas_abund.mass(reactants[1])) * krate/Omega0
            elif len(reactants)==1:
                norm_rates[react] = (gas_abund[reactants[0]]/gas_abund.mass(reactants[0])) * krate/Omega0
            else:
                raise NotImplementedError
                
        ## Return timestep 
        if dt is None:
            # Grain surface
            for react, rate in zip(self._ice_reactions,self._ice_rates):
                reactants, products = react.replace(' ','').split('->')
                reactants = reactants.split('+')
                weights_r = [float(reactant.split('*')[0]) if '*' in reactant else 1.0 for reactant in reactants]
                reactants = [reactant.split('*')[-1] for reactant in reactants]
                if 'CR' in reactants:
                    raise NotImplementedError
                elif "H" in reactants:
                    weights_r.pop(reactants.index('H'))
                    reactants.pop(reactants.index('H'))
                for r, w in zip(reactants, weights_r):
                    calc_dt = min(calc_dt, np.nanmin(ice_abund[r]/(ice_abund.mass(r) * norm_rates[react] * w)))
            # Gas phase
            for react, rate in zip(self._gas_reactions,self._gas_rates):
                reactants, products = react.replace(' ','').split('->')
                reactants = reactants.split('+')
                weights_r = [float(reactant.split('*')[0]) if '*' in reactant else 1.0 for reactant in reactants]
                reactants = [reactant.split('*')[-1] for reactant in reactants]
                if 'CR' in reactants:
                    weights_r.pop(reactants.index('CR'))
                    reactants.pop(reactants.index('CR'))
                elif 'Hej' in reactants:
                    reactants[reactants.index('Hej')]='He'
                    norm_rates[react] *= 0.65*self._zetaCR/He_sinks
                for r, w in zip(reactants, weights_r):
                    calc_dt = min(calc_dt, np.nanmin(gas_abund[r]/(gas_abund.mass(r) * norm_rates[react] * w)))
            return calc_dt
                
        ## Apply reaction rates
        # Grain surface
        for react, rate in zip(self._ice_reactions,self._ice_rates):
            reactants, products = react.replace(' ','').split('->')
            reactants = reactants.split('+')
            weights_r = [float(reactant.split('*')[0]) if '*' in reactant else 1.0 for reactant in reactants]
            reactants = [reactant.split('*')[-1] for reactant in reactants]
            products  = products.split('+')
            weights_p = [float(product.split('*')[0]) if '*' in product else 1.0 for product in products]
            products  = [product.split('*')[-1] for product in products]
            if 'CR' in reactants:
                raise NotImplementedError
            elif "H" in reactants:
                weights_r[reactants.index('H')]/=2.0
                reactants[reactants.index('H')]='H2'
            for r, w in zip(reactants, weights_r):
                ice_abund[r] -= ice_abund.mass(r) * norm_rates[react] * w * dt
            for p, w in zip(products, weights_p):
                ice_abund[p] += ice_abund.mass(p) * norm_rates[react] * w * dt
        # Gas phase
        for react, rate in zip(self._gas_reactions,self._gas_rates):
            reactants, products = react.replace(' ','').split('->')
            reactants = reactants.split('+')
            weights_r = [float(reactant.split('*')[0]) if '*' in reactant else 1.0 for reactant in reactants]
            reactants = [reactant.split('*')[-1] for reactant in reactants]
            products  = products.split('+')
            weights_p = [float(product.split('*')[0]) if '*' in product else 1.0 for product in products]
            products  = [product.split('*')[-1] for product in products]
            if 'CR' in reactants:
                weights_r.pop(reactants.index('CR'))
                reactants.pop(reactants.index('CR'))
            elif 'Hej' in reactants:
                reactants[reactants.index('Hej')]='He'
                norm_rates[react] *= 0.65*self._zetaCR/He_sinks
            for r, w in zip(reactants, weights_r):
                gas_abund[r] -= gas_abund.mass(r) * norm_rates[react] * w * dt
            for p, w in zip(products, weights_p):
                gas_abund[p] += gas_abund.mass(p) * norm_rates[react] * w * dt

        return ice_abund, gas_abund
        
    def max_timestep(self, disc):
        """Compute the maximum timestep that does not completely deplete a reactant

        args:
             disc         : disc object

        returns:
            dt : timestep (default=infinity if no reactions)
        """
        dt = np.inf
        if self._Nreacts==0:
            return dt
            
        T = disc.T
        rho = disc.midplane_gas_density
        ice_abund = disc.chem.ice
        gas_abund = disc.chem.gas
        tot_abund = ice_abund.copy()
        tot_abund += gas_abund.copy()
        self._mu = tot_abund.mu()
        
        dt = self.convert_molecular_abundance(T, rho, ice_abund, gas_abund, None)
                
        return dt/np.e  # Add a factor of e to avoid going exactly to 0 in non-empty cells
        
###############################################################################
# Combined Models
###############################################################################
class EquilibriumChemExtended(ChemExtended, EquilibriumChem):
    def __init__(self, fix_ratios=True, ratesFile=None, **kwargs):
        #assert fix_ratios,"For Extended chem, no option to reset implemented, cannot run a model with fix_ratios=False"
        ChemExtended.__init__(self, ratesFile)
        EquilibriumChem.__init__(self, fix_ratios=fix_ratios, **kwargs)

class TimeDepChemExtended(ChemExtended, TimeDependentChem):
    def __init__(self, **kwargs):
        raise NotImplementedError
        ChemExtended.__init__(self)
        TimeDependentChem.__init__(self, **kwargs)

###############################################################################
# Tests
############################################################################### 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..eos import LocallyIsothermalEOS
    from ..star import SimpleStar
    from ..grid import Grid

    # Compare equilibrium chem with equilibrium of TD chem:

    # DISC model
    GM = 1.
    cs0 = (1 / 30.)
    q = -0.25
    Mdot = 1e-8
    alpha = 1e-3

    Mdot *= Msun / (2 * np.pi)
    Mdot /= AU ** 2

    Rin = 0.01
    Rout = 5e2
    Rd = 100.

    t0 = (2 * np.pi)

    d2g = 0.01

    muH = 1.42

    grid = Grid(0.01, 1000, 1000, spacing='log')
    R = grid.Rc
    eos = LocallyIsothermalEOS(SimpleStar(), cs0, q, alpha)
    eos.set_grid(grid)
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R / Rd)
    rho = Sigma / (np.sqrt(2 * np.pi) * eos.H * AU)

    T = eos.T
    n = rho / (2.4 * m_H)

    EQ_chem = SimpleChemExtended()
    TD_chem = TimeDepChemExtended(a=1e-5)

    X_solar = SimpleAtomAbund(n.shape[0])
    X_solar.set_solar_abundances()

    # Simple chemistry of Madhu:
    plt.subplot(211)
    S_chem = SimpleCNOChemMadhu()
    EQ_chem = EquilibriumCNOChemMadhu()

    S_mol = S_chem.equilibrium_chem(T, rho, d2g, X_solar)
    EQ_mol = EQ_chem.equilibrium_chem(T, rho, d2g, X_solar)

    S_atom = S_mol.gas.atomic_abundance()
    EQ_atom = EQ_mol.gas.atomic_abundance()
    plt.semilogx(R, S_atom.number_abund('C') * 1e4 * muH, 'r-')
    plt.semilogx(R, S_atom.number_abund('N') * 1e4 * muH, 'g-')
    plt.semilogx(R, S_atom.number_abund('O') * 1e4 * muH, 'b-')
    plt.semilogx(R, EQ_atom.number_abund('C') * 1e4 * muH, 'r:')
    plt.semilogx(R, EQ_atom.number_abund('N') * 1e4 * muH, 'g:')
    plt.semilogx(R, EQ_atom.number_abund('O') * 1e4 * muH, 'b:')
    plt.ylabel(r'$[X/H]\,(\times 10^4)$')

    plt.subplot(212)
    S_chem = SimpleChemExtended()
    EQ_chem = EquilibriumChemExtended()

    S_mol = S_chem.equilibrium_chem(T, rho, d2g, X_solar)
    EQ_mol = EQ_chem.equilibrium_chem(T, rho, d2g, X_solar)

    S_atom = S_mol.gas.atomic_abundance()
    EQ_atom = EQ_mol.gas.atomic_abundance()

    plt.semilogx(R, S_atom.number_abund('C') * 1e4 * muH, 'r-')
    plt.semilogx(R, S_atom.number_abund('N') * 1e4 * muH, 'g-')
    plt.semilogx(R, S_atom.number_abund('O') * 1e4 * muH, 'b-')
    plt.semilogx(R, EQ_atom.number_abund('C') * 1e4 * muH, 'r:')
    plt.semilogx(R, EQ_atom.number_abund('N') * 1e4 * muH, 'g:')
    plt.semilogx(R, EQ_atom.number_abund('O') * 1e4 * muH, 'b:')
    plt.ylabel(r'$[X/H]\,(\times 10^4)$')
    plt.xlabel('$R\,[\mathrm{au}]$')

    mol_solar = S_chem.molecular_abundance(T, rho, d2g, 1.0, R, Sigma, X_solar)

    # Test the time-evolution
    plt.figure()
    times = np.array([1e0, 1e2, 1e4, 1e6, 1e7]) * t0
    H_eff = np.sqrt(2 * np.pi) * eos.H * AU

    chem = MolecularIceAbund(gas=mol_solar.copy(), ice=mol_solar.copy())
    if 1:
        for spec in chem:
            chem.ice[spec] = 0
    else:
        for spec in chem:
            chem.gas[spec] = 0

    t = 0.
    for ti in times:
        dt = ti - t
        TD_chem.update(dt, T, rho, d2g, chem)

        t = ti

        l, = plt.semilogx(R, chem.gas['H2O'] / mol_solar['H2O'], '-')
        plt.semilogx(R, chem.ice['H2O'] / mol_solar['H2O'], '--', c=l.get_color())

    plt.semilogx(R, EQ_mol.gas['H2O'] / mol_solar['H2O'], 'k-')
    plt.semilogx(R, EQ_mol.ice['H2O'] / mol_solar['H2O'], 'k:')
    plt.xlabel('$R\,[\mathrm{au}}$')

    plt.show()
