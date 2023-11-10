from __future__ import print_function
from operator import xor
import numpy as np
from ..constants import *
from .base_chem import ChemicalAbund, MolecularIceAbund
from .base_chem import SimpleChemBase, StaticChem, ThermalChem
from .base_chem import TimeDependentChem, EquilibriumChem

################################################################################
# Simple CNO Chemistry wrappers
################################################################################
class SimpleCNOIsotopeAbund(ChemicalAbund):
    """Class to hold the raw atomic abundaces of C/N/O/Si/S for the CNO chemistry"""

    def __init__(self, *sizes):
        self.atom_ids = ['H', 'D', 'C', '13C', 'N', 'O', '18O', 'Si', 'S']
        masses = [1., 2., 12., 13., 14., 16., 18., 28., 32.]

        super(SimpleCNOIsotopeAbund, self).__init__(self.atom_ids, masses, *sizes)

    def set_solar_abundances(self, muH=1.28, isotopes=False):
        """Solar mass fractions of C, N, O, Si and S (Asplund 2021; Table 2)

        args:
            muH : mean atomic mass, default = 1.28
        """
        if isotopes:
            log12 = np.array([12.00, 12.00+np.log(0.002/99.998), 8.46+np.log(0.98893), 8.46+np.log(0.01107), 7.83, 8.69+np.log(0.99776), 8.69+np.log(0.00188), 7.51, 7.12])
        else:
            log12 = np.array([12.00, 0.,                         8.46,                 0.,                   7.83, 8.69,                 0.,                   7.51, 7.12])
        A_abund = np.power(10,log12-12.0)
        m_abund = np.array(self.masses)*A_abund
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH
        self.update_isotopic_fractions(fC13 = 0.1107*isotopes, fO18=0.00188*isotopes)
        
    def set_protosolar_abundances(self, muH=1.28, isotopes=False):
        """Proto-solar mass fractions of C, N, O, Si and S (Asplund 2021; Table B.1)
        Main difference, besides listing isotopes separately, is that all are adjsuted up by ~0.06 to account for atomic diffusion

        args:
            muH : mean atomic mass, default = 1.28
        """
        if isotopes:
            log12 = np.array([12.00, 7.22,    8.52, 6.57,    7.89, 8.75, 6.03,    7.57, 7.16])
        else:
            log12 = np.array([12.00, -np.inf, 8.52, -np.inf, 7.89, 8.75, -np.inf, 7.57, 7.16])
        A_abund = np.power(10,log12-12.0)
        m_abund = np.array(self.masses)*A_abund
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH
        self.update_isotopic_fractions()
        
    def set_mass_abundances(self, muH=1.28):
        """Mass abundance fractions of C, N, O, Si and S.

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([1., 2., 12., 13., 14., 16., 18., 28., 32.])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH

    def set_unit_abundances(self, muH=1.28):
        """Unit abundance fractions of C, N, O and Si.

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH
        
    def update_isotopic_fractions(self, fC13=None, fO18=None):
        ## Default values from Asplund 2021:
        # self._fC13 = 0.1107
        # self._fO18 = 0.00188
        if fC13:
            self._fC13
        else:
            self._fC13 = self.number_abund('13C')/(self.number_abund('C')+self.number_abund('13C'))
        if fO18:
            self._fO18
        else:
            self._fO18 = self.number_abund('18O')/(self.number_abund('O')+self.number_abund('18O'))

class SimpleCNOIsotopologueAbund(ChemicalAbund):
    """Class that holds the abundances of molecules needed for C/O chemistry"""

    def __init__(self, *sizes, fC13=0.):
        mol_ids = ['CO', 'CO2', '13CO', '13CO2', 'H2O', 'N2', 'NH3',
                   'C-grain', 'Si-grain']
        mol_mass = [28., 44., 29., 45., 18., 28., 17.,
                    12., 100.]

        super(SimpleCNOIsotopologueAbund, self).__init__(mol_ids, mol_mass, *sizes)

        # Atomic make up of the molecules:
        self._n_spec = {'CO': {'C': 1, 'O': 1, }, '13CO': {'13C': 1, 'O': 1, },
                        'CO2': {'C': 1, 'O': 2, }, '13CO2': {'13C': 1, 'O': 2, },
                        'CH3OH': {'C': 1, 'O': 4, 'H': 4, },
                        'CH4': {'C': 1, 'H': 4, },
                        'C2H': {'C': 2, 'H': 1, },
                        'C2H2': {'C': 2, 'H': 2, },
                        'C2H6': {'C': 2, 'H': 2, },
                        'H2O': {'O': 1, 'H': 2, },
                        'O2': {'O': 2, },
                        'N2': {'N': 2, },
                        'NH3': {'N': 1, },
                        'HCOOH': {'C': 1, 'O': 2, 'H': 2, },
                        'NH4HCOO': {'N': 1, 'C': 1, 'O': 2, 'H': 5,},
                        'C-grain': {'C': 1-fC13, '13C': fC13, },
                        'Si-grain': {'O': 3, 'Si': 1, },
                        }

    def atomic_abundance(self):
        """Compute the mass abundances of atomic species in the molecules"""

        atomic_abund = SimpleCNOIsotopeAbund(self.data.shape[1])
        for mol in self.species:
            nspec = self._n_spec[mol]
            for atom in nspec:
                n_atom = (self[mol] / self.mass(mol)) * nspec[atom]
                atomic_abund[atom] += n_atom * atomic_abund.mass(atom)
        atomic_abund.update_isotopic_fractions()

        return atomic_abund


###############################################################################
# Specific Chemical models
###############################################################################
class CNOChemOberg(object):
    """Chemical ratios from Oberg+ (2011)

    args:
        fNH3 : Fraction of nitrogen in NH_3
        fix_grains : Whether to fix the dust grain abundances when recomputing
                     the molecular abundances
        fix_NH3    : Whether to fix the nitrogen abundance when recomputing the
                     molecular abundances
    """

    def __init__(self, fNH3=None, fix_grains=True, fix_N=False):
        if fNH3 is None: fNH3 = 0.07
        self._fNH3 = fNH3
        self._fix_grains = fix_grains
        self._fix_N = fix_N

    def ASCII_header(self):
        """CNO Oberg chem header"""
        return (super(CNOChemOberg, self).ASCII_header() +
                ', f_NH3: {}, fix_grains: {}, fix_N: {}'
                ''.format(self._fNH3, self._fix_grains, self._fix_N))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        __, header = super(CNOChemOberg, self).HDF5_attributes()
        header['f_NH3'] = '{}'.format(self._fNH3)
        header['fix_grains'] = "{}".format(self._fix_grains)
        header['fix_N'] = "{}".format(self._fix_N)

        return self.__class__.__name__, header

    def molecular_abundance(self, T, rho, dust_frac, f_small, R, SigmaG,
                            atomic_abund=None, mol_abund=None):
        """Compute the fractions of species present given total abundances

        args:
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleCNOIsotopeAbund object

        returns:
            nmol : array(3, N) molecular mass-densities
        """
        assert (xor(atomic_abund is None, mol_abund is None))
        if atomic_abund is None:
            atomic_abund = mol_abund.atomic_abundance()

        C = atomic_abund.number_abund('C')
        C13 = atomic_abund.number_abund('13C')
        fC13 = atomic_abund._fC13
        N = atomic_abund.number_abund('N')
        O = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')

        # Set up the number-density abundances
        initial_abund = False
        if mol_abund is None:
            initial_abund = True
            mol_abund = SimpleCNOIsotopologueAbund(atomic_abund.size, fC13=fC13)
        else:
            #  Convert to number abundances
            for spec in mol_abund.species:
                mol_abund[spec] = mol_abund.number_abund(spec)

        # If grain abundance provided use that value, otherwise set
        # the grain abundance
        if initial_abund or not self._fix_grains:
            mol_abund['C-grain'] = 0.2 * C / (1-fC13)
            mol_abund['Si-grain'] = Si
            C -= mol_abund['C-grain'] * (1-fC13)
            C13 -= mol_abund['C-grain'] * fC13
            O -= 3 * Si
        else:
            C -= mol_abund['C-grain'] * mol_abund._n_spec['C-grain']['C']
            C13 -= mol_abund['C-grain'] * mol_abund._n_spec['C-grain']['13C']
            O -= 3 * mol_abund['Si-grain']

        # From the amount of O available work out how much CO/CO_2 we can
        # have
        fCO2 = 0.15 / (0.65 + 0.15)
        mol_abund['CO2'] = np.minimum(C * fCO2, O * (1-fC13) - C)
        mol_abund['13CO2'] = np.minimum(C13 * fCO2, O * fC13 - C13)
        mol_abund['CO'] = C - mol_abund['CO2']
        mol_abund['13CO'] = C13 - mol_abund['13CO2']

        # Put the remaining O in water (if any)
        O -= mol_abund['CO'] + 2 * mol_abund['CO2'] + mol_abund['13CO'] + 2 * mol_abund['13CO2']
        mol_abund['H2O'] = np.maximum(O, 0)

        # Nitrogen
        if initial_abund or not self._fix_N:
            mol_abund['NH3'] = self._fNH3 * N
            mol_abund['N2'] = 0.5 * (N - 1 * mol_abund['NH3'])

        #  Convert to mass abundances
        for spec in mol_abund.species:
            mol_abund[spec] *= mol_abund.mass(spec)

        return mol_abund

###############################################################################
# Combined Models
###############################################################################
class SimpleCNOChemObergI(CNOChemOberg, StaticChem):
    def __init__(self, fNH3=None, **kwargs):
        CNOChemOberg.__init__(self, fNH3)
        StaticChem.__init__(self, **kwargs)


class EquilibriumCNOChemObergI(CNOChemOberg, EquilibriumChem):
    def __init__(self, fNH3=None, fix_ratios=False, fix_grains=True,
                 fix_N=False, **kwargs):
        CNOChemOberg.__init__(self, fNH3, fix_grains, fix_N)
        EquilibriumChem.__init__(self,
                                 fix_ratios=fix_ratios,
                                 **kwargs)
                                 

class TimeDepCNOChemObergI(CNOChemOberg, TimeDependentChem):
    def __init__(self, fNH3=None, **kwargs):
        CNOChemOberg.__init__(self, fNH3)
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

    muH = 1.28

    grid = Grid(0.01, 1000, 1000, spacing='log')
    R = grid.Rc
    eos = LocallyIsothermalEOS(SimpleStar(), cs0, q, alpha)
    eos.set_grid(grid)
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R / Rd)
    rho = Sigma / (np.sqrt(2 * np.pi) * eos.H * AU)

    T = eos.T
    n = rho / (2.4 * m_H)

    EQ_chem = SimpleCNOChemOberg()
    TD_chem = TimeDepCNOChemOberg(a=1e-5)

    X_solar = SimpleCNOIsotopeAbund(n.shape[0])
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
    S_chem = SimpleCNOChemOberg()
    EQ_chem = EquilibriumCNOChemOberg()

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
