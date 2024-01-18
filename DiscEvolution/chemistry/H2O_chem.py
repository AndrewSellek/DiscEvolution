from __future__ import print_function
from operator import xor
import numpy as np
from ..constants import *
from .base_chem import ChemicalAbund, MolecularIceAbund
from .base_chem import SimpleChemBase, ThermalChem
from .base_chem import TimeDependentChem, EquilibriumChem

################################################################################
# Simple CNO Chemistry wrappers
################################################################################
class SimpleH2OAtomAbund(ChemicalAbund):
    """Class to hold the raw atomic abundaces of O/Si for the CNO chemistry"""

    def __init__(self, *sizes):
        self.atom_ids = ['O', 'Si']
        masses = [16., 28.]

        super(SimpleH2OAtomAbund, self).__init__(self.atom_ids, masses, *sizes)

    def set_Kalyaan_abundances(self, muH=1.28):
        """Solar mass fractions of C, N, O and Si.

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([(16/18+48/100) * 0.005, 28/100*0.005])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) #/ muH

    def set_solar_abundances(self, muH=1.28):
        """Solar mass fractions of C, N, O and Si.

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([16 * 4.9e-4, 28 * 3.2e-5])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH

    def set_WTTS_abundances(self, muH=1.28):
        """WTTS abundance fractions of C, N, O and Si (see Booth & Clarke 2018).
        Based on data from Ardila et al. 2013
        Set N to 1
        Use average of doublet flux ratios for Si and C(nb =/= ratio of averages)
        No O measurements - assume 1

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([16 * 1, 28 * 1.70])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH

    def set_mass_abundances(self, muH=1.28):
        """Mass abundance fractions of C, N, O and Si.

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([16 * 1, 28 * 1])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH

    def set_unit_abundances(self, muH=1.28):
        """Unit abundance fractions of C, N, O and Si.

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([1, 1])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH

class SimpleH2OMolAbund(ChemicalAbund):
    """Class that holds the abundances of molecules needed for C/O chemistry"""

    def __init__(self, *sizes):
        mol_ids = ['H2O', 'Si-grain']
        mol_mass = [18., 100.]

        super(SimpleH2OMolAbund, self).__init__(mol_ids, mol_mass, *sizes)

        # Atomic make up of the molecules:
        self._n_spec = {'H2O': {'O': 1, },
                        'Si-grain': {'O': 3, 'Si': 1},
                        }

    def atomic_abundance(self):
        """Compute the mass abundances of atomic species in the molecules"""

        atomic_abund = SimpleH2OAtomAbund(self.data.shape[1])
        for mol in self.species:
            nspec = self._n_spec[mol]
            for atom in nspec:
                n_atom = (self[mol] / self.mass(mol)) * nspec[atom]
                atomic_abund[atom] += n_atom * atomic_abund.mass(atom)

        return atomic_abund


###############################################################################
# Specific Chemical models
###############################################################################
class H2OChemKalyaan(object):
    """Chemical ratios from Oberg+ (2011)

    args:
        fix_grains : Whether to fix the dust grain abundances when recomputing
                     the molecular abundances
    """
    def __init__(self, fix_grains=True):
        self._fix_grains = fix_grains

    def ASCII_header(self):
        """H2O Kalyaan chem header"""
        return (super(H2OChemKalyaan, self).ASCII_header() +
                ', fix_grains: {}'
                ''.format(self._fix_grains))
                
    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        __, header = super(H2OChemKalyaan, self).HDF5_attributes()
        header['fix_grains'] = "{}".format(self._fix_grains)

        return self.__class__.__name__, header
      
    def molecular_abundance(self, T, rho, dust_frac, f_small, R, SigmaG,
                            atomic_abund=None, mol_abund=None):
        """Compute the fractions of species present given total abundances

        args:
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleH2OAtomAbund object

        returns:
            nmol : array(3, N) molecular mass-densities
        """
        assert (xor(atomic_abund is None, mol_abund is None))
        if atomic_abund is None:
            atomic_abund = mol_abund.atomic_abundance()

        O = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')

        # Set up the number-density abundances
        initial_abund = False
        if mol_abund is None:
            initial_abund = True
            mol_abund = SimpleH2OMolAbund(atomic_abund.size)
        else:
            #  Convert to number abundances
            for spec in mol_abund.species:
                mol_abund[spec] = mol_abund.number_abund(spec)

        # If grain abundance provided use that value, otherwise set
        # the grain abundance
        if initial_abund or not self._fix_grains:
            mol_abund['Si-grain'] = Si
            O -= 3 * Si
        else:
            O -= 3 * mol_abund['Si-grain']

        # Put the remaining O in water (if any)
        mol_abund['H2O'] = np.maximum(O, 0)

        #  Convert to mass abundances
        for spec in mol_abund.species:
            mol_abund[spec] *= mol_abund.mass(spec)

        return mol_abund
        

###############################################################################
# Combined Models
###############################################################################
class EquilibriumH2OChemKalyaan(H2OChemKalyaan, EquilibriumChem):
    def __init__(self, fix_ratios=False, fix_grains=True, **kwargs):
        H2OChemKalyaan.__init__(self, fix_grains)
        EquilibriumChem.__init__(self,
                                 fix_ratios=fix_ratios,
                                 **kwargs)

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

    X_solar = SimpleCNOAtomAbund(n.shape[0])
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
