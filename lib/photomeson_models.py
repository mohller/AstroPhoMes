""" Defining a class for models each photomeson model
"""

import numpy as np
import sys
from os.path import join

sys.path.append('../')
sys.path.append('../utils')
from config import *
from utils import *

class SuperpositionModel(object):
    """ Implementation of the Superposition Model 

    The base class uses cross sections for protons and neutrons obtained 
    from SOPHIA.

    The cross sections are tabulated in microbarn and the energy in 
    gigaelectronvolts.

    Serves as base class for the rest of the photomeson model classes
    """
    def __init__(self):
        self._load_SOPHIA_data()

    def _load_SOPHIA_data(self):
        # loading the SOPHIA crossections from file
        self.egrid, self.cs_proton_grid, self.cs_neutron_grid = \
            np.load(join(global_path, 'data/sophia_crosssec.npy'))

        epsr_grid, self.xbins, self.redist_proton, self.redist_neutron = \
            np.load(join(global_path, 
                         'data/sophia_redistribution_logbins.npy'))

        if np.all(self.egrid != epsr_grid):
            # Raise exception
            pass

    @property
    def xcenters(self):
        """Returns centers of the grid in x.

        Returns:
            (numpy.array): x grid
        """

        return 0.5 * (self.xbins[1:] + self.xbins[:-1])

    def cs_nonel(self, species):
        r"""Returns the non-elastic cross section.

        Args:
            species (int): Particle id of interacting species

        Returns:
           Returns:
            (numpy.array, numpy.array): energy, cross section
        """

        _, Z, N = get_AZN(species)

        cgrid = Z * self.cs_proton_grid + N * self.cs_neutron_grid
        return self.egrid, cgrid

    def cs_incl(self, species, product):
        r"""Returns the inclusive cross section.

        Args:
            species (int): Particle id of interacting species
            product (int): Particle id of produced species

        Returns:
           Returns:
            (numpy.array, numpy.array): energy, cross section
        """
        from scipy.integrate import trapz

        _, Z, N = get_AZN(species)

        if product in [2, 3, 4, 100, 101]:
            egr_incl, cs_diff = self.cs_incl_diff(species, product)
            cgrid = trapz(cs_diff, x=self.xcenters,
                          dx=bin_widths(self.xbins), axis=0)
        elif product == species - 101:
            cgrid = Z * self.cs_proton_grid
        elif product == species - 100:
            cgrid = N * self.cs_neutron_grid
        else:
            cgrid = np.zeros_like(self.egrid)
        
        return self.egrid, cgrid

    def cs_incl_diff(self, species, product):
        r"""Returns the non-elastic cross section.

        Args:
            species (int): Particle id of interacting species
            product (int): Particle id of produced species

        Returns:
           Returns:
            (numpy.array, numpy.array): energy, cross section
        """

        _, Z, N = get_AZN(species)

        csec_diff = np.zeros_like(self.redist_proton[2].T)

        if product in self.redist_proton:
            cgrid = Z * self.cs_proton_grid
            csec_diff = self.redist_proton[product].T * cgrid

        if product in self.redist_neutron:
            cgrid = N * self.cs_neutron_grid
            csec_diff += self.redist_neutron[product].T * cgrid

        return self.egrid, csec_diff


class EmpiricalModel(SuperpositionModel):
    """Implements the Empirical photomeson Model as in article"""
    def __init__(self):
        SuperpositionModel.__init__(self)
