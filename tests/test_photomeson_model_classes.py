"""Automated tests for the photmeson model classes
"""

import numpy as np
import sys
import unittest
sys.path.append('../')
from config import *
from photomeson_lib.photomeson_models import *

class Test_SingleParticleModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_SingleParticleModel, self).__init__(*args, **kwargs)
        
        # creating class instance for testing
        self.pm = SingleParticleModel()

    def test_nonel_nucleons(self):
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid
        
        e, cs = self.pm.cs_nonel(101)
        self.assertTrue(np.all(cs == cs_proton))
        
        e, cs = self.pm.cs_nonel(100)
        self.assertTrue(np.all(cs == cs_neutron))

    def test_nonel_nuclei(self):
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid
        
        e, cs = self.pm.cs_nonel(1406)
        cs_mix = 6 * cs_proton + 8 * cs_neutron
        self.assertTrue(np.all(cs == cs_mix))

    def test_incl_nucleons(self):
        """
            Test that nonel works
        """
        from scipy.integrate import trapz

        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid

        redist_p = self.pm.redist_proton
        redist_n = self.pm.redist_neutron

        for spec, cs_tot, redist in zip([100, 101], 
            [cs_neutron, cs_proton],
            [redist_n, redist_p]):
            for prod in [2, 3, 4, 100, 101]:
                e, cs = self.pm.cs_incl(spec, prod)
                cs_val_diff = redist[prod].T * cs_tot

                cs_val = trapz(cs_val_diff, x=self.pm.xcenters,
                              dx=bin_widths(self.pm.xbins), axis=0)
                if not np.all(cs == cs_val):
                    print spec, prod
                self.assertTrue(np.all(cs == cs_val))

    def test_incl_nuclei(self):
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid

        e, cs = self.pm.cs_incl(502, 402)
        cs_val = 3./5*(3 * cs_neutron + 2 * cs_proton)

        self.assertTrue(np.all(cs == cs_val))
                
        e, cs = self.pm.cs_incl(704, 603)
        cs_val = 4./7*(3 * cs_neutron + 4 * cs_proton)
        self.assertTrue(np.all(cs == cs_val))
        
        e, cs = self.pm.cs_incl(1407, 402)
        cs_val = np.zeros_like(cs)
        self.assertTrue(np.all(cs == cs_val))        

    def test_incl_diff_nucleons(self):
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid

        redist_p = self.pm.redist_proton
        redist_n = self.pm.redist_neutron

        for spec, cs_tot, redist in zip([100, 101], 
            [cs_neutron, cs_proton],
            [redist_n, redist_p]):
            for prod in [2, 3, 4, 100, 101]:
                e, cs = self.pm.cs_incl_diff(spec, prod)
                cs_val = cs_tot * redist[prod].T
                self.assertTrue(np.all(cs == cs_val))        

    def test_incl_diff_nuclei(self) :
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid
    
        e, cs = self.pm.cs_incl_diff(1407, 100)
        cs_val = 7 * cs_neutron * self.pm.redist_neutron[100].T + \
                 7 * cs_proton * self.pm.redist_proton[100].T
        self.assertTrue(np.all(cs == cs_val))

        e, cs = self.pm.cs_incl(4018, 100)
        cs_val = np.zeros_like(self.pm.cs_proton_grid)

        self.assertTrue(np.all(cs.shape == cs_val.shape))

        e, cs = self.pm.cs_incl_diff(1407, 402)
        cs_val = np.zeros_like(self.pm.redist_proton[2].T)
        self.assertTrue(np.all(cs == cs_val))


class Test_EmpiricalModel(Test_SingleParticleModel):
    def __init__(self, *args, **kwargs):
        super(Test_EmpiricalModel, self).__init__(*args, **kwargs)
        
        # creating class instance for testing
        self.pm = EmpiricalModel()

    def test_nonel_nuclei(self):
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid
        
        e, cs = self.pm.cs_nonel(1406)
        # cs_mix = 6 * cs_proton + 8 * cs_neutron

        egrid = self.pm.egrid
        cs_mix = 1e30 * self.pm.univ_spl(egrid)  # universal function
        cs_mix *= 14.**self.pm.alpha(self.pm.egrid)  # mass scaling

        idcs = np.argwhere((egrid > .3) * (egrid < 1.2))  # universal function range
        
        self.assertTrue(np.all(abs(cs[idcs] - cs_mix[idcs]) < 1e-10))

    def test_incl_nuclei(self):
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid

        e, cs = self.pm.cs_incl(502, 402)
        cs_val = 3 * cs_neutron
        self.assertTrue(np.all(cs != cs_val))

    def test_incl_diff_nuclei(self) :
        """
            Test that inclusive differential cross section work
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid
    
        e, cs = self.pm.cs_incl_diff(1407, 3)
        cs_val = 7 * cs_neutron * self.pm.redist_neutron[3].T + \
                 7 * cs_proton * self.pm.redist_proton[3].T
        self.assertTrue(np.any(cs != cs_val))
        self.assertTrue(np.any(cs[-50:] == cs_val[-50:]))
        self.assertTrue(np.any(cs[:-50] != cs_val[-50]))

        e, cs = self.pm.cs_incl_diff(1407, 100)
        cs_val = 7 * cs_neutron * self.pm.redist_neutron[100].T + \
                 7 * cs_proton * self.pm.redist_proton[100].T
        self.assertTrue(np.any(cs != cs_val))

        e, cs = self.pm.cs_incl(4018, 101)
        cs_val = np.zeros_like(self.pm.cs_proton_grid)

        self.assertTrue(np.all(cs.shape == cs_val.shape))

        e, cs = self.pm.cs_incl_diff(1407, 402)
        cs_val = np.zeros_like(self.pm.redist_proton[2].T)
        self.assertTrue(np.all(cs == cs_val))


class Test_ResidualDecayModel(Test_SingleParticleModel):
    def __init__(self, *args, **kwargs):
        super(Test_ResidualDecayModel, self).__init__(*args, **kwargs)
        
        # creating class instance for testing
        self.pm = ResidualDecayModel()

    def test_nonel_nuclei(self):
        pass

    def test_incl_nuclei(self):
        pass

    def test_incl_diff_nuclei(self):
        pass


if __name__ == '__main__':
    unittest.main()