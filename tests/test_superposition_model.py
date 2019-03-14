import unittest

import numpy as np
import sys
sys.path.append('../lib/')

from photomeson_models import *
    

class Test_SuperpositionModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_SuperpositionModel, self).__init__(*args, **kwargs)
        
        # creating class instance for testing
        self.pm = SuperpositionModel()

    def test_nonel(self):
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid
        
        e, cs = self.pm.cs_nonel(101)
        self.assertTrue(np.all(cs == cs_proton))
        
        e, cs = self.pm.cs_nonel(100)
        self.assertTrue(np.all(cs == cs_neutron))

        e, cs = self.pm.cs_nonel(1406)
        cs_mix = 6 * cs_proton + 8 * cs_neutron
        self.assertTrue(np.all(cs == cs_mix))

    def test_incl(self):
        """
            Test that nonel works
        """
        cs_proton = self.pm.cs_proton_grid
        cs_neutron = self.pm.cs_neutron_grid

        e, cs = self.pm.cs_incl(502, 402)
        cs_val = 3 * cs_neutron
        self.assertTrue(np.all(cs == cs_val))        
                
        e, cs = self.pm.cs_incl(704, 603)
        cs_val = 4 * cs_proton
        self.assertTrue(np.all(cs == cs_val))
        
        e, cs = self.pm.cs_incl(1407, 402)
        cs_val = np.zeros_like(cs)
        self.assertTrue(np.all(cs == cs_val))        

    def test_incl_diff(self):
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


if __name__ == '__main__':
    unittest.main()