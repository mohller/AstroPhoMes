""" Defining a class for models each photomeson model
"""

import numpy as np
import sys
from os.path import join

sys.path.append('../')
sys.path.append('../utils')
from config import *
from utils import *
from scaling_models_from_data import med as alpha_med

class GeneralPhotomesonModel(object):
    """Base class for all photomeson models which enhance the 
    Superposition Model
       
    Parameters
    ----------
    universal_function : bool, optional, default: True
        This parameter controls if the universal function (spline fit to data)
        should be used for describing the nonelastic cross section for nuclei
        in the low energy range. When False, the Superposition Model 
        assumptions are used.

    alpha: callable, default: alpha_med
        The function describing de dependence of the mass scaling with energy.
        The function alpha(er) should receive as parameter er the energies in
        GeV units and return the mass scaling exponent.
        By default , T
        he Superposition Model corresponds to a function returning 1 
        regardless of the energy.

    pion_function : bool, optional, default: True
        This parameter controls if the pion function (spline fit to data) 
        should be used for describing the inclusive differential cross 
        sections for pion production off nuclei in the low energy range. When
        False, the Superposition Model assumptions are used.
    
    multiplicity_source : bool, default: False
        Determines if the multiplicities for producing intermediate fragments
        should be loaded. This keyword is intended for models that include 
        this feature. The Superposition Model does not include it and is the 
        default option. When the keyword is given, the method 
        _fill_multiplicity() is called, and should be implemented in the class
        from which it will be called.
    """

    def __init__(self, **kwargs):
        object.__init__(self)

        self._load_SOPHIA_data()

        if ('universal_function' in kwargs) and \
            kwargs['universal_function'] == False:
            info(2, 'Using Superposition Model assumptions for the total '+\
                'cross section at low energies.')
            self.univ_spl = None
        else:
            info(2, 'Using Universal-function is on.')
            self._load_universal_function()

        if 'alpha' not in kwargs:
            info(2, 'Default function chosen for assymptotic scaling of total cross section.')
            def scaling_function(A, egrid):
                """Returns the effective A which results from the ratio of nuclear
                to nucleon total inelastic cross section. Based on Weise (1974).
                Returns :math:`A(E) = A^{\alfa(E)}`

                Assumes that Glauber regime :math:`A^{2/3}` is reached at 30 GeV.
                NOTE: Temporarily implemented as :math:`\alpha(E) = 1 - 1/3*E/E_{max}`
                """
                E = egrid
                Emax = egrid[-1]
                # alpha = np.max(2./3., 1 - 1 / 3 * E / 30.)
                alpha = 1 - 1 / 3 * E / Emax

                return A**alpha
        else:
            info(2, 'Custom function chosen for assymptotic scaling of total cross section.')
            alpha = kwargs['alpha']
        self.alpha = alpha

        if ('pion_function' in kwargs) and \
            kwargs['pion_function'] == False:
            info(2, 'Using Superposition Model assumptions for the ' + \
                'inclusive cross sections of pions produced from nuclei.')
            self.pion_spl = None
        else:
            info(2, 'Using data-based-spline function for the inclusive \
                cross sections of pions produced from nuclei.')
            self._load_pion_function()

            def fade(cs1, cs2, indices=None):
                """Smoothes the transition from cs1 to cs2
                Uses a sigmoid to smothen the transition between differential 
                cross sections cs1 and cs2 in the energy range defined by the
                indices.
                """
                if indices is None:
                    return cs1
                    
                x = -100 * np.ones_like(cs1)
                x[..., indices[-1]:] = 100
                x[..., indices] = np.linspace(-5, 5, len(indices))
                
                def sigmoid(x):
                    return 1./(1 + np.exp(-x))

                return cs1[..., :]*(1 - sigmoid(x)) + cs2[..., :]*sigmoid(x)

            self.fade = fade

        if ('multiplicity_source' in kwargs) and \
            kwargs['multiplicity_source'] == True:
            self._fill_multiplicity()

    def _load_SOPHIA_data(self):
        """Loading data sampled from SOPHIA

        The class uses cross sections for protons and neutrons obtained from 
        SOPHIA for describing the neutron and proton cross sections and some parts
        of the nuclei's cross sections (dependeing on the inheriting class).

        The cross sections are tabulated in microbarn and the energy in 
        gigaelectronvolts.
        """
        self.egrid, self.cs_proton_grid, self.cs_neutron_grid = \
            np.load(join(global_path, 'data/sophia_crosssec.npy'))

        epsr_grid, self.xbins, self.redist_proton, self.redist_neutron = \
            np.load(join(global_path, 
                         'data/sophia_redistribution_logbins.npy'))

    def _load_universal_function(self):
        """Returns the universal function on a fixed energy range
        """
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        uf_file = join(global_path, 'data/universal-spline.pkl')
        with open(uf_file, 'r') as f:
            tck = pickle_load(f)

        self.univ_spl = UnivariateSpline._from_tck(tck)

    def _load_pion_function(self):
        """Returns the universal function on a fixed energy range
        """
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        uf_file = join(global_path, 'data/pion_spline.pkl')
        with open(uf_file, 'r') as f:
            tck = pickle_load(f)

        self.pion_spl = UnivariateSpline._from_tck(tck)
        
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

        A, Z, N = get_AZN(species)

        cgrid = Z * self.cs_proton_grid + N * self.cs_neutron_grid
        
        if (A > 1) and self.univ_spl:
            e_max = 1.2 # energy at which universal function ends
            e_scale = .3  # energy at which mass scaling starts
            egrid = self.egrid

            cs_univ = self.univ_spl(egrid)
            cgrid[egrid <= e_max] = A * self.univ_spl(egrid[egrid <= e_max])
            cgrid[egrid > e_scale] = (cgrid * A**self.alpha(egrid))[egrid > e_scale]/A

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
        
        if (species, product) in self.multiplicity:
            cgrid *= self.multiplicity[species, product]

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

        A, Z, N = get_AZN(species)

        csec_diff = np.zeros_like(self.redist_proton[2].T)

        if product in self.redist_proton:
            cgrid = Z * self.cs_proton_grid
            csec_diff = self.redist_proton[product].T * cgrid

        if product in self.redist_neutron:
            cgrid = N * self.cs_neutron_grid
            csec_diff += self.redist_neutron[product].T * cgrid

        # if model is beyond superposition, include multiplicity table
        if (species, product) in self.multiplicity:
            xw = self.xwidths[-1]  # accounting for bin width
            _, cs_nonel = self.nonel(species)
            csec_diff[-1, :] += \
                self.multiplicity[species, product] * cs_nonel / xw
        elif (product in [2, 3, 4]) and self.pion_spl:
            spm = SophiaSuperposition()

            csec_diff *= float(A)**(-1/3.)  # ... rescaling SpM to A^2/3
            _, cs_incl = spm.incl(species, product)

            cspi = 1e-30*self.pion_spl(egrid * 1e3)*A**(2./3)

            _, M_pi = spm.multiplicities(species, product)
            _, M_pi0 = spm.multiplicities(species, 4)

            renorm = M_pi / M_pi0 * cspi / cs_incl

            csec_diff = self.fade(csec_diff, csec_diff*renorm, range(32)) # hardcoded, found manually
            csec_diff = self.fade(csec_diff*renorm, csec_diff, range(55, 105)) # hardcoded, found manually

        return self.egrid, csec_diff


class SuperpositionModel(GeneralPhotomesonModel):
    """Implementation of the Superposition Model 

    The model uses cross sections for protons and neutrons obtained from 
    SOPHIA.

    Serves as comparison for the rest of the photomeson models
    """
    def __init__(self):
        def alpha_SM(er):
            """Mass scaling exponent function for Superposition Model
            """
            return np.ones_like(er)

        GeneralPhotomesonModel.__init__(self, universal_function=False,
                                        alpha=alpha_SM,
                                        pion_function=False)


class EmpiricalModel(GeneralPhotomesonModel):
    """Implements the Empirical photomeson Model as in article.

    The _fill_multiplicity function is redefined to load the cross sections
    calculated with the empirical functions. 
    """
    def __init__(self):
        GeneralPhotomesonModel.__init__(self, alpha=alpha_med,
                                        multiplicity_source=True)


    def _fill_multiplicity(self, *args, **kwargs):
        """Loads the data and creates the multiplicity table
        """
        from phenom_relations import multiplicity_table

        self._nonel_tab = {100:(), 101:()}
        self._incl_tab = {}
        self._incl_diff_tab = {}

        for mom in self._nonel_tab:
            for dau in [2, 3, 4, 100, 101]:
                self._incl_diff_tab[mom, dau] = ()
                
        new_multiplicity = {}
        for mom in sorted(spec_data.keys()):
            if (mom < 101) or isinstance(mom, str) or \
                (spec_data[mom]['lifetime'] < tau_dec_threshold):
                continue                          
            
            mults = multiplicity_table(mom)
            dau_list, csincl_list = zip(*((k, v) for k, v in mults.iteritems()))
            
            self._nonel_tab[mom] = ()
            for dau in [2, 3, 4, 100, 101]:
                self._incl_diff_tab[mom, dau] = ()
            
            for dau, mult in mults.iteritems():
                new_multiplicity[mom, dau] = mult
                self._incl_tab[mom, dau] = np.array([])
            
        # Sophia cross section are loaded by now, and they are used to build all the tabs
        self.multiplicity = new_multiplicity
        

class ResidualDecayModel(object):
    """Implements the Residual Decay Model as in article.

    The _fill_multiplicity function is redefined to use the multiplicities of
    a given disintegration model. The multiplicities are extracted for an 
    excitation energy (defined by the parameter excitation_function), and are 
    used for calculating the inclusive cross sections.

    Parameter
    ---------
    excitation_function : callable
        The function that calculates the excitation energy left in the 
        residual nucleus. Technically it returns the energy at which the 
        mutiplicities are obtained from the disintegration model. Here the 
        disintegration model is a table obtained from sampling TALYS.
    """
    def __init__(self, arg):
        super(ResidualDecayModel, self).__init__()

        # defining excitation function
        if 'excitation_function' not in kwargs:
            info(2, 'Default function chosen for residual excitation energy.')
            def excitation_energy_index(A, egrid):
                """Returns the index corresponding to the excitation energy of
                the residual nucleus.
                The excitation energy is calculated using formula 1.3.11a in
                Rachen PhD Thesis (1996)
                """
                constant = .017  # GeV
                Eexc = constant * A ** (1. / 3.)
                return np.argmin(abs(Eexc - egrid))
            self.Eexc = excitation_energy_index
        else:
            info(2, 'Custom function chosen for residual excitation energy.')
            self.Eexc = kwargs['excitation_function']

        
    def _fill_multiplicity(self, model_prefix):
        """Crates a dictionary with multiplicity values
        """
        info(9, 'Creating multiplicity table from base photodisintegration model.')
        photodis_model = TabulatedCrossSection(model_prefix)

        # populate nuclei list (already reduced by decay time)...
        self._nonel_tab = photodis_model._nonel_tab
        for mom in self._nonel_tab:
            for dau in [2, 3, 4]:
                self._incl_diff_tab[mom, dau] = ()
        
        for dau in [2, 3, 4, 100, 101]:
            self._incl_diff_tab[100, dau] = ()
            self._incl_diff_tab[101, dau] = ()

        # populate multiplicities...
        multiplicity = {}


        for mom in self._nonel_tab:
            Am, Zm, Nm = get_AZN(mom)
            idx = self.Eexc(Am, photodis_model.egrid)

            new_channels = []
            if mom - 101 in photodis_model.reactions:
                new_channels += [d for _, d in
                                 photodis_model.reactions[mom - 101]]

            if mom - 100 in photodis_model.reactions:
                new_channels += [d for _, d in
                                 photodis_model.reactions[mom - 100]]

            new_channels = set(new_channels)  # avoid repetitions

            for dau in new_channels:
                multip_value = 0

                if mom - 100 in photodis_model.reactions:
                    if (mom - 100, dau) in photodis_model.reactions[mom - 100]:
                        multip_value += float(Nm) / Am * photodis_model.multiplicities(mom - 100, dau)[1][idx]

                if mom - 101 in photodis_model.reactions:
                    if (mom - 101, dau) in photodis_model.reactions[mom - 101]:
                        multip_value += float(Zm) / Am * photodis_model.multiplicities(mom - 101, dau)[1][idx]

                if (multip_value > 0):
                    multiplicity[mom, dau] = multip_value

                    # preparing tabs to work with _optimize_and_generate_index
                    if dau > 101:
                        self._incl_tab[mom, dau] = ()
                    else:
                        self._incl_diff_tab[mom, dau] = ()

                    if mom not in self._nonel_tab:
                        self._nonel_tab[mom] = ()

        # correcting nuclei woth only one path
        for ch in multiplicity:
            Am, Zm, Nm = get_AZN(ch[0])
            if ch[0] - 100 not in photodis_model.reactions:
                multiplicity[ch] *= float(Am) / Zm
            if ch[0] - 101 not in photodis_model.reactions:
                multiplicity[ch] *= float(Am) / Nm

        self.multiplicity = multiplicity

