""" Defining a class for models each photomeson model
"""

import numpy as np
import sys
from os.path import join

sys.path.append('../')
from config import *
from utils.utils import *
from utils.scaling_models_from_data import med as alpha_med
from phenom_relations import *

class GeneralPhotomesonModel(object):
    """Base class for all photomeson models which enhance the 
    Superposition Model
    """

    def __init__(self, **kwargs):
        """This function defines the type of photomeson model by loading the 
        relevant data and functions based on the keyword arguments. 
        The default model (call without arguments) returns the published version
        of the photomeson model (Empirical Model)        
                
        Parameters
        ----------
        universal_function : bool, optional, default: True
            This parameter controls if the universal function (spline fit to data)
            should be used for describing the nonelastic cross section for nuclei
            in the low energy range. When False, the Superposition Model 
            assumptions are used.

        alpha: callable, default: alpha_med
            The function describing de dependence of the mass scaling with energy.
            The function alpha should receive as parameter the energies in
            GeV units and return the mass scaling exponent.
            By default the Superposition Model corresponds to a function returning 1 
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
        object.__init__(self)

        self.nonel_idcs = []
        self.incl_idcs = []
        self.incl_diff_idcs = []

        self.multiplicity = {}

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
            alpha = alpha_med
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
            info(2, 'Using data-based-spline function for the inclusive ' + \
                'cross sections of pions produced from nuclei.')
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

        if 'pion_scaling' not in kwargs:
            info(2, 'Using custom pion scaling function.')
            self._load_pion_scaling_function()
        else:
            info(2, 'Using custom pion scaling function.')

        if ('multiplicity_source' in kwargs) and \
            kwargs['multiplicity_source'] == True:
            self._fill_multiplicity()

        self._fill_idcs()

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

    def _load_pion_scaling_function(self):
        """Returns the pion scaling function
        """
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        uf_file = join(global_path, 'data/alphapi-spline.pkl')
        with open(uf_file, 'r') as f:
            tck = pickle_load(f)

        def alphapi(energies):
            """Returnes the scaling coefficient for pions as a function of energy
            
            Arguments
            ---------
            energies -- {array, float} energies where to evaluate, in GeV
            """
            tp1 = 10**(-.18)
            tp2 = 10**(.55)
            he_alphapi = UnivariateSpline._from_tck(tck, ext=3)  # ext='const'
            api = 2./3 * np.ones_like(energies)
            api[(tp1 < energies) * (energies < tp2)] = sigm(
                np.log10(energies[(tp1 < energies) * (energies < tp2)]), .2, .25, 8, .65, True) 
            api[tp2 <= energies] = he_alphapi(np.log10(energies[tp2 <= energies]))

            return api

        self.alphapi_spl = alphapi

    def _load_pion_function(self):
        """Returns the universal function on a fixed energy range
        """
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        uf_file = join(global_path, 'data/pion_spline.pkl')
        with open(uf_file, 'r') as f:
            tck = pickle_load(f)

        self.pion_spl = UnivariateSpline._from_tck(tck)
    
    def _fill_idcs(self):
        for mom in sorted(spec_data.keys()):
            if (mom <= 101) or isinstance(mom, str) or \
                (spec_data[mom]['lifetime'] < tau_dec_threshold):
                continue
            
            if mom not in self.nonel_idcs:
                self.nonel_idcs.append(mom)

            for dau in (d for m,d in self.multiplicity if m == mom):
                if (mom, dau) not in self.incl_diff_idcs:
                    self.incl_idcs.append((mom, dau))

            for dau in [2, 3, 4, 100, 101]:
                if (mom, dau) not in self.incl_diff_idcs:
                    self.incl_diff_idcs.append((mom, dau))

        if self.incl_idcs == []:
            for mom in self.nonel_idcs:
                _, Z, N = get_AZN(mom)
                if N > 1:
                    self.incl_idcs.append((mom, mom - 100))
                if Z > 1:
                    self.incl_idcs.append((mom, mom - 101))

    @property
    def xcenters(self):
        """Returns centers of the grid in x.

        Returns:
            (numpy.array): x grid
        """

        return 0.5 * (self.xbins[1:] + self.xbins[:-1])

    @property
    def xwidths(self):
        """Returns bin widths of the grid in x.

        Returns:
            (numpy.array): x widths
        """

        return self.xbins[1:] - self.xbins[:-1]

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

            cs_univ = 1e30 * self.univ_spl(egrid)
            cgrid[egrid <= e_max] = A * cs_univ[egrid <= e_max]
            cgrid[egrid > e_scale] = (cgrid * A**self.alpha(egrid))[egrid > e_scale] / A

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
        elif (species, product) in self.multiplicity:
            _, cgrid = self.cs_nonel(species)
            cgrid *= self.multiplicity[species, product]
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
        from scipy.integrate import trapz

        if species == 100:
            cgrid = self.cs_neutron_grid
            csec_diff = self.redist_neutron[product].T * cgrid
        elif species == 101:
            cgrid = self.cs_proton_grid
            csec_diff = self.redist_proton[product].T * cgrid
        elif (species > 101) and (product <= 101):
        # include redistributed particles in multiplicity table 
        # and pion renormalizations if pion_spl is defined
            A, Z, N = get_AZN(species)
            def spm_incl_diff(product):
                csec_diff = np.zeros_like(self.redist_proton[2].T)

                if product in self.redist_proton:
                    cgrid = Z * self.cs_proton_grid
                    csec_diff += self.redist_proton[product].T * cgrid

                if product in self.redist_neutron:
                    cgrid = N * self.cs_neutron_grid
                    csec_diff += self.redist_neutron[product].T * cgrid

                return csec_diff

            csec_diff = spm_incl_diff(product)  # initialize as in Single Particle Model

            if (species, product) in self.multiplicity:  # only p and n
                xw = self.xwidths[-1]  # only on last x bin
                _, cs_nonel = self.cs_nonel(species)

                if (species, product) in self._incl_diff_tab:
                    if self._incl_diff_tab[species, product]:
                        cs_incl = trapz(csec_diff, x=self.xcenters,
                            dx=bin_widths(self.xbins), axis=0)
                        csec_diff *= \
                            self._incl_diff_tab[species, product] / cs_incl

                csec_diff[-1, :] += \
                    self.multiplicity[species, product] * cs_nonel / xw

            elif self.pion_spl:  # only pions (product ID in [2, 3, 4])
                csec_diff = spm_incl_diff(product)

                csec_diff_ref = spm_incl_diff(4)
                cs_incl_ref = trapz(csec_diff_ref, x=self.xcenters,
                              dx=bin_widths(self.xbins), axis=0)

                cs_incl_prod = trapz(csec_diff, x=self.xcenters,
                              dx=bin_widths(self.xbins), axis=0)
                alphapi = self.alphapi_spl(self.egrid)
                csec_diff *= A**(alphapi - 1.)  # rescaling Single Particle Model values to alphapi function
                cs_incl = trapz(csec_diff, x=self.xcenters,
                              dx=bin_widths(self.xbins), axis=0)

                cspi = self.pion_spl(self.egrid * 1e3)*A**alphapi # rescaling Single Particle Model values to alphapi function

                renorm = cs_incl_prod / cs_incl_ref * cspi / cs_incl
                csec_diff_renormed = csec_diff * renorm
                csec_diff_renormed = self.fade(csec_diff, csec_diff_renormed, range(32)) # hardcoded, found manually
                csec_diff = self.fade(csec_diff_renormed, csec_diff, range(55, 105)) # hardcoded, found manually
        else:
            csec_diff = np.zeros_like(self.redist_neutron[2].T)

        return self.egrid, csec_diff


class SingleParticleModel(GeneralPhotomesonModel):
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
                                        pion_function=False,
                                        multiplicity_source=True)

    def _fill_multiplicity(self):
        """Adds multiplicity of non distributed products
        
        For products in A-1 fills the multiplicity to be used
        by method cs_incl 
        """
        multiplicity_table = {}

        for nucleus in spec_data:
            if (nucleus < 200) or (type(nucleus) is str):
                continue
            
            A, Z, N = get_AZN(nucleus)

            multiplicity_table[nucleus, nucleus - 100] = N/float(A)
            multiplicity_table[nucleus, nucleus - 101] = Z/float(A)

        self.multiplicity = multiplicity_table


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

        self._nonel_tab = {100:(), 101:()}
        self._incl_tab = {}
        self._incl_diff_tab = {}

        for mom in self._nonel_tab:
            for dau in [2, 3, 4, 100, 101]:
                self._incl_diff_tab[mom, dau] = ()
                
        new_multiplicity = {}
        for mom in sorted(spec_data.keys()):
            if (mom <= 101) or isinstance(mom, str) or \
                (spec_data[mom]['lifetime'] < tau_dec_threshold):
                continue                          
            
            mults = multiplicity_table(mom)
            dau_list, csincl_list = zip(*((k, v) for k, v in mults.iteritems()))
            
            self._nonel_tab[mom] = ()
            for dau in [2, 3, 4]:
                self._incl_diff_tab[mom, dau] = ()
            
            A, Z, _ = get_AZN(mom)
            self._incl_diff_tab[mom, 100] = cs_gn(A) / cs_tot(A, False)
            self._incl_diff_tab[mom, 101] = cs_gp(Z) / cs_tot(A, False)
            
            for dau, mult in mults.iteritems():
                new_multiplicity[mom, dau] = mult
                self._incl_tab[mom, dau] = np.array([])
            
        # Sophia cross section are loaded by now, and they are used to build all the tabs
        self.multiplicity = new_multiplicity
        

class ResidualDecayModel(GeneralPhotomesonModel):
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
    def __init__(self, **kwargs):
        self._nonel_tab = {}
        self._incl_tab = {}
        self._incl_diff_tab = {}

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

        GeneralPhotomesonModel.__init__(self, alpha=alpha_med,
                                        multiplicity_source=True)

        
    def _fill_multiplicity(self):
        """Crates a dictionary with multiplicity values
        """
        info(9, 'Creating multiplicity table from base photodisintegration model.')
        photodis_model_egrid = np.load(join(global_path, 
                         'data/CRP2_TALYS_egrid_reduced.npy'))
        
        for dau in [2, 3, 4, 100, 101]:
            self._incl_diff_tab[100, dau] = ()
            self._incl_diff_tab[101, dau] = ()

        reactions = {}
        _incl_tab = {}
        for row in np.load(join(global_path, 'data/CRP2_TALYS_incl_i_j_reduced.npy')):
            spec, prod = int(row[0]), int(row[1])
            _incl_tab[spec, prod] = row[2:]

            if spec in reactions:
                reactions[spec].append((spec, prod))
            else:
                reactions[spec] = [(spec, prod)]

        _nonel_tab = {}
        for row in np.load(join(global_path, 'data/CRP2_TALYS_nonel_reduced.npy')):
            spec = int(row[0])
            _nonel_tab[spec] = row[1:] 

            for pion in [2, 3, 4]:
                self._incl_diff_tab[spec, pion] = ()
        
        # populate multiplicities...
        multiplicity_table = {}

        for mom in _nonel_tab:
            Am, Zm, Nm = get_AZN(mom)
            idx = self.Eexc(Am, photodis_model_egrid)

            new_channels = []
            if mom - 101 in reactions:
                new_channels += [d for _, d in reactions[mom - 101]]

            if mom - 100 in reactions:
                new_channels += [d for _, d in reactions[mom - 100]]

            new_channels = set(new_channels)  # avoid repetitions

            for dau in new_channels:
                multip_value = 0

                if mom - 100 in reactions:
                    if (mom - 100, dau) in reactions[mom - 100]:
                        pdis_multiplicity = (_incl_tab[mom - 100, dau] / \
                            np.where(_nonel_tab[mom - 100] == 0, np.inf, 
                                _nonel_tab[mom - 100]))[idx]
                        multip_value += float(Nm) / Am * pdis_multiplicity

                if mom - 101 in reactions:
                    if (mom - 101, dau) in reactions[mom - 101]:
                        pdis_multiplicity = (_incl_tab[mom - 101, dau] / \
                            np.where(_nonel_tab[mom - 101] == 0, np.inf, 
                                _nonel_tab[mom - 101]))[idx]
                        multip_value += float(Zm) / Am * pdis_multiplicity

                if (multip_value > 0):
                    multiplicity_table[mom, dau] = multip_value

                    # preparing tabs to work with _optimize_and_generate_index
                    if dau > 101:
                        self._incl_tab[mom, dau] = ()
                    elif dau == 100:
                        self._incl_diff_tab[mom, 100] = Nm / Am
                    elif dau == 101:
                        self._incl_diff_tab[mom, 101] = Zm / Am
                    else:
                        self._incl_diff_tab[mom, dau] = ()

                    if mom not in self._nonel_tab:
                        self._nonel_tab[mom] = ()

        # correcting nuclei with only one path
        for ch in multiplicity_table:
            Am, Zm, Nm = get_AZN(ch[0])
            if ch[0] - 100 not in reactions:
                multiplicity_table[ch] *= float(Am) / Zm
            if ch[0] - 101 not in reactions:
                multiplicity_table[ch] *= float(Am) / Nm

        self.multiplicity = multiplicity_table

