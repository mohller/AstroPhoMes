""" Configuration settings
"""
import os
import pickle
import sys
import os.path as path

global_path = path.dirname(path.abspath(__file__))

# so that utils.nubase resolves however config was loaded, including by the
# table generators of other projects, which import this file by path
if global_path not in sys.path:
    sys.path.insert(0, global_path)

from utils import nubase

debug_level = 3 # defines the information output of the code

print_module = False

tau_dec_threshold = 0  # lifetime threshold to consider a species

max_A = None  # maximum mass of the model. None means the nuclide table decides

# Lightest nucleus the spallation formula cs_gSp is applied to.  Eq. (A.8) of
# Morejon et al., JCAP 11 (2019) 007, defines spallation as the loss of x > 1
# protons and y > 1 neutrons leaving a residual of mass A_r >= A/2.  At least
# four nucleons must therefore leave, and at most half the nucleus may, so no
# such reaction exists below A = 8.  Below that the empirical relations describe
# only direct nucleon emission, multineutron emission and pion production, and
# the remainder of the 0.28*A total is left unattributed rather than
# redistributed, since Eqs. (A.4)-(A.9) are absolute predictions and not shares.
cs_gSp_min_A = 8

# How the light fragment combinations of a spalled group are counted.
#
#   False -- a combination is the multiset of species it contains, so (t, He3)
#            and (He3, t) are one combination.  This is Eq. (A.13) of the paper,
#            which specifies a combination by its per species counts, and it is
#            what the model was meant to do.
#   True  -- combinations are told apart by the order of the parts within a mass
#            class, which is what the published enumeration did: it ran
#            itertools.product over an ordered list of part masses.  Reproduces
#            data/small_frags_relative_yields.pkl and the published tables.
#
# The two differ wherever a mass class holds more than one species, and the
# normalization does not absorb it: ordering favours those classes, moving the
# relative yields by up to a factor 1.9.  Only the light fragments are affected;
# <dA>, the mass budget and the heavy residual channels are identical either
# way.  See ResidualMultiplicities.
ordered_combinations = False

# Which nuclide table backs spec_data:
#   'nubase' -- data/nubase2020.txt, cut down by the selection below.  Covers
#               the whole chart, which is what the model needs above iron.
#   'legacy' -- data/particle_data.ppo, the 479 nuclide table the published
#               A <= 56 results were produced with.
# The ASTROPHOMES_NUCLIDES environment variable overrides this, so that a
# process can pick a table without editing the file.
nuclide_source = os.environ.get('ASTROPHOMES_NUCLIDES', 'nubase')

# Which nuclides the NUBASE selection keeps. The lifetime threshold sets the
# coverage: the default of 2 s matches CRPropa's photodisintegration tables and
# leaves the light fragment basis at the physical [n, p, d, t, He3, He4], while
# a lower value fills in short lived nuclides near the drip lines at the cost of
# daughters other codes may not know. See utils/nubase.py.
nuclide_tau_min = 2.
nuclide_zmax = None
crpropa_isotopes = None  # path to a CRPropa isotopes-*.txt to intersect with

# non-nuclear particle ids, kept for compatibility with the legacy table
non_nuclear_species = [0, 2, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 20, 21, 50,
                       100, 101]

# nucleons, which every nuclide table carries regardless of its coverage
_NUCLEONS = {
    100: {'name': 'n0', 'charge': 0, 'mass': 0.93957, 'stable': False,
          'lifetime': 885.6126727510937, 'branchings': [(1.0, [12, 20, 101])],
          'incomplete': False},
    101: {'name': 'p+', 'charge': 1, 'mass': 0.93827, 'stable': True,
          'lifetime': float('inf'), 'branchings': [], 'incomplete': False},
}


def load_legacy_particle_data(filename=None):
    """Reads the pickled particle table shipped with the published model.

    Returns (spec_data, provenance).  The pickle was written by python 2, so
    both its keys and the keys of the per species dictionaries come back as
    bytes and have to be decoded.
    """
    if filename is None:
        filename = path.join(global_path, 'data/particle_data.ppo')

    with open(filename, 'rb') as ppofile:
        raw = pickle.load(ppofile, encoding='bytes')

    data = {}
    for datakey, dataval in raw.items():
        if type(datakey) is int:
            data[datakey] = {bkey.decode('ascii'): dataval.get(bkey)
                             for bkey in dataval.keys()}
        else:
            data[datakey.decode('ascii')] = raw.get(datakey)

    nuclei = [k for k in data if isinstance(k, int) and k >= 200]
    provenance = ('source: data/particle_data.ppo (legacy table of the published '
                  'model)\nnuclides: %d, A = %d..%d'
                  % (len(nuclei), min(nuclei) // 100, max(nuclei) // 100))

    return data, provenance


def load_nuclide_table(filename=None, tau_min=None, amax=None, zmax=None,
                       crpropa=None):
    """Builds spec_data from NUBASE2020 under the selection set above.

    NUBASE carries Z, A, half life and mass excess; the remaining fields of the
    returned dictionaries exist so that consumers written against the legacy
    table keep working. Only 'lifetime' and 'mass' are ever read.

    Returns (spec_data, provenance). The provenance records exactly which
    selection was applied, so that whoever generates cross section tables can
    copy it into their own file headers.
    """
    if filename is None:
        filename = os.environ.get('ASTROPHOMES_NUBASE', nubase.DEFAULT_TABLE)
    if tau_min is None:
        tau_min = nuclide_tau_min
    if amax is None:
        amax = max_A
    if zmax is None:
        zmax = nuclide_zmax
    if crpropa is None:
        crpropa = crpropa_isotopes

    isotopes = nubase.read_crpropa_isotopes(crpropa) if crpropa else None
    nuclides = nubase.select(nubase.parse(filename)[0], tau_min, amax, zmax,
                             isotopes)

    data = {ncoid: dict(fields) for ncoid, fields in _NUCLEONS.items()}
    data['non_nuclear_species'] = list(non_nuclear_species)

    for Z, A, tau, mass in nuclides:
        data[100 * A + Z] = {'charge': Z, 'mass': mass,
                             'lifetime': tau, 'stable': tau == float('inf'),
                             'branchings': [], 'incomplete': False}

    return data, nubase.describe(nuclides, tau_min, amax, zmax, crpropa)


if nuclide_source == 'legacy':
    spec_data, nuclide_table_provenance = load_legacy_particle_data()
elif nuclide_source == 'nubase':
    spec_data, nuclide_table_provenance = load_nuclide_table()
else:
    raise ValueError("nuclide_source must be 'nubase' or 'legacy', got %r"
                     % nuclide_source)
