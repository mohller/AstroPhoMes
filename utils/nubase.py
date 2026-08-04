"""Reader for the NUBASE2020 evaluation of nuclear properties.

The photomeson models need, for every nuclide they may produce or destroy, only
a mass number, a charge, a lifetime and a ground state mass. This module pulls
those four numbers out of data/nubase2020.txt and applies the coverage cuts that
decide which species the model knows about.

Deliberately free of project imports and of third party dependencies, so that
config.py can use it before anything else is importable.

Run it directly to inspect a selection:

    python3 utils/nubase.py --tau-min 2

The table is the one distributed by the Atomic Mass Data Center, retrieved from
https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt, and it documents its
own column layout in the header. Note that the layout is NOT the same as
NUBASE2016's: the mass excess field is wider and the half life moved from
columns 62-69 to 70-78, so the older column map silently reads nonsense.

Derived from the NUBASE reader in CRISP (crisp/data/nucleardecays.py), which is
where the half life unit factors, the nuclear mass convention below and the rule
for picking ground states out of the isomer ids come from. The parsing itself
was rewritten on the standard library, since CRISP reads the table with pandas
and this module has to be importable before anything else is; the column map is
NUBASE2020's own rather than CRISP's 2016 one, and the selection is new.

Reference:
    F. G. Kondev, M. Wang, W. J. Huang, S. Naimi and G. Audi, "The NUBASE2020
    evaluation of nuclear physics properties", Chinese Physics C 45 (2021)
    030001.
"""

import os.path as path
from math import log

# atomic mass constant and electron mass, CODATA 2018, in GeV
U_GEV = 0.93149410242
ME_GEV = 0.51099895000e-3

DEFAULT_TABLE = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                          'data/nubase2020.txt')

# NUBASE2020 is a fixed width format, documented in the header of the file
# itself.  These are the fields we need, as zero based half open slices of the
# one based inclusive columns given there.  The isomer id encodes the charge and
# the isomeric level as ZZZi, so the ground state of a nuclide is the row whose
# id equals 10 * Z; i = 1..9 are isomers, levels, resonances and IAS.
_COLUMNS = {
    'A': (0, 3),                    # columns   1:3    AAA
    'isomer_id': (4, 8),            # columns   5:8    ZZZi
    'mass_excess_keV': (18, 31),    # columns  19:31   Mass excess in keV
    'half_life': (69, 78),          # columns  70:78   Half-life
    'half_life_units': (78, 80),    # columns  79:80   Half-life unit
}

_YEAR = 365 * 24 * 3600.   # as in CRISP; the 0.07% against a Julian year is far
                           # below anything a lifetime threshold can resolve
_TIME_UNITS = {
    '': 1., 'ys': 1e-24, 'zs': 1e-21, 'as': 1e-18, 'fs': 1e-15, 'ps': 1e-12,
    'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1., 'm': 60., 'h': 3600.,
    'd': 24 * 3600., 'y': _YEAR, 'ky': 1e3 * _YEAR, 'My': 1e6 * _YEAR,
    'Gy': 1e9 * _YEAR, 'Ty': 1e12 * _YEAR, 'Py': 1e15 * _YEAR,
    'Ey': 1e18 * _YEAR, 'Zy': 1e21 * _YEAR, 'Yy': 1e24 * _YEAR,
}

# The models have no use for a nuclide they cannot spall into anything, and
# neutrons and protons are handled outside the nuclide table.
MIN_A = 2

# Particle ids are 100*A + Z, so a charge of 100 would be indistinguishable from
# one more nucleon and no charge: Fm-250 and a bag of 251 neutrons would share
# the id 25100. This is a hard limit of the identification scheme, not a physics
# choice, so it is applied whatever the caller asks for. NUBASE reaches Z = 118;
# everything above fermium is dropped.
MAX_Z = 99


def parse(filename=None):
    """Reads NUBASE2020 and returns [(Z, A, tau [s], mass [GeV]), ...].

    Isomeric states are dropped, as are the rows whose half life is not a
    number: unknown, given only as a limit, or the literal 'p-unst' for
    particle unstable. A nuclide with no measured lifetime cannot be placed
    relative to a lifetime threshold.

    Masses are nuclear, not atomic: ``m = A*u + mass_excess - Z*m_e``, matching
    the convention of the legacy ``particle_data.ppo`` (Fe-56: 52.10 GeV).
    Mass excesses flagged as extrapolated ('#') are used as given.
    """
    if filename is None:
        filename = DEFAULT_TABLE

    nuclides, skipped = [], 0

    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            def field(name):
                start, stop = _COLUMNS[name]
                return line[start:stop].strip()

            isomer_id = field('isomer_id')
            if 'W' in isomer_id or not isomer_id.isdigit():
                continue

            Z = int(isomer_id[:3])
            if int(isomer_id) != 10 * Z:   # an isomeric level, not the ground state
                continue

            A = int(field('A'))

            half_life = field('half_life').replace('stbl', 'inf')
            try:
                half_life = float(half_life)
            except ValueError:
                skipped += 1
                continue

            if half_life == float('inf'):
                tau = float('inf')
            else:
                unit = _TIME_UNITS.get(field('half_life_units'))
                if unit is None:
                    skipped += 1
                    continue
                tau = half_life * unit / log(2)

            excess = field('mass_excess_keV').split('#')[0]
            excess = float(excess) if excess else 0.
            mass = A * U_GEV + excess * 1e-6 - Z * ME_GEV

            nuclides.append((Z, A, tau, mass))

    return nuclides, skipped


def read_crpropa_isotopes(filename):
    """Reads a CRPropa 'Z N A' isotope list into a set of (Z, A) pairs."""
    isotopes = set()
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            Z, _, A = (int(v) for v in line.split()[:3])
            isotopes.add((Z, A))

    return isotopes


def select(nuclides, tau_min=2., amax=None, zmax=None, crpropa=None):
    """Applies the coverage cuts, returning the surviving nuclides sorted."""
    kept = []
    for Z, A, tau, mass in nuclides:
        if A < MIN_A or Z < 1 or Z > MAX_Z:
            continue
        if tau <= tau_min:
            continue
        if (amax is not None) and (A > amax):
            continue
        if (zmax is not None) and (Z > zmax):
            continue
        if (crpropa is not None) and ((Z, A) not in crpropa):
            continue
        kept.append((Z, A, tau, mass))

    return sorted(kept, key=lambda row: (row[1], row[0]))


CAVEAT = """\
CAVEAT -- above A = 56 the empirical photomeson model is EXTRAPOLATION, not
validated physics.  cs_gSp, the Rudstam-type spallation formula, is documented
for A <= 90; cs_gpi, cs_gn and cs_gxn were fitted on data up to iron; and the
mass scaling exponents alpha and alpha_pi were calibrated on the same range,
though the universal function itself is fitted to nuclear photoabsorption data
that does extend to heavy nuclei.  Tables generated from nuclides above A = 56
must be labelled as superheavy extrapolations by their consumers.

The lifetime cut also shapes the results.  Spallation residuals have to be
nuclides of this selection, so close to the drip lines, where the neighbours of
a mother are too short lived to be listed, the nearest reachable residual is
several mass units away and the mean mass loss of that mother comes out
inflated.  Lowering tau_min populates those gaps and smooths the trend of <dA>
with A, at the cost of daughters that CRPropa may not know."""


def describe(nuclides, tau_min, amax=None, zmax=None, crpropa=None):
    """Returns a provenance block recording how a selection was made."""
    masses = sorted(set(A for _, A, _, _ in nuclides))
    gaps = [A for A in range(MIN_A, masses[-1] + 1) if A not in masses]

    return '\n'.join([
        'source: NUBASE2020 (Kondev et al., Chin. Phys. C 45 (2021) 030001),',
        '        ground states only, isomers excluded',
        'selection: tau > %g s, A >= %d, Z <= %d (particle ids are 100*A + Z)%s%s%s' % (
            tau_min, MIN_A, MAX_Z,
            ', A <= %d' % amax if amax else '',
            ', Z <= %d' % zmax if zmax else '',
            ', intersected with %s' % path.basename(crpropa) if crpropa else ''),
        'nuclides: %d, A = %d..%d, Z <= %d' % (
            len(nuclides), masses[0], masses[-1],
            max(Z for Z, _, _, _ in nuclides)),
        'unpopulated mass numbers: %s' % (gaps if gaps else 'none'),
        'masses: nuclear ground state, A*u + mass_excess - Z*m_e, in GeV',
        '',
        CAVEAT,
    ])


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nubase', default=DEFAULT_TABLE)
    ap.add_argument('--tau-min', type=float, default=2.,
                    help='keep nuclides with mean lifetime above this, in seconds '
                         '(default 2, matching CRPropa\'s photodisintegration coverage)')
    ap.add_argument('--amax', type=int, default=None,
                    help='drop nuclides above this mass number (default: no cap)')
    ap.add_argument('--zmax', type=int, default=None,
                    help='drop nuclides above this charge (default: no cap)')
    ap.add_argument('--crpropa-isotopes', default=None,
                    help='intersect with a CRPropa isotopes-*.txt list, so that every '
                         'daughter the model can produce is a species CRPropa knows')
    args = ap.parse_args()

    nuclides, skipped = parse(args.nubase)
    print('%s: %d ground states (%d rows without a usable half life)'
          % (path.basename(args.nubase), len(nuclides), skipped))

    crpropa = (read_crpropa_isotopes(args.crpropa_isotopes)
               if args.crpropa_isotopes else None)
    kept = select(nuclides, args.tau_min, args.amax, args.zmax, crpropa)

    print(describe(kept, args.tau_min, args.amax, args.zmax, args.crpropa_isotopes))


if __name__ == '__main__':
    main()
