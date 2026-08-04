"""Tests for the extension of the empirical photomeson model above A = 56

The model was published for nuclei up to iron. Extending it upwards meant
widening the nuclide table and replacing the explicit enumeration of light
fragment combinations, which does not scale, with a closed form count. These
tests pin both: that the replacement reproduces the published table exactly, and
that the empirical relations stay well behaved over the whole nuclide chart.
"""

import numpy as np
from collections import Counter
import pickle
import sys
import unittest
from os.path import join

sys.path.append('../')
from config import (global_path, load_legacy_particle_data, load_nuclide_table,
                    nuclide_source, spec_data, nuclide_table_provenance)
from photomeson_lib.phenom_relations import (
    ResidualMultiplicities, cs_gn, cs_gp, cs_gpi, cs_gSp, cs_gSp_min_A,
    cs_gxn_all, light_fragment_classes, list_species_by_mass, modelled_species,
    multiplicity_table, resmul, spallation_multiplicities, species_by_mass, xm)
from utils.utils import get_AZN


# The legacy table stops at iron by construction, so the tests that exercise the
# widened chart have nothing to say about it. The acceptance gate below runs
# either way: it builds the legacy fragment set explicitly rather than relying on
# whichever table config loaded.
needs_wide_table = unittest.skipIf(
    nuclide_source == 'legacy',
    'the legacy nuclide table stops at A = 56')


def group_by_mass(data):
    """Groups a spec_data mapping into {A: [nco_id, ...]}, as the model does."""
    species = {}
    for nuc in sorted(k for k in data if isinstance(k, int)):
        if nuc < 100:
            continue
        At, _, _ = get_AZN(nuc)
        species.setdefault(At, []).append(nuc)

    return species


class Test_ResidualMultiplicities(unittest.TestCase):
    """The closed form count against the enumeration it replaces."""

    def test_reproduces_published_table(self):
        """Acceptance gate: reproduce data/small_frags_relative_yields.pkl.

        That file was produced by the published enumeration over the legacy
        nuclide table, so the comparison has to be made with the legacy light
        fragment set, which includes the unphysical species (diproton, Li-4)
        that a modern table drops.
        """
        with open(join(global_path, 'data/small_frags_relative_yields.pkl'), 'rb') as f:
            published = pickle.load(f)

        legacy, _ = load_legacy_particle_data()
        counts = ResidualMultiplicities(group_by_mass(legacy), ordered=True)

        self.assertEqual(len(published), 501)
        for ncos_id, expected in published.items():
            obtained = counts[ncos_id]
            for species in set(expected) | set(obtained):
                self.assertAlmostEqual(float(obtained.get(species, 0.)),
                                       float(expected.get(species, 0.)),
                                       delta=1e-12 * abs(float(expected.get(species, 1.))),
                                       msg='spalled group %d, species %s' % (ncos_id, species))

    def test_unphysical_groups_are_empty(self):
        """Groups with a negative proton or neutron count yield nothing."""
        counts = ResidualMultiplicities()
        for ncos_id in (203, 209, 305):
            self.assertEqual(len(counts[ncos_id]), 0)

    def test_yields_account_for_the_spalled_nucleons(self):
        """The normalization makes the yields add up to the spalled group."""
        counts = ResidualMultiplicities()
        for x, y in ((1, 1), (3, 5), (12, 20), (41, 63)):
            yields = counts[100 * (x + y) + x]
            total = sum(nuc * value for nuc, value in yields.items())
            self.assertAlmostEqual(total, 100. * (x + y) + x, places=6)

    def test_unordered_matches_a_direct_multiset_enumeration(self):
        """With ordered=False a combination is the multiset of its species.

        Checked against the obvious enumeration: walk the species, take every
        number of copies of each that still fits inside (x, y), and tally. That
        is Eq. (A.13) of the paper read literally, a combination being specified
        by its per species counts.
        """
        counts = ResidualMultiplicities(ordered=False)
        fragments = [(nuc, Z, N) for members in counts.classes.values()
                     for nuc, Z, N in members]

        def enumerate_multisets(x, y):
            tally = Counter()

            def walk(index, Z, N, chosen):
                if index == len(fragments):
                    for nuc, copies in chosen.items():
                        tally[nuc] += copies
                    tally[101] += x - Z
                    tally[100] += y - N
                    return

                nuc, Zs, Ns = fragments[index]
                copies = 0
                while (Z + copies * Zs <= x) and (N + copies * Ns <= y):
                    if copies:
                        chosen[nuc] = copies
                    walk(index + 1, Z + copies * Zs, N + copies * Ns, chosen)
                    chosen.pop(nuc, None)
                    copies += 1

            walk(0, 0, 0, {})
            tally = Counter({k: v for k, v in tally.items() if v})
            suma = sum(k * v for k, v in tally.items())
            for k in tally:
                tally[k] *= (100 * (x + y) + x) / float(suma)

            return tally

        for x, y in ((1, 1), (2, 3), (4, 4), (6, 9), (7, 7)):
            obtained = counts.counts(x, y)
            expected = enumerate_multisets(x, y)
            for nuc in set(obtained) | set(expected):
                self.assertAlmostEqual(float(obtained.get(nuc, 0.)),
                                       float(expected.get(nuc, 0.)), places=9,
                                       msg='(x, y) = (%d, %d), species %s' % (x, y, nuc))

    def test_the_two_readings_disagree_where_a_class_has_two_species(self):
        """Ordering only matters for a mass class holding more than one species.

        Pinned on an explicit basis rather than the loaded one, since the point
        is the mechanism: mass 3 is the only class here with two members, so it
        is the only one ordering can enhance, and everything else is depleted
        once the yields are renormalized. A class with a single species has
        1/(1 - q_s) either way.
        """
        basis = {2: [201], 3: [301, 302], 4: [402]}
        ordered = ResidualMultiplicities(basis, ordered=True)
        unordered = ResidualMultiplicities(basis, ordered=False)

        a, b = ordered.counts(12, 20), unordered.counts(12, 20)
        self.assertTrue(a[301] > b[301])
        self.assertTrue(a[302] > b[302])
        for single in (201, 402, 100, 101):
            self.assertTrue(a[single] < b[single], 'species %d' % single)

        # both still account for exactly the spalled nucleons
        for yields in (a, b):
            self.assertAlmostEqual(sum(k * v for k, v in yields.items()),
                                   100. * 32 + 12, places=6)

    def test_grid_grows_on_demand(self):
        """A query outside the tabulated grid rebuilds it instead of failing."""
        counts = ResidualMultiplicities(species_by_mass, Zmax=4, Nmax=4)
        yields = counts[100 * 40 + 15]
        self.assertTrue(len(yields) > 0)
        self.assertTrue(counts.Zmax >= 15)


@needs_wide_table
class Test_NuclideTable(unittest.TestCase):
    """The widened nuclide universe."""

    def test_light_fragment_basis_is_physical(self):
        """No unbound light species survive a lifetime cut.

        The legacy table carries the diproton (202) and two isotopes of Li-4
        (303, 403); a table built with a positive lifetime threshold leaves the
        basis at CRPropa's own [n, p, d, t, He3, He4].
        """
        classes = light_fragment_classes()
        fragments = [nuc for members in classes.values() for nuc, _, _ in members]

        for unbound in (202, 303, 403):
            self.assertNotIn(unbound, fragments)

        self.assertEqual(sorted(fragments), [201, 301, 302, 402])

    def test_covers_the_chart_above_iron(self):
        """The table reaches at least lead, which is the point of the exercise."""
        heaviest = max(get_AZN(nuc)[0] for nuc in modelled_species())
        self.assertTrue(heaviest >= 208,
                        'nuclide table stops at A = %d' % heaviest)
        self.assertIn(20882, spec_data)

    def test_unpopulated_masses_are_tolerated(self):
        """There is no bound A = 5 or A = 8, and the model must cope.

        Mothers in this range enumerate big fragments of those masses, so an
        unguarded lookup would raise rather than skip them.
        """
        self.assertNotIn(5, species_by_mass)
        self.assertNotIn(8, species_by_mass)

        for mom in (704, 904, 1004, 1206):
            if mom in spec_data:
                self.assertTrue(len(multiplicity_table(mom)) > 0)

    def test_amax_is_honoured(self):
        """list_species_by_mass applies its Amax argument."""
        capped = list_species_by_mass(56)
        self.assertTrue(max(capped) <= 56)
        self.assertTrue(max(species_by_mass) >= 208)

    def test_provenance_is_recorded(self):
        """The table says where it came from and that it extrapolates."""
        self.assertIn('NUBASE', nuclide_table_provenance)
        self.assertIn('EXTRAPOLATION', nuclide_table_provenance)

    def test_nuclide_table_matches_spec_data(self):
        """The plain text table and the loaded dictionary agree."""
        data, _ = load_nuclide_table()
        nuclides = [k for k in data if isinstance(k, int) and k > 101]
        self.assertTrue(len(nuclides) > 1500)
        for nuc in nuclides:
            A, Z, N = get_AZN(nuc)
            self.assertEqual(data[nuc]['charge'], Z)
            self.assertTrue(data[nuc]['mass'] > 0)
            self.assertTrue(N >= 0)


@needs_wide_table
class Test_EmpiricalRelationsAboveIron(unittest.TestCase):
    """The empirical formulas evaluated over the whole widened table."""

    @classmethod
    def setUpClass(cls):
        cls.species = modelled_species()

    def test_cross_section_components_are_positive(self):
        """Every component, and the spallation remainder, stays positive.

        The remainder is what multiplicity_table hands to spallation; a sign
        change there would silently invert the fragment yields.
        """
        for mom in self.species:
            A, Z, _ = get_AZN(mom)
            components = [cs_gpi(A), cs_gp(A=A), cs_gn(A), cs_gxn_all(A)]
            remainder = .28 * A - sum(components)

            for value in components[:3]:
                self.assertTrue(value > 0, 'nuclide %d' % mom)

            # multineutron emission only opens up once xm(A) > 2, which takes
            # A >= 6; below that the channel is legitimately absent
            self.assertTrue(cs_gxn_all(A) >= 0, 'nuclide %d' % mom)
            if A >= 6:
                self.assertTrue(cs_gxn_all(A) > 0, 'nuclide %d' % mom)

            if A >= cs_gSp_min_A:
                self.assertTrue(remainder > 0,
                                'spallation remainder %.3f mb for nuclide %d'
                                % (remainder, mom))

    def test_neutron_ladder_is_sane(self):
        """(g,xn) never emits more neutrons than the nucleus has, halved."""
        for mom in self.species:
            A, _, _ = get_AZN(mom)
            if A > 6:
                self.assertTrue(xm(A) < A / 2., 'nuclide %d' % mom)

    def test_mass_is_conserved_within_tolerance(self):
        """The inclusive yields carry between 90% and 100% of the mother mass.

        Not exactly 100%: the empirical relations are independent fits, so the
        residual and the fragments do not close the mass budget exactly. Fe-56
        gives 0.948, and the extrapolation must not drift away from that.

        Only where spallation applies. Below cs_gSp_min_A the fragments carry
        far less, because the share of the total that Eq. (A.10) would call
        spallation is left unattributed; see test_no_spallation_below_min_A.
        """
        for mom in self.species:
            A, _, _ = get_AZN(mom)
            if A < cs_gSp_min_A:
                continue
            mults = multiplicity_table(mom)
            carried = sum(get_AZN(dau)[0] * mult for dau, mult in mults.items()) / A
            self.assertTrue(0.90 <= carried <= 1.00,
                            'nuclide %d carries %.4f of its mass' % (mom, carried))

    def test_no_spallation_below_min_A(self):
        """cs_gSp is not applied below cs_gSp_min_A.

        Eq. (A.8) of the paper defines spallation as losing x > 1 protons and
        y > 1 neutrons while keeping a residual of mass A_r >= A/2. Four
        nucleons must leave and at most half the nucleus may, so no such
        reaction exists below A = 8 and the formula has no meaning there.
        """
        self.assertEqual(cs_gSp_min_A, 8)

        for mom in self.species:
            A, Z, _ = get_AZN(mom)
            if A < cs_gSp_min_A:
                self.assertEqual(cs_gSp(Z, A, 1, 1), 0., 'nuclide %d' % mom)
                self.assertEqual(spallation_multiplicities(mom), {},
                                 'nuclide %d' % mom)
            else:
                self.assertTrue(cs_gSp(Z, A, 2, 2) > 0, 'nuclide %d' % mom)

    def test_light_nuclei_keep_their_nucleon_channels(self):
        """Removing spallation must not remove direct emission with it."""
        for mom in (301, 402, 603, 704):
            if mom not in spec_data:
                continue
            mults = multiplicity_table(mom)
            self.assertIn(100, mults)
            self.assertIn(101, mults)
            self.assertTrue(mults[100] > 0 and mults[101] > 0)

    def test_lead_is_tractable(self):
        """Pb-208, the nuclide the whole exercise is about."""
        mults = multiplicity_table(20882)

        self.assertTrue(len(mults) > 500)
        self.assertAlmostEqual(.28 * 208 - (cs_gpi(208) + cs_gp(A=208)
                                            + cs_gn(208) + cs_gxn_all(208)),
                               31.00, places=1)
        self.assertEqual(xm(208), 16)

        # every daughter is a nuclide the model can name, or a nucleon
        for dau in mults:
            self.assertTrue(dau >= 100)
            A, Z, N = get_AZN(dau)
            self.assertTrue((Z >= 0) and (N >= 0), 'daughter %d' % dau)


if __name__ == '__main__':
    unittest.main()
