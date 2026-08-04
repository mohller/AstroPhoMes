"""Module implementing empirical relations from paper

Functions that implement the formulas from 
Ref = http://stacks.iop.org/1402-4896/49/i=3/a=004
and others. This modeule calculates the inclusive cross
sections for the Empirical Photomeson Model.
"""

import itertools
from collections import Counter
from numpy import exp, sum, inf, where, zeros, arange
import sys
sys.path.append('../')
from config import *
from utils.utils import get_AZN

# listing functions to create tables of nuclear species
def list_species_by_mass(Amax=None, tau=inf):
	'''Returns a dictionary with the species stable enough
	to be produced in spallation.

	Arguments:
		Amax {int}   -- if given, nuclides heavier than this are left out
		tau {float}  -- minimum lifetime, in seconds, for a species to count
	'''
	species = {}
	for nuc in sorted([k for k in spec_data.keys() if isinstance(k, int)]):
		if (nuc < 100) or (spec_data[nuc]['lifetime'] < tau):
			continue
		At, _, _ = get_AZN(nuc)

		if (Amax is not None) and (At > Amax):
			continue

		if At in species:
			species[At].append(nuc)
		else:
			species[At] = [nuc]

	return species


def modelled_species(Amax=None, tau=None):
	'''Returns the sorted ids of the nuclides the photomeson models describe.

	Nucleons and lighter particles are left out: they are covered by the SOPHIA
	tables rather than by the empirical relations. Defaults come from config, so
	that max_A caps every sweep over the nuclide table in the same way.
	'''
	if Amax is None:
		Amax = max_A
	if tau is None:
		tau = tau_dec_threshold

	species = []
	for nuc in sorted([k for k in spec_data.keys() if isinstance(k, int)]):
		if (nuc <= 101) or (spec_data[nuc]['lifetime'] < tau):
			continue
		if (Amax is not None) and (get_AZN(nuc)[0] > Amax):
			continue
		species.append(nuc)

	return species


def light_fragment_classes(species=None):
	'''Groups the bound fragments of mass 2 to 4 by mass number.

	Returns {A: [(nco_id, Z, N), ...]}, read from the nuclide table so that the
	fragment basis follows whatever selection config.py loaded rather than being
	hardcoded. With a physically sensible selection this is CRPropa's basis,
	d, t, He3 and He4, plus the free nucleons handled separately below.
	'''
	if species is None:
		species = species_by_mass

	classes = {}
	for Af in (2, 3, 4):
		fragments = [(nuc,) + get_AZN(nuc)[1:] for nuc in species.get(Af, ())]
		if fragments:
			classes[Af] = fragments

	return classes


def _accumulate(grid, fragments):
	'''In place grid[Z, N] += sum_s grid[Z - Z_s, N - N_s] over the fragments.

	Sweeping in increasing (Z, N) lets each updated cell feed back into the sums
	of the cells after it, so a single pass turns grid into grid / (1 - P),
	where P is the generating polynomial sum_s z^Z_s w^N_s of the fragments.
	That is the geometric series over any number of them.
	'''
	for Z in range(grid.shape[0]):
		for N in range(grid.shape[1]):
			total = grid[Z, N]
			for _, Zs, Ns in fragments:
				if (Z >= Zs) and (N >= Ns):
					total += grid[Z - Zs, N - Ns]
			grid[Z, N] = total


class ResidualMultiplicities(dict):
	'''Yields of the light fragments produced together with a spallation residue.

	Keyed by the nco id of the spalled nucleon group, each entry gives the
	expected number of each species of mass 1 to 4, normalized so that they
	account for exactly the x protons and y neutrons that left the nucleus.

	The original model obtained these by enumerating, for every partition of
	x + y into parts of at most 4, every assignment of species to the parts, and
	counting how often each species came up. That enumeration grows like
	2^((x+y)/3) and is too slow above iron: A = 208 spalls up to 104 nucleons.

	However, the expected counts required scale polynomially. Writing
	P_A for the generating polynomial of the fragments of mass A, the number of
	assignments with total charge Z and neutron number N is the coefficient of

		T = 1 / ((1 - P_2) (1 - P_3) (1 - P_4)),

	and the number of them containing a given fragment of mass A, counted with
	multiplicity, is the coefficient of that fragment's monomial times

		U_A = T / (1 - P_A).

	The original code used enumerated each partition of x + y as a LIST of part 
	masses, collected the candidate species for every position in that list, and 
	ran itertools.product over them. For the partition [3, 3, 2] that is
	product(S_3, S_3, S_2), which yields (t, He3, d) and (He3, t, d) as two
	separate tuples. Counter then tallied species across every tuple, so both 
	were counted.

	The normalization below is a common rescaling and so does not absorb the
	difference. Ordering favours the classes with more species, and the relative
	yields move accordingly: spalling (x, y) = (12, 20) gives 1.9 times as much
	He3 and half as much He4 as the unordered reading would.

	Ordered is the published behaviour, and reproducing the shipped
	data/small_frags_relative_yields.pkl requires it. However, Eq. (A.13) of the 
	paper specifies a combination by its per species counts, which reads as 
	unordered.

	Both arrays are built once over the whole (Z, N) grid, since neither depends
	on x or y; a query is then a sum over the box Z <= x, N <= y, and the free
	protons and neutrons that pad every assignment out to (x, y) come from the
	same box weighted by (x - Z) and (y - N).
	'''

	def __init__(self, species=None, Zmax=None, Nmax=None, ordered=None):
		dict.__init__(self)

		if species is None:
			species = species_by_mass
		if ordered is None:
			ordered = ordered_combinations
		self.species = species
		self.ordered = ordered
		self.classes = light_fragment_classes(species)

		# one geometric series per mass class counts the assignments to its
		# parts in order; one per species counts them as multisets instead
		if ordered:
			self.groups = [(Af, self.classes[Af]) for Af in sorted(self.classes)]
		else:
			self.groups = [(frag[0], [frag]) for Af in sorted(self.classes)
						   for frag in self.classes[Af]]

		if (Zmax is None) or (Nmax is None):
			nuclides = [nuc for Af in species for nuc in species[Af]]
			charges = [get_AZN(nuc)[1] for nuc in nuclides]
			neutrons = [get_AZN(nuc)[2] for nuc in nuclides]
			Zmax = max(charges) if Zmax is None else Zmax
			Nmax = max(neutrons) if Nmax is None else Nmax

		self._build(Zmax, Nmax)

	def _build(self, Zmax, Nmax):
		'''Tabulates the assignment counts over the (Z, N) grid.'''
		self.Zmax, self.Nmax = Zmax, Nmax

		combinations = zeros((Zmax + 1, Nmax + 1))
		combinations[0, 0] = 1.
		for _, fragments in self.groups:
			_accumulate(combinations, fragments)

		expected = {}
		for key, fragments in self.groups:
			expected[key] = combinations.copy()
			_accumulate(expected[key], fragments)

		self.combinations, self.expected = combinations, expected
		self.clear()

	def _grow(self, x, y):
		'''Rebuilds on a wider grid if a query falls outside the current one.'''
		if (x > self.Zmax) or (y > self.Nmax):
			self._build(max(x, self.Zmax), max(y, self.Nmax))

	def counts(self, x, y):
		'''Expected yields when x protons and y neutrons are spalled off.'''
		counts = Counter()
		if (x < 0) or (y < 0):
			# unphysical spalled groups do occur as intermediate keys, and the
			# enumeration they replace produced nothing for them either
			return counts

		self._grow(x, y)

		for key, fragments in self.groups:
			expected = self.expected[key]
			for nuc, Zs, Ns in fragments:
				if (Zs > x) or (Ns > y):
					continue
				value = expected[:x - Zs + 1, :y - Ns + 1].sum()
				if value:
					counts[nuc] += value

		box = self.combinations[:x + 1, :y + 1]
		free_protons = (box * (x - arange(x + 1))[:, None]).sum()
		free_neutrons = (box * (y - arange(y + 1))[None, :]).sum()
		if free_protons:
			counts[101] = free_protons
		if free_neutrons:
			counts[100] = free_neutrons

		# normalization of the published model: the yields are rescaled so that
		# the ids they carry add up to the id of the spalled group, that is, so
		# that they account for exactly x protons and y neutrons
		suma = 0
		for k, v in counts.items():
			suma += k * v
		if suma:
			ncos_id = 100 * (x + y) + x
			for k in counts:
				counts[k] *= ncos_id / float(suma)

		return counts

	def __missing__(self, ncos_id):
		_, x, y = get_AZN(ncos_id)
		self[ncos_id] = self.counts(x, y)

		return self[ncos_id]


def residual_multiplicities(species=None):
	'''Returns the table of light fragment yields keyed by spalled nco id.

	The table fills itself on demand, so it costs nothing for the nuclides a
	given run never touches and never needs regenerating when the nuclide
	selection changes.
	'''
	return ResidualMultiplicities(species)


# local lookup tables for efficiency
species_by_mass = list_species_by_mass(max_A, tau_dec_threshold)
resmul = residual_multiplicities()


#### empirical relations from Ref...
def cs_gpi(A):
	"""Cross section for pion photoproduction averaged over 
	E[.14, 1.] GeV
	
	Average cross section values obtained by Monte Carlo
	simulations, for nuclei of different masses, starting at
	7Li. It includes the production of pions of all types.
	
	Arguments:
		A {int} -- Number of nucleons in the target nucleus
	
	Returns:
		float -- Mean cross section in miliibarn
	"""
	return 0.027 * A**.847


def cs_gn(A):
	"""Cross section for A(g,n)X averaged over E[.3, 1.] GeV
	
	Returns cross section of photoneutron production averaged
	over the energy range [.3, 1.] GeV, in milibarn units.
	
	Arguments:
		A {int} -- Nucleon number of the target nucleus
	"""

	return 0.104 * A**0.81


def xm(A):
	"""Returns the maximum number of emmited neutrons in a 
	A(g,xn)X reaction
	
	See function cs_gxn()
	
	Arguments:
		A {int} -- 
	
	Returns:
		int -- Maximum number of neutrons to be produced
	"""
	return int(1.4 * A**.457)


def cs_gxn(A, x=2):
	"""Cross section for A(g,xn)X averaged over E[.2, 1.] GeV
	
	Returns cross section of photoneutrons production (x > 1)
	averaged over the energy range [.3, 1.] GeV, in milibarn 
	units. The number of neutrons x is checked to be lower than
	xm (see function definition xm).
	
	Arguments:
		A {int} -- Nucleon number of the target nucleus
		x {int} -- Number of neutrons produced (1 < x < xm)
	"""

	if 1 < x < xm(A):
		k = 37. * A ** -.924
		return 0.187 * A**0.684 * exp(-k * (x - 1)**1.25)
	else:
		return 0


def cs_gxn_all(A):
	"""Cross section for A(g,xn)X averaged over E[.2, 1.] GeV
	
	Returns cross section of photoneutrons production (x > 1)
	averaged over the energy range [.3, 1.] GeV, in milibarn 
	units. The number of neutrons x is checked to be lower than
	xm (see function definition xm).
	
	Arguments:
		A {int} -- Nucleon number of the target nucleus
		x {int} -- Number of neutrons produced (1 < x < xm)
	"""
	cs_gxn_summed = 0
	for xi in range(2, xm(A)):
		cs_gxn_summed += cs_gxn(A, xi)

	return cs_gxn_summed


def cs_gp(Z=1, **kwargs):
	"""Cross section for A(g,p)X averaged over E[.3, .8] GeV
	
	Returns cross section of photoproduction of a proton averaged
	over the energy range [.3, .8] GeV, in milibarn units.
	
	Arguments:
		Z {int} -- Number of protons of the target nucleus
		A {int} -- (optional) 
	"""
	if 'A' in kwargs:
		return 0.078 * kwargs['A']**0.5
	else:
		return 0.115 * Z**0.5


def cs_gSp(Z, A, x=1, y=1):
	"""Cross section for spallation averaged over E[.2, 1.] GeV

	Returns cross section of photoproduction of multiple
	protons and neutrons in a spallation procees, averaged
	over the energy range [.2, 1.] GeV, in milibarn units.

	Eq. (A.8) of Morejon et al., JCAP 11 (2019) 007.  The reaction it describes
	loses x > 1 protons and y > 1 neutrons while leaving a residual of mass
	A_r >= A/2, so at least four nucleons have to depart and at most half the
	nucleus may: below cs_gSp_min_A = 8 there is no such reaction and the
	formula returns zero.  It is not merely small there, it is undefined --
	E = 446/A leaves the range where the fitted parameters vary at all, cs_M
	sticking at 0.248 below A = 21 and B at 0.25 below A = 45.

	Arguments:
		Z {int} -- Number of protons of the target nucleus
		A {int} -- Number of nucleons of the target nucleus
		           cs_gSp_min_A <= A <= 90
		x {int} -- Number of protons produced. Default is 1.
		           1 <= x <= Z/2
		y {int} -- Number of neutron produced. Default is 1.
		           1 <= y <= (A - Z)/2

	"""
	if A < cs_gSp_min_A:
		return 0.

	K = 0.466
	a = float(Z) / (A - Z)
	C = 2.3 * a - 1.044
	E = 446. / A
	if E < 21.:
		cs_M = 15.7 / E**1.356
	else:
		cs_M = 0.248
	if E < 10.:
		B = 3.03 / E**1.06
	else:
		B = 0.25

	return cs_M * exp(-B * (x - 1) - K*(x - C*a*y)**2)


def cs_gSp_all(Z, A):
	"""Cross section summed for all possible spallation events
	"""
	mother = 100*A + Z
	cs_tot = 0
	for A_big_frag in range(A//2, A-1):
		for big_frag in species_by_mass.get(A_big_frag, ()):
			_, x, y = get_AZN(mother - big_frag)
			spalled_id = 100*(x+y) + x
			
			if (x < 1) or (y < 1):
				# in spallation at least a neutron and proton escape
				continue
			
			cs_frag = cs_gSp(Z, A, x, y)
			cs_tot += cs_frag
			
	return cs_tot


def cs_gSp_all_inA(A):
    """Cross section summed for all possible spallation events
    """
    cs_vals = []
    for nuc in species_by_mass.get(A, ()):
        _, Zi, _ = get_AZN(nuc)
        cs_vals.append(cs_gSp_all(Zi, A))

    if not cs_vals:
        # no bound nuclide of this mass, so nothing can spall off one
        return 0.

    return max(cs_vals)


def cs_tot(A, sumed=True):
	'''Determines the norm for the empirical formulas
	such that the addition of inclusive cross sections
	does not fluctuate with A. Renormalization value 
	depends on the mass and is set to the mean of the
	total empirical cross section per nucleon over 
	a range of A in A=4-55.

	Arguments:
		A {int}         -- Number of nucleons of the target nucleus
		sumed {boolean} -- If True, returns the total cross section as the sum
		                   of all the different components, otherwise it 
						   provides the average value according to the formula.
	'''
	if sumed:
		csp = cs_gp(A=A)
		cspi = cs_gpi(A)
		csn = cs_gn(A)
		csxn = cs_gxn_all(A)
		csSpal = cs_gSp_all_inA(A)
		
		cstot = csp + cspi + csn + csxn + csSpal
	else:
		cstot = .28 * A

	return cstot


#### inclusive cross sections derived from relations above, and related functions

def partition_probability(partition, A, beta=.1):
	"""Gives partitions probabilities as exp(-beta*Ei)
	where Ei is the sum of binding energies of the components of the partition.
	
	Given a partition (list of mass fragments {A_j}), the sets of all possible
	species per mass {S^A_k} are taken. Then all possible combinations corresponding
	to the partition are built C_l={S^j_m} and the combination is given a probability
	based on it's energy: P_l = exp(-beta*E_l). The probablities of all combinations
	are normalized to unity (P_l = P_l / sum(P_l))
	
	Arguments:
		partition {[list]} -- list of [A_k] masses into which the original nucleus was split

	Returns:
		combinations -- a list of possible nuclei that make up the A partition given
		yields -- the probabilities of each of the partitions, obtained by a statistical
			argument (i.e. canonical ensemble )
	"""

	species = [species_by_mass[Af] for Af in partition if Af in species_by_mass]
	combinations = list(itertools.product(*species))

	yields = []
	for combination in combinations:
		Ei = A
		for nuc in combination:
			Ei -= spec_data[nuc]['mass']
		
		yields.append(exp(-beta*Ei))

	tot_yields = sum(yields)
	yields = [y/tot_yields for y in yields]
	
	return combinations, yields


def partition_probability(combinations, A, beta=.1):
	"""Gives partitions probabilities as exp(-beta*Ei)
	where Ei is the sum of binding energies of the components of the partition.
	
	Given a partition (list of mass fragments {A_j}), the sets of all possible
	species per mass {S^A_k} are taken. Then all possible combinations corresponding
	to the partition are built C_l={S^j_m} and the combination is given a probability
	based on it's energy: P_l = exp(-beta*E_l). The probablities of all combinations
	are normalized to unity (P_l = P_l / sum(P_l))
	
	Arguments:
		partition {[list]} -- list of [A_k] masses into which the original nucleus was split

	Returns:
		combinations -- a list of possible nuclei that make up the A partition given
		yields -- the probabilities of each of the partitions, obtained by a statistical
			argument (i.e. canonical ensemble )
	"""

	yields = []
	for combination in combinations:
		Ei = A
		for nuc in combination:
			Ei -= spec_data[nuc]['mass']
		
		yields.append(exp(-beta*Ei))

	tot_yields = sum(yields)
	yields = [y/tot_yields for y in yields]
	
	return combinations, yields


def gxn_multiplicities(mother):
	"""Multiplicities for multineutron emission A(g,xn)X
	
	Arguments:
		A {int} -- Nucleon number of the target nucleus
	"""
	cs_sum = 0
	cs_gxn_incl = {100:0}
	Am, _, _ = get_AZN(mother)

	for xi in range(2, xm(Am)):
		cs = cs_gxn(Am, xi)
		cs_gxn_incl[100] += xi*cs
		cs_gxn_incl[mother - xi*100] = cs

		cs_sum += cs

	for dau in cs_gxn_incl:
		if cs_gxn_incl[dau] != 0:
			cs_gxn_incl[dau] /= cs_sum
	
	return cs_gxn_incl


def spallation_multiplicities(mother):
	'''Calculates the inclusive cross sections of all fragments
	of mothter species (moter is a neucos id) for spallation
	'''
	Am, Zm, _ = get_AZN(mother)

	incl_tab = {}
	cs_sum = 0

	if Am < cs_gSp_min_A:
		# no spallation channel exists this light, see cs_gSp.  Returning early
		# rather than letting the zero cross sections through keeps daughters
		# with a multiplicity of exactly zero out of the tables
		return incl_tab

	for A_big_frag in range(Am//2, Am-1):
		for big_frag in species_by_mass.get(A_big_frag, ()):
			_, x, y = get_AZN(mother - big_frag)
			spalled_id = 100*(x+y) + x

			if (x < 1) or (y < 1):
				# in spallation at least a neutron and proton escape
				continue

			cs_frag = cs_gSp(Zm, Am, x, y)
			cs_sum += cs_frag  # sum of all cross sections to normalize incl_tab

			if big_frag in incl_tab:
				incl_tab[big_frag] += cs_frag
			else:
				incl_tab[big_frag] = cs_frag

			# get low fragment incl_tab from using Counter on a prepared list with x, y outputs
			spalled_mult = resmul[spalled_id]
			for dau in spalled_mult:
				if dau in incl_tab:
					incl_tab[dau] += cs_frag * spalled_mult[dau]
				else:
					incl_tab[dau] = cs_frag * spalled_mult[dau]
	for dau in incl_tab:
		# all spallation cross section should match total spallation cross section
		incl_tab[dau] /= where(cs_sum == 0, inf, cs_sum)

	return incl_tab


def cs_Rincl(Z, A, yields):
	"""Returns incl of residual production
	
	[description]
	
	Arguments:
		Z {[type]} -- [description]
		A {[type]} -- [description]
	"""
	Amax = max(yields.keys())
	nuclist = [100, 101]
	csilist = [cs_nincl(Z, A), cs_pincl(Z, A)]
	sub_frags = {}
	for nz in range(0, int(Z / 2.) + 1):
		for nn in range(0, int((A - Z) / 2.) + 1):
			Ared = min(Amax, nn + nz)

			# print nn, nz, A-Z, Z
			if nn + nz == 0:
				# add pion contribution
				nuclist += [A*100 + Z,
							A*100 + Z - 1,
							A*100 + Z + 1]
				csilist += [cs_gpi(A) / 3,] * 3
				continue

			cs_incl = 0
			new_frag = 0
			if (nn == 1) and (nz == 0):
				cs_incl = cs_gn(A)
			elif (nn > 1) and (nz == 0):
				cs_incl = cs_gxn(A, nn)
			elif (nn == 0) and (nz == 1):
				cs_incl = cs_gp(Z)
			elif (nn >= 1) and (nz >= 1):
				cs_incl = cs_gSp(Z, A, nz, nn)
				new_frag = 100 * Ared + Ared//2 - nz
				if new_frag > 0:
					csilist.append(cs_incl)
					nuclist.append(new_frag)				
				sub_frags = yields[Ared]

			if cs_incl > 0:
				csilist.append(cs_incl)
				nuclist.append(100 * (A - nn - nz) + Z - nz)
				if sub_frags:
					suma = 0
					for nuc, val in sub_frags.items():
						Af, _, _ = get_AZN(nuc)
						suma += Af * val
					norm = Ared / suma

					for nuc, val in sub_frags.items():
						if nuc in nuclist:
							csilist[nuclist.index(nuc)] += val * norm * cs_incl
						else:
							csilist.append(val * norm * cs_incl)
				
	return nuclist, csilist


def multiplicity_table(mother):
	'''Returns a dict with the multiplicities
	for all fragments from mother is contained.
	The differentence with spallation_inclusive is that
	here all processes are contained
	'''
	gxn_mult = gxn_multiplicities(mother)
	sp_mult = spallation_multiplicities(mother)
	
	Am, Zm, _ = get_AZN(mother)

	cspi = cs_gpi(Am)
	csp = cs_gp(A=Am)
	csn = cs_gn(Am)
	csxn = cs_gxn_all(Am)
	cs_tot = .28 * Am
	if Am < cs_gSp_min_A:
		# Eq. (A.10) reads the remainder of the total as spallation, but there
		# is no spallation this light.  It stays unattributed: the other
		# relations are absolute cross sections, not shares of the total, so
		# there is nothing to give it to
		csSp = 0.
	else:
		csSp = cs_tot - (cspi + csp + csn + csxn)

	multiplicities = {100: 1.*csn/cs_tot,
					  101: 1.*csp/cs_tot,
					  mother - 100: 1.*csn/cs_tot,
					  mother - 101: 1.*csp/cs_tot,}

	for dau, mult in gxn_mult.items():
		if dau in multiplicities:
			multiplicities[dau] += mult * csxn / cs_tot
		else:
			multiplicities[dau] = mult * csxn / cs_tot

	for dau, mult in sp_mult.items():
		if dau in multiplicities:
			multiplicities[dau] += mult * csSp / cs_tot
		else:
			multiplicities[dau] = mult * csSp / cs_tot

	return multiplicities


def main():

	print('light fragment basis:', light_fragment_classes())
	print(multiplicity_table(1407))
	print('spalled groups evaluated so far:', sorted(resmul.keys()))

if __name__ == '__main__':
	main()