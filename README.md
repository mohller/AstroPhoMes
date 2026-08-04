# AstroPhoMes

A python library implementing astrophysical photomeson models for usage in UHECR sources simulations and related physical scenarios

## Description:

This repository contains tools used for calculating photo-nuclear interactions.
The code provides cross sections for the interaction between photons and nuclei for astrophysical problems.
This is of interest for modeling Ultra-High Energy Cosmic Ray sources and transport.

## Usage:

For details see the publication [arXiv](https://arxiv.org/abs/1904.07999), and for usage refer to the examples folder.

The lib folder contains the main classes which implement the photomeson models.

The model contains methods to obtain the cross sections for a variety of nuclei. 

The particle identification follows the convention below

#### Particle ID convention:

- 2: pi plus
- 3: pi minus
- 4: pi zero
- 100\*A+Z: for nuclei, where Z and A are the proton and total nucleon numbers. For example
	- 100: neutron
	- 101: proton
	- ...
	- 1407: Nitrogen 14 (Z=7, A=14)

Note that this convention caps the charge at Z=99: an id with Z=100 is indistinguishable from one more nucleon and no charge.

## Nuclide table:

The species the models know about come from the NUBASE2020 evaluation in `data/nubase2020.txt` ([AMDC](https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt)), read at import time by `config.py` and cut down by the selection set there:

```python
nuclide_tau_min = 2.     # keep nuclides with mean lifetime above this, in seconds
max_A = None             # no cap on mass number
nuclide_zmax = None      # no cap on charge, beyond the Z <= 99 the ids impose
crpropa_isotopes = None  # path to a CRPropa isotopes-*.txt to intersect with
```

The lifetime threshold sets the coverage. The default of 2 s matches CRPropa's photodisintegration tables and leaves the light fragment basis at the physical `[n, p, d, t, He3, He4]`; a lower threshold fills in short lived nuclides near the drip lines, at the cost of daughters other codes may not know. To inspect a selection without loading the model:

```bash
python3 utils/nubase.py --tau-min 2
```

`config.nuclide_table_provenance` records the selection that is in force, for copying into the headers of generated cross section tables.

Setting `nuclide_source = 'legacy'`, or `ASTROPHOMES_NUCLIDES=legacy` in the environment, restores the 479 nuclide table the published results were produced with.

**Going above A=56 is not an extrapolation of the empirical relations.** They are reproduced from [Terranova and Tavares (1994)](https://doi.org/10.1088/0031-8949/49/3/004), whose subject is photoabsorption *for nuclei throughout the periodic table*, and the restriction the model's paper states on them is on photon energy (0.2–1 GeV) rather than on mass; the universal function is fitted to data spanning A=7–208. The old A=56 ceiling came from the nuclide table the code shipped with, not from the physics.

What is genuinely absent above iron is **fission**, which the paper excludes explicitly and notes "needs to be included for much higher masses". Beyond that, the published model was validated against data and against Fluka for A≤56 only, and `cs_gSp` carries its own documented ceiling of A=90. `config.nuclide_table_provenance` records all of this in full.

Spallation has a lower bound as well. Eq. (A.8) of the paper defines it as losing more than one proton and more than one neutron while keeping a residual of at least half the mass, so it cannot occur below A=8 and `cs_gSp` returns zero there. `config.cs_gSp_min_A` holds that bound.

## Difference from the published tables:

`config.ordered_combinations` selects how the light fragments of a spalled group are counted. The default, `False`, treats a combination as the multiset of species it contains, per Eq. (A.13). The published enumeration instead distinguished the order of parts within a mass class, which counted `(t, He3)` and `(He3, t)` twice; set `ordered_combinations = True` to reproduce it.

The two agree on `<dA>`, on the mass budget and on every heavy residual channel — the normalisation of Eq. (A.19) pins the nucleon count. They differ only in light fragment composition, by up to 36%.

## Dependencies:

It has been tested with the following versions (it should also work with newer versions):
- python 3.8+
- numpy 1.23+
- scipy 1.10+

Additionally this code uses interaction tables generated with [SOPHIA](https://www.uibk.ac.at/projects/he-cosmic-sources/tools/sophia/index.html.en) and [TALYS](http://www.talys.eu). These tables can be updated or substituted by other tables containing the analogous information, provided they are in the same format.

## How to cite the code:

Research works using this code should cite the following references:
 - [L. Morejon *et al.* 2019](https://arxiv.org/abs/1904.07999)
 - [L. Morejon "AstroPhoMes: Photomeson models ..." 2019 (DOI:10.5281/zenodo.2600177)](https://doi.org/10.5281/zenodo.2600177)
 - [J. Heinze, A. Fedynitch *et al.* 2019](https://arxiv.org/abs/1901.03338)

### Aknowledgements:
Portions of this code are based on the code PriNCe from Jonas Heinze and Anatoli Fedynitch [link](https://arxiv.org/abs/1901.03338)

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement no. 646623.
