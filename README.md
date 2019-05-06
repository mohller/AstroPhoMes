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

## Dependencies:

It has been tested with the following versions (it should also work with newer versions):
- python 2.7.15
- numpy 1.15.1
- scipy 1.1.0

Additionally this code uses interaction tables generated with [SOPHIA](https://www.uibk.ac.at/projects/he-cosmic-sources/tools/sophia/index.html.en) and [TALYS](http://www.talys.eu). These tables can be updated or substituted by other tables containing the analogous information, provided they are in the same format.

## How to cite the code:

Research works using this code should cite the following references:
 - [L. Morejon *et al.* 2019](https://arxiv.org/abs/1904.07999)
 - [L. Morejon "AstroPhoMes: Photomeson models ..." 2019 (DOI:10.5281/zenodo.2600177)](https://doi.org/10.5281/zenodo.2600177)
 - [J. Heinze, A. Fedynitch *et al.* 2019](https://arxiv.org/abs/1901.03338)

### Aknowledgements:
Portions of this code are based on the code PriNCe from Jonas Heinze and Anatoli Fedynitch [link](https://arxiv.org/abs/1901.03338)

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement no. 646623.
