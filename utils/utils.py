""" Utility functions not related to physics
"""

import numpy as np

def get_AZN(nco_id):
    """Returns mass number :math:`A`, charge :math:`Z` and neutron
    number :math:`N` of ``nco_id``.

    Args:
        nco_id (int): corsika id of nucleus/mass group
    Returns:
        (int,int,int): (Z,A) tuple
    """
    Z, A = 1, 1

    if nco_id >= 100:
        Z = nco_id % 100
        A = (nco_id - Z) / 100
    else:
        Z, A = 0, 0

    return A, Z, A - Z


def bin_widths(bin_edges):
    """Computes and returns bin widths from given edges."""
    edg = np.array(bin_edges)

    return np.abs(edg[1:, ...] - edg[:-1, ...])
