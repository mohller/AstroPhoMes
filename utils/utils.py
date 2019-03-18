""" Utility functions not related to physics
"""

import numpy as np
sys.path.append('../')
from config import *

def info(min_dbg_level, *message):
    """Print to console if `min_debug_level <= config["debug_level"]`

    The fuction determines automatically the name of caller and appends
    the message to it. Message can be a tuple of strings or objects
    which can be converted to string using `str()`.

    Args:
        min_dbg_level (int): Minimum debug level in config for printing
        message (tuple): Any argument or list of arguments that casts to str
    """

    if min_dbg_level <= debug_level:
        message = [str(m) for m in message]
        print caller_name() + " ".join(message)

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
