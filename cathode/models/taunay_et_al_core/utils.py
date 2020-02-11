# "cathode" Python package
# Version: 1.0
# A package of various cathode models that have been published throughout the
# years. Associated publications:
# Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
# review of hollow cathode modeling: 0-D models," Journal of Propulsion and
# Power, in preparation.
# Taunay, P.-Y. C. R., Wordingham, C. J., and Choueiri, E.Y., "Physics of
# thermionic, orificed hollow cathodes," Plasma Sources Science and Technology,
# in preparation.
#
# Copyright (C) 2019 Christopher J. Wordingham and Pierre-Yves C. R. Taunay
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https:/www.gnu.org/licenses/>.
#
# Contact info: https:/github.com/pytaunay
#

import os

def find_number_processes(list_case):
    """Find the number of parallel processes to spawn based on a list size.

    If the length of the list is less than the total number of CPUs we have,
    then just spawn as many elements as there are in the list. Otherwise,
    set the number of processes to None because multiprocessing.Pool then 
    uses all available CPUs
    """
    procnum = None
    if len(list_case) < os.cpu_count():
        procnum = len(list_case) 

    return procnum


def check_answer(ng_i,ret_all):
    """Check that the proposed solution is valid. 
    Checks that ionization fractions are within bounds, that orifice neutral 
    density is less than insert neutral density, and that orifice electron
    temperature is larger than that of the insert.
    """
        ng_o = ret_all['ng_o']
        alpha_o = ret_all['alpha_o']
        Te_o = ret_all['Te_o']
        
        alpha_i = ret_all['alpha_i']
        Te_i = ret_all['Te_i']

        b_ai = alpha_i < 0 or alpha_i > 1
        b_ao = alpha_o < 0 or alpha_o > 1
        b_ng = ng_i < ng_o
        b_Te = Te_i > Te_o    
        
        if(b_ai or b_ao or b_ng or b_Te):
            return False
        else:
            return True
