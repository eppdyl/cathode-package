# "cathode" Python package
# Version: 1.0
# A package of various cathode models that have been published throughout the
# years. Associated publication:
# Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
# review of hollow cathode modeling: 0-D models," Journal of Propulsion and
# Power, in preparation.
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

"""
This submodule contains functions related to the computation of collision
frequencies
"""

import cathode.physics as cp

def nu_ei(ne, TeV):
    """
    Returns the electron-ion collision frequency when both species are
    near-Maxwellian. See, e.g. NRL plasma formulary, section "Collisions and
    transport."
    Inputs:
        - Electron number density (1/m^3)
        - Electron temperature (eV)
    Output:
        - Collision frequency (s)
    """
    return 2.9e-12*ne*cp.coulomb_log(ne, TeV)/TeV**(3/2)

def nu_en_mk(ng, TeV):
    """
    Function: nu_en
    Mandell and Katz' expression for the electron-neutral collision frequency
    Reference:
    - Mandell, M. J. and Katz, I., "Theory of Hollow Cathode Operation in Spot
    and Plume Modes," 30th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &
    Exhibit, 1994.
    Inputs:
        - ng: neutral density (1/m3)
        - Te: electron temperature (eV)
    Ouputs:
        - Electron-neutral collision frequency (s)
    """
    # Thermal velocity of electrons
    vte = cp.thermal_velocity(TeV)
    return 5e-19*ng*vte
