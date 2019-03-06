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
import cathode.collisions.cross_section as xsec

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

def nu_en_xe_mk(ng, TeV, xsec_type='variable'):
    """
    Mandell and Katz' expression for the electron-neutral collision frequency
    of xenon. Can either use the simple collision cross section (constant) or
    the variable one.

    Inputs:
    - ng: neutral density (1/m3)
    - TeV: electron temperature (eV)
    - xsec_type: The type of cross section model to use. Can either be
      "constant" or "variable". If "variable" uses the model from year 1999 and
      above. Otherwise uses a set value of 5 10^{-19} m2.

    Ouputs:
    - Electron-neutral collision frequency (s)

    References:
    - Mandell, M. J. and Katz, I., "Theory of Hollow Cathode Operation in Spot
    and Plume Modes," 30th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &
    Exhibit, 1994.
    - Katz, I., et al, "Sensitivity of Hollow Cathode Performance to Design and
      Operating Parameters," 35th AIAA/ASME/SAE/ASEE Joint Propulsion Conference
      & Exhibit, 1999. http://arc.aiaa.org/doi/pdf/10.2514/6.1999-2576
    - Goebel, D. M. and Katz, I., "Fundamentals of Electric Propulsion,"
      Appendix D p.475, John Wiley and Sons, 2008.
    """
    # Thermal velocity of electrons
    vte = cp.thermal_velocity(TeV)
    # Cross section
    en_xsec = xsec.electron_neutral_xe_mk(TeV, xsec_type)
    # Collision freq.
    nu_en = ng * en_xsec * vte

    return nu_en
