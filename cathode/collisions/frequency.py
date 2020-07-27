###############################################################################
#
# "cathode" Python package
# Version: 1.0.0
# A package of various cathode models that have been published throughout the
# years. Associated publications:
# - Taunay, P.-Y. C. R., Wordingham, C. J., and Choueiri, E. Y., "A 0-D model 
# for orificed hollow cathodes with application to the scaling of total 
# pressure," AIAA Propulsion and Energy Forum, 2019, AIAA-2019-4246
# - Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
# review of hollow cathode modeling: 0-D models," 53rd AIAA/ASME/SAE/ASEE Joint 
# Propulsion Conference, 2017, AIAA-2017-4888 
# 
###############################################################################
# Copyright (C) 2017-2020 Chris Wordingham, Pierre-Yves Taunay
#  
# This file is part of the Python "cathode" package.
#
# The Python "cathode" package is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License (LGPL) as 
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The Python "cathode" package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License (LGPL) for more details.
#
# You should have received a copy of the GNU Lesser General Public License 
# (LGPL) along with the Python "cathode" package.  
# If not, see <https://www.gnu.org/licenses/>.
###############################################################################
"""
This submodule contains functions related to the computation of collision
frequencies
"""

import cathode.physics as cp
import cathode.collisions.cross_section as xsec

def nu_ei(ne, TeV):
    """
    Returns the electron-ion collision frequency when both species are
    near-Maxwellian. See, e.g., NRL plasma formulary, section "Collisions and
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
    # Depending on the type of cross-section the velocity is different
    # Mandell and Katz with a constant cross section use the thermal velocity
    # of electrons. Later models use the Maxwellian-averaged cross section and
    # therefore the mean Maxwellian velocity.
    if xsec_type == 'constant':
        # Thermal velocity of electrons
        ve = cp.thermal_velocity(TeV)
    elif xsec_type == 'variable':
        ve = cp.mean_velocity(TeV, species='e')
    else:
        raise ValueError

    # Cross section
    en_xsec = xsec.electron_neutral_xe_mk(TeV, xsec_type)
    # Collision freq.
    nu_en = ng * en_xsec * ve

    return nu_en
