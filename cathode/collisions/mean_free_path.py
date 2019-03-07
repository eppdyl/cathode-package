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
This submodule contains functions related to the computation of mean free paths.
"""

import cathode.collisions.reaction_rate as rr

def lambda_pr_hg(ne, ng, phi_p):
    """
     Computes the mean free path for the primary electrons.
     Applicable to mercury only! Based on Siegfried and Wilbur's computer
     model.

     TODO: Add citation for computer model
     Inputs:
     - ne: Electron density (1/m3)
     - ng: Gas density (1/m3)
     - phi_p: Plasma potential (V)

     References:
    - D. E.  Siegfried and P. J. Wilbur, "Phenomenological Model Describing
    Orificed, Hollow Cathode Operation," AIAA Journal, Vol. 21, No. 1, 1983,
    pp. 5–6.
    - P. J. Wilbur, "Advanced Ion Thruster Research," Tech. Rep. CR-168340,
    NASA, 1984.
    - D. E.  Siegfried and P. J. Wilbur, "Studies on an experimental quartz
    tube hollow cathode," 14th International Electric Propulsion Conference,
    1979
    - D. E.  Siegfried, "A Phenomenological Model for Orificed Hollow
    Cathodes," Ph.D., Colorado State University, 1982.
    """
    inv = 6.5e-17*ne/phi_p**2 + 1e3*ng*phi_p / (2.83e23 - 1.5*ng)

    return 1/inv

def mean_free_path(xsec_spline, TeV, target_species_density):
    '''
    Returns the mean free path for the given collision type assuming a
    maxwellian distribution of test particles with temperature TeV and
    collision cross section as a function of energy given by xsec_spline.
    '''
    _, xsec_avg = rr.reaction_rate(xsec_spline, TeV, output_xsec=True)

    return 1/(target_species_density*xsec_avg)
