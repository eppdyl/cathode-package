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
This submodule contains functions related to the computation of reaction rates.
"""

import numpy as np

import cathode.constants as cc
from scipy.integrate import quad

###############################################################################
#                     Reaction Rate Integrals
###############################################################################
@np.vectorize
def reaction_rate(xsec_spline, TeV, Emin=None, Emax=None, output_xsec=False):
    """
    Returns the reaction rate for the process with cross section xsec_spline
    for Maxwellian electrons with temperature TeV.  Emin and Emax specify the
    minimum and maximum energy for integration (Emin should be the threshold
    energy for the process, Emax should be the useful limit of the spline)
    Inputs:
        - xsec_spline: cross section spline, created by
        create_cross_section_spline() (CrossSection object)
        - TeV: electron temperature (eV)
        - Emin: minimum energy, i.e. threshold for process of interest (eV)
        - Emax: maximum energy, i.e. limit of data or maximum extrapolation
        for spline (eV)
    Optional Input:
        - output_xsec: can be set to true to return a tuple of
        (reaction rate, cross section). Defaults to FALSE, in which case it
        outputs only reaction rate.
    Outputs:
        - output_xsec true: (reaction rate, cross section) (m^3/s, m^2)
        - output_xsec false: (reaction rate) (m^3/s)
    """
    # Check whether integration bounds have been specified, if not, use
    # minimum and maximum bounds for spline data (flux integral should always
    # be evaluated from 0 eV)
    if not Emin:
        Emin = np.min(xsec_spline.emins)
    if not Emax:
        Emax = np.max(xsec_spline.emaxs)

    # Normalization factor for reaction rate integral
    normalization = 8*np.pi*cc.e**2/np.sqrt(cc.me)
    normalization /= (2*np.pi*cc.e*TeV)**(3/2)

    scaling = np.max(xsec_spline.scalings)

    # Define integrand lambda functions
    energy_integrand = lambda E: E*xsec_spline(E)*np.exp(-E/TeV)*scaling
    flux_integrand = lambda E: E*np.exp(-E/TeV)

    # Integrate (note that these return the error estimate as the second output)
    energy_integral = quad(energy_integrand, Emin, Emax, epsabs=1.0E-30,
                           limit=5000)
    flux_integral = quad(flux_integrand, 0.0, Emax, epsabs=1.0E-30, limit=5000)

    # Reaction rate
    K = normalization*energy_integral[0]/scaling

    # Maxwellian-averaged cross section
    xsec_avg = energy_integral[0]/flux_integral[0]/scaling

    #if specified, output both the cross section and reaction rate
    if output_xsec:
        return K, xsec_avg
    else:
        return K


@np.vectorize
def beam_reaction_rate(xsec_spline, Ebeam):
    """
    Returns the reaction rate for monoenergetic beam
    electrons and cross section described by xsec_spline.
    Inputs:
        - xsec_pline: cross section spline created by
        create_cross_section_spline()
        - Ebeam: Beam energy (eV)
    Output:
        Reaction rate (m^3/s)
    """

    # Beam electron velocity
    v = np.sqrt(2.0*cc.e*Ebeam/cc.me)

    return v*xsec_spline(Ebeam)


def finite_temperature_beam_reaction_rate(xsec_spline, Ebeam, Tbeam):
    return NotImplemented

def domonkos_beam_reaction_rate():
    return NotImplemented
