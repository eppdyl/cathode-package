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

###############################################################################
#                     Reaction Rate Integrals
###############################################################################

@np.vectorize
def reaction_rate(xsec_spline,TeV,Emin=None,Emax=None,output_xsec=False):
    """
    Returns the reaction rate for the process with cross section xsec_spline
    for Maxwellian electrons with temperature TeV.  Emin and Emax specify the 
    minimum and maximum energy for integration (Emin should be the threshold
    energy for the process, Emax should be the useful limit of the spline)
    Inputs:
        Cross section spline, created by create_cross_section_spline() (CrossSection object)
        Electron temperature, eV
        Minimum energy, eV (threshold for process of interest)
        Maximum energy, eV (limit of data or maximum extrapolation for spline)
    Optional Input:
        output_xsec can be set to true to return a tuple of (reaction rate, cross section)
            Defaults to FALSE, -> outputs only reaction rate unless specified
    Outputs:
        if output_xsec is true, (reaction rate, cross section) (m^3/s, m^2)
        otherwise, (reaction rate) (m^3/s)
    """
    
    #check whether integration bounds have been specified, if not, 
    #use minimum and maximum bounds for spline data (flux integral should always be
    #evaluated from 0 eV)
    if not Emin:
        Emin = np.min(xsec_spline.Emins)
    if not Emax:
        Emax = np.max(xsec_spline.Emaxs)
    
    #normalization factor for reaction rate integral
    normalization = (8.0*np.pi*cc.e**2.0/np.sqrt(cc.me))/(
            (2.0*np.pi*cc.e*TeV)**(3.0/2.0))
    
    scaling = np.max(xsec_spline.scalings)
    
    #define integrand lambda functions
    energy_integrand = lambda E: E*xsec_spline(E)*np.exp(-E/TeV)*scaling
    flux_integrand = lambda E: E*np.exp(-E/TeV)
    
    #integrate (note that these return the error estimate as the second output)
    energy_integral = quad(energy_integrand,Emin,Emax,epsabs=1.0E-30,limit=5000)
    flux_integral = quad(flux_integrand,0.0,Emax,epsabs=1.0E-30,limit=5000)
    
    #reaction rate
    K = normalization*energy_integral[0]/scaling
    
    #Maxwellian-averaged cross section
    xsec_avg = energy_integral[0]/flux_integral[0]/scaling
    
    #if specified, output both the cross section and reaction rate
    if output_xsec:
        return K,xsec_avg
    else:
        return K    


@np.vectorize
def beam_reaction_rate(xsec_spline,Ebeam):
    """
    Returns the reaction rate for monoenergetic beam 
    electrons and cross section described by xsec_spline.
    Inputs:
        Cross section spline created by create_cross_section_spline()
        Beam energy in eV
    Output:
        Reaction rate (m^3/s)
    """
    
    #beam electron velocity
    v = np.sqrt(2.0*cc.e*Ebeam/cc.me)
    
    return v*xsec_spline(Ebeam)


def finite_temperature_beam_reaction_rate(xsec_spline,Ebeam,Tbeam):
    return NotImplemented

def domonkos_beam_reaction_rate():
    return NotImplemented

def mean_free_path(xsec_spline,TeV,target_species_density):
    '''
    Returns the mean free path for the given collision type assuming a
    maxwellian distribution of test particles with temperature TeV and
    collision cross section as a function of energy given by xsec_spline.
    '''
    rate,xsec_avg = reaction_rate(xsec_spline,TeV,output_xsec=True)

    return 1/(target_species_density*xsec_avg)
