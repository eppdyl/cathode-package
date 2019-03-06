# -*- coding: utf-8 -*-
"""
Created on Thu Jun 1 00:58 2017
Submodule of the cathode package containing common cathode parameter 
calculations and atomic physics methods.

@author: cjw
"""

import numpy as np
import re
from scipy.interpolate import splrep,splev
from scipy.integrate import quad
import cathode.constants as cc

###############################################################################
#                         Basic Plasma Physics
###############################################################################

def debye_length(ne,TeV,species='e'):
    """
    Returns the debye length, sqrt(eps0*k*T/(e^2 ne)) for the species specified.
    Inputs:
        Number density (1/m^3)
        Temperature (eV)
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        plasma frequency (rad/s)
    """
    return np.sqrt((cc.epsilon0*cc.e*TeV)/(cc.e**2*ne))

def plasma_frequency(ne,species='e'):
    """
    Returns the plasma frequency, sqrt(e^2 ne/(eps0* m)) for the species specified.
    Inputs:
        number density (1/m^3)
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        plasma frequency (rad/s)
    """
    return np.sqrt((cc.e**2)*ne/(cc.epsilon0*cc.M.species(species)))

def thermal_velocity(TeV,species='e'):
    """
    Returns the thermal velocity, sqrt(kB T/m) for the species specified.
    Inputs:
        Temperature in eV
        
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        thermal velocity in m/s
    """
    return np.sqrt((cc.e*TeV)/(cc.M.species(species)))

def mean_velocity(TeV,species='e'):
    """
    Returns the distribution-averaged species velocity assuming a Maxwellian 
    distribution at a temperature of TeV (in electron volts).
    Inputs:
        Temperature in eV
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        mean velocity in m/s
    """
    return np.sqrt((8.0*cc.e*TeV)/(np.pi*cc.M.species(species)))

def bohm_velocity(TeV,species='Xe'):
    """
    Returns the Bohm velocity (ion acoustic velocity) for a given electron
    temperature and ion species.
    Input:
        electron temperature in eV
    Optional Input:
        species string, defaults to Xe
    Output:
        Bohm velocity, m/s
    """
    return np.sqrt(cc.e*TeV/cc.M.species(species))

@np.vectorize
def coulomb_log(ne,TeV,collision_type='ei'):
    """
    Returns the Coulomb logarithm value as a function of plasma density and 
    electron temperature.  Collision type can be either electron-ion (default)
    or electron-electron.
    Inputs:
        plasma number density (1/m^3)
        plasma electron temperature (eV)
    Optional Inputs:
        collision type string ('ei' or 'ee')
    Output:
        Value of Coulomb Logarithm
        
    Ref: NRL Plasma Formulary (values translated from CGS to SI)
    """
    if collision_type == 'ei':
        return (23.0 - 0.5*np.log(1E-6*ne/(TeV**3.0))) 
    elif collision_type == 'ee':
        return (23.5 - 0.5*np.log(1E-6*ne/(TeV**(5.0/2.0)))-np.sqrt(1E-5 + (np.log(TeV)-2.0)**2/16.0))

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

