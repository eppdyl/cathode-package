# -*- coding: utf-8 -*-
"""
Created on Thu Jun 1 00:58 2017
Submodule of the cathode package containing common cathode parameter 
calculations and atomic physics methods.

@author: cjw
"""

import numpy as np
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
