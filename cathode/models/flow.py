# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:02:05 2017

cathode.models.flow
Collected flow model expressions used by the various cathode models
or those that would be of use to future models.

@author: cjw
"""

import cathode.constants as cc
import numpy as np

@np.vectorize
def viscosity(T,species='Xe-Goebel',units='poise'):
    """
    Calculates the gas species viscosity in poises given the temperature in 
    Kelvin and the species name.
    Inputs:
        temperature in Kelvin
    Optional Inputs:
        species - string with abbreviated identifier for each gas species
            -defaults to Goebel's Xe fit
        units - desired viscosity output unit
            -defaults to poise
    Output:
        viscosity in chosen unit (default poise)
        
    Refs:
        Goebel's 2008 Textbook for Xe-Goebel fit
        Stiel and Thodos 1961 for remaining gases
    """
    #species dictionary: [Tc,upsilon]
    species_dict = {'Xe-Goebel' :   [289.7,None],
                    'Xe'        :   [289.8,0.0151],
                    'Ar'        :   [151.2,0.0276],
                    'Kr'        :   [209.4,0.0184],
                    'Ne'        :   [44.5,0.0466],
                    'N2'        :   [126.2,0.0407]}
    
    #unpack species data
    Tc, upsilon = species_dict[species]
    Tr = T/Tc
    
    #apply the appropriate fit equation
    if species == 'Xe-Goebel':
        zeta = 2.3E-4*Tr**(0.71+0.29/Tr)
    else:
        zeta = (17.78E-5*(4.58*Tr-1.67)**(5.0/8.0))/(upsilon*100.0)
        
    #convert to chosen units
    units_dict = {'poise' : 1.0,
                  'centipoise' : 100.0,
                  'Pa-s' : 0.1,
                  'kg/(m-s)' : 0.1}
        
    
    return zeta*units_dict[units]
    

def poiseuille_flow(length,diameter,flow_rate_sccm,T,P_outlet,species='Xe-Goebel'):
    """
    Returns the upstream pressure in Torr for a flow rate of flow_rate_sccm in 
    standard cubic centimeters per minute with a gas temperature (within the cathode)
    of T (Kelvin) and an outlet pressure of P_outlet (Torr).
    Inputs:
        length of cathode/orifice channel, meters
        diameter of channel, meters
        flow rate in sccm
        gas temperature in Kelvin
        outlet pressure in Torr
    Optional Inputs:
        gas species string, default value of 'Xe-Goebel' to use Goebel's fits
    """
    #convert lengths to cm to match formula in Goebel's Appendix
    length_cm = length/cc.cm
    diameter_cm = diameter/cc.cm
    
    #calculate Tm, ratio to measurement temperature
    Tm = T/289.7
    
    return np.sqrt(P_outlet**2 + 0.78*flow_rate_sccm*viscosity(T,species,units='poise')*Tm*length_cm/diameter_cm**4)

def sonic_orifice():
    return NotImplemented

