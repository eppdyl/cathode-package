"""
Created on Tue Jun 6 17:21 2017
cathode.models.goebel_katz
Defines the model equations and solution procedure for Goebel and Katz's 0D
hollow cathode model described in 
Fundamentals of Electric Propulsion: Hall and Ion Thrusters (2008)

"""
import cathode.constants as cc
import cathode.physics as cp
import numpy as np
from scipy.optimize import fsolve,root

@np.vectorize
def ambipolar_diffusion_model(cathode_radius,neutral_density,TiV,species='Xe'):
    """
    Implementation of Goebel's ambipolar diffusion model used for determination
    of the orifice- or insert-region plasma electron temperature based on the 
    cathode geometry, neutral density and ion temperature (usually taken to be 
    some constant value 2-4x the insert temperature).
    Inputs:
        cathode radius, m
        neutral density, #/m^3
        ion temperature, eV
    Optional Input:
        species, defaults to 'Xe'
            -NOTE: currently only works for xenon
    Output:
        electron temperature, eV
    """
    #NOTE: THIS SECTION WILL CURRENTLY ONLY WORK FOR XENON!!
    lhs = lambda TeV: ((cathode_radius/cc.besselJ01)**2*(neutral_density*
                       cp.goebel_ionization_xsec(TeV)*
                       np.sqrt(8*cc.e*TeV/(np.pi*cc.me))))
    
    rhs = lambda TeV: ((cc.e/cc.M.species(species))*(TiV+TeV)/
                       (neutral_density*cp.charge_exchange_xsec(TiV,species)*
                        np.sqrt(cc.e*TiV/cc.M.species(species))))
    
    goal = lambda x: lhs(x) - rhs(x)
    
    electron_temperature = fsolve(goal,x0=2.0)
    
    return electron_temperature
