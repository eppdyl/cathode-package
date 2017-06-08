"""
Created on Tue Jun 6 17:21 2017
cathode.models.goebel_katz
Defines the model equations and solution procedure for Goebel and Katz's 0D
hollow cathode model described in Fundamentals of Electric Propulsion: Hall and Ion Thrusters (2008)

"""
import cathode.constants as cc
import cathode.physics as cp
import numpy as np
from scipy.optimize import fsolve

def ambipolar_diffusion_model(cathode_radius,neutral_density,TiV,species='Xe'):
    lhs = lambda TeV: ((cathode_radius/cc.BesselJ01)*(neutral_density*
                       cp.goebel_ionization_xsec(TeV)*
                       np.sqrt(8*cc.e*TeV/(np.pi*cc.me))))
    
    rhs = lambda TeV: ((cc.e/cc.M.species(species))*(TiV+TeV)/
                       (neutral_density*cp.charge_exchange_xsec(TiV,species)*
                        np.sqrt(cc.e*TiV/cc.M.species(species))))
    
    goal = lambda x: lhs(x) - rhs(x)
    
    electron_temperature = fsolve(goal,x0=1.4)
    
    return electron_temperature
