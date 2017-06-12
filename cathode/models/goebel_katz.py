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
from scipy.optimize import fsolve #,root

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
    lhs = lambda TeV: ((cathode_radius/cc.BesselJ01)**2*(neutral_density*
                       cp.goebel_ionization_xsec(TeV)*
                       cp.mean_velocity(TeV,'e')))
    
    rhs = lambda TeV: ((cc.e/cc.M.species(species))*(TiV+TeV)/
                       (neutral_density*
                        cp.charge_exchange_xsec(TiV,species)*
                        np.sqrt(cc.e*TiV/cc.M.species(species))))
    
    goal = lambda x: lhs(x) - rhs(x)
    
    electron_temperature = fsolve(goal,x0=2.0)
    
    return electron_temperature

@np.vectorize
def electron_ion_collision_frequency(ne,TeV):
    return 2.9E-12 * ne * TeV**(-3/2) * cp.coulomb_log(ne,TeV,'ei')

@np.vectorize
def electron_neutral_collision_frequency(neutral_density,TeV):
    return cp.goebel_electron_neutral_xsec(TeV)*neutral_density*cp.mean_velocity(TeV,'e')

@np.vectorize
def resistivity(ne,neutral_density,TeV):
    return (electron_ion_collision_frequency(ne,TeV)+
            electron_neutral_collision_frequency(neutral_density,TeV))/(
            cc.epsilon0*cp.plasma_frequency(ne,'e')**2)
    
    
def thermionic_current_density(Tw,phi_wf,D=cc.A0):
    return D*Tw**2*np.exp(-cc.e*phi_wf/(cc.kB*Tw))

def ion_current_density(ne,TeV,species='Xe'):
    return cc.e*ne*cp.bohm_velocity(TeV,species)

def plasma_resistance(length,diameter,ne,neutral_density,TeV):
    return length*resistivity(ne,neutral_density,TeV)/(cc.pi*(diameter/2)**2)

def random_electron_current_density(ne,TeV,phi_s):
    return cc.e*ne*cp.mean_velocity(TeV,'e')*np.exp(-phi_s/TeV)/4.0

def heat_loss(Tw,method='fixed'):
    if (method == 'fixed'):
        return 13
    
def insert_plasma_power_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density): #may want to add species later
    A_emit = cc.pi*length*diameter**2/4.0
    
    lhs = (thermionic_current_density(Tw,phi_wf,D)*A_emit*phi_s + 
           plasma_resistance(length,diameter,ne,neutral_density,TeV)*Id**2)
    
    rhs = (ion_current_density(ne,TeV)*A_emit*E_iz + 2.5*TeV*Id + 
           (2.0*TeV + phi_s)*random_electron_current_density(ne,TeV,phi_s)*A_emit)

    return (lhs - rhs)

def current_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density):
    A_emit = cc.pi*length*diameter**2/4.0
    
    lhs = Id
    
    rhs = (thermionic_current_density(Tw,phi_wf,D)*A_emit + 
           ion_current_density(ne,TeV)*A_emit - 
           random_electron_current_density(ne,TeV,phi_s)*A_emit)
    
    return (lhs - rhs)

def emitter_power_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density):
    A_emit = cc.pi*length*diameter**2/4.0
    
    lhs = heat_loss(Tw) + thermionic_current_density(Tw,phi_wf,D)*phi_wf*A_emit
    
    rhs = (ion_current_density(ne,TeV)*(E_iz + phi_s + 0.5*TeV - phi_wf)*A_emit +
           (2.0*TeV + phi_wf)*random_electron_current_density(ne,TeV,phi_s)*A_emit)
    
    return (lhs-rhs)
    
def zerofun(x,args):
    ne=x[0]*1E21
    Tw=x[1]*1E3
    phi_s=x[2]
    
    #unpack arguments
    Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density = args
    
    goal=np.zeros(3)
    goal[0]=insert_plasma_power_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density)/10.0
    goal[1]=current_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density)/10.0
    goal[2]=emitter_power_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density)/10.0
    
    print('GOAL FUN:',goal)
    print('ARGS:',x)
    
    return goal

def approx_solve():
    
    #first guess value for ne
    ne_0 = 1.5E21
    
    #resisitivity at initial step
    Rp = plasma_resistance(length,diameter,ne_0,neutral_density,TeV)

