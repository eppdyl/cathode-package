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
Created on Tue Jun 6 17:21 2017
cathode.models.goebel_katz
Defines the model equations and solution procedure for Goebel and Katz's 0D
hollow cathode model described in 
Fundamentals of Electric Propulsion: Hall and Ion Thrusters (2008)

"""
import cathode.constants as cc
import cathode.physics as cp
import cathode.models.flow as flow
import numpy as np
from scipy.optimize import fsolve #,root
from scipy.special import jn 

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
    
    electron_temperature = fsolve(goal,x0=5.0)
    
    return electron_temperature

@np.vectorize
def electron_ion_collision_frequency(ne,TeV):
    """
    Returns the plasma electron-ion collision frequency as expressed in the 
    NRL Plasma Formulary.
    Inputs:
        electron number density, #/m^3
        electron temperature, eV
    Output:
        electron-ion collision frequency, 1/s
    """
    return 2.9E-12 * ne * TeV**(-3/2) * cp.coulomb_log(ne,TeV,'ei')

@np.vectorize
def electron_neutral_collision_frequency(neutral_density,TeV,species='Xe'):
    """
    Returns the electron-neutral collision frequency using the fits given 
    in Goebel's textbook.
    Inputs:
        neutral number density, #/m^3
        electron temperature, eV
    Optional Input:
        species string, defaults to Xenon (only works for Xenon right now)
    Output:
        electron-neutral collision frequency, 1/s
    """
    return cp.goebel_electron_neutral_xsec(TeV)*neutral_density*cp.mean_velocity(TeV,'e')

@np.vectorize
def resistivity(ne,neutral_density,TeV):
    """
    Returns the plasma resistivity, \eta, as a function of plasma density,
    neutral density, and electron temperature.
    Inputs:
        plasma density, #/m^3
        neutral gas density, #/m^3
        electron temperature, eV
    Output:
        plasma resistivity, Ohm-m
    """
    return (electron_ion_collision_frequency(ne,TeV)+
            electron_neutral_collision_frequency(neutral_density,TeV))/(
            cc.epsilon0*cp.plasma_frequency(ne,'e')**2)
    
    
def thermionic_current_density(Tw,phi_wf,D=cc.A0):
    """
    Returns the thermionic electron emission current density as defined by the
    Richardson-Dushman equation with the option of incorporating an experimentally
    determined temperature coefficient through the use of D.
    Inputs:
        wall temperature, K
        work function, V
    Optional Input:
        D, experimental coefficient A/(m-K)^2 defaults to the universal constant A0
    Output:
        current density in A/m^2
    """
    return D*Tw**2*np.exp(-cc.e*phi_wf/(cc.kB*Tw))

def ion_current_density(ne,TeV,species='Xe'):
    return cc.e*ne*cp.bohm_velocity(TeV,species)

def plasma_resistance(length,diameter,ne,neutral_density,TeV):
    return length*resistivity(ne,neutral_density,TeV)/(cc.pi*(diameter/2.0)**2)

def random_electron_current_density(ne,TeV,phi_s):
    return cc.e*ne*cp.mean_velocity(TeV,'e')*np.exp(-phi_s/TeV)/4.0

def heat_loss(Tw=None,method='fixed',curve=None):
    if (method == 'fixed'):
        return 13
    if (method == 'spline'):
        return curve(Tw)
    
def insert_plasma_power_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density): #may want to add species later
    A_emit = cc.pi*length*diameter
    
    lhs = (thermionic_current_density(Tw,phi_wf,D)*A_emit*phi_s + 
           plasma_resistance(length,diameter,ne,neutral_density,TeV)*Id**2)
    
    rhs = (ion_current_density(ne,TeV)*A_emit*E_iz + 2.5*TeV*Id + 
           (2.0*TeV + phi_s)*random_electron_current_density(ne,TeV,phi_s)*A_emit)

    return (lhs - rhs)

def current_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density):
    A_emit = cc.pi*length*diameter
    
    lhs = Id
    
    rhs = (thermionic_current_density(Tw,phi_wf,D)*A_emit + 
           ion_current_density(ne,TeV)*A_emit - 
           random_electron_current_density(ne,TeV,phi_s)*A_emit)
    
    return (lhs - rhs)

def emitter_power_balance(ne,Tw,phi_s,Id,TeV,D,phi_wf,E_iz,length,diameter,neutral_density):
    A_emit = cc.pi*length*diameter
    
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

def sheath_voltage(Id,TeV,phi_wf,length,diameter,ne,neutral_density,h_loss=heat_loss()):
    return ((h_loss/Id) + 2.5*TeV + phi_wf - Id*plasma_resistance(length,diameter,ne,neutral_density,TeV))

def average_plasma_density_model(Id,TeV,phi_wf,length,diameter,ne,neutral_density,plasma_potential,E_iz,h_loss=heat_loss()):
    
    phi_s = sheath_voltage(Id,TeV,phi_wf,length,diameter,ne,neutral_density,h_loss)
    Rp = plasma_resistance(length,diameter,ne,neutral_density,TeV)
    f_n = np.exp(-(plasma_potential-phi_s)/TeV) #edge-to-average ratio as defined by Goebel
    #f_n = neutral_density*cp.goebel_ionization_xsec(TeV)*cp.mean_velocity(TeV,'e')*diameter/(4*np.sqrt(cc.e*TeV/cc.M.species('Xe')))
    
    A_emit = length*cc.pi*diameter
    V_emit = length*cc.pi*diameter**2/4.0
    
    ne_bar = (Rp*Id**2 - 2.5*TeV*Id + phi_s*Id)/(
            cc.e*f_n*TeV*np.sqrt(cc.e*TeV/(2*cc.pi*cc.me))*A_emit*np.exp(-phi_s/TeV)+
            cc.e*neutral_density*V_emit*(E_iz+phi_s)*cp.goebel_ionization_xsec(TeV)*
            cp.mean_velocity(TeV,'e'))
    
    return ne_bar,phi_s

def orifice_plasma_density_model(Id,TeV,TeV_insert,length,diameter,ne,neutral_density,E_iz):
    
    Rp = plasma_resistance(length,diameter,ne,neutral_density,TeV)
    #print(Rp)
    V_ori = cc.pi*diameter**2*length/4.0
    #print(Id**2*Rp)
    #print(2.5*(TeV-TeV_insert)*Id)
    
    return ((Id**2*Rp - 2.5*Id*(TeV-TeV_insert))/(cc.e*neutral_density*
           cp.goebel_ionization_xsec(TeV)*cp.mean_velocity(TeV,'e')*E_iz*V_ori))
    

def approx_solve(Id,orifice_length,orifice_diameter,
                 insert_length,insert_diameter,flow_rate_sccm,Tgas,P_outlet,
                 E_iz,phi_wf,plasma_potential,h_loss=heat_loss(),
                 solver_tol = 1E-8,solver_out = False,
                 verbose=False):
    
    if verbose:
        print('-------------------INSERT-----------------------')
    #use the orifice dimensions and the flow rate to get P_ins
    P_insert_downstream = flow.poiseuille_flow(orifice_length,orifice_diameter,flow_rate_sccm,Tgas,P_outlet)
    P_insert_upstream = flow.poiseuille_flow(insert_length,insert_diameter,flow_rate_sccm,Tgas,P_insert_downstream)
    
    P_insert = (P_insert_downstream+P_insert_upstream)/2.0
    
    if verbose:
        print('Pressure:\t\t\t{:.3f} Torr (upstream)\n\t\t\t\t{:.3f} Torr (downstream)\n\t\t\t\t{:.3f} Torr (average)'.format(
            P_insert_upstream,P_insert_downstream,P_insert))
    
    neutral_density = P_insert*cc.Torr2eVm3/(Tgas*cc.Kelvin2eV)
    
    TeV = ambipolar_diffusion_model(insert_diameter/2.0,neutral_density,Tgas*cc.Kelvin2eV)[0]
    if verbose:
        print('Electron Temperature:\t\t{:.3f} eV'.format(TeV))
    
    #first guess value for ne
    #sheath voltage at initial step
    ne_bar = 1.5E21
    phi_s = sheath_voltage(Id,TeV,phi_wf,insert_length,insert_diameter,ne_bar,neutral_density,h_loss)
    delta = 1.0
    
    while(delta>=solver_tol):
        phi_s_old = phi_s
        ne_old = ne_bar
        ne_bar,phi_s = average_plasma_density_model(Id,TeV,phi_wf,insert_length,insert_diameter,
                                                    ne_old,neutral_density,plasma_potential,E_iz,h_loss)
        delta = np.max(np.abs([1-phi_s/phi_s_old,1-ne_bar/ne_old]))
        if solver_out:
            print(delta,ne_bar,phi_s)
        
    avg_to_peak = 2*jn(1,cc.BesselJ01)/cc.BesselJ01
    
    if verbose:
        print('Plasma Density:\t\t\t{:.3E} /m^3'.format(ne_bar))
        print('Peak Density:\t\t\t{:.3E} /m^3'.format(ne_bar/avg_to_peak))
        print('Sheath Voltage:\t\t\t{:.3f} V'.format(phi_s))
    
    P_orifice_downstream = P_outlet
    P_orifice_upstream = P_insert_downstream
    
    P_orifice = (P_orifice_upstream+P_orifice_downstream)/2.0
    
    orifice_neutral_density = P_orifice*cc.Torr2eVm3/(Tgas*cc.Kelvin2eV)
    
    if verbose:
        print('-------------------ORIFICE----------------------')
        print('Pressure:\t\t\t{:.3f} Torr (upstream)\n\t\t\t\t{:.3f} Torr (downstream)\n\t\t\t\t{:.3f} Torr (average)'.format(
                P_orifice_upstream,P_orifice_downstream,P_orifice))
    
    TeV_orifice = ambipolar_diffusion_model(orifice_diameter/2.0,orifice_neutral_density,
                                            Tgas*cc.Kelvin2eV)[0]
    
    if verbose:
        print('Electron Temperature:\t\t{:.3f} eV'.format(TeV_orifice))
    
    solve_fun = lambda n: n - orifice_plasma_density_model(Id,TeV_orifice,TeV,orifice_length,
                                                      orifice_diameter,n,
                                                      orifice_neutral_density,E_iz)
    ne_bar_orifice = fsolve(solve_fun,1E18)[0]
    
    if verbose:
        print('Plasma Density:\t\t\t{:.3E} /m^3'.format(ne_bar_orifice))
        print('Peak Density:\t\t\t{:.3E} /m^3'.format(ne_bar_orifice/avg_to_peak))
    

    return P_insert,TeV,ne_bar,phi_s,P_orifice,TeV_orifice,ne_bar_orifice









