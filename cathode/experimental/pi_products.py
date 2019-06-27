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

import cathode.constants as cc
from cathode.models.flow import reynolds_number

import numpy as np
import pandas as pd

C_corr = 65599289061.02858
exp_vec = np.array([ 1.2228393974651621 ,
        0.3575019476462662, 
        -0.6643001188405657, 
        1.412319447779662, 
        0.10178328743856285,
        0.23638093259230672])

def generate_pi_products(pdf):
    '''
    Generate the pi products as defined in the empirical relationship paper.
    Inputs:
        - pdf: the pandas dataframe that contains all of the information for
        the calculations
    Returns:
        List of pi products
    '''
    # Temperature of gas approximately 3x the wall temperature
    T_SI = (pdf.Tw + 273.15)*3
    
    # Speed of sound
    gam = 5/3
    a = ( gam * cc.gas_constant * 1e3 / pdf.mass * T_SI)**(0.5) 

    # Convert data to SI
    mdot_SI = pdf.mdot * (pdf.mass * cc.atomic_mass/cc.e)
    do_SI = pdf.do * 1e-3
    dc_SI = pdf.dc * 1e-3
    Lo_SI = pdf.Lo * 1e-3
    M_SI = pdf.mass*cc.atomic_mass
    P_SI = pdf.P * cc.Torr
    P_gd =  mdot_SI * a / do_SI**2
    P_mag = cc.mu0 * pdf.Id**2 / do_SI**2

    # Calculate PI products
    # P/P_mag
    PI1 = P_SI / P_mag

    ## (do/dc)
    PI2 = pdf.do/pdf.dc
    
    ## (do/Lo)
    PI3 = pdf.do/pdf.Lo

    ## (mdot/Id)^2 * (M*do/mu0 * q^2)
    PI4 = (pdf.mdot / pdf.Id)**2
    PI4 *= M_SI * do_SI / (cc.mu0 * cc.e**2)

    ## P_gd / P_mag 
    PI5 = P_gd/P_mag

    ## P_iz / P_mag 
    PI6 = cc.e*pdf.eiz / (do_SI**2 * Lo_SI)
    PI6 /= P_mag

    # Re
    PI7 = pd.Series(index=pdf.index,dtype = np.object)
    for cat in pdf.index:
        mass = np.unique(pdf.mass[cat])[0]
        do_unique = np.unique(do_SI[cat])[0]
    
    
        Xe = (mass == cc.M.species('Xe')/cc.atomic_mass)
        Ar = (mass == cc.M.species('Ar')/cc.atomic_mass)
        Hg = (mass == cc.M.species('Hg')/cc.atomic_mass)
    
        if Xe or Ar:
            if Xe:
                species = 'Xe'
            elif Ar:
                species = 'Ar'
    
            Re = reynolds_number(mdot_SI[cat],
                                 do_unique,
                                 T_SI[cat],
                                 species)
        elif Hg:
            T_lj, mu_lj = cc.LJ.viscosity('Hg')
            Re = reynolds_number(mdot_SI[cat],
                                 do_unique,
                                 T_SI[cat],
                                 species,
                                 T_lj,
                                 mu_lj)      
        PI7[cat] = np.copy(Re)

    PIlist = [PI1,PI2,PI3,PI4,PI5,PI6,PI7]

    return PIlist

def linearize_pi_products(PIvec, pdf):
    '''
    Transforms the PI products out of a pandas data series to a single linear
    array.
    '''
    PI1,PI2,PI3,PI4,PI5,PI6,PI7 = PIvec
    
    M_SI = pdf.mass*cc.atomic_mass
    P_SI = pdf.P * cc.Torr
    # Temperature of gas approximately 3x the wall temperature
    T_SI = (pdf.Tw + 273.15)*3
    
    # Speed of sound
    gam = 5/3
    a = ( gam * cc.gas_constant * 1e3 / pdf.mass * T_SI)**(0.5) 
    
    ## Linearize the data
    PI1_lin = np.array([],dtype=np.float64)
    PI2_lin = np.array([],dtype=np.float64)
    PI3_lin = np.array([],dtype=np.float64)
    PI4_lin = np.array([],dtype=np.float64)
    PI5_lin = np.array([],dtype=np.float64)
    PI6_lin = np.array([],dtype=np.float64)
    PI7_lin = np.array([],dtype=np.float64)
    a_lin = np.array([],dtype=np.float64)
    M_lin = np.array([],dtype=np.float64)
    P_lin = np.array([],dtype=np.float64)
    
    
    size_tot = 0
    for idx,name in enumerate(pdf.index):
        PI1_lin = np.append(PI1_lin,PI1.values[idx].flatten())
        PI2_lin = np.append(PI2_lin,PI2.values[idx].flatten())
        PI3_lin = np.append(PI3_lin,PI3.values[idx].flatten())
        PI4_lin = np.append(PI4_lin,PI4.values[idx].flatten())
        PI5_lin = np.append(PI5_lin,PI5.values[idx].flatten())
        PI6_lin = np.append(PI6_lin,PI6.values[idx].flatten())
        PI7_lin = np.append(PI7_lin,PI7.values[idx].flatten())
        a_lin = np.append(a_lin,a.values[idx].flatten())
        M_lin = np.append(M_lin,M_SI.values[idx].flatten())
        P_lin = np.append(P_lin,P_SI.values[idx].flatten())
        
        size_tot = size_tot + len(PI1.values[idx])

    
    linearized = np.array([PI1_lin,
                           PI2_lin,
                           PI3_lin,
                           PI4_lin,
                           PI5_lin,
                           PI6_lin,
                           PI7_lin,
                           a_lin,
                           M_lin,
                           P_lin
                           ]).T
    
    return linearized

def compute_correlation(PIlist):
    X = C_corr
    for idx, val in enumerate(exp_vec):
        X *= PIlist[idx+1]**val
    
    return X