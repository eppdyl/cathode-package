# A first approach at theoretical scaling relationships for the total pressure
# in hollow cathodes
# Associated publication:
# Taunay, P.-Y. C. R., Wordingham, C. J., and Choueiri, E. Y., "Scaling
# relationships for the total pressure in orificed hollow cathodes", 
# AIAA Propulsion and Energy forum, 2019.
#
# Copyright (C) 2019 Pierre-Yves C. R. Taunay and Christopher J. Wordingham 
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

import cathode.constants as cc
from cathode.math.bisect_next import bisect_next

from correlation import Te_insert, Te_orifice, TeK_orifice

import numpy as np

def orifice_density_wrapper(mdot,
                            Id,
                            M,
                            dc,do,Lo,
                            eiz,
                            log_ng_i,
                            species,
                            chold,
                            TgK,
                            verbose):
    ''' Calculates the neutral density in the orifice based on a given gas, 
    mass flow rate, discharge current, geometry, and insert density.
    The insert density is necessary because of the insert electron temperature
    that appears in the energy equation for electrons in the orifice
    Inputs:
        - mdot: mass flow rate (kg/s)
        - Id: discharge current (A)
        - M: mass of gas (kg)
        - dc, do, Lo: insert diameter, orifice diameter, orifice length, resp. (m)
        - eiz: ionization potential (eV)
        - log_ng_i: logarithm base 10 of insert neutral density (1/m3)
        - species: species of interest
        - chold: the collision cross section holder object
        - TgK: the neutral gas temperature assumed (K)
    '''
    
    ro = do/2 # Orifice radius
    ng_i = 10**log_ng_i
    
    ### Insert electron temperature
    Te_i = Te_insert(ng_i,dc,species)

    ### Sanity check: if the electron temperature for the orifice evaluated with
    ### the INSERT density is NaN then the neutral insert density is too low,
    ### and so will be the orifice density (ng_o < ng_i). Throw out that case.
    Te_o = Te_orifice(ng_i,do,TgK,species)
    
    if np.isnan(Te_o):
        ng_o = np.nan
        return np.array([mdot,Id,ng_i,ng_o])    

    ### Cross-sections
    rr_iz = lambda ng: chold.rr('iz',Te_orifice(ng,do,TgK,species))
    rr_en = lambda ng: chold.rr('en',Te_orifice(ng,do,TgK,species))
    rr_ex = lambda ng: chold.rr('ex',Te_orifice(ng,do,TgK,species))
    rr_ei = lambda ng: chold.rr('ei',Te_orifice(ng,do,TgK,species))
    eex = lambda ng: chold.ex_avg_energy(Te_orifice(ng,do,TgK,species)) # 
    
    ### ORIFICE EQUATIONS
    Rg = cc.R_specific(species) 
    gam = 5/3

    # TODO; this could be moved into a single function
    Tbar = lambda ng: TgK / TeK_orifice(ng,do,TgK,species)
    aTe = lambda ng: np.sqrt(gam * Rg * TeK_orifice(ng,do,TgK,species))
    vg = lambda ng: mdot / (np.pi * ro**2 * ng * M)
    delta = lambda ng: (vg(ng)/aTe(ng))**2

    sqrt_alpha = lambda ng: np.sqrt(4*delta(ng)*(1+Tbar(ng))+1)
    alpha_o = lambda ng: 1 + 1/(2*delta(ng)) * (1-sqrt_alpha(ng))
    
    # First term in alpha^2
    alpha2_o1 = lambda ng: cc.e * ng**2 * rr_iz(ng) * np.pi * Lo * ro**2 * eiz
    alpha2_o2 = lambda ng: cc.me * Lo / (np.pi*ro**2*cc.e**2) * rr_ei(ng) * Id**2
    alpha2_o3 = lambda ng: 5/2 * Id * (Te_orifice(ng,do,TgK,species) - Te_i)
    alpha2_o4 = lambda ng: cc.me * Lo / (np.pi*ro**2*cc.e**2) * rr_en(ng) * Id**2
    alpha2_o5 = lambda ng: cc.e * ng**2 * rr_ex(ng) * np.pi * Lo * ro**2 * eex(ng)
    
    alpha2_o = lambda al,ng: al(ng)**2 * (alpha2_o1(ng) + alpha2_o2(ng) - alpha2_o3(ng) - alpha2_o4(ng) + alpha2_o5(ng))
    
    # Terms in alpha^1
    alpha1_o1 = lambda ng: alpha2_o3(ng)
    alpha1_o2 = lambda ng: alpha2_o2(ng)
    alpha1_o3 = lambda ng: alpha2_o4(ng)
    
    alpha1_o = lambda al,ng: al(ng) * (alpha1_o1(ng) - alpha1_o2(ng) + 2*alpha1_o3(ng))
    
    # Term in alpha^0
    alpha0_o = lambda ng: alpha2_o4(ng)
    
    # Resulting equation
    core_orifice_equation = lambda ng: alpha2_o(alpha_o,ng) + alpha1_o(alpha_o,ng) - alpha0_o(ng)
    orifice_equation = lambda log_ng: core_orifice_equation(10**log_ng)    
    
    
    ### Solve the orifice equation for ng
    # Simple bounding of ng_orifice
    # Upper bound for ng_orifice: cannot exceed insert density, and have to 
    # ensure that alpha is always positive
    # We check the latter condition later
    # Lower bound for ng_orifice: hard value
    lngo_max = log_ng_i
    lngo_min = 17
    
    ### Bisect the equation; if we can't, then there are no solutions
    try:
        l_ngo = next(bisect_next(lambda log_ng: orifice_equation(log_ng),
                                             lngo_min,
                                             lngo_max,
                                             0.1,
                                             xtol = 1e-8,
                                             atol = 1e-8))[0]  
        
    except:
        ng_o = np.nan
        return np.array([mdot,Id,ng_i,ng_o])
    
    # Verify that the solution works with the ionization fraction
    ng_o = -1
    l_ao = alpha_o(10**l_ngo)
        
    if l_ao < 0 or l_ao > 1:
        ng_o = -1
    else:
        ng_o = 10 ** l_ngo
        
    # If no solutions are satisfactory, then return a NaN 
    if ng_o < 0:
        ng_o = np.nan
    
    if verbose:
        print(l_ngo,np.array([mdot * cc.e / M  / cc.sccm2eqA,Id,ng_i,ng_o]),l_ao)
    return np.array([mdot,Id,ng_i,ng_o])
