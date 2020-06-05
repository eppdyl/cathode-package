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

import numpy as np

from correlation import Te_insert, TeK_orifice, Lem

import cathode.constants as cc
from cathode.math.bisect_next import bisect_next
from cathode.models.flow import knudsen_number, santeler_theta
from cathode.models.taunay_et_al_core.utils import check_answer 

def ng_target(lng_i,params):
    '''
    Bisection target. Returns the difference between the neutral pressure and
    the pressure as calculated by the sum of each individual pressures.
    '''
    # We bisect on the logarithm but work in the core with the actual density
    ng_i = 10**lng_i
    ret = ng_core(ng_i,params)
        
#    print(lng_i,params['phi_s'],ret['goal'])   

    return ret['goal']
    
def ng_core(ng_i,params):   
    '''
    Core function that computes all quantities necessary for the bisection
    and for the main outputs. 
    
    This function is separated from ng_target because it will be called
    once more once we have a solution from the bisection method.
    '''
    ### Extract parameters
    dc = params['dc'] # m
    do = params['do'] # m
    Rg = params['Rg'] # S.I.
    gam = params['gamma']
    TgK = params['TgK'] # K
    mdot = params['mdot'] # kg/s
    rr_iz = params['rr_iz'] # m3/s for the sheath-edge factor
    Id = params['Id'] # A
    M = params['M'] # kg
    phi_s = params['phi_s'] # V
    ngo_fn = params['ng_o'] # 1/m3
    species = params['species'] 
    
    PI2 = do/dc
    
    alpha_i_p = params['alpha_i_p']
    alpha_i_m = params['alpha_i_m']

    Kn = params['Kn']
    
    ### Return dictionary
    ret_obj = {}
    
    ### Insert quantities
    # Get insert electron temperature and ionization fraction
    Te_i = Te_insert(ng_i,dc,species)

    # Grab the MINIMUM alpha
    try:
        ai_p = alpha_i_p(ng_i,phi_s)
        ai_m = alpha_i_m(ng_i,phi_s)
           
        # TODO: More meaningful variable names
        at_p = 1e5
        at_m = 1e5
        
        if ai_p > 0 and ai_p < 1:
            at_p = ai_p
        
        if ai_m > 0 and ai_m < 1:
            at_m = ai_m
        
        alpha_i = np.min([at_p,at_m])
        
        if alpha_i > 1 or alpha_i < 0:
            raise ValueError("No valid insert ionization fraction found")
    except:
        alpha_i = 1e-10
        
    ### ORIFICE QUANTITIES
    ro = do / 2
    rc = dc / 2
    # Get the orifice neutral density from the input parameters
    ng_o_in = np.array([mdot,Id,ng_i])
    ng_o = ngo_fn(ng_o_in)    

    # Get the corresponding orifice ionization fraction
    alpha_o = params['alpha_o']
    al_o = alpha_o(ng_o)
    
    # Get the orifice electron temperature in K
    Te_o = TeK_orifice(ng_o,do,TgK,species)

    # Orifice exit plane speed of sound    
    a_o = gam * Rg * ( TgK + al_o * Te_o )
    a_o = np.sqrt(a_o)        

    ### Total pressure
    ## Gasdynamic pressure
    Pgd = mdot * a_o / (np.pi*ro**2) # Gas dynamic
    
    ## Magnetic pressure
    P_mag = cc.mu0 * Id**2 / (np.pi**2*do**2) * (1/4 + np.log(dc/do))

    ## Static pressures on the wall
    # Quantities "w/o" correction are the static species pressure used in the
    # perfect gas law
    # Electron pressure w/o correction
    P_e = alpha_i / (1-alpha_i) * cc.e * Te_i * ng_i

    # Ion pressure w/o correction
    P_i = alpha_i / (1-alpha_i) * cc.Boltzmann * TgK * ng_i
    
    # Correction for sheath-edge
    vb = np.sqrt(cc.e*Te_i/M)
    nes_fac = ng_i * rc * 1/(2*vb) * rr_iz(ng_i,dc) # Sheath-edge factor

    # Momentum pressure on the orifice
    P_or = P_e * (1 + alpha_i / (1-alpha_i) * nes_fac)
    P_or *= nes_fac * (1/PI2**2-1)
   
    # Exit plane static pressure
    lKn = Kn(ng_o,do)  # Orifice Knudsen number    
    th = santeler_theta(lKn) # Theta
    Teq = TgK + al_o * Te_o # Mixture temperature
    Teq_stag = Teq * (gam+1)/2 # Stagnation mixture temperature
    
    # Throughput, Pa m3/s: defined with stagnation quantities
    Qput = mdot * cc.R_specific(species) * Teq_stag 
    
    Ca = np.pi * ro**2 * np.sqrt(cc.Boltzmann * Teq / (2 * np.pi * M)) # Aperture conductance (n * cbar / 4)
    Cm = Ca # Molecular flow conductance
    exponent = (gam+1)/(gam-1)
    
    Ca_v = np.pi * ro**2 * np.sqrt(cc.Boltzmann * Teq_stag / (2 * np.pi * M))
    Ggam = np.sqrt(gam * (2/(gam+1))**exponent)
    Cz = np.sqrt(2*np.pi) * Ca_v
    Cv = Ggam * Cz
    Pexit = Qput / (th * Cm + (1-th)*Cv)
    
    Pexit_static = Pexit * (2/(gam+1)) ** (gam/(gam-1))

    # Total pressure
    P_g = TgK * ng_i * cc.Boltzmann # Target neutral static pressure
    P_ie = (P_e + P_i) # Target plasma static pressure
    
    P_tot = Pgd + P_or + P_mag + Pexit_static # Total pressure as calculated

    ### Target
    ret = 1-(P_tot/(P_g + P_ie))
    
    ret_obj = {}
    # Plasma quantities: gas and electron density, temperature, ionization 
    # fraction
    ret_obj['ng_i'] = ng_i
    ret_obj['ne_i'] = alpha_i / (1-alpha_i) * ng_i
    ret_obj['Te_i'] = Te_i
    ret_obj['alpha_i'] = alpha_i

    ret_obj['ng_o'] = ng_o
    ret_obj['ne_o'] = al_o / (1-al_o) * ng_o
    ret_obj['Te_o'] = Te_o*cc.Kelvin2eV
    ret_obj['alpha_o'] = al_o

    ret_obj['sheath_edge_correction'] = nes_fac
     
    # Total pressure components 
    ret_obj['Pgd'] = Pgd # Gasdynamic
    ret_obj['Pmag'] = P_mag # Magnetic
    ret_obj['Por'] = P_or # Orifice plate
    ret_obj['Pexit'] = Pexit_static # Exit static
    ret_obj['Ptot'] = P_tot
    
    # Perfect gas law
    ret_obj['Pe'] = P_e
    ret_obj['Pi'] = P_i
    ret_obj['Pg'] = P_g
    
    
    ret_obj['ratio'] = P_tot / P_mag # Total pressure to magnetic pressure 
    ret_obj['phi_s'] = phi_s # Value of sheath voltage
    ret_obj['goal'] = ret # Value of goal function
    
    return ret_obj


def insert_density_wrapper(mdot,
                              Id,
                              M,
                              dc,do,Lo,
                              eiz,
                              ngo_fn,
                              species,
                              chold,
                              TgK,
                              phi_s,
                              verbose):
    '''
    Outer-most wrapper for the insert density calculation. 
    This function is in charge of setting-up the insert equations and 
    performing the bisection method for specified sheath voltages.
    
    This wrapper is necessary in order to parallelize the calculations.
    
    Inputs:
        - mdot: mass flow rate (kg/s)
        - Id: discharge current (A)
        - M: species mass (kg)
        - dc, do, Lo: cathode insert diameter, orifice diameter, orifice length
        respectively (m)
        - eiz: ionization energy (eV)
        - ngo_fn: a "RegularGridInterpolator" function used to calculate the
        orifice neutral density from an input mass flow rate, discharge current
        and insert neutral density
        - species: a string to describe the species used
        - chold: a collision cross section holder object
        - TgK: neutral gas temperature assumed (K)
        - phi_s: vector of sheath potentials to try
    '''
    ### A holder for the bisection method
    params = {}

    ### Read the input values (they are in SI units already)
    rc = dc/2
    params['M'] = M 
    params['dc'] = dc
    params['do'] = do
    
    params['mdot'] = mdot
    params['Id'] = Id    
    params['ng_o'] = ngo_fn
    params['TgK'] = TgK
    params['species'] = species
    
    ### Cross-sections
    rr_iz = lambda ng,ds: chold.rr('iz',Te_insert(ng,ds,species))
    rr_en = lambda ng,ds: chold.rr('en',Te_insert(ng,ds,species))
    rr_ex = lambda ng,ds: chold.rr('ex',Te_insert(ng,ds,species))
    rr_ei = lambda ng,ds: chold.rr('ei',Te_insert(ng,ds,species))
    eex = lambda ng: chold.ex_avg_energy(Te_insert(ng,dc,species)) # 
    
    params['rr_iz'] = rr_iz
    
    ### Inputs to the main equations
    ## Flow stuff
    params['Rg'] = cc.R_specific(species) # Specific gas constant
    params['gamma'] = 5/3 # We only deal with monatomic particles for now
    params['Kn'] = lambda ng, ds: knudsen_number(ng, ds, species)
    
    # Correlations
    fLem = lambda ng, ds: Lem(ng,ds,species)
    
    ### BUILD THE EQUATIONS TO SOLVE
    ### INSERT EQUATIONS
    # Common factor to ion and random electron heating
    f1_i_fac = lambda ng: cc.e * ng**2 * rr_iz(ng,dc) * np.pi * fLem(ng,dc) * rc**2 
    f1_i_ion = lambda phi_s: eiz + phi_s
    f1_i_el = lambda ng,phi_s: 1/4*np.sqrt(8*M/(np.pi*cc.me))*2*Te_insert(ng,dc,species)*np.exp(-phi_s/Te_insert(ng,dc,species))

    # First term in alpha^2
    alpha2_i1 = lambda ng,phi_s: f1_i_fac(ng) * (f1_i_ion(phi_s) + f1_i_el(ng,phi_s))
    alpha2_i2 = lambda ng: cc.me*fLem(ng,dc)/(cc.e**2*np.pi*rc**2)*rr_ei(ng,dc)*Id**2
    alpha2_i3 = lambda ng,phi_s: (5/2*Te_insert(ng,dc,species)-phi_s)*Id
    alpha2_i4 = lambda ng: cc.me*fLem(ng,dc)/(cc.e**2*np.pi*rc**2)*rr_en(ng,dc)*Id**2
    alpha2_i5 = lambda ng: cc.e * ng**2 * rr_ex(ng,dc) * np.pi * fLem(ng,dc) * rc**2 * eex(ng)
    
    alpha_i_a = lambda ng,phi_s: (alpha2_i1(ng,phi_s) + alpha2_i2(ng) - alpha2_i3(ng,phi_s) - alpha2_i4(ng) + alpha2_i5(ng))
        
    # Terms in alpha^1
    alpha1_i1 = lambda ng,phi_s: alpha2_i3(ng,phi_s)
    alpha1_i2 = lambda ng: alpha2_i2(ng)
    alpha1_i3 = lambda ng: alpha2_i4(ng)
    
    alpha_i_b = lambda ng,phi_s: (alpha1_i1(ng,phi_s) - alpha1_i2(ng) + 2*alpha1_i3(ng))
    
    # Term in alpha^0
    alpha_i_c = lambda ng: alpha2_i4(ng)
        
    # Quadratic formula for the insert density   
    # delta = b^2 - 4 * a * c 
    # alpha = 1/(2*a) * (-b +/- sqrt(delta))
    # we already included the negative sign on c, hence the +4ac
    alpha_del = lambda ng,phi_s: alpha_i_b(ng,phi_s)**2 + 4 * alpha_i_a(ng,phi_s) * alpha_i_c(ng)
    alpha_i_p = lambda ng,phi_s: (-alpha_i_b(ng,phi_s) + np.sqrt(alpha_del(ng,phi_s))) / (2*alpha_i_a(ng,phi_s))
    alpha_i_m = lambda ng,phi_s: (-alpha_i_b(ng,phi_s) - np.sqrt(alpha_del(ng,phi_s))) / (2*alpha_i_a(ng,phi_s))
    
    params['alpha_i_p'] = alpha_i_p
    params['alpha_i_m'] = alpha_i_m

    ### ORIFICE EQUATIONS
    ro = do/2
    Tbar = lambda ng: TgK / TeK_orifice(ng,do,TgK,species)
    aTe = lambda ng: np.sqrt(params['gamma'] * params['Rg'] * TeK_orifice(ng,do,TgK,species))
    vg = lambda ng: mdot / (np.pi * ro**2 * ng * M)
    delta = lambda ng: (vg(ng)/aTe(ng))**2

    sqrt_alpha = lambda ng: np.sqrt(4*delta(ng)*(1+Tbar(ng))+1)
    alpha_o = lambda ng: 1 + 1/(2*delta(ng)) * (1-sqrt_alpha(ng))
    params['alpha_o'] = alpha_o

    ### BISECT WITH A PHI_S SET
    ret_array = []
    for phis in phi_s:
        params['phi_s'] = phis
        ng_i = np.nan
        # Bisect on the logarithm of ng_i between 10^18 and 10^25
        lng_i_list = list(bisect_next(lambda lng_i: ng_target(lng_i,params),
                                     20.,
                                     23.,
                                     0.1,
                                     xtol = 1e-6,
                                     atol = 1e-6))
        # TODO: zip the tuple and ** to unpack the output of the list into
        # two different vectors

        # Sanity check: do we have multiple solutions (this should not happen)
        # Case 1: no solution
        if(len(lng_i_list) == 0):
            ng_i = np.nan
        # Case 2: single solution
        elif(len(lng_i_list) == 1):
            lng_i = lng_i_list[0][0]
            ret_all = ng_core(10**lng_i,params)
            b_answer = check_answer(10**lng_i,ret_all)    
            
            if not b_answer:
                ng_i = np.nan
            else:
                ng_i = 10**lng_i
                
        # Case 3: multiple solutions
        else:
            # TODO: check that the lowest solution is the best
            lng_i_array = np.array(lng_i_list)
            lng_i_array = np.sort(lng_i_array,axis=0)
                    
            # Do any of the solutions work with the ionization fraction?
            for l_ngi in lng_i_array[:,0]:
                ret_all = ng_core(10**l_ngi,params)
                
                # Test out of bounds for ionization fraction and if the orifice
                # density is greater than the insert one
                b_answer = check_answer(10**l_ngi,ret_all)    

                if(b_answer):
                    # If it is a satisfactory answer, break out of loop
                    ng_i = 10**l_ngi
                    break
                else:
                    # Otherwise, try the next one
                    ng_i = np.nan
                    continue

        
        ret_obj = {}
        #ret_obj['output'] = np.array([ng_i])
        ret_obj['input'] = np.array([mdot,Id,M,dc,do,Lo,eiz])
        ret_obj['bisection_out'] = lng_i_list # Make sure we have a single solution
        try:
            ret_obj['complete'] = ng_core(ng_i,params)
        except:
            ret_obj['complete'] = []
 
        if verbose:
            print("======")    
            print("mdot(sccm),Id(A),M(amu),dc(mm),do(mm),Lo(mm),eiz(eV),phi_S (V),P_model (Torr)")
            print(mdot * cc.e / M / cc.sccm2eqA,
                  Id,
                  M/cc.atomic_mass,
                  dc/1e-3,
                  do/1e-3,
                  Lo/1e-3,
                  eiz,
                  phis,
                  ret_obj['complete']['Ptot']/cc.Torr)
                  
            print("bisection_out")
            print(ret_obj['bisection_out'])  
        
        ret_array.append(ret_obj)
    
    return ret_array
