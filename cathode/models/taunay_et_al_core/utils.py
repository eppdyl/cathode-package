###############################################################################
#
# "cathode" Python package
# Version: 1.0
# A package of various cathode models that have been published throughout the
# years. Associated publications:
# - Taunay, P.-Y. C. R., Wordingham, C. J., and Choueiri, E. Y., "A 0-D model 
# for orificed hollow cathodes with application to the scaling of total 
# pressure," AIAA Propulsion and Energy Forum, 2019, AIAA-2019-4246
# - Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
# review of hollow cathode modeling: 0-D models," 53rd AIAA/ASME/SAE/ASEE Joint 
# Propulsion Conference, 2017, AIAA-2017-4888 
# 
###############################################################################
# Copyright (C) 2019-2020 Pierre-Yves Taunay, Chris Wordingham
#  
# This file is part of the Python "cathode" package.
#
# The Python "cathode" package is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License (LGPL) as 
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The Python "cathode" package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License (LGPL) for more details.
#
# You should have received a copy of the GNU Lesser General Public License 
# (LGPL) along with the Python "cathode" package.  
# If not, see <https://www.gnu.org/licenses/>.
###############################################################################

import os
import numpy as np
import pandas as pd
import cathode.constants as cc

from cathode.models.taunay_et_al_core.correlation import Lem
from cathode.models.flow import poiseuille_flow

from scipy.optimize import root

### Columns of the pandas dataframe
columns = np.dtype([ # Start with inputs
           ('dischargeCurrent',float),  # A
           ('massFlowRate_SI',float),   # kg/s
           ('massFlowRate_eqA',float),  # eqA
           ('massFlowRate_sccm',float), # sccm
           ('species',str),           # string
           ('mass',float),              # kg
           ('ionizationEnergy',float),  # eV
           ('insertDiameter',float),    # m
           ('orificeDiameter',float),   # m
           ('orificeLength',float),     # m
           ('sheathVoltage',float),     # V
           ('neutralGasTemperature',float), # K
           ('upstreamPressureTap',float), # m
           ('emitterLength',float),     # m
           ('workFunctionMaterial',str), # string
            # Continue with outputs
            ('bisectionOutput',np.ndarray), # List of (result,tolerance)
            ('electronPressure',float),     # Pa
            ('ionPressure',float),          # Pa
            ('neutralPressure',float),      # Pa
            ('exitStaticPressure',float),   # Pa
            ('gasdynamicPressure',float),   # Pa
            ('magneticPressure',float),     # Pa
            ('momentumFluxPressure',float), # Pa
            ('totalPressure',float),        # Pa
            ('totalPressure_Torr',float),   # Torr
            ('totalPressureCorr',float),    # Pa
            ('totalPressureCorr_Torr',float),    # Torr 
            ('insertElectronTemperature',float),    # eV
            ('orificeElectronTemperature',float),   # eV
            ('insertTemperature',float),            # degC 
            ('emissionLength',float),               # m 
            ('insertElectronDensity',float),        # 1/m3
            ('orificeElectronDensity',float),       # 1/m3
            ('insertNeutralDensity',float),         # 1/m3
            ('orificeNeutralDensity',float),        # 1/m3
            ('insertIonizationFraction',float),     # 1
            ('orificeIonizationFraction',float),    # 1
            ('totalToMagneticRatio',float),         # 1
            ('sheathEdgeFactor',float),             # 1
            ('goal',float)])


def find_number_processes(list_case):
    """
    Find the number of parallel processes to spawn based on a list size.

    If the length of the list is less than the total number of CPUs we have,
    then just spawn as many elements as there are in the list. Otherwise,
    set the number of processes to None because multiprocessing.Pool then 
    uses all available CPUs
    """
    procnum = None
    if len(list_case) < os.cpu_count():
        procnum = len(list_case) 

    return procnum


def check_answer(ng_i,ret_all):
    """
    Check that the proposed solution is valid. 

    Checks that ionization fractions are within bounds, that orifice neutral 
    density is less than insert neutral density, and that orifice electron
    temperature is larger than that of the insert.
    """
    ng_o = ret_all['ng_o']
    alpha_o = ret_all['alpha_o']
    Te_o = ret_all['Te_o']
    
    alpha_i = ret_all['alpha_i']
    Te_i = ret_all['Te_i']

    b_ai = alpha_i < 0 or alpha_i > 1
    b_ao = alpha_o < 0 or alpha_o > 1
    b_ng = ng_i < ng_o
    b_Te = Te_i > Te_o    
    
    if(b_ai or b_ao or b_ng or b_Te):
        return False
    else:
        return True

def emitter_material(emitterMaterial):
    if emitterMaterial == 'BaO-W':
        RDConstant = 120e4
        phi_wf = lambda Tw: 1.67+2.87e-4 * Tw
    elif emitterMaterial == 'LaB6':
        RDconstant = 29e4
        phi_wf = lambda Tw: 2.70

    return RDConstant, phi_wf


def unpack_results(results,
        Lupstream,
        Lemitter,
        species,
        chold,
        TgK,
        emitterMaterial,
        phisvec):
    """ 
    Unpacks the results of the simulations.

    Traverses the array of results and stores them into a Pandas dataframe.
    Required inputs:
        - Lupstream
        - Lemitter
        - species
        - chold
        - TgK
        - emitterMaterial
        - phisvec
    """
    df = np.empty(0,dtype=columns)
    df = pd.DataFrame(df)
    RDConstant, phi_wf = emitter_material(emitterMaterial)

    # The results are in an N by 1 array of objects
    # where N is the total number of (mdot,Id) cases 
    for elem in results:
        # Each element is *also* an M by 1 array of dictionaries, where M
        # is the total number of sheath voltages tested
        for idx, phi_s in enumerate(phisvec):
            lelem = elem[idx]
            
            ### Unpack the inputs:
            # 0: mdot (kg/s)
            # 1: Id
            # 2: M (kg)
            # 3: dc (m)
            # 4: do (m)
            # 5: Lo (m)
            # 6: eiz (eV)
            mdot_SI = lelem['input'][0]
            Id = lelem['input'][1]
            M = cc.M.species(species)
            dc = lelem['input'][3]
            do = lelem['input'][4]
            Lo = lelem['input'][5]
            eiz = lelem['input'][6]
 
            mdot_eqA = mdot_SI * cc.e / M 
            mdot_sccm = mdot_eqA / cc.sccm2eqA

            ### Unpack the results of the algorithm
            bisection_out = lelem['bisection_out']
            goal = lelem['complete']['goal'][0]

            ### Unpack plasma quantities           
            Te_i = lelem['complete']['Te_i']
            ng_i = lelem['complete']['ng_i']
            ne_i = lelem['complete']['ne_i']
            al_i = lelem['complete']['alpha_i']
            Lem_i = Lem(ng_i,dc,species)
            sheath_edge_factor = lelem['complete']['sheath_edge_correction']

            Te_o = lelem['complete']['Te_o'][0]
            ng_o = lelem['complete']['ng_o'][0]
            ne_o = lelem['complete']['ne_o'][0]
            al_o = lelem['complete']['alpha_o'][0]

            ### Unpack the pressures
            # Static pressures for perfect gas law
            Pg = lelem['complete']['Pg']
            Pe = lelem['complete']['Pe']
            Pi = lelem['complete']['Pi']
            # Pressures for momentum balance
            Pmag = lelem['complete']['Pmag']
            Pgd = lelem['complete']['Pgd'][0]
            Pexit = lelem['complete']['Pexit'][0] # Static
            Pmf = lelem['complete']['Por']
            # Total pressure
            Ptot = lelem['complete']['Ptot'][0]
            Ptot_torr = Ptot/cc.Torr

            # Total pressure w/ viscous correction
            Pout = lelem['complete']['Ptot'][0]
            # Poiseuille flow correction to the measurement location
            Lp = Lupstream + Lemitter - Lem_i
            Pout = poiseuille_flow(Lp,dc,mdot_sccm,TgK,Pout/cc.Torr,species=species) * cc.Torr
            Pout_torr = Pout/cc.Torr
            
            # Ratio of P/Pmagnetic
            Pratio = Pout/Pmag

            ### Solve for the wall temperature     
            V_i = np.pi * Lem_i * (dc/2)**2
            f_Ir = 1/4 * np.sqrt(8*M/(np.pi*cc.me)) * np.exp(-phi_s/Te_i)
            f_Ii = 1
            
            rhs = al_i / (1-al_i) * ng_i**2 * cc.e * chold.xsec('iz',Te_i) * V_i
            rhs *= (f_Ir-f_Ii)
            rhs += Id
            
            rhs /= (np.pi * Lem_i * dc * RDConstant)
            lhs = lambda Tw: Tw**2 * np.exp(-cc.e*(phi_wf(Tw))/(cc.kB*Tw))
            
            sol = root(lambda Tw: lhs(Tw)-rhs,3000)
            if np.isnan(Pout):
                Tw = np.nan
            else:
                Tw = sol.x[0] - 273.15
            
            ### BUILD THE CORRESPONDING LINE IN PANDA FRAME
            arr = np.array([ # Start with inputs
                            Id,
                            mdot_SI,
                            mdot_eqA,
                            mdot_sccm,
                            species,
                            M,
                            eiz,
                            dc,
                            do,
                            Lo,
                            phi_s,
                            TgK,
                            Lupstream,
                            Lemitter,
                            emitterMaterial,
                            # Continue with outputs
                            bisection_out,
                            Pe,
                            Pi,
                            Pg,
                            Pexit,
                            Pgd,
                            Pmag,
                            Pmf,
                            Ptot,
                            Ptot_torr,
                            Pout,
                            Pout_torr,
                            Te_i,
                            Te_o,
                            Tw,
                            Lem_i,
                            ne_i,
                            ne_o,
                            ng_i,
                            ng_o,
                            al_i,
                            al_o,
                            Pratio,
                            sheath_edge_factor,
                            goal],dtype=np.object)

            df.loc[len(df)] = np.copy(arr)

    return df
