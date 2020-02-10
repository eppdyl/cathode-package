# "cathode" Python package
# Version: 1.0
# A package of various cathode models that have been published throughout the
# years. Associated publications:
# Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
# review of hollow cathode modeling: 0-D models," Journal of Propulsion and
# Power, in preparation.
# Taunay, P.-Y. C. R., Wordingham, C. J., and Choueiri, E.Y., "Physics of
# thermionic, orificed hollow cathodes," Plasma Sources Science and Technology,
# in preparation.
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
import numpy as np
import pickle
import multiprocessing as mp
import itertools


from scipy.interpolate import RegularGridInterpolator

import cathode.constants as cc

from collision_holder import collision_holder
from compute_core_orifice import orifice_density_wrapper
from compute_core_insert import insert_density_wrapper

def solve(Idvec,mdotvec,
                                   M_db,
                                   dc_db,
                                   do_db,
                                   Lo_db,
                                   eiz_db,
                                   TgK,
                                   phi_s,
                                   file_ngo = None
                                   ):          
    '''
     Compute the case from an Id and mdot vector
     Assume:
         - single gas (M and eiz are unique)
         - single geometry (dc, do, Lo are unique)
    Inputs:
        - Idvec: vector of discharge currents to test (A)
        - mdotvec: vector of mass flow rates to test (eqA)
        - M_db: gas mass from the database (amu)
        - dc_db: cathode insert diameter (mm)
        - do_db: cathode orifice diameter (mm)
        - Lo_db: cathode orifice length (mm)
        - eiz_db: ionization energy of the gas (eV)
        - TgK: neutral gas temperature assumed (K)
        - phi_s: vector of sheath potentials to test (V)
        - file_ngo: pickle file that contains the results of a sweep calculation
        of the orifice neutral gas density as a function of current, mass flow
        rate, and insert neutral density
    '''      
    ### Check that we have a single gas, single geometry, single temperature
    try:
        for x in list([M_db, dc_db, do_db, Lo_db, eiz_db, TgK]):
            if type(x) is list or type(x) is np.ndarray:
                str_err = "ERROR "
                str_err += "No more than one gas, geometry, and temperature "
                str_err += "may be specified."
                raise TypeError(str_err)
    except Exception as e:
        print(str(e))
        return

    ### Convert to SI the data from the database / input
    M = M_db * cc.atomic_mass
    dc = dc_db * 1e-3
    do = do_db * 1e-3
    Lo = Lo_db * 1e-3
    mdot_SI = mdotvec * M / cc.e

    # Find the species of interest with a reverse lookup 
    # TODO: Simplify this
    species = next(key for key in cc.M.__dict__ if cc.M.__dict__[key] == M_db)

    # Create a cross-section object
    chold = collision_holder(species)

    ### Pre-compute the orifice density for a number of insert densities, 
    ### discharge currents, and mass flow rates
    ngi_vec = np.linspace(20.,23.,10)
    if file_ngo is None:
        print("Finding all orifice solutions...")
        ngo_case = itertools.product(mdot_SI,
                Idvec,
                [M],
                [dc], [do], [Lo],
                [eiz_db],
                ngi_vec,
                [species],
                [chold],
                [TgK])
        print(list(ngo_case))

        ngo_case = []
        for mdot in mdot_SI:
            for Id in Idvec:
                for ngi in ngi_vec:
                    ngo_case.append((mdot,
                                     Id,
                                     M,
                                     dc,do,Lo,
                                     eiz_db,
                                     ngi,
                                     species,
                                     chold,
                                     TgK))
        res = np.zeros((len(ngo_case),4))
        print(ngo_case)
#        with mp.Pool(processes=12) as pool:
#            res = pool.starmap(orifice_density_wrapper,ngo_case)
#
#        ### TODO: BETTER SOLUTION THAN DUMPING A PKL FILE
#        print("...done")
#        pickle.dump(res,open('tmp.pkl','wb'))
#    else:
#        res = pickle.load(open(file_ngo,'rb'))
#    
#    # x,y,z,V: mdot,Id,ng_i,ng_o
#    res = np.array(res)
#    x = np.unique(res[:,0])
#    x = np.sort(x)
#    y = np.unique(res[:,1])
#    y = np.sort(y)
#    z = np.unique(res[:,2])
#    z = np.sort(z)
#    
#    print("Creating interpolating function...")
#    V = np.zeros((len(x),len(y),len(z)))
#    for ii, mdot in enumerate(x):
#        for jj, Id in enumerate(y):
#            for kk, ngi in enumerate(z):
#                #print(ii,jj,kk)
#                idx = kk + jj * len(z) + ii * len(z)*len(y)
#                V[ii,jj,kk] = res[idx][-1]  
#                
#    ngo_fn = RegularGridInterpolator((x,y,z), V, 
#                                     bounds_error = False)
#    print("...done")
#    
#    ### Run all cases 
#    # Create a list of all cases to run
#    list_case = []
#    for mdot in mdot_SI:
#        for Id in Idvec:
#            list_case.append((mdot,
#                              Id,
#                              M,
#                              dc,do,Lo,
#                              eiz_db,
#                              ngo_fn,
#                              species,
#                              chold,
#                              TgK,
#                              phi_s))
#
#    print("Running cases...")
#    with mp.Pool() as pool:
#        res = pool.starmap(insert_density_wrapper,list_case)
#    print("...done")
#
#    return res
#    
#
#    
#    


