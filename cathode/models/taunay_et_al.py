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
import h5py
import os


from scipy.interpolate import RegularGridInterpolator

import cathode.constants as cc
from cathode.models.taunay_et_al_core.collision_holder import collision_holder
from cathode.models.taunay_et_al_core.compute_orifice import orifice_density_wrapper
from cathode.models.taunay_et_al_core.compute_insert import insert_density_wrapper
from cathode.models.taunay_et_al_core.utils import find_number_processes

def create_h5file(Idvec, mdotvec, dc_db, do_db, Lo_db, Lupstream, Lemitter, eiz_db, TgK,
        species, data_file):

    f = h5py.File(data_file, 'w')

    conditions_path = species + '/simulations' + '/conditions'
    results_path = species + '/simulations' + '/results'
    f.create_group(conditions_path)
    f.create_group(species + '/simulations' + '/results')

    # Create the geometry dataset
    geometry_names = ['insert_diameter', 'insert_length', 'orifice_diameter', 'orifice_length',
            'pressure_tap_position']
    geometry_dt = np.dtype({'names': geometry_names,
        'formats':[(np.float64)]*len(geometry_names)})
    geometry = np.rec.fromarrays([dc_db, Lemitter, do_db, Lo_db, Lupstream],
            dtype=geometry_dt)
    f.create_dataset('geometry', data=geometry)

    # Create the conditions dataset
    f.create_dataset(conditions_path + '/Id_orifice', data=Idvec)
    f.create_dataset(conditions_path + '/mdot_orifice', data=mdotvec)

    # Create the results groups
    for Tg in [2000,3000,4000]:
        f.create_group(results_path + "/" + str(Tg))

    # Close the file
    f.close()

def solve(Idvec,mdotvec,
                                   M_db,
                                   dc_db,
                                   do_db,
                                   Lo_db,
                                   Lupstream,
                                   Lemitter,
                                   eiz_db,
                                   TgK,
                                   data_file,
                                   phi_s = None
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
        - data_file: location of HDF5 file that contains results.
        - phi_s: vector of sheath potentials to test (V). Optional.
    If phi_s is NOT specified, then the code will run only the orifice cases.
    If phi_s is specified, then the code will try to read data_file to obtain
    orifice results. If they do not exist, they will be created, saved, and
    used for the insert simulation.
    '''    
    ### Check that we have a single gas, single geometry, single temperature
    try:
        list_test = list([M_db, dc_db, do_db, Lo_db, Lupstream, Lemitter, 
            eiz_db, TgK]) 

        for x in list_test:
            if type(x) is list or type(x) is np.ndarray:
                str_err = "ERROR "
                str_err += "No more than one gas, geometry, and temperature "
                str_err += "may be specified."
                raise TypeError(str_err)
    except Exception as e:
        print(str(e))
        return

    ### If the temperature is passed as a float, truncate to integer
    ### otherwise it trips up the correlation function
    TgK = (int)(TgK)

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

    ###########################################################################
    ### Orifice results #######################################################
    ###########################################################################
    run_orifice = False
    orifice_results = None
    f = None
    results_path = species + '/simulations' + '/results' + '/' + str(TgK)

    ### Does the file exist?
    if os.path.exists(data_file):
        ### If so, try to open it. If it throws an error, print it and stop
        try:
            f = h5py.File(data_file,'a')
        except Exception as e:
            print(str(e))
            return

        ### Are there orifice results already?
        try:
            # If there are, grab them and convert to Numpy array
            orifice_results = f.get(results_path + '/orifice')
            orifice_results = np.array(orifice_results)
        except:
            # If not, then we will have to run the orifice simulation
            run_orifice = True
    else:
        # The file did not exist: we just have to create it and populate it
        # We will then need to run the orifice cases and use Idvec and mdotvec 
        # as the Id and mdot in the HDF5 file
        create_h5file(Idvec, mdotvec, dc_db, do_db, Lo_db, Lupstream,
                Lemitter, eiz_db, TgK, species, data_file)
        run_orifice = True

    if run_orifice:
        ### Pre-compute the orifice density for a number of insert densities, 
        ### discharge currents, and mass flow rates
        ngi_vec = np.linspace(20.,23.,10)

        print("Finding all orifice solutions...")
        it = itertools.product(mdot_SI,
                Idvec,
                [M],
                [dc], [do], [Lo],
                [eiz_db],
                ngi_vec,
                [species],
                [chold],
                [TgK])
        ngo_case = list(it)
        res = np.zeros((len(ngo_case),4))

        ### Compute cases in parallel
        # Make sure we don't spawn more processes than we need
        procnum = find_number_processes(ngo_case) 

        #with mp.Pool(processes=procnum) as pool:
        #    res = pool.starmap(orifice_density_wrapper,ngo_case)
        orifice_results = orifice_density_wrapper(*ngo_case[0]) 
        orifice_results = np.array(orifice_results)

        print("...done")

        ### Save the results in the HDF5 file
        f.create_dataset(results_path + "/orifice", 
                data=np.copy(orifice_results))
    
    ###########################################################################
    ### Insert results ########################################################
    ###########################################################################
    ### We can just solve for the orifice solutions if we do not specify the
    ### set of sheath potentials to use
    ### If we do specify the sheath potentials, then we create an interpolator
    ### based on orifice data, then solve for the insert solution.
    if (phi_s is not None):
        ### Create an interpolator to find orifice density from mass flow rate,
        ### discharge current, and insert density. 
        # Inputs: x,y,z are mass flow rate, discharge current, insert density
        # Output: orifice density
        # x,y,z,V: mdot,Id,ng_i,ng_o
        #res = np.array(res)
        x = np.unique(orifice_results[:,0])
        x = np.sort(x)
        y = np.unique(orifice_results[:,1])
        y = np.sort(y)
        z = np.unique(orifice_results[:,2])
        z = np.sort(z)
        
        print("Creating interpolating function...")
        V = np.zeros((len(x),len(y),len(z)))
        for ii, mdot in enumerate(x):
            for jj, Id in enumerate(y):
                for kk, ngi in enumerate(z):
                    #print(ii,jj,kk)
                    idx = kk + jj * len(z) + ii * len(z)*len(y)
                    V[ii,jj,kk] = res[idx][-1]  
                    
        ngo_fn = RegularGridInterpolator((x,y,z), V, 
                                         bounds_error = False)
        print("...done")
        
        ### Run all cases 
        # Create a list of all cases to run
        it = itertools.product(mdot_SI,
                    Idvec,
                    [M],
                    [dc], [do], [Lo],
                    [eiz_db],
                    [ngo_fn],
                    [species],
                    [chold],
                    [TgK],
                    phi_s)

        insert_case = list(it) 

        # Find number of processes
        procnum = find_number_processes(insert_case) 
        # Run the cases
        print("Running cases...")
        #with mp.Pool(processes=procnum) as pool:
        #    res = pool.starmap(insert_density_wrapper,insert_case)
        res = insert_density_wrapper(*insert_case[0])
        print("...done")

        ### Save into the HDF5 file
        ### TODO: TEST THIS BC. I'M PRETTY SURE THIS IS A DICTIONARY BEING RETURNED
        results_path = species + '/simulations' + '/results' + '/' + str(TgK)
        f.create_dataset(results_path + "/insert", data=np.array(res))

    else:
        return
