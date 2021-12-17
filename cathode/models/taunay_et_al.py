###############################################################################
#
# "cathode" Python package
# Version: 1.0.0
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
import numpy as np
import multiprocessing as mp
import itertools
import h5py
import os
import datetime
import pickle


from scipy.interpolate import RegularGridInterpolator 

import cathode.constants as cc
from cathode.models.taunay_et_al_core.collision_holder import collision_holder
from cathode.models.taunay_et_al_core.compute_orifice import orifice_density_wrapper
from cathode.models.taunay_et_al_core.compute_insert import insert_density_wrapper
from cathode.models.taunay_et_al_core.utils import find_number_processes,unpack_results
from cathode.models.taunay_et_al_core.SingleCoordInterpolator import SingleCoordInterpolator 

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
    geometry = np.recarray((len(geometry_names),), # shape
            dtype=geometry_dt, # data type
            names=geometry_names, # data name
            buf=np.array([dc_db, Lemitter, do_db, Lo_db, Lupstream], dtype=geometry_dt) # actual data
            )
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
                                   phi_s = None,
                                   force_orifice_calc = False,
                                   verbose = True 
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
        - Lupstream: upstream distance to the pressure measurement ()
        - Lemitter: length of the emitter ()
        - eiz_db: ionization energy of the gas (eV)
        - TgK: neutral gas temperature assumed (K)
        - data_file: location of HDF5 file that contains results.
        - phi_s: vector of sheath potentials to test (V). Defaults to None. 
          Optional.
        - force_orifice_calc: a boolean to force computations of orifice
          results. Defaults to False. Optional.
        - verbose: Controls verbosity. Defaults to True. Optional.
    If phi_s is NOT specified, then the code will run only the orifice cases.
    If phi_s is specified, then the code will try to read data_file to obtain
    orifice results. If they do not exist, they will be created, saved, and
    used for the insert simulation.
    If force_orifice_calc is used, the orifice cases will be computed
    regardless of the presence of existing results. If orifice results already
    exist, they will be overwritten. Be careful!

    If verbose is "True" then all printouts will be displayed.
    '''    
    ### Check that we have a single gas, single geometry, single temperature
    try:
        list_test = list([M_db, dc_db, do_db, Lo_db, Lupstream, Lemitter, 
            eiz_db, TgK, verbose]) 

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
            print("Attempting to open ", data_file, " ...")
            f = h5py.File(data_file,'a')
        except Exception as e:
            print(str(e))
            return

        print("...success!")

        ### Are there orifice results already?
        orifice_results = f.get(results_path + '/orifice')
        # If there are, grab them and convert to Numpy array
        if orifice_results is not None:
            print("Orifice results already exist in ", data_file)
            print("Please make sure they can be used for the insert cases!")
            orifice_results = np.array(orifice_results)
        else:
            # If not, then we will have to run the orifice simulation
            run_orifice = True
    else:
        print(data_file," does not exist! Creating file.")
        # The file did not exist: we just have to create it and populate it
        # We will then need to run the orifice cases and use Idvec and mdotvec 
        # as the Id and mdot in the HDF5 file
        create_h5file(Idvec, mdotvec, dc_db, do_db, Lo_db, Lupstream,
                Lemitter, eiz_db, TgK, species, data_file)
        run_orifice = True
        f = h5py.File(data_file,'a')

    if run_orifice or force_orifice_calc:
        ### Pre-compute the orifice density for a number of insert densities, 
        ### discharge currents, and mass flow rates
        ngi_vec = np.linspace(20.,24.,10)

        print("Finding all orifice solutions...")
        it = itertools.product(mdot_SI,
                Idvec,
                [M],
                [dc], [do], [Lo],
                [eiz_db],
                ngi_vec,
                [species],
                [chold],
                [TgK],
                [verbose])
        ngo_case = list(it)
        orifice_results = np.zeros((len(ngo_case),4))

        ### Compute cases in parallel
        # Make sure we don't spawn more processes than we need
        procnum = find_number_processes(ngo_case) 

        with mp.Pool(processes=procnum) as pool:
            orifice_results = pool.starmap(orifice_density_wrapper,ngo_case)

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
        ### Check that the specified Idvec and mdotvec are within the orifice
        ### Idvec and mdotvec
        conditions_path = species + '/simulations' + '/conditions'
        Id_o = f.get(conditions_path + '/Id_orifice') 
        Id_o = np.array(Id_o)
        mdot_o = f.get(conditions_path + '/mdot_orifice') 
        mdot_o = np.array(mdot_o)

        Id_o_min = np.min(Id_o)
        Id_o_max = np.max(Id_o)
        mdot_o_min = np.min(mdot_o)
        mdot_o_max = np.max(mdot_o)

        ### Note that these are *strict* inequalities
        ### Turns out that the RegularGridInterpolator does not work well with
        ### values that are exactly on the boundary
        b_Id = (np.min(Idvec) < Id_o_min) or (np.max(Idvec) > Id_o_max)
        b_mdot = (np.min(mdotvec) < mdot_o_min) or (np.max(mdotvec) >
                mdot_o_max)
        if b_Id or b_mdot:
            str_err = "ERROR "
            str_err += "Cannot run insert cases that are out of the existing "
            str_err += "bounds of the orifice cases: "
            str_err += "(" + str(Id_o_min) + " < Id < " + str(Id_o_max) 
            str_err += " and "
            str_err += str(mdot_o_min /cc.sccm2eqA) + " < mdot < " 
            str_err += str(mdot_o_max/cc.sccm2eqA) + ")"
            raise RuntimeError(str_err)

        ### Create an interpolator to find orifice density from mass flow rate,
        ### discharge current, and insert density. 
        # Inputs: x,y,z are mass flow rate, discharge current, insert density
        # Output: orifice density
        # x,y,z,V: mdot,Id,ng_i,ng_o
        x = np.unique(orifice_results[:,0])
        x = np.sort(x)
        y = np.unique(orifice_results[:,1])
        y = np.sort(y)
        z = np.unique(orifice_results[:,2])
        z = np.sort(z)

        print(x,y,z)

        ### From scipy documentation on RegularGridInterpolator:
        ### "If any of points have a dimension of size 1, linear interpolation 
        ### will return an array of nan values. Nearest-neighbor interpolation 
        ### will work as usual in this case."

        print("Creating interpolating function...")
        V = np.zeros((len(x),len(y),len(z)))
        Vlin = np.zeros(len(x)*len(y)*len(z))
        for ii, mdot in enumerate(x):
            for jj, Id in enumerate(y):
                for kk, ngi in enumerate(z):
                    idx = kk + jj * len(z) + ii * len(z)*len(y)
                    V[ii,jj,kk] = orifice_results[idx][-1]  
                    Vlin[idx] = orifice_results[idx][-1]
                    
        ### We have different cases we have to deal with because of the quirks
        ### of RegularGridInterpolator
        ### The only possible cases are that mass flow rate and discharge 
        ### current have single entry. The insert density values *always* have
        ### more than 1 element by construction
        if len(x) == 1 or len(y) == 1:
            print("WARNING: The single coordinate interpolator has not been "
                    "thoroughly tested yet. Use at own risk.")
            ngo_fn = SingleCoordInterpolator((x,y,z), V, 
                                             bounds_error = False)
        else:
            ngo_fn = RegularGridInterpolator((x,y,z), V, 
                                             bounds_error = False)
        print("...done")
        
        ### Run all cases 
        # Create a list of all cases to run
        # Note: can't use itertools.product here bc. otherwise the vector of
        # sheath voltages, phi_s, gets expanded and gums up the interface in
        # insert_density_wrapper.
        insert_case = []
        for mdot in mdot_SI:
            for Id in Idvec:
                insert_case.append((mdot,
                    Id,
                    M,
                    dc,do,Lo,
                    eiz_db,
                    ngo_fn,
                    species,
                    chold,
                    TgK,
                    phi_s,
                    verbose))

        # Find number of processes
        procnum = find_number_processes(insert_case) 
        # Run the cases
        print("Running cases...")

        # If procnum is None, we use *all* threads available 
        if procnum is None:
            with mp.Pool(processes=procnum) as pool:
                res = pool.starmap(insert_density_wrapper,insert_case)
        # Otherwise, procnum is an integer: we use fewer threads than there are available
        else:
            # We use more than one CPU thread on all cases
            if procnum > 1:
                with mp.Pool(processes=procnum) as pool:
                    res = pool.starmap(insert_density_wrapper,insert_case)
            # Or we just use a single one but loop over all cases
            else:
                res = []
                for case in insert_case:
                    t = insert_density_wrapper(*case)
                    res.append(t)

        print("...done")

        ### Convert to a Pandas dataframe
        # TODO Pass the work function upstream. Now everything is BaO-W
        emitterMaterial = 'BaO-W'
        df = unpack_results(res,Lupstream,Lemitter,species,chold,TgK,emitterMaterial,phi_s)

        ### Create a timestamp 
        # and remove milliseconds
        ts = datetime.datetime.utcnow()
        ts = ts.strftime('%Y%m%d%H%M%S')
        results_name = 'r' + ts

        ### Save into the HDF5 file
        insert_path = results_path + '/insert'
        # Did we create the "insert" group already?
        insert_results = f.get(insert_path)
        if insert_results is None:
            print("INFO No insert results exist.")
            print("INFO Creating insert group in the HDF5 file.")
            f.create_group(insert_path)

        # Ensure the file is closed before we use .to_hdf() from Pandas
        f.close()

        # Save using Pandas to_hdf(). It will create a new group under which 
        # the data lives
        insert_results_path = insert_path + '/' + results_name 
        df.to_hdf(data_file,insert_results_path)

        return insert_results_path, df

    else:
        f.close()
        return None,None
