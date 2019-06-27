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

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import files  # relative-import the *package* containing the templates

import cathode.constants as cc
import pandas as pd
import numpy as np
import os

def load_all_data():
    '''
    Loads the data for all the cathodes specified by name in cathode_idx. 
    Stores the corresponding results in the dataframe pdf ("pressure 
    dataframe")
    
    FYI - There are NO error checks. The cathode_idx variable should correspond
    to the same idx specified in the datafile_idx.pkl...
    
    '''
    ### Pandas representation...
    # idx: name of the cathode
    idx = ['NSTAR','NEXIS','Salhi','Salhi-Ar','Salhi-Xe','Salhi-Ar-1.21','Salhi-Ar-0.76','Siegfried','AR3','EK6','SC012','Friedly','T6']
    cathode_idx = ['NSTAR','NEXIS','Salhi','Siegfried','AR3','EK6','SC012','Friedly','T6']

    # col: name of the columns
    # Id -> discharge current
    # mdot -> mass flow rate
    # P -> pressure
    # do -> orifice diameter
    # Lo -> orifice length
    # species -> gas used
    # corr -> P/mdot * do^2 for Siegfried and Wilbur
    col = ['Id','mdot','P','do','Lo','mass','Tw','To','dc','eiz','corr','corr_up','corr_lo']

    pdf = pd.DataFrame(index=idx, columns=col)

    #### Load data
    #root_folder = 'cathode-data'
    root_folder = os.path.dirname(__file__)
    root_folder = os.path.join(root_folder,'files','cathode-data')

    di_str = pkg_resources.open_binary(files,'datafile_index.pkl')
    df = pd.read_pickle(di_str)

    for cat in cathode_idx:
        folder = root_folder + '/' + df.folder[cat]
        datafile = folder + '/' + df.datafile[cat]
        nskip = df.skip_header[cat]
        dtype = df.dtype[cat]

        load_single_cathode(cat,datafile,nskip,dtype,pdf)

    ## Make sure we fill the temperature array with an arbitrary temperature
    ## when data is not available
    for name in idx:
        arr = pdf.Tw[name]
        arr_idx = np.isnan(arr)
        arr[arr_idx] = 1000
        pdf.Tw[name] = arr

    return pdf

def load_single_cathode(cat,datafile,nskip,dtype,pdf):
    '''
    Loads the data from a single cathode. Since they each differ, there are 
    tests to determine which cathode we are considering. 
    The corresponding fields are then filled in the dataframe used
    Inputs:
        - cat: the cathode name of interest
        - datafile: the datafile we are loading
        - nskip: number of header lines to skipe
        - dtype: the datatype loaded
        - pdf: the pandas frame to fill
    Note about the units for the final panda frame
        - Id: current (A)
        - mdot: mass flow rate (A)
        - P: pressure (Torr)
        - do: Orifice diameter (mm)
        - Lo: Orifice length (mm)
        - mass: propellant mass (amu)
        - Tw: wall temperature (degC)
    '''
    # Read data
    print(cat)
    data = np.genfromtxt(datafile,delimiter=',',dtype=dtype,names=True,skip_header=nskip)
    
    pdf.Id[cat] = data['Id']
    pdf.P[cat] = data['P']
    pdf.do[cat] = data['do']
    pdf.Lo[cat] = data['Lo']
    pdf.mass[cat] = data['mass']
    pdf.Tw[cat] = data['Tw']
    pdf.dc[cat] = data['dc']
    
    
    # Deduce the ionization energy from the mass
    # TODO: Add the ionization energy to the cathode.constants package
    pdf.eiz[cat] = (data['mass'] == 131.293) * 12.1298  # Xe
    pdf.eiz[cat] += (data['mass'] == 39.948) * 15.75962 # Ar
    pdf.eiz[cat] += (data['mass'] == 200.59) * 10.43750 # Hg
    
    
    if cat != 'Siegfried':
        pdf.mdot[cat] = data['mdot']

    if cat == 'NSTAR' or cat == 'NEXIS':        
        pdf.mdot[cat] *= cc.sccm2eqA # sccm to equivalent amperes

    elif cat == 'Salhi':
        # Get a couple of datasets separately
        separate_salhi(pdf)

    elif cat == 'Siegfried':     
        # Round to 102 to have similar correlation
        pdf.mdot[cat] = [102,102,102,102,102,78,62,35,25] 
        
        # mA to A
        pdf.mdot[cat]  = np.asarray([mdot * 1e-3 for mdot in pdf.mdot[cat]]) 

    elif cat == 'AR3' or cat == 'EK6' or cat == 'SC012':
        pdf.P[cat] *= 1e3/cc.Torr
        pdf.mdot[cat] *= cc.sccm2eqA # sccm to A

    elif cat == 'Friedly':
        pdf.mdot[cat] *= 1e-3 # mA to A

    elif cat == 'T6':
        mg2sccm = 22.413996 * 1e3 * 60.0 / (1e6*6.02214179e23) * 1.0 / cc.M.species('Xe')
        mg2eqA = mg2sccm * cc.sccm2eqA
        pdf.mdot[cat] *= mg2eqA # mg to A



def separate_salhi(pdf):
    salhi_ar = pdf.mass['Salhi'] == 39.948
    salhi_xe = pdf.mass['Salhi'] == 131.293
    
    pdf.Id['Salhi-Ar'] = pdf.Id['Salhi'][salhi_ar]
    pdf.mdot['Salhi-Ar'] = pdf.mdot['Salhi'][salhi_ar]
    pdf.P['Salhi-Ar'] = pdf.P['Salhi'][salhi_ar]
    pdf.do['Salhi-Ar'] = pdf.do['Salhi'][salhi_ar]
    pdf.Lo['Salhi-Ar'] = pdf.Lo['Salhi'][salhi_ar]
    pdf.mass['Salhi-Ar'] = pdf.mass['Salhi'][salhi_ar]
    pdf.Tw['Salhi-Ar'] = pdf.Tw['Salhi'][salhi_ar]
    pdf.dc['Salhi-Ar'] = pdf.dc['Salhi'][salhi_ar]
    pdf.eiz['Salhi-Ar'] = pdf.eiz['Salhi'][salhi_ar]
        
    pdf.Id['Salhi-Xe'] = pdf.Id['Salhi'][salhi_xe]
    pdf.mdot['Salhi-Xe'] = pdf.mdot['Salhi'][salhi_xe]
    pdf.P['Salhi-Xe'] = pdf.P['Salhi'][salhi_xe]
    pdf.do['Salhi-Xe'] = pdf.do['Salhi'][salhi_xe]
    pdf.Lo['Salhi-Xe'] = pdf.Lo['Salhi'][salhi_xe]
    pdf.mass['Salhi-Xe'] = pdf.mass['Salhi'][salhi_xe]
    pdf.Tw['Salhi-Xe'] = pdf.Tw['Salhi'][salhi_xe]
    pdf.dc['Salhi-Xe'] = pdf.dc['Salhi'][salhi_xe]
    pdf.eiz['Salhi-Xe'] = pdf.eiz['Salhi'][salhi_xe]

    # Get the 1.21 mm separately
    salhi_ar121 = pdf.do['Salhi-Ar'] == 1.21
    
    pdf.Id['Salhi-Ar-1.21'] = pdf.Id['Salhi-Ar'][salhi_ar121]
    pdf.mdot['Salhi-Ar-1.21'] = pdf.mdot['Salhi-Ar'][salhi_ar121]
    pdf.P['Salhi-Ar-1.21'] = pdf.P['Salhi-Ar'][salhi_ar121]
    pdf.do['Salhi-Ar-1.21'] = pdf.do['Salhi-Ar'][salhi_ar121]
    pdf.Lo['Salhi-Ar-1.21'] = pdf.Lo['Salhi-Ar'][salhi_ar121]
    pdf.mass['Salhi-Ar-1.21'] = pdf.mass['Salhi-Ar'][salhi_ar121]
    pdf.Tw['Salhi-Ar-1.21'] = pdf.Tw['Salhi-Ar'][salhi_ar121]
    pdf.dc['Salhi-Ar-1.21'] = pdf.dc['Salhi-Ar'][salhi_ar121]
    pdf.eiz['Salhi-Ar-1.21'] = pdf.eiz['Salhi-Ar'][salhi_ar121]
    
    salhi_ar076 = pdf.do['Salhi-Ar'] == 0.76
    
    pdf.Id['Salhi-Ar-0.76'] = pdf.Id['Salhi-Ar'][salhi_ar076]
    pdf.mdot['Salhi-Ar-0.76'] = pdf.mdot['Salhi-Ar'][salhi_ar076]
    pdf.P['Salhi-Ar-0.76'] = pdf.P['Salhi-Ar'][salhi_ar076]
    pdf.do['Salhi-Ar-0.76'] = pdf.do['Salhi-Ar'][salhi_ar076]
    pdf.Lo['Salhi-Ar-0.76'] = pdf.Lo['Salhi-Ar'][salhi_ar076]
    pdf.mass['Salhi-Ar-0.76'] = pdf.mass['Salhi-Ar'][salhi_ar076]
    pdf.Tw['Salhi-Ar-0.76'] = pdf.Tw['Salhi-Ar'][salhi_ar076]
    pdf.dc['Salhi-Ar-0.76'] = pdf.dc['Salhi-Ar'][salhi_ar076]
    pdf.eiz['Salhi-Ar-0.76'] = pdf.eiz['Salhi-Ar'][salhi_ar076]


