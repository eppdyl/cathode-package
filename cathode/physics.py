# -*- coding: utf-8 -*-
"""
Created on Thu Jun 1 00:58 2017
Submodule of the cathode package containing common cathode parameter 
calculations and atomic physics methods.

@author: cjw
"""

import numpy as np
import re
from scipy.interpolate import splrep,splev
from scipy.integrate import quad
import constants as cc

###############################################################################
#%%                         Basic Plasma Physics
###############################################################################

def debye_length(ne,TeV):
    return NotImplemented

def plasma_frequency(ne):
    return NotImplemented

def thermal_velocity(TeV,species='e'):
    return NotImplemented

def mean_velocity(TeV,species='e'):
    return NotImplemented

def electron_ion_collision_frequency():
    return NotImplemented

def electron_electron_collision_frequency():
    return NotImplemented

###############################################################################
#%%                             Cross Section Fits
###############################################################################

def charge_exchange_xsec(TeV,species='Xe'):
    """
    Returns charge exchange cross section for the specified species in m^2 for 
    the specified ION temperature in eV.
    Applies only to ion - neutral collisions.
    Inputs:
        Ion temperature/energy in eV
        Ion/Neutral species (e.g. 'Xe')
    Outputs:
        charge exchange xsec m^2
        
    Refs: Miller et al. 2002 (Xe)
          Hause et al. 2013 (Kr)
    """
    consts={
            'Xe':[87.3,13.6],
            'Xe2+':[45.7,]
            'Ar':[0,0],
            'Kr':[80.7,14.7]}
    
    A,B = consts[species]
    
    return A*cc.angstrom**2 - B*np.log(TeV)*cc.angstrom**2
    

def goebel_electron_neutral_xsec(TeV):
    """ 
    Electron-neutral collision cross-section from Goebel's textbook.
    VALID ONLY FOR XENON
    
    Inputs: Electron temperature in eV
    Output: cross section in m^2
    """
    return (6.6E-19)*(((TeV/4)-0.1)/(1+(TeV/4)**(1.6)))

def goebel_ionization_xsec(TeV):
    """
    Electron-impact ionization cross-section from Goebel's textbook.
    VALID ONLY FOR XENON
    
    Inputs: Electron temperature in eV
    Output: cross section in m^2
    """
    return (1.0E-20)*((3.97+0.643*TeV-0.0368*TeV**2)*np.exp(-12.127/TeV))


###############################################################################
#%%                           Cross Section Import
###############################################################################

def _fetch_data(lines,match_index):
    """Private helper function for cross section creation"""
    separator = re.compile('-{5,}\n')
    
    #join all lines following match index
    bulk = ''.join(lines[match_index:])
    
    #split at separators, data is 1st grouping after match
    #split again for import to numpy
    data = re.split(separator,bulk)[1]        
    data = np.loadtxt(re.split('\n',data))
    print('\tImported level: '+str(data[0,0])+' eV')
    
    #rescale and spline data
    _scaling = np.max(data[:,1])
    data[:,1] = data[:,1]/_scaling
    
    _spline = splrep(data[:,0],data[:,1])
    
    return (lambda x: _scaling*splev(x,_spline))

def create_cross_section_spline(filename,xsec_type,chosen=None):
    """
    Finds the string associated with xsec_type in filename and extracts the
    numeric cross-section data following it.
    If multiple matches are found, user is prompted to select one or all of them.
    ALL combines the cross sections into a lumped value by splining each individually
    and then comibining to make a lumped spline.
    
    Inputs: 
            filename of lxcat data
            cross-section type (e.g. 'IONIZATION')
    Optional Inputs:
            chosen (number corresponding to selected cross section or 'ALL')
    Output: 
            spline representation function
    """
    with open(filename,'r') as f:
        lines = f.readlines()
        
    if xsec_type.upper() not in ['EXCITATION','IONIZATION','ELASTIC']:
        print('Must select a valid cross section type.')
        return None
        
    pattern = re.compile(xsec_type.upper())
    match_num = 0
    matches = {}
    
    #find each possible match and print a description
    for index,line in enumerate(lines):
        if re.match(pattern,line):
            match_num += 1
            print(str(match_num)+'.')
            print(lines[index+4].strip())
            print(lines[index+6].strip()+'\n')
            matches[match_num]=index
    
    #If no matches, print warning and return None      
    if len(matches)==0:
        print("WARNING: No Cross Section match found.")
        return None
    
    #If only a single match, import it directly
    elif len(matches)==1:
        print("Importing data...")
        return _fetch_data(lines,matches[1])
    
    else:
        print("Multiple matches found.")
        ########################## Input loop #################################
        failure = True
        while failure:
            if not chosen:
                chosen = input("Select cross section to import: (or enter ALL to lump)\n").upper()
        
            if chosen=='ALL':
                print("Importing and lumping all cross sections...")
                failure = False
            
            else:
                #Make sure the input number is valid
                try:
                    chosen = int(chosen)
                    
                    if (chosen not in matches.keys()):
                        raise ValueError
                        
                    failure = False
                    print("Importing cross section No."+str(chosen))
                    
                except ValueError:
                    chosen = None
                    print("Please enter a cross section number or ALL.")
        #######################################################################
        
        #If a single cross section was selected, import it and return
        if chosen!='ALL':
            return _fetch_data(lines,matches[chosen])
        #If ALL was selected, import each spline, then sum to lump
        else:
            splines={}
            for m in matches.keys():
                splines[m]=_fetch_data(lines,matches[m])
            
            return (lambda x: np.sum(sp(x) for sp in splines.values()))
            
            
###############################################################################
#%%                       Reaction Rate Integrals
###############################################################################

def reaction_rate():
    return NotImplemented

def beam_reaction_rate():
    return NotImplemented
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            