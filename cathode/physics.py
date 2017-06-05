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
import cathode.constants as cc

###############################################################################
#                         Basic Plasma Physics
###############################################################################

def debye_length(ne,TeV,species='e'):
    """
    Returns the debye length, sqrt(eps0*k*T/(e^2 ne)) for the species specified.
    Inputs:
        Number density (1/m^3)
        Temperature (eV)
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        plasma frequency (rad/s)
    """
    return np.sqrt((cc.epsilon0*cc.e*TeV)/(cc.e**2*ne))

def plasma_frequency(ne,species='e'):
    """
    Returns the plasma frequency, sqrt(e^2 ne/(eps0* m)) for the species specified.
    Inputs:
        number density (1/m^3)
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        plasma frequency (rad/s)
    """
    return np.sqrt((cc.e**2)*ne/(cc.epsilon0*cc.M.species(species)))

def thermal_velocity(TeV,species='e'):
    """
    Returns the thermal velocity, sqrt(kB T/m) for the species specified.
    Inputs:
        Temperature in eV
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        thermal velocity in m/s
    """
    return np.sqrt((cc.e*TeV)/(cc.M.species(species)))

def mean_velocity(TeV,species='e'):
    """
    Returns the distribution-averaged species velocity assuming a Maxwellian 
    distribution at a temperature of TeV (in electron volts).
    Inputs:
        Temperature in eV
    Optional Input:
        species, string specifying standard abbreviation for elemental species
            - DEFAULTS TO ELECTRON
    Output:
        mean velocity in m/s
    """
    return np.sqrt((8.0*cc.e*TeV)/(np.pi*cc.M.species(species)))

@np.vectorize
def coulomb_log(ne,TeV,collision_type='ei'):
    """
    Returns the Coulomb logarithm value as a function of plasma density and 
    electron temperature.  Collision type can be either electron-ion (default)
    or electron-electron.
    Inputs:
        plasma number density (1/m^3)
        plasma electron temperature (eV)
    Optional Inputs:
        collision type string ('ei' or 'ee')
    Output:
        Value of Coulomb Logarithm
        
    Ref: NRL Plasma Formulary (values translated from CGS to SI)
    """
    if collision_type == 'ei':
        return (23.0 - 0.5*np.log(1E-6*ne/(TeV**3.0))) 
    elif collision_type == 'ee':
        return (23.5 - 0.5*np.log(1E-6*ne/(TeV**(5.0/2.0)))-np.sqrt(1E-5 + (np.log(TeV)-2.0)**2/16.0))

def electron_ion_collision_frequency():
    return NotImplemented

def electron_electron_collision_frequency():
    return NotImplemented

###############################################################################
#                             Cross Section Fits
###############################################################################

@np.vectorize
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
          Nichols and Witteborn 1966 (Ar,N2)
    """
    consts={
            'Xe':[87.3,13.6],
            'Xe2+':[45.7,8.9],
            'Ar':[7.49,0.73],
            'N2':[6.48,0.24],
            'Kr':[80.7,14.7],
            'Kr2+':[44.6,9.8]}
    
    A,B = consts[species]
    
    #special case for Argon and N2 fits:
    if species == 'Ar' or species == 'N2':
        return (A*cc.angstrom - B*cc.angstrom*np.log(TeV))**2
    else:
        return (A*cc.angstrom**2 - B*np.log(TeV)*cc.angstrom**2)
    

def goebel_electron_neutral_xsec(TeV):
    """ 
    Electron-neutral collision cross-section from Goebel's textbook.
    VALID ONLY FOR XENON
    
    Inputs: Electron temperature in eV
    Output: cross section in m^2
    """
    return (6.6E-19)*(((TeV/4.0)-0.1)/(1.0+(TeV/4.0)**(1.6)))

def goebel_ionization_xsec(TeV):
    """
    Electron-impact ionization cross-section from Goebel's textbook.
    VALID ONLY FOR XENON
    
    Inputs: Electron temperature in eV
    Output: cross section in m^2
    """
    return (1.0E-20)*((3.97+0.643*TeV-0.0368*TeV**2)*np.exp(-12.127/TeV))


###############################################################################
#                           Cross Section Import
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
            cross-section type (e.g. any of these: 'IONIZATION','Excitation','elastic')
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
#                     Reaction Rate Integrals
###############################################################################

@np.vectorize
def reaction_rate(xsec_spline,TeV,Emin,Emax,output_xsec=False):
    """
    Returns the reaction rate for the process with cross section xsec_spline
    for Maxwellian electrons with temperature TeV.  Emin and Emax specify the 
    minimum and maximum energy for integration (Emin should be the threshold
    energy for the process, Emax should be the useful limit of the spline)
    Inputs:
        Cross section spline, created by create_cross_section_spline()
        Electron temperature, eV
        Minimum energy, eV (threshold for process of interest)
        Maximum energy, eV (limit of data or maximum extrapolation for spline)
    Optional Input:
        output_xsec can be set to true to return a tuple of (reaction rate, cross section)
            Defaults to FALSE, -> outputs only reaction rate unless specified
    Outputs:
        if output_xsec is true, (reaction rate, cross section) (m^3/s, m^2)
        otherwise, (reaction rate) (m^3/s)
    """
    #normalization factor for reaction rate integral
    normalization = (8.0*np.pi*cc.e**2.0/np.sqrt(cc.me))/(
            (2.0*np.pi*cc.e*TeV)**(3.0/2.0))
    
    #define integrand lambda functions
    energy_integrand = lambda E: E*xsec_spline(E)*np.exp(-E/TeV)
    flux_integrand = lambda E: E*np.exp(-E/TeV)
    
    #integrate (note that these return the error estimate as the second output)
    energy_integral = quad(energy_integrand,Emin,Emax,epsabs=1.0E-30)
    flux_integral = quad(flux_integrand,0.0,Emax,epsabs=1.0E-30)
    
    #reaction rate
    K = normalization*energy_integral[0]
    
    #Maxwellian-averaged cross section
    xsec_avg = energy_integral[0]/flux_integral[0]
    
    #if specified, output both the cross section and reaction rate
    if output_xsec:
        return K,xsec_avg
    else:
        return K    
    

@np.vectorize
def beam_reaction_rate(xsec_spline,Ebeam):
    """
    Returns the reaction rate for monoenergetic beam electrons and cross section
    described by xsec_spline.
    Inputs:
        Cross section spline created by create_cross_section_spline()
        Beam energy in eV
    Output:
        Reaction rate (m^3/s)
    """
    
    #beam electron velocity
    v = np.sqrt(2.0*cc.e*Ebeam/cc.me)
    
    return v*xsec_spline(Ebeam)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            