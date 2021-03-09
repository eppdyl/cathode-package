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
# Copyright (C) 2019-2020 Chris Wordingham, Pierre-Yves Taunay
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

import cathode.constants as cc
import numpy as np

Tgcorr = 3000 # Temperature at which the correlation was computed
Te_orifice_coeff = {'Ar': 
    {'2000': np.array([ 1.88931256, -0.01973419,  0.28725302,  0.79287866]),
     '3000': np.array([ 1.94138013, -0.02498979, 0.31979282,  0.93520214]),
     '4000': np.array([ 1.72255801, -0.02568448,  0.40125467,  1.25021734])
    },
    'Xe':
    {'2000': np.array([1.23039818, -0.00521554,  0.31250671,  0.42937491]),
     '3000': np.array([1.29009197, -0.0061753,   0.33723824,  0.50344054]),
     '4000': np.array([1.30035341, -0.00678322,  0.36503613,  0.59095683])
    }
}

def Te_insert(ng,ds,species):
    ''' Correlation to calculate the electron temperature in eV for xenon gas. 
    Assumes a gas temperature of 3000 K
    Inputs: 
        - ng: density (1/m3)
        - ds: diameter of tube (m)
    '''
    kbT = cc.kB * Tgcorr
    if species == 'Xe':
        ret = 1.20072 / (kbT * ng * ds * 1e2/cc.Torr)**0.35592 + 0.52523
    elif species == 'Ar':
        ret = 1.66426 /  (kbT * ng * ds * 1e2/cc.Torr)**0.38159 + 1.12521
    else:
        raise NotImplemented
    return ret

def TeK_insert(ng,ds,species):
    ''' Correlation to calculate the electron temperature in K for xenon gas. 
    Assumes a gas temperature of 3000 K
    Inputs: 
        - ng: density (1/m3)
        - ds: diameter of tube (m)
    '''
    return Te_insert(ng,ds,species)*cc.e/cc.Boltzmann

def compute_Te_orifice(Pd,cvec):
    a,b,c,d = cvec[0:]
#    print(cvec,a,b,c,d,Pd)
    
    return a / (Pd + b) ** c + d

def Te_orifice(ng,ds,TgK,species):
    ''' Correlation to calculate the electorn temperature in eV in the orifice
    Inputs:
        - ng: density (1/m3)
        - ds: diameter of tube (m)
        - TgK: neutral gas temperature (K)
    '''
    Pd = cc.kB * TgK * ng * ds * 1e2/cc.Torr
    cvec = Te_orifice_coeff[species][str(TgK)]
    
    TeV = compute_Te_orifice(Pd,cvec)
    
    return TeV
  
def TeK_orifice(ng,ds,TgK,species):
    return Te_orifice(ng,ds,TgK,species)*cc.e/cc.Boltzmann

def Lem(ng,ds,species):
    ''' Correlation to calculate the attachment length in m for xenon gas. 
    Assumes a gas temperature of 3000 K
    Inputs: 
        - ng: density (1/m3)
        - ds: diameter of insert (m)
    '''
    kbT = cc.kB * Tgcorr
    if species == 'Xe':
        ret = ds/2 * (0.72389 + 0.17565 / (kbT * ng * ds * 1e2/cc.Torr)**1.22140)
    elif species == 'Ar':
        ret = ds/2 * (0.71827 + 0.34198 / (kbT * ng * ds * 1e2/cc.Torr)**1.19716)
    else:
        raise NotImplemented
    return ret
