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
        ret =  1.3 / (kbT * ng * ds * 1e2/cc.Torr)**0.34 + 0.48
    elif species == 'Ar':
#        ret = 10**0.4588/(kbT * ng * ds * 1e2/cc.Torr)**0.30
        ret = 1.91 / (kbT * ng * ds * 1e2/cc.Torr)**0.341 + 0.945
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
        ret = ds/2 * (0.75 + 1/np.log(kbT * ng / cc.Torr * ds * 1e2 + 3)**6) 
    elif species == 'Ar':
        ret = ds/2 * (0.86 + 0.613/np.log(kbT * ng / cc.Torr * ds * 1e2 + 1.89)**6) 
    else:
        raise NotImplemented
    return ret