"""
cathode.models.mandell_katz

Defines the model equations and solution procedure for Mandell and Katz's
0D hollow cathode model of the orifice described in:
- Mandell, M. J. and Katz, I., "Theory of Hollow Cathode Operation in Spot and 
Plume Modes," 30th AIAA/ASME/SAE/ASEE Joint Propulsion Conference & Exhibit, 
1994.
- Katz, I., Gardner, B., Jongeward, G., Patterson, M., and Myers, R., "A model 
of plasma contactor behavior in the laboratory,” 34th Aerospace Sciences 
Meeting and Exhibit, 1996.
- Katz, I., Gardner, B. M., Mandell, M. J., Jongeward, G. A., Patterson, M., 
and Myers, R. M., "Model of Plasma Contactor Performance," Journal of 
Spacecraft and Rockets, Vol. 34, No. 6, 1997.
- Katz, I., Mandell, M. J., Patterson, M., and Domonkos, M., "Sensitivity of 
Hollow Cathode Performance to Design and Operating Parameters," 35th 
AIAA/ASME/SAE/ASEE Joint Propulsion Conference & Exhibit, 1999. 
"""
import cathode.constants as cc
import numpy as np

from scipy.optimize import root
from itertools import product



def goal_function(X,args):
    
    q = cc.elementary_charge
    kb = cc.Boltzmann
    
    # Rescale
    ne = X[0]*1e20
    ng = X[1]*1e20
    phi_p = X[2]
    Tc = X[3]*1e3

    goal = np.zeros(4)
    
    # Unpack arguments    
    dc, eps_i, mass, phi_wf, DRD, TeV, P, Id, lambda_pr, qth = args
    
    # Effective length, areas
    rc = dc/2
    lpr = lambda_pr(ne,ng,phi_p)
    Leff = 2*lpr

    Ae = 2*np.pi*rc*Leff
    Ac = np.pi*rc**2
    As = 2*Ac + Ae

    # Power loss per unit length (W/m)
    qth_val = qth(Tc-273.15) # W/mm
    qth_val = qth_val*1e3 # W/m
    
    # Ion Current density, ion current
    ji = q*ne*np.sqrt(q*TeV/mass)
    Ii = ji*As

    # Electron current
    Ie = jth_rd(DRD,Tc,phi_wf,True,ne,phi_p,TeV)*Ae

    # Equations
    # Perfect gas law
    goal[0] = P - kb*(ne*(TeV*q/kb + Tc) + ng*Tc)

    # Current conservation
    goal[1] = Id-Ie-Ii

    # Insert surface power balance
    goal[2] = ji*Ae*(phi_p + eps_i - phi_wf) 
    goal[2] -= qth_val * Leff 
    goal[2] -= Ie*phi_eff(ne,phi_p,TeV,phi_wf)

    # Plasma power balance
    goal[3] = phi_p*Ie - eps_i * Ii - 5/2*TeV*Id

    return goal

def sw_pressure_correlation(mdot,Id,orifice_diameter,mass):
    """
    Function: sw_pressure_correlation
    Calculates the pressure correlation proposed by Siegfried and Wilbur
    """
    amu_mass = mass/cc.atomic_mass
    if amu_mass == cc.M.Hg:
        c1 = 13.7
        c2 = 7.82
    elif amu_mass == cc.M.Xe:
        c1 = 9.0
        c2 = 4.0
    elif amu_mass == cc.M.Ar:   
        c1 = 5.6
        c2 = 1.2

    Ptorr = mdot / (orifice_diameter*1e3)**2 * (c1 + c2*Id)*1e-3
    
    return Ptorr * cc.Torr

def solve(cathode_diameter, 
          eps_i, mass,
          phi_wf, DRD,
          TeV, 
          P = None, mdot = None, orifice_diameter = None,
          Id = None, Pfunc = sw_pressure_correlation,
          lambda_pr = hg_lambda_pr, qth = None,
          solver_tol = 1E-8,solver_out = False):
    """
    Solves for the electron and neutral densities, plasma potential, and
    neutral gas temperature. Uses:
        1. Perfect gas law
        2. Current balance
        3. Insert surface power balance
        4. Plasma volume power balance
    Inputs:
        1. Geometry: cathode_diameter, orifice_diameter (optional). Both in m
        2. Gas: ionization potential "eps_i" (eV), particle mass "mass" (kg)
        3. Emitter info: work function "phi_wf" (eV), Richardson-Dushman 
        constant for the material considered "DRD" (A/m^2)
        4. Experimental info: electron temperature "TeV" (eV)
        5. Operating conditions: either the total pressure "P" (Pa) or the 
        discharge current "Id" (A) and mass flow rate "mdot" (milli-eqA)
        6. Necessary functions: a pressure correlation "Pfunc" if Id and mdot 
        are used to determine the total pressure, an evaluation of the mean 
        free path for primary electrons "lambda_pr", and an estimate of the 
        heat loss "qth"
    """   
    ### Sanity checks...
    if (P is None) and (mdot is None) and (Id is None):
        raise ValueError("Either [P] or [mdot AND Id] must be specified")
        
    if (P is not None) and ((mdot is not None) or (Id is not None)):
        raise ValueError("Only [P] or [mdot and Id] can be specified. Not both")
   
    ### Constants
    q = cc.elementary_charge # Electron charge 
    kb = cc.Boltzmann

    ### Id case
    if (Id is not None) and (mdot is not None):
        if(len(Id) == len(mdot)):
            cases = zip(Id,mdot)
        else:
            cases = product(Id,mdot)
        
        solvec = []
        for lId,lmdot in cases:
            lP = Pfunc(lmdot,lId,orifice_diameter,mass)
            
            # Initial guess: densities and temperature are scaled
            Tc0 = 1050 + 273.15 # Wall temperature, K
            ne0 = 1.5e20 # Electron density, 1/m3
            nc0 = 1./(kb*Tc0)*(lP - ne0*kb*(TeV*q/kb + Tc0)) # Neutral density 
            # from perfect gas law, 1/m3
            phi_p0 = 7 # Plasma potential, V
            
            x0 = [ne0*1e-20,nc0*1e-20,phi_p0,Tc0*1e-3]
    
            # Optimizer options
            root_options = {'maxiter':int(1e6),
                            'xtol':solver_tol,'ftol':solver_tol}

            # Arguments
            dc = cathode_diameter
            args = [dc, eps_i, mass, phi_wf, DRD, TeV, lP, lId, lambda_pr, qth]
            
            # Solve!
            optimize_results = root(goal_function,x0,args=args,
                                    method='lm',options = root_options)
            
            # Extract and rescale results
            ne,nc,phi_p,Tc = optimize_results.x
            
            rescaled_results = [ne*1e20,nc*1e20,phi_p,Tc*1e3-273.15]
            
            solvec.append(rescaled_results)
            
    return np.array(solvec)




# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:51:24 2017

@author: Arbiter
"""

from __future__ import division # Because reasons.
import numpy as np
from scipy.optimize import fsolve,root
import matplotlib.pyplot as plt
import scipy.constants as cs


### Constants
cm = cs.centi
angstrom = 1E-10
e = cs.elementary_charge
me = cs.electron_mass
kB = cs.k
pi = np.pi
eps0 = cs.epsilon_0

### Cathode info
r = 0.0368*cm
L = 0.076*cm

#r = 0.014*cm
#L = 0.075*cm
### Gas info
E_i = 12.2
E_rad = 10.0
T_n = 0.1 
M = cs.u*131.293 # Mass of Xe, kg

### Experimental data
T_e_ins = 0.8 # Insert electron temp., eV
#T_e_ins = 0.46 # Insert electron temp., eV



def sigma_iz(T_e):
    return (3.97+0.643*T_e-0.0368*T_e**2)*np.exp(-12.127/T_e)*angstrom**2

def sigma_rad(T_e):
    return 1.93E-19*np.exp(-11.6/T_e)/np.sqrt(T_e)

def sigma_cond(n_e,T_e,N_n):
    return eps0*omega_p(n_e)**2/(nu_ei(n_e,T_e)+nu_en(N_n,T_e))

def coulombLog(n_e,T_e):
    return 30 - (1/2)*np.log(n_e/(T_e**3))

def nu_ei(n_e,T_e):
    return 2.9E-12*n_e*coulombLog(n_e,T_e)/(T_e**(3/2))

def nu_en(N_n,T_e):
    return 5E-19*N_n*np.sqrt(e*T_e/me)

def omega_p(n_e):
    return np.sqrt(n_e*e**2/(eps0*me))

def resistance(n_e,T_e,N_n):
    return (L/(pi*r**2))/sigma_cond(n_e,T_e,N_n)

def J_e(n_e,T_e):
    return e*n_e*np.sqrt(e*T_e/(2*pi*me))

def J_i(n_e,T_e):
    return np.sqrt(me/M)*J_e(n_e,T_e)

def ion_production(n_e,T_e,N_n):
    return pi*r**2*L*4*sigma_iz(T_e)*J_e(n_e,T_e)*N_n

def ion_loss(n_e,T_e):
    return 2*pi*r*(r+L)*J_i(n_e,T_e)

def ionization_loss(n_e,T_e,N_n):
    return E_i*ion_production(n_e,T_e,N_n)

def radiation_loss(n_e,T_e,N_n):
    return pi*r**2*L*E_rad*4*sigma_rad(T_e)*J_e(n_e,T_e)*N_n

def convection_loss(T_e,T_e_ins,I_d):
    return I_d*(T_e-T_e_ins)

def ohmic_heating(n_e,T_e,N_n,I_d):
    return I_d**2*resistance(n_e,T_e,N_n)

def power_balance(n_e,N_n,T_e,T_e_ins,I_d):
    #print  ohmic_heating(n_e,T_e,N_n),ionization_loss(n_e,T_e,N_n),radiation_loss(n_e,T_e,N_n),convection_loss(T_e)
    return ohmic_heating(n_e,T_e,N_n,I_d)-ionization_loss(n_e,T_e,N_n)-radiation_loss(n_e,T_e,N_n)-convection_loss(T_e,T_e_ins,I_d)

def ion_balance(n_e,T_e,N_n):
    return ion_production(n_e,T_e,N_n)-ion_loss(n_e,T_e)

def flow_balance(n_e,N_n,T_e,F):
    return pi*r**2*e*N_n*np.sqrt(e*T_n/(2*pi*M))-(0.0718*F-pi*r**2*J_i(n_e,T_e))

def zerofun(x,T_e_ins,I_d,F):
    n_e=x[0]*1E21
    T_e=x[1]
    N_n=x[2]*1E22
    goal=np.zeros(3)
    goal[0]=power_balance(n_e,N_n,T_e,T_e_ins,I_d)/1E6
    goal[1]=ion_balance(n_e,T_e,N_n)*100
    goal[2]=flow_balance(n_e,N_n,T_e,F)*100
    
    return goal

def solve(do, Lo,
          eps_i, eps_x, mass,
          TeV_ins, TgV,
          Id, mdot,
          sig_iz=sig_iz_xe_mk, sig_ex=sig_ex_xe_mk,
          convection='MK',
          solver_tol = 1E-8,solver_out = False):
    """
    Solves for the electron and neutral densities and electron temperature in 
    the orifice.
    Uses:
        - Mass conservation
        - Charge conservation
        - Plasma power balance

    Inputs:
        1. Geometry: orifice diameter "do" and orifice length "Lo" (m)
        2. Gas: ionization potential "eps_i" (eV), excitation potential "eps_x"
        (eV), particle mass "mass" (kg)
        3. Emitter info: None
        4. Experimental info: electron insert temperature "TeV_ins" (eV),
        neutral gas temperature "TgV" (eV)
        5. Operating conditions: discharge current "Id" (A), mass flow rate 
        "mdot" (milli-eqA)
        6. Necessary functions: ionization and excitaiton cross-sections 
        "sig_iz" and "sig_ex" (m2). Defaults to the fits proposed by Mandell
        and Katz for xenon.
        7. Other: "convection" chooses either the Mandell and Katz convection
        term without a factor of 5/2 ('MK') or the correct convection term with
        the factor of 5/2 ('corrected')
    """   
    


### Operating condition
Idvec = np.arange(1.,10.,0.1) # A
mdotvec = np.array([1,6,10]) # sccm

### Storage for all solutions
# 3 mass flow rates
# size(Idvec) currents
# store ne,ng,Te,convergence
solvec = np.zeros([3,np.size(Idvec),4])


#x0 = np.array([1,2.5,1])
x0vec = np.array( [ [1,2.5,1],[2,1.8,2],[0.5,1.7,1]])
TeInsvec = np.array([0.848866477452,0.779318325967,0.791571666652])

#mdot_idx = 0
#Id_idx = 0
#filename = 'mandell_solvec_mdot-'
#for mdot in np.nditer(mdotvec):
#		print mdot
#		x0 = x0vec[mdot_idx]
#		filename += str(mdot)
#		Id_idx = 0
#
#		#T_e_ins = TeInsvec[mdot_idx]
#		T_e_ins = 0.8
#
#		for Id in np.nditer(Idvec):
#				data = (T_e_ins,Id,mdot) 
#				optimize_results = root(zerofun,x0,data,method='lm',options={'maxiter':1000000,'xtol':1e-8,'ftol':1e-8})
#				n_e,T_e,N_n = optimize_results.x
#				n_e *= 1E21
#				N_n *= 1E22
#				solvec[mdot_idx,Id_idx,:] = [n_e,N_n,T_e,optimize_results.success]
#				Id_idx += 1
#
#		np.save(filename,solvec[mdot_idx,:,:])
#		mdot_idx += 1
#		filename = 'mandell_solvec_mdot-'
#
#np.save('mandell_solvec-all',solvec)
x0 = np.array([1.0,1.6,1])
data = (T_e_ins,3.26,6)
optimize_results = root(zerofun,x0,data,method='lm',options={'maxiter':1000000,'xtol':1e-8,'ftol':1e-8})
n_e,T_e,N_n = optimize_results.x
n_e *= 1e21
N_n *= 1e22
print n_e,T_e,N_n

#ion_output = pi*r**2*J_i(n_e,T_e)
#ion_eff = ion_output/(resistance(n_e,T_e,N_n)*I_d**2)
#utilization = ion_output/(0.0718*F)
#voltage_drop = resistance(n_e,T_e,N_n)*I_d

#print('Electron Density\tElectron Temperature\tNeutral Density')
#print(str(n_e)+'\t'+str(T_e)+'\t\t'+str(N_n))
#print('Orifice Vd\t\tIon Output\t\tIon Output/Power\tUtilitzation')
#print(str(voltage_drop)+'\t\t'+str(ion_output)+'\t\t'+str(ion_eff)+'\t'+str(utilization))

