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

def goal_function(x,args):
    # Unpack inputs
    n_e=x[0]*1E21
    T_e=x[1]
    N_n=x[2]*1E22
    
    Id, mdot, Te_ins = args
    
    # Compute goal vector
    goal=np.zeros(3)

    goal[0] = power_balance(n_e,N_n,T_e,T_e_ins,I_d)
    goal[1]=ion_balance(n_e,T_e,N_n)   
    goal[2]=flow_balance(n_e,N_n,T_e,F)
    
    # Rescale goal vector
    goal[0] /= 1e6
    goal[1] *= 100
    goal[2] *= 100
    
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
    
    if(len(Id) == len(mdot) == len(TeV_ins)):
        cases = zip(Id,mdot,TeV_ins)
    else:
        cases = product(Id,mdot,TeV_ins)

    solvec = []
    for lId,lmdot,lTeV_ins in cases:
        # Initial guess: densities are scaled
        ne0 = 1e21 # 1/m3
        Te0 = 2 # eV
        ng0 = 1e22 # 1/m3
        
        x0 = np.array([ne0*1e-21,Te0,ng0*1e-22])
    

        root_options = {'maxiter':int(1e6),
                        'xtol':solver_tol,'ftol':solver_tol}

        # Arguments
        args = [lId,lmdot,lTeV_ins]
        
        # Solve!
        optimize_results = root(goal_function,x0,args=args,
                                method='lm',options = root_options)
        
        # Extract and rescale results
        ne,TeV,ng = optimize_results.x
        
        rescaled_results = [ne*1e21,Te0,ng*1e22]
        
        solvec.append(rescaled_results)        

#### Operating condition
#Idvec = np.arange(1.,10.,0.1) # A
#mdotvec = np.array([1,6,10]) # sccm
#
#### Storage for all solutions
## 3 mass flow rates
## size(Idvec) currents
## store ne,ng,Te,convergence
#solvec = np.zeros([3,np.size(Idvec),4])
#
#
##x0 = np.array([1,2.5,1])
#x0vec = np.array( [ [1,2.5,1],[2,1.8,2],[0.5,1.7,1]])
#TeInsvec = np.array([0.848866477452,0.779318325967,0.791571666652])
#
##mdot_idx = 0
##Id_idx = 0
##filename = 'mandell_solvec_mdot-'
##for mdot in np.nditer(mdotvec):
##		print mdot
##		x0 = x0vec[mdot_idx]
##		filename += str(mdot)
##		Id_idx = 0
##
##		#T_e_ins = TeInsvec[mdot_idx]
##		T_e_ins = 0.8
##
##		for Id in np.nditer(Idvec):
##				data = (T_e_ins,Id,mdot) 
##				optimize_results = root(zerofun,x0,data,method='lm',options={'maxiter':1000000,'xtol':1e-8,'ftol':1e-8})
##				n_e,T_e,N_n = optimize_results.x
##				n_e *= 1E21
##				N_n *= 1E22
##				solvec[mdot_idx,Id_idx,:] = [n_e,N_n,T_e,optimize_results.success]
##				Id_idx += 1
##
##		np.save(filename,solvec[mdot_idx,:,:])
##		mdot_idx += 1
##		filename = 'mandell_solvec_mdot-'
##
##np.save('mandell_solvec-all',solvec)
#x0 = np.array([1.0,1.6,1])
#data = (T_e_ins,3.26,6)
#optimize_results = root(zerofun,x0,data,method='lm',options={'maxiter':1000000,'xtol':1e-8,'ftol':1e-8})
#n_e,T_e,N_n = optimize_results.x
#n_e *= 1e21
#N_n *= 1e22
#print n_e,T_e,N_n
#
##ion_output = pi*r**2*J_i(n_e,T_e)
##ion_eff = ion_output/(resistance(n_e,T_e,N_n)*I_d**2)
##utilization = ion_output/(0.0718*F)
##voltage_drop = resistance(n_e,T_e,N_n)*I_d
#
##print('Electron Density\tElectron Temperature\tNeutral Density')
##print(str(n_e)+'\t'+str(T_e)+'\t\t'+str(N_n))
##print('Orifice Vd\t\tIon Output\t\tIon Output/Power\tUtilitzation')
##print(str(voltage_drop)+'\t\t'+str(ion_output)+'\t\t'+str(ion_eff)+'\t'+str(utilization))

