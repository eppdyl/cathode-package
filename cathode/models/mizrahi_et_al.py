"""
cathode.models.mizrahi_et_al

Defines the model equations and solution procedure for Mandell and Katz's
0D hollow cathode model of the orifice described in:
- Mizrahi, J. P., Vekselman, V., Krasik, Y., & Gurovich, V., "0-D Plasma Model 
for Orificed Hollow Cathodes," 32nd International Electric Propulsion 
Conference, IEPC-2011-334, 2011.
- Mizrahi, J., Vekselman, V., Gurovich, V., & Krasik, Y. E., "Simulation of 
Plasma Parameters During Hollow Cathodes Operation," Journal of Propulsion and 
Power, 28(5), 1134–1137, 2012.

"""
import numpy as np
from scipy.optimize import root

import cathode.constants as cc

#from cathode.physics import charge_exchange_xsec 
#from cathode.models.flow import viscosity
#from cathode.constants import eV2Kelvin

#### Experimental data
#T_i = 0.4 # Ion temperature, eV
#T_e_ins = 0.8 # Insert electron temp., eV
#delta = 0.2 # Free parameter, ratio of neutral density downstream to upstream
#beta = (1.-delta)/(1.+delta)
##xi = 1.7e-4 # Viscosity, Pa.s
#xi = viscosity(0.4*eV2Kelvin,units='Pa-s') # Viscosity, Pa.s

#sigma_cex = charge_exchange_xsec(T_i,'Xe') 

### Cross sections
def sigma_iz(T_e):
    return (3.97+0.643*T_e-0.0368*T_e**2)*np.exp(-12.127/T_e)*angstrom**2

def sigma_rad(T_e):
    return 1.93E-19*np.exp(-11.6/T_e)/np.sqrt(T_e)

def sigma_cond(n_e,T_e,N_n):
    return eps0*omega_p(n_e)**2/(nu_ei(n_e,T_e)+nu_en(N_n,T_e))

def sigma_en(T_e):
		return (0.25*T_e - 0.1) / ( 1. + (0.25*T_e)**(1.6)) * 6.6e-19

### Coulomb logarithm / plasma
def omega_p(n_e):
    return np.sqrt(n_e*e**2/(eps0*me))

def lambdad(n_e,T_e):
		return np.sqrt(eps0*T_e/(n_e*e))

def impact_param(T_e):
		ue = np.sqrt(8.*e*T_e/(pi*me))
		return e**2. / (4.*pi*eps0*me*ue**2.)

def coulombLog(n_e,T_e):
		return np.log(lambdad(n_e,T_e)/impact_param(T_e))

def ambi_diff(T_e,N_n):
		return (1. + T_e / T_i) * e * T_i / (M*nu_in(N_n))

### Collision frequencies
def nu_ei(n_e,T_e):
    return 2.9E-12*n_e*coulombLog(n_e,T_e)/(T_e**(3/2))

def nu_en(N_n,T_e):
    return sigma_en(T_e)*N_n*np.sqrt(e*T_e/me)

def nu_in(N_n):
		#print N_n * sigma_cex * np.sqrt( e * T_i / M)
		return N_n * sigma_cex * np.sqrt( e * T_i / M)

### Helper functions
def resistance(n_e,T_e,N_n):
    return (L/(pi*r**2))/sigma_cond(n_e,T_e,N_n)

def average_vel(N_n):
		ug = beta*e*T_n*N_n*r**2./(4.*xi*L)
		return ug

def J_e(n_e,T_e):
    return e*n_e*np.sqrt(8*e*T_e/(pi*me))

def J_i(n_e,T_e):
    return np.sqrt(me/M)*J_e(n_e,T_e)

### Physical functions
def ion_production(n_e,T_e,N_n):
    return pi*r**2*L*sigma_iz(T_e)*J_e(n_e,T_e)*N_n

def ion_loss(n_e,T_e):
    return 2*pi*r*(r+L)*J_i(n_e,T_e)

def ionization_loss(n_e,T_e,N_n):
    return E_i*ion_production(n_e,T_e,N_n)

def radiation_loss(n_e,T_e,N_n):
    return pi*r**2.*L*E_rad*sigma_rad(T_e)*J_e(n_e,T_e)*N_n

def convection_loss(T_e,T_e_ins,I_d):
    return 5./2.*I_d*(T_e-T_e_ins)

def ohmic_heating(n_e,T_e,N_n,I_d):
    return I_d**2.*resistance(n_e,T_e,N_n)

### Balance equations
def power_balance(n_e,N_n,T_e,T_e_ins,I_d):
    return ohmic_heating(n_e,T_e,N_n,I_d)-ionization_loss(n_e,T_e,N_n)-radiation_loss(n_e,T_e,N_n)-convection_loss(T_e,T_e_ins,I_d)

def ion_balance(n_e,T_e,N_n):
		Da = ambi_diff(T_e,N_n)
		ue = np.sqrt(8.*e*T_e/(pi*me))
#		print Da,ue
		return N_n * sigma_iz(T_e) * ue - 2.*Da/r**2.*(1. + 2.*(r/L)**2.) 

def flow_balance(n_e,N_n,T_e,F):
		ug = average_vel(N_n)
		return F*7.43583e-10*131.293-M*pi*r**2.*(N_n + n_e)*ug




#def zerofun(x,T_e_ins,I_d,F):
def goal_function(x,args):
		ne = x[0]*1E21
		TeV = x[1]
		ng = x[2]*1E22

		goal = np.zeros(3)
		goal[0] = power_balance(n_e,N_n,T_e,T_e_ins,I_d)
		goal[1] = ion_balance(n_e,T_e,N_n)
		goal[2] = flow_balance(n_e,N_n,T_e,F)

		return goal

def solve(do, Lo,
          eps_i, eps_x, mass,
          TeV_ins, TgV,
          Id, mdot,
          delta,
          sig_iz=sig_iz_xe, sig_ex = sig_ex_xe, sig_cex = sig_cex_xe,
          nu_en = nu_en_xe, mu = mu_xe,
          solver_tol = 1e-8,solver_out = False):
    print("Test")

#### Operating condition
#mdot = 92e-3 /7.174486e-2 
#Idvec = np.arange(1.,5.01,0.1)  
#
#### Storage for all solutions
## 1 mass flow rates
## size(Idvec) currents
## store ne,ng,Te,convergence
#solvec = np.zeros(np.size(Idvec))
#solvec.resize((np.size(Idvec),4))
#
#### Actual solving
## Scaling
## - ne / 1e21
## - Te / 1
## - nc / 1e22
#x0 = np.array([1,2.5,1])
#
#Id_idx = 0
#for Id in np.nditer(Idvec):
#    print Id
#    data = (T_e_ins,Id,mdot) 
#    optimize_results = root(zerofun,x0,data,method='lm',options={'maxiter':100000,'xtol':1e-8,'ftol':1e-8})
#    n_e,T_e,N_n = optimize_results.x
#    n_e *= 1E21
#    N_n *= 1E22
#    solvec[Id_idx,:] = [n_e,N_n,T_e,optimize_results.success]
#    Id_idx += 1
#
#
#alpha = solvec[:,0]/(solvec[:,0] + solvec[:,1]) * 100
#
#
#print solvec
#
#np.save('mizrahi_sw-cathode',solvec)


