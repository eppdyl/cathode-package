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

def hg_lambda_pr(ne,ng,phi_p):
    """
     Function: hg_lambda_pr
     Description: computes the mean free path for the primary electrons.
     Applicable to mercury only! Based on Siegfried and Wilbur's computer model
     
     TODO: Add citation for computer model
     Inputs: 
     - ne: Electron density (1/m3)
     - ng: Gas density (1/m3)
     - phi_p: Plasma potential (V)
    """
    inv = 6.5e-17*ne/phi_p**2 + 1e3*ng*phi_p / (2.83e23 - 1.5*ng)
    
    return 1/inv


def jth_rd(DRD,Tc,phi_wf,schottky=False,ne=1,phi_p=1,TeV=1):
    """
    Function: jth_rd
    Description: computers the thermionic current density, based on
    Richardson-Dushman's law, with an optional Schottky effect
    """
    kb = cc.Boltzmann
    q = cc.elementary_charge
     
    fac = DRD*Tc**2

    # If we consider the Schottky effect, calculate the effective work function
    if(schottky):
        phi = phi_eff(ne,phi_p,TeV,phi_wf)

    jth = fac*np.exp(-q*phi/(kb*Tc))

    return jth


def phi_eff(ne,phi_p,TeV,phi_wf):
    """
    Function: phi_eff
    Description: computes the effective work function, based on a model of a
    thermionic double sheath by Prewett and Allen    
    """ 
    q = cc.elementary_charge
    eps0 = cc.epsilon0
    
    # Cathode surface field
    Ec = np.sqrt( ne*q*TeV / eps0 );

    tmp = 2*np.sqrt(1 + 2*phi_p/TeV);
    tmp = tmp - 4;

    Ec = Ec*np.sqrt(tmp);

    # Effective work function
    return phi_wf - np.sqrt(q/(4*np.pi*eps0)*Ec) 


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
