"""
cathode.models.siegfried_wilbur_simple

Defines the model equations and solution procedure for Siegfried and Wilbur's
0D hollow cathode model described in:
D. Siegfried and P. J. Wilbur, "An Investigation of Mercury Hollow Cathode
Phenomena," 13th International Electric Propulsion Conference, 1978

"""
import cathode.constants as cc
import numpy as np

from scipy.optimize import root


def goal_function(alpha,args):
    P,TeV,TiV,TgV,Sigi,Sig0,epsi = args 
    
    # Constants
    q = cc.e # Electron charge 
    me = cc.me # Electron mass
    h = cc.h # Planck's constant
    pi = np.pi # 3.14159...

    # Temperature ratios and multipliers
    tau = TgV/TeV
    etoT = epsi/TeV
    Teg = TgV * TeV**(3/2)

    # Factor of constants
    cfac = (2*pi*me)**(3/2) / h**3 * q**(5/2)

    # Ratio of sigmas
    s = Sigi/Sig0
        
    # Left-hand side
    lhs = alpha**(1+tau) / ((1-alpha)**tau * ( 1 + alpha/tau))
    rhs = 1/P * cfac * Teg * s**tau * np.exp(-etoT)
    goal = lhs-rhs

    return goal


def solve(eps_i, Sig_i, Sig_0, TeV, TgV, 
          P = None, mdot = None,orifice_diameter=None,
          solver_tol = 1E-8,solver_out = False):
    """
    Solves for the ionization fraction and electron density using:
        1. Choked orifice flow
        2. Two-temperature Saha equation
        3. Perfect gas law
    Iputs:
        1. eps_i: Ionization potential (eV)
        2. Sig_i, Sig_0: Internal partition function for the singly ionized
        state and neutral state, respectively (1)
        3. TeV: Electron temperature (eV) (can be a vector)
        4. TgV: Gas temperature (eV)
        5. mdot or P: mass flow rate (kg/s) or total pressure (Pa)
        6. orifice_diameter: Orifice radius (m)
    """   
    ### Sanity check
    if P == None and mdot == None:
        raise ValueError("Either P or mdot must be specified")
        
    if (P != None) and (mdot != None):
        raise ValueError("Only P or mdot can be specified. Not both")
    
    ### Create vectors
    alpha_out = []
    ne_out = []
    
    ### Algorithm
    for lTeV in TeV: 
        # TODO Calculate the pressure if necessary
        
        # Arguments for our goal function
        args = (P,TeV,TgV,TgV,Sig_i,Sig_0,eps_i)
        
        # Solving for the ionization fraction
        root_options = {'maxiter':int(1e5),'xtol':1e-8,'ftol':1e-8}
        alpha0 = 1e-3
            
        sol = root(goal_function,alpha0,args=args,
                method='lm',options=root_options)
        
        lalpha = sol.x
        lne = P/cc.e*1/(TeV + TgV*(1/lalpha-1))
        alpha_out.append(lalpha)
        ne_out.append(lne)

    alpha_out = np.array(alpha_out)
    ne_out = np.array(ne_out)

    return alpha_out, ne_out
