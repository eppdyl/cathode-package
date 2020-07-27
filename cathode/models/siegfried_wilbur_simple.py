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
# Copyright (C) 2017-2020 Chris Wordingham, Pierre-Yves Taunay
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
        args = [P,lTeV,TgV,TgV,Sig_i,Sig_0,eps_i]
        
        # Solving for the ionization fraction
        root_options = {'maxiter':int(1e5),'xtol':1e-8,'ftol':1e-8}
        alpha0 = 1e-3
            
        sol = root(goal_function,alpha0,args=args,
                method='lm',options=root_options)
        
        lalpha = sol.x
        lne = P/cc.e*1/(lTeV + TgV*(1/lalpha-1))
        alpha_out.append(lalpha)
        ne_out.append(lne)

    alpha_out = np.array(alpha_out)
    ne_out = np.array(ne_out)

    return alpha_out, ne_out
