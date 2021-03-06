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
cathode.models.siegfried_wilbur

Defines the model equations and solution procedure for Siegfried and Wilbur's
0D hollow cathode model described in:
    - D. E.  Siegfried and P. J. Wilbur, "A model for mercury orificed hollow
    cathodes - Theory and experiment," AIAA journal,Vol. 22, No. 10, 1984,
    pp. 1405–1412.
    - D. E.  Siegfried and P. J. Wilbur, "Phenomenological Model Describing
    Orificed, Hollow Cathode Operation," AIAA Journal, Vol. 21, No. 1, 1983,
    pp. 5–6.
    - P. J. Wilbur, "Advanced Ion Thruster Research," Tech. Rep. CR-168340,
    NASA, 1984.
    - D. E.  Siegfried and P. J. Wilbur, "Studies on an experimental quartz
    tube hollow cathode," 14th International Electric Propulsion Conference,
    1979
    - D. E.  Siegfried, "A Phenomenological Model for Orificed Hollow
    Cathodes," Ph.D., Colorado State University, 1982.
"""
from itertools import product

import cathode.constants as cc
import cathode.collisions.mean_free_path as mfp
import numpy as np

from scipy.optimize import root

NUNK = 4 # Number of unknowns

def jth_rd(DRD, Tc, phi_wf, schottky=False, ne=1, phi_p=1, TeV=1):
    """
    Function: jth_rd
    Description: computers the thermionic current density, based on
    Richardson-Dushman's law, with an optional Schottky effect
    """
    kb = cc.Boltzmann
    q = cc.elementary_charge

    fac = DRD*Tc**2

    # If we consider the Schottky effect, calculate the effective work function
    if schottky:
        phi = phi_eff(ne, phi_p, TeV, phi_wf)

    jth = fac*np.exp(-q*phi/(kb*Tc))

    return jth


def phi_eff(ne, phi_p, TeV, phi_wf):
    """
    Function: phi_eff
    Description: computes the effective work function, based on a model of a
    thermionic double sheath by Prewett and Allen
    """
    q = cc.elementary_charge
    eps0 = cc.epsilon0

    # Cathode surface field
    Ec = np.sqrt( ne*q*TeV / eps0 )

    tmp = 2*np.sqrt(1 + 2*phi_p/TeV)
    tmp = tmp - 4

    Ec = Ec*np.sqrt(tmp)

    # Effective work function
    return phi_wf - np.sqrt(q/(4*np.pi*eps0)*Ec)


def goal_function(X, args):

    q = cc.elementary_charge
    kb = cc.Boltzmann

    # Rescale
    ne = X[0]*1e20
    ng = X[1]*1e20
    phi_p = X[2]
    Tc = X[3]*1e3

    goal = np.zeros(4)

    # Unpack arguments    
    dc, eps_i, mass, phi_wf, DRD, TeV, P, Id, lambda_pr, qth, weights = args

    # Effective length, areas
    rc = dc/2
    lpr = lambda_pr(ne, ng, phi_p)
    Leff = 2*lpr

    Ae = 2*np.pi*rc*Leff
    Ac = np.pi*rc**2
    As = 2*Ac + Ae

    # Power loss per unit length (W/m)
    # Function signature:
    # - Tc: wall temperature (K)
    # - Leff: emission length (m)
    # - dc: emitter diameter (m)
    qth_val = qth(Tc, Leff, dc) # W

    # Ion Current density, ion current
    ji = q*ne*np.sqrt(q*TeV/mass)
    Ii = ji*As

    # Electron current
    Ie = jth_rd(DRD, Tc, phi_wf, True, ne, phi_p, TeV)*Ae

    # Equations
    # Perfect gas law
    goal[0] = P - kb*(ne*(TeV*q/kb + Tc) + ng*Tc)

    # Current conservation
    goal[1] = Id-Ie-Ii

    # Insert surface power balance
    goal[2] = ji*Ae*(phi_p + eps_i - phi_wf)
    goal[2] -= qth_val
    goal[2] -= Ie*phi_eff(ne, phi_p, TeV, phi_wf)

    # Plasma power balance
    goal[3] = phi_p*Ie - eps_i * Ii - 5/2*TeV*Id

    for g, w in zip(goal, weights):
        g *= w

    return goal

def sw_pressure_correlation(mdot, Id, do, mass):
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

    Ptorr = mdot / (do*1e3)**2 * (c1 + c2*Id)*1e-3

    return Ptorr * cc.Torr

def solve(dc,
          eps_i, mass,
          phi_wf, DRD,
          TeV,
          X0=None,
          P=None, mdot=None, do=None,
          Id=None, Pfunc=sw_pressure_correlation,
          lambda_pr=mfp.lambda_pr_hg, qth=None,
          solver_tol=1E-8,
          weights=np.ones(NUNK)):
    """
    Solves for the electron and neutral densities, plasma potential, and
    neutral gas temperature. Uses:
        1. Perfect gas law
        2. Current balance
        3. Insert surface power balance
        4. Plasma volume power balance
    Inputs:
        1. Geometry: cathode diameter dc, orifice diameter do (optional).
        Both in m
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
        if len(Id) == len(mdot):
            cases = zip(Id, mdot)
        else:
            cases = product(Id, mdot)

        solvec = []
        idx = 0
        for lId, lmdot in cases:
            lP = Pfunc(lmdot, lId, do, mass)

            # Initial guess: densities and temperature are scaled
            if X0 is None:
                Tc0 = 1000 + 273.15 # Wall temperature, K
                ne0 = 2.5e20 # Electron density, 1/m3
                nc0 = 1/(kb*Tc0)*(lP - ne0*kb*(TeV*q/kb + Tc0)) # Neutral density
                # from perfect gas law, 1/m3
                phi_p0 = 7 # Plasma potential, V

                x0 = [ne0*1e-20, nc0*1e-20, phi_p0, Tc0*1e-3]
            else:
                ne0 = X0[0][idx]
                nc0 = X0[1][idx]
                phi_p0 = X0[2][idx]
                Tc0 = X0[3][idx]
                x0 = [ne0, nc0, phi_p0, Tc0]

            # Optimizer options
            root_options = {'maxiter':int(1e6),
                            'xtol':solver_tol, 'ftol':solver_tol}

            # Arguments
            dc = dc
            args = [dc,
                    eps_i, mass,
                    phi_wf, DRD,
                    TeV, lP, lId,
                    lambda_pr, qth, weights]

            # Solve!
            optimize_results = root(goal_function, x0, args=args,
                                    method='lm', options=root_options)

            # Extract and rescale results
            ne, nc, phi_p, Tc = optimize_results.x

            rescaled_results = [ne*1e20, nc*1e20, phi_p, Tc*1e3-273.15]

            solvec.append(rescaled_results)

            idx = idx + 1

    return np.array(solvec)
