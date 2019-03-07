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
We use the typical plasma physics Coulomb logarithm expression, which differs
only by 0.3% from the expression proposed by Mandell and Katz.
"""
from itertools import product

import cathode.constants as cc
import cathode.physics as cp
import cathode.collisions.frequency as nu
import numpy as np

from scipy.optimize import root

def resistance(ne, TeV, ng, L, r):
    '''
    Function: resistance
    Calculate the resistance of a plasma column of length L and radius r
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - L,r: length and radius of plasma column (m)
    Outputs:
        - Resistance (Ohms)
    '''
    # Resistivity
    eta = nu.nu_ei(ne, TeV) + nu.nu_en_xe_mk(ng, TeV, 'constant')
    eta *= cc.me / (ne * cc.e**2)

    # Resistance
    Rp = L/(np.pi*r**2) * eta

    return Rp

def ohmic_heating(ne, TeV, ng, Id, L, r):
    '''
    Function: ohmic_heating
    Calculate the Ohmic heating power of a plasma column of length L and
    radius r
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - Id: discharge current (A)
        - L,r: length and radius of plasma column (m)
    Outputs:
        - Ohmic heating power (W)
    '''
    Rp = resistance(ne, TeV, ng, L, r)
    return Rp*Id**2

def ion_production(ne, TeV, ng, L, r, sigma_iz):
    '''
    Function: ion_production
    Calculate the total amount of ions produced in the volume by direct-impact
    ionization.
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - L,r: length and radius of plasma column (m)
        - sigma_iz: ionization cross-section function (m2)
    Outputs:
        - Ion production (1/s)
    '''
    vol = np.pi * r**2 * L # Volume 

    sig_iz = sigma_iz(TeV) # Cross-section term

    ve = cp.mean_velocity(TeV, 'e')

    return vol * sig_iz * cc.e * ve * ng

def ionization_loss(ne, TeV, ng, L, r, eps_i, sigma_iz):
    '''
    Function: ionization_loss
    Calculates the total amount of power spent in ionization.
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - L,r: length and radius of plasma column (m)
        - eps_i: ionization energy (eV)
        - sigma_iz: ionization cross-section function (m2)
    Outputs:
        - Ionization power loss (W)
    '''
    ip = ion_production(ne,TeV,ng,L,r,sigma_iz)

    il = eps_i * ip

    return il

def excitation_loss(ne, TeV, ng, L, r, eps_x, sigma_ex):
    '''
    Function: excitation_loss
    Calculates the total amount of power spent in excitation.
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - L,r: length and radius of plasma column (m)
        - eps_x: excitation energy (eV)
        - sigma_iz: ionization cross-section function (m2)
    Outputs:
        - Excitation power loss (W)
    '''
    vol = np.pi * r**2 * L # Volume 

    sig_ex = sigma_ex(TeV) # Cross-section term

    ve = cp.mean_velocity(TeV, 'e')

    el = eps_x * vol * sig_ex * cc.e * ve * ng

    return el

def convection_loss(TeV, TeV_ins, Id, convection='MK'):
    '''
    Function: convection_loss
    Calculates the total convection losses.
    Inputs:
        - TeV: electron temperature (eV)
        - TeV_ins: insert electron temperature (eV)
        - Id: discharge current (A)
        - convection: a string describing either the original Mandell and Katz
        equations ('MK') or a flag ('corrected') to use the correct value 
        for the factor in front of the convection term (should be 5/2)
    Outputs:
        - Convection loss (W)
    '''
    ret = Id*(TeV-TeV_ins)

    if convection == 'MK':
        fac = 1.0
    elif convection == 'corrected':
        fac = 5/2
    else:
        raise ValueError

    return fac*ret

def power_balance(ne, TeV, ng, args):
    '''
    Function: power_balance
    Difference of power input to and output from the plasma volume. Should be
    zero (input = output.)
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - args: list of arguments necessary for the subfunctions called
    Outputs:
        - Power balance (W)
    '''
    TeV_ins, Id, L, r, eps_i, eps_x, sigma_iz, sigma_ex, convection = args

    oh = ohmic_heating(ne, TeV, ng, Id, L, r)
    il = ionization_loss(ne, TeV, ng, L, r, eps_i, sigma_iz)
    rl = excitation_loss(ne, TeV, ng, L, r, eps_x, sigma_ex)
    cl = convection_loss(TeV, TeV_ins, Id, convection)

    return oh-il-rl-cl


def J_i(ne, TeV, M):
    ve = cp.mean_velocity(TeV, 'e')

    return 1/4 * cc.e * np.sqrt(cc.me/M) * ve

def ion_loss(ne, TeV, L, r, M):
    '''
    Function: ion_loss
    Total loss of ions to the walls and inflow/outflow
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - L,r: length and radius of plasma column (m)
        - M: mass of an ion (kg)
    Outputs:
        - Total ion loss (1/s)
    '''
    Aeff = 2*np.pi*r*(r+L)
    ji = J_i(ne, TeV, M)

    return Aeff * ji

def ion_balance(ne, TeV, ng, args):
    '''
    Function: ion_balance
    Difference of ion production and loss. Should be zero (creation=loss)
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - args: list of arguments necessary for the subfunctions called
    Outputs:
        - Ion species balance (1/s)
    '''
    L, r, M, sigma_iz = args

    ip = ion_production(ne, TeV, ng, L, r, sigma_iz)
    il = ion_loss(ne, TeV, L, r, M)

    return ip-il

def flow_balance(ne, TeV, ng, args):
    '''
    Function: flow_balance
    Difference of "charge flow rate" between inlet and outlets of the orifice.
    Should be zero (what comes in = what comes out)
    Inputs:
        - ne: plasma density (1/m3)
        - TeV: electron temperature (eV)
        - ng: neutral density (1/m3)
        - args: list of arguments necessary for the subfunctions called
    Outputs:
        - "Charge" flow rate balance (C/s = A)
    '''

    TgV, r, M, mdot = args

    mdot_A = mdot * cc.sccm2eqA # Flow rate (eq-A)
    Ao = np.pi*r**2 # Orifice area (m2)

    # Neutral particle flux (A)
    vg = np.sqrt(cc.me/M) * cp.mean_velocity(TgV, 'e')
    gam_g = 1/4 * cc.e * ng * vg
    gam_g *= Ao

    # Ion flux (A)
    gam_i = J_i(ne,TeV,M)
    gam_i *= Ao

    # Balance
    ret = gam_g + gam_i
    ret -= mdot_A

    return ret

def goal_function(x, args):
    # Unpack inputs
    ne = x[0]*1E21
    TeV = x[1]
    ng = x[2]*1E22

    Id, mdot, Te_ins = args

    # Goal vector
    goal=np.zeros(3)

    # Create the argument list for each balance
    args_pb = args['pb']
    args_ib = args['ib']
    args_fb = args['fb']

    # Compute the balances
    goal[0] = power_balance(ne, TeV, ng, args_pb)
    goal[1] = ion_balance(ne, TeV, ng, args_ib)
    goal[2] = flow_balance(ne, TeV, ng, args_fb)

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
          solver_tol = 1e-8,solver_out = False):
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
        "mdot" (sccm)
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
        Te0 = 2.5 # eV
        ng0 = 1e22 # 1/m3

        # M&K do not give their initial guesses and solver is very sensitive
        # to I.V.
        # TODO: Find a better solution here
        if lmdot == 1:
            x0 = np.array([1,2.5,1])
        elif lmdot == 6:
            x0 = np.array([2,1.8,2])
        elif lmdot == 10:
            x0 = np.array([0.5,1.7,1])
        else:
            x0 = np.array([ne0*1e-21,Te0,ng0*1e-22])

        root_options = {'maxiter':int(1e6),
                        'xtol':solver_tol,'ftol':solver_tol}

        # Arguments
        ro = do/2

        args_pb = [lTeV_ins,lId,
                   Lo,ro,
                   eps_i,eps_x,
                   sig_iz,sig_ex,
                   convection]
        args_ib = [Lo,ro,mass,sig_iz]
        args_fb = [TgV,ro,mass,lmdot]

        args = {}
        args['pb'] = args_pb
        args['ib'] = args_ib
        args['fb'] = args_fb
        # Solve!
        optimize_results = root(goal_function,x0,args=args,
                                method='lm',options = root_options)

        # Extract and rescale results
        ne,TeV,ng = optimize_results.x

        rescaled_results = [lId,lmdot,lTeV_ins,ne*1e21,TeV,ng*1e22]

        solvec.append(rescaled_results)

    solvec = np.array(solvec)

    return solvec
