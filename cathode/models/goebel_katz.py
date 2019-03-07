# "cathode" Python package
# Version: 1.0
# A package of various cathode models that have been published throughout the
# years. Associated publication:
# Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
# review of hollow cathode modeling: 0-D models," Journal of Propulsion and
# Power, in preparation.
#
# Copyright (C) 2019 Christopher J. Wordingham and Pierre-Yves C. R. Taunay
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https:/www.gnu.org/licenses/>.
#
# Contact info: https:/github.com/pytaunay
#
"""
Created on Tue Jun 6 17:21 2017
cathode.models.goebel_katz
Defines the model equations and solution procedure for Goebel and Katz's 0D
hollow cathode model described in "Fundamentals of Electric Propulsion: Hall 
and Ion Thrusters" (2008)

"""
import cathode.constants as cc
import cathode.physics as cp
import cathode.models.flow as flow
import cathode.collisions.cross_section as xsec
import cathode.collisions.frequency as nu

import numpy as np
from scipy.optimize import fsolve #,root
from scipy.special import jn

@np.vectorize
def ambipolar_diffusion_model(rc, ng, TiV, species='Xe'):
    """
    Implementation of Goebel's ambipolar diffusion model used for determination
    of the orifice- or insert-region plasma electron temperature based on the
    cathode geometry, neutral density and ion temperature (usually taken to be
    some constant value 2-4x the insert temperature).
    Inputs:
        - rc: cathode radius (m)
        - ng: neutral density (#/m^3)
        - TiV: ion temperature  (eV)
    Optional Input:
        - species, defaults to 'Xe'
            -NOTE: currently only works for xenon
    Output:
        Electron temperature (eV)
    """
    #NOTE: THIS SECTION WILL CURRENTLY ONLY WORK FOR XENON!!
    lhs = lambda TeV: (rc/cc.BesselJ01)**2 * (ng * xsec.ionization_xe_mk(TeV) *
                                              cp.mean_velocity(TeV, 'e'))

    rhs = lambda TeV: ((cc.e/cc.M.species(species)) * (TiV+TeV) /
                       (ng * xsec.charge_exchange(TiV, species) *
                        cp.thermal_velocity(TiV, species)))

    goal = lambda x: lhs(x) - rhs(x)

    TeV = fsolve(goal, x0=5.0)

    return TeV

@np.vectorize
def resistivity(ne, ng, TeV):
    """
    Returns the plasma resistivity, \eta, as a function of plasma density,
    neutral density, and electron temperature.
    Inputs:
        - ne: plasma density (#/m^3)
        - ng: neutral gas density (#/m^3)
        - TeV: electron temperature (eV)
    Output:
        plasma resistivity, Ohm-m
    """
    nu_ei = nu.nu_ei(ne, TeV)
    nu_en = nu.nu_en_xe_mk(ng, TeV) # Use the variable cross section
    nu_m = nu_ei + nu_en

    ret = cc.me / (ne * cc.e**2) * nu_m

    return ret


def thermionic_current_density(Tw, phi_wf, D=cc.A0):
    """
    Returns the thermionic electron emission current density as defined by the
    Richardson-Dushman equation with the option of incorporating an experimentally
    determined temperature coefficient through the use of D.
    Inputs:
        - Tw: wall temperature (K)
        - phi_wf: work function (eV)
    Optional Input:
        - D: experimental coefficient (A/(m-K)^2). Defaults to the universal 
        constant A0
    Output:
        current density (A/m^2)
    """
    return D*Tw**2*np.exp(-cc.e*phi_wf/(cc.kB*Tw))

def ion_current_density(ne, TeV, species='Xe'):
    return cc.e*ne*cp.bohm_velocity(TeV, species)

def plasma_resistance(L, d, ne, ng, TeV):
    return L*resistivity(ne, ng, TeV)/(cc.pi*(d/2)**2)

def random_electron_current_density(ne, TeV, phi_s):
    return cc.e*ne*cp.mean_velocity(TeV, 'e') * np.exp(-phi_s/TeV)/4

def heat_loss(Tw=None, method='fixed', curve=None):
    if (method == 'fixed'):
        return 13
    if (method == 'spline'):
        return curve(Tw)

def insert_plasma_power_balance(ne, Tw, phi_s, Id, TeV, D, phi_wf, eps_i, L, d,
                                ng): #may want to add species later
    Aemit = cc.pi*L*d

    lhs = (thermionic_current_density(Tw, phi_wf, D)*Aemit * phi_s +
           plasma_resistance(L, d, ne, ng, TeV) * Id**2
          )

    rhs = (ion_current_density(ne, TeV) * Aemit * eps_i
           + 5/2*TeV*Id
           + (2*TeV + phi_s)*
           random_electron_current_density(ne, TeV, phi_s) * Aemit
          )

    ret = lhs-rhs
    return ret

def current_balance(ne, Tw, phi_s, Id, TeV, D, phi_wf, eps_i, L, d,
                    ng):
    Aemit = cc.pi*L*d

    lhs = Id

    rhs = (thermionic_current_density(Tw, phi_wf, D) * Aemit +
           ion_current_density(ne, TeV) * Aemit -
           random_electron_current_density(ne, TeV, phi_s) * Aemit)

    ret = lhs-rhs
    return ret

def emitter_power_balance(ne, Tw, phi_s, Id, TeV, D, phi_wf, eps_i, L, d,
                          ng):
    Aemit = cc.pi*L*d

    lhs = (heat_loss(Tw) +
           thermionic_current_density(Tw, phi_wf, D) * phi_wf * Aemit
          )

    rhs = (ion_current_density(ne, TeV)*
           (eps_i + phi_s + 1/2*TeV - phi_wf)*Aemit +
           (2*TeV + phi_wf)*
           random_electron_current_density(ne, TeV, phi_s) * Aemit
          )

    ret = lhs-rhs
    return ret

def zerofun(x, args):
    ne = x[0]*1E21
    Tw = x[1]*1E3
    phi_s = x[2]

    #unpack arguments
    Id, TeV, D, phi_wf, eps_i, L, d, ng = args

    goal = np.zeros(3)
    goal[0] = insert_plasma_power_balance(ne, Tw, phi_s, Id, TeV, D, phi_wf,
                                          eps_i, L, d, ng)
    goal[1] = current_balance(ne, Tw, phi_s, Id, TeV, D, phi_wf, eps_i, L, d,
                              ng)
    goal[2] = emitter_power_balance(ne, Tw, phi_s, Id, TeV, D, phi_wf, eps_i, L,
                                    d, ng)

    goal = goal/10

    print('GOAL FUN:',goal)
    print('ARGS:',x)

    return goal

def sheath_voltage(Id, TeV, phi_wf, L, d, ne, ng, h_loss=heat_loss()):
    Rp = plasma_resistance(L, d, ne, ng, TeV)

    t1 = h_loss/Id
    t2 = 5/2 * TeV + phi_wf
    t3 = Rp * Id

    phi_s = t1+t2-t3
    return phi_s

def average_plasma_density_model(Id, TeV, phi_wf, L, d, ne, ng,
                                 phi_p, eps_i, h_loss=heat_loss()):

    phi_s = sheath_voltage(Id, TeV, phi_wf, L, d, ne, ng, h_loss)
    Rp = plasma_resistance(L, d, ne, ng, TeV)
    f_n = np.exp(-(phi_p-phi_s)/TeV) #edge-to-average ratio as defined by Goebel

    Aemit = L*cc.pi*d
    Vemit = L*cc.pi*d**2/4

    ### Calculate average plasma density
    # Numerator
    num = (Rp*Id**2 - 5/2*TeV*Id + phi_s*Id)

    # Denominator
    ve = cp.mean_velocity(TeV, species='e')

    t1 = 1/4 * cc.e *f_n * TeV * ve
    t1 *= Aemit * np.exp(-phi_s/TeV)

    t2 = cc.e *ng * Vemit * (eps_i + phi_s)*xsec.ionization_xe_mk(TeV) * ve

    den = t1 + t2

    # Results
    ne_bar = num/den

    return ne_bar, phi_s

def orifice_plasma_density_model(Id, TeV, TeV_insert, L, d, ne, ng, eps_i):

    Rp = plasma_resistance(L, d, ne, ng, TeV)
    Vori = cc.pi*d**2*L/4

    ve = cp.mean_velocity(TeV,'e')


    num = Rp*Id**2 - 5/2*Id*(TeV-TeV_insert)
    den = cc.e * ng * xsec.ionization_xe_mk(TeV) * ve * eps_i * Vori

    ne_bar = num/den
    return ne_bar


def solve(Id, Lo, do, Lc, dc,
          mdot, TgK, Pout,
          eps_i, phi_wf, phi_p, h_loss=heat_loss(),
          solver_tol = 1E-8,
          solver_out = False,
          verbose=False):
    """
    Solves for the average electron density and electron temperature in both
    the insert and orifice. Also outputs the insert and orifice pressure, from
    which the neutral gas density can be deduced.
    Uses:
        - Ambipolar diffusion for TeV
        - Charge conservation
        - Plasma power balance
        - Emitter power balance

    Inputs:
        1. Geometry: orifice diameter "do", orifice length "Lo" (m), insert
        diameter "dc" (m)
        2. Gas: ionization potential "eps_i" (eV), particle mass "mass" (kg).
        3. Emitter info: work function phi_wf (eV). The Richardson-Dushman
        constant D is assumed to be 120e4 (A/(m-K)^2).
        4. Experimental info: plasma potential "phi_p", neutral gas temperature
        "TgK" (K), emission length "Lc" (m).
        5. Operating conditions: discharge current "Id" (A), mass flow rate
        "mdot" (sccm)
        6. Necessary functions: ionization, electron-neutral collision, and
        charge-exchange cross-sections. They are all defined internally and
        default to the fits proposed by Mandell and Katz. A Poiseuille flow
        model is used to deduce the insert pressure from a specified outlet
        pressure "Pout" (Torr). The heat loss for the emitter power balance is
        required as well and is defined as h_loss. It defaults to an the NSTAR
        '13 W' value if nothing is specified.
        7. Other: solver tolerance "solver_tol" and verbosity "verbose".
        Setting "verbose" to True generates a lot of output on the terminal.
    """

    if verbose:
        print('-------------------INSERT-----------------------')
    #use the orifice dimensions and the flow rate to get P_ins
    P_insert_downstream = flow.poiseuille_flow(Lo, do, mdot, TgK, Pout)
    P_insert_upstream = flow.poiseuille_flow(Lc, dc, mdot, TgK,
                                             P_insert_downstream)

    P_insert = (P_insert_downstream + P_insert_upstream)/2

    if verbose:
        print('Pressure:\t\t\t{:.3f} Torr (upstream)\n\t\t\t\t{:.3f} Torr (downstream)\n\t\t\t\t{:.3f} Torr (average)'.format(
            P_insert_upstream,P_insert_downstream,P_insert))

    ng = P_insert * cc.Torr2eVm3 / (TgK*cc.Kelvin2eV)

    TeV = ambipolar_diffusion_model(dc/2, ng, TgK*cc.Kelvin2eV)[0]

    if verbose:
        print('Electron Temperature:\t\t{:.3f} eV'.format(TeV))

    #first guess value for ne
    #sheath voltage at initial step
    ne_bar = 1.5E21
    phi_s = sheath_voltage(Id, TeV, phi_wf, Lc, dc, ne_bar, ng ,h_loss)
    delta = 1.0

    while delta >= solver_tol:
        phi_s_old = phi_s
        ne_old = ne_bar
        ne_bar, phi_s = average_plasma_density_model(Id, TeV, phi_wf, Lc, dc,
                                                     ne_old, ng, phi_p,
                                                     eps_i, h_loss)
        delta = np.max(np.abs([1-phi_s/phi_s_old, 1-ne_bar/ne_old]))
        if solver_out:
            print(delta, ne_bar, phi_s)

    avg_to_peak = 2*jn(1, cc.BesselJ01)/cc.BesselJ01

    if verbose:
        print('Plasma Density:\t\t\t{:.3E} /m^3'.format(ne_bar))
        print('Peak Density:\t\t\t{:.3E} /m^3'.format(ne_bar/avg_to_peak))
        print('Sheath Voltage:\t\t\t{:.3f} V'.format(phi_s))

    P_orifice_downstream = Pout
    P_orifice_upstream = P_insert_downstream

    P_orifice = (P_orifice_upstream + P_orifice_downstream)/2

    ng_o = P_orifice*cc.Torr2eVm3/(TgK*cc.Kelvin2eV)

    if verbose:
        print('-------------------ORIFICE----------------------')
        print('Pressure:\t\t\t{:.3f} Torr (upstream)\n\t\t\t\t{:.3f} Torr (downstream)\n\t\t\t\t{:.3f} Torr (average)'.format(
                P_orifice_upstream,P_orifice_downstream,P_orifice))

    TeV_o = ambipolar_diffusion_model(do/2, ng_o, TgK*cc.Kelvin2eV)[0]

    if verbose:
        print('Electron Temperature:\t\t{:.3f} eV'.format(TeV_o))

    solve_fun = lambda ne_o: ne_o - orifice_plasma_density_model(Id, TeV_o, TeV,
                                                                 Lo, do, ne_o,
                                                                 ng_o, eps_i)
    ne_bar_orifice = fsolve(solve_fun, 1E18)[0]

    if verbose:
        print('Plasma Density:\t\t\t{:.3E} /m^3'.format(ne_bar_orifice))
        print('Peak Density:\t\t\t{:.3E} /m^3'.format(ne_bar_orifice/avg_to_peak))


    return P_insert, TeV, ne_bar, phi_s, P_orifice, TeV_o, ne_bar_orifice
