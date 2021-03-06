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
Created on Wed May 31 19:14:18 2017

cathode.constants
Defines physical constants for use in cathode models.
Static class M is used to store elemental atomic masses in amu.

@author: cjw
"""

#define constants:
from scipy.constants import (speed_of_light,Boltzmann,electron_volt,
                             elementary_charge,electron_mass,torr,
                             atomic_mass,epsilon_0,gas_constant,h,pi)
from scipy.special import jn_zeros
from scipy.interpolate import splrep,splev

import numpy as np
import os
import math

###############################################################################
#       Physical constants/Energy conversions/Mass conversions
###############################################################################

#centimeter (cm to m)
cm = 1.0E-2

#millimeter (mm to m)
mm = 1.0E-3

#angstrom (angstrom to m)
angstrom = 1.0E-10

#Bohr radius
a0 = 0.52917721067*angstrom

#electron volt (eV to J)
eV = electron_volt

#permittivity of free space (SI)
epsilon0 = epsilon_0

# permeability of free space (SI)
mu0 = 4*np.pi*1e-7

#electron charge (SI)
e = elementary_charge

#speed of light (SI)
c = speed_of_light

#electron mass (SI)
me = electron_mass

#electron mass (eV/c^2)
me_eV = electron_mass/(eV/(c**2))

#Boltzmann constant (SI)
kB = Boltzmann

#Boltzmann constant (eV/K)
kB_eV = kB/eV

#Gas constant (J/(mol*K))
R0 = gas_constant

#Richardson-Dushman Constant (A/(m^2-K^2))
A0 = 4.0*me*pi*kB**2*e/h**3

#Unified Atomic mass Unit (kg)
u = atomic_mass

#Unified Atomic mass Unit (eV/c^2)
u_eV = u/(eV/(c**2))

#Temperature conversions
Kelvin2eV = kB/e
eV2Kelvin = e/kB

###############################################################################
#                           Pressure/Flow conversions
###############################################################################

#Torr (Torr to Pascal)
Torr = torr

#Pascals to eV/m^3
Pa2eVm3 = (1/e)

#Torr to eV/m^3
Torr2eVm3 = Torr/e

#sccm to equivalent amperes
sccm2eqA = (e*760.0*Torr2eVm3*cm**3/60.0)/(273.15*Kelvin2eV)

###############################################################################
#                                   Math
###############################################################################

BesselJ01 = jn_zeros(0,1)[0]

###############################################################################
#                               Atomic Data
###############################################################################
#Elemental mass numbers embedded in "static" class
class M:
    #electron (for use in species access)
    e = electron_mass/u # electron mass in amu, M.species returns mass in kg

    #Period 1
    H=1.007940
    He=4.002602

    #Period 2
    Li=6.941000
    Be=9.012182
    B=10.811000
    C=12.010700
    N=14.006700
    O=15.999400
    F=18.998403
    Ne=20.179700

    #Period 3
    Na=22.989769
    Mg=24.305000
    Al=26.981539
    Si=28.085500
    P=30.973762
    S=32.065000
    Cl=35.453000
    Ar=39.948000

    #Period 4
    K=39.098300
    Ca=40.078000
    Sc=44.955912
    Ti=47.867000
    V=50.941500
    Cr=51.996100
    Mn=54.938045
    Fe=55.845000
    Co=58.933195
    Ni=58.693400
    Cu=63.546000
    Zn=65.380000
    Ga=69.723000
    Ge=72.640000
    As=74.921600
    Se=78.960000
    Br=79.904000
    Kr=83.798000

    #Period 5
    Rb=85.467800
    Sr=87.620000
    Y=88.905850
    Zr=91.224000
    Nb=92.906380
    Mo=95.960000
    Tc=98.000000
    Ru=101.070000
    Rh=102.905500
    Pd=106.420000
    Ag=107.868200
    Cd=112.411000
    In=114.818000
    Sn=118.710000
    Sb=121.760000
    Te=127.600000
    I=126.904470
    Xe=131.293000

    #Period 6
    Cs=132.905452
    Ba=137.327000

    #Lanthanides
    La=138.905470
    Ce=140.116000
    Pr=140.907650
    Nd=144.242000
    Pm=145.000000
    Sm=150.360000
    Eu=151.964000
    Gd=157.250000
    Tb=158.925350
    Dy=162.500000
    Ho=164.930320
    Er=167.259000
    Tm=168.934210
    Yb=173.054000
    Lu=174.966800

    Hf=178.490000
    Ta=180.947880
    W=183.840000
    Re=186.207000
    Os=190.230000
    Ir=192.217000
    Pt=195.084000
    Au=196.966569
    Hg=200.590000
    Tl=204.383300
    Pb=207.200000
    Bi=208.980400
    Po=209.000000
    At=210.000000
    Rn=222.000000

    #Period 7
    Fr=223.000000
    Ra=226.000000

    #Actinides
    Ac=227.000000
    Th=232.038060
    Pa=231.035880
    U=238.028910
    Np=237.000000
    Pu=244.000000
    Am=243.000000
    Cm=247.000000
    Bk=247.000000
    Cf=251.000000
    Es=252.000000
    Fm=257.000000
    Md=258.000000
    No=259.000000
    Lr=262.000000

    Rf=267.000000
    Db=268.000000
    Sg=271.000000
    Bh=272.000000
    Hs=270.000000
    Mt=276.000000
    Ds=281.000000
    Rg=280.000000
    Cn=285.000000
    Uut=284.000000
    Uuq=289.000000
    Uup=288.000000
    Uuh=293.000000
    Uus='?'
    Uuo=294.000000

    # Allow for dictionary-like access of element data
    @classmethod
    def species(self,name):
        if name in self.__dict__:
            return self.__dict__[name]*u
        else:
            raise KeyError

# Lennard-Jones parameters for some select elements
# Sorted as [sigma,kb/eps]
class LJ:
    Hg=[2.898e-10,851.]

    # Allow for dictionary-like access of element data
    @classmethod
    def species(self,name):
        if name in self.__dict__:
            return np.array(self.__dict__[name])
        else:
            raise KeyError

    @classmethod
    def viscosity(self,name):
        ### Hg viscosity
        folder = os.path.dirname(__file__)
        folder = os.path.join(folder,'resources')
        fname = os.path.join(folder,'collision-integrals-lj.csv')
        data = np.genfromtxt(fname,delimiter=',',names=True)

        Mass = M.species(name)
        sig_lj, kbeps = self.species(name)

        Tlj = data['Tstar']
        omega22_data = splrep(Tlj,data['omega22'])
        omega23_data = splrep(Tlj,data['omega23'])
        omega24_data = splrep(Tlj,data['omega24'])

        Tvec = np.arange(300,4001,10)

        omega_hs = lambda l,s: math.factorial(s+1)/2. * (1. - 1./2.*(1. + (-1)**l)/(1.+l))*np.pi*sig_lj**2
        omega_hs22 = omega_hs(2,2)
        omega_hs23 = omega_hs(2,3)
        omega_hs24 = omega_hs(2,4)

        fac = np.sqrt(Boltzmann*Tvec/(np.pi*Mass))
        omega22 = fac * splev(Tvec/kbeps,omega22_data) * omega_hs22
        omega23 = fac * splev(Tvec/kbeps,omega23_data) * omega_hs23
        omega24 = fac * splev(Tvec/kbeps,omega24_data) * omega_hs24

        b11 = 4.* omega22
        b12 = 7.*omega22 - 2*omega23
        b22 = 301./12.*omega22 - 7*omega23 + omega24

        mu_lj = 5.*Boltzmann*Tvec/2.*(1./b11 + b12**2./b11 * 1./(b11*b22-b12**2.))

        return Tvec,mu_lj



def R_specific(species):
    #specific gas constant in J/(kg-K) for the species given
    return kB/M.species(species)


