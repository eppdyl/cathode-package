# -*- coding: utf-8 -*-
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
                             atomic_mass,epsilon_0,gas_constant)

###################################################################
#  Physical constants/Energy conversions/Mass conversions
###################################################################

#centimeter (cm to m)
cm = 1.0E-2

#millimeter (mm to m)
mm = 1.0E-3

#angstrom (angstrom to m)
angstrom = 1.0E-10

#electron volt (eV to J)
eV = electron_volt

#permittivity of free space (SI)
epsilon0 = epsilon_0

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

#Unified Atomic mass Unit (kg)
u = atomic_mass 

#Unified Atomic mass Unit (eV/c^2)
u_eV = u/(eV/(c**2))

#Temperature conversions
Kelvin2eV = kB/e
eV2Kelvin = e/kB

###################################################################
#  Pressure/Flow conversions
###################################################################

#Torr (Torr to Pascal)
Torr = torr

#Pascals to eV/m^3
Pa2eVm3 = (1/e)

#Torr to eV/m^3
Torr2eVm3 = Torr/e

###################################################################
#  Atomic Data
###################################################################

#Elemental mass numbers embedded in "static" class
class M:
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
