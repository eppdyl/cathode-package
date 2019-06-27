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

import cathode.constants as cc
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cathode.models.flow as cmf

mu0 = 4*np.pi*1e-7 # Permeability of vac.

from scipy.interpolate import splrep,splev
import math


def plot_pi_products(cat,pdf,df,C,opti_vec,Ymin,Ymax):
    ### Hg viscosity
    data = np.genfromtxt('pressure_theory/collision-integrals-lj.csv',delimiter=',',names=True)
    
    MHg = cs.M.species('Hg')
    
    Tlj = data['Tstar']
    omega22_data = splrep(Tlj,data['omega22'])
    omega23_data = splrep(Tlj,data['omega23'])
    omega24_data = splrep(Tlj,data['omega24'])
    
    Tvec = np.arange(300,4001,10)
    
    sig_lj = 2.5e-10
    omega_hs = lambda l,s:  math.factorial(s+1)/2. * (1. - 1./2.*(1. + (-1)**l)/(1.+l))*np.pi*sig_lj**2
    omega_hs22 = omega_hs(2,2)
    omega_hs23 = omega_hs(2,3)
    omega_hs24 = omega_hs(2,4)
    #np.sqrt(cs.Boltzmann*Tvec/(pi*MXe)) *
    
    kbeps = 1522.
    
    omega22 = np.sqrt(cs.Boltzmann*Tvec/(np.pi*MHg))* splev(Tvec/kbeps,omega22_data) * omega_hs22
    omega23 = np.sqrt(cs.Boltzmann*Tvec/(np.pi*MHg))*splev(Tvec/kbeps,omega23_data) * omega_hs23
    omega24 = np.sqrt(cs.Boltzmann*Tvec/(np.pi*MHg))*splev(Tvec/kbeps,omega24_data) * omega_hs24
    
    b11 = 4.* omega22
    b12 = 7.*omega22 - 2*omega23
    b22 = 301./12.*omega22 - 7*omega23 + omega24
    
    mu_lj = 5.*cs.Boltzmann*Tvec/2.*(1./b11 + b12**2./b11 * 1./(b11*b22-b12**2.))


    exp_vec = opti_vec[0:]
    
    # Get the unique species for each cathode
    # Each index should operate on a single species...
    # species_uniq = np.unique(pdf.mass[cat])
    # mass = cs.M.species(species_uniq[0])
    colors = df.colors_mf[cat] 
    markers = df.markers_mf[cat]
    

    # Plot P / (mu0 * Id**2 / (pi*do**2)) as function of mdot / Id
    plt.figure(1)

    # How many times does a unique mass flow rate appear?
    mdot_index = Counter(pdf.mdot[cat])
    mdot_index = sorted(mdot_index.items())

    cnt = 0
    a = ( 5/3 * cs.gas_constant * 1e3 / pdf.mass[cat] * (pdf.Tw[cat] + 273.15)*3)**(0.5) # Speed of sound

    alpha = exp_vec[0]
    beta = exp_vec[1]
    gamma = exp_vec[2]
    delta = exp_vec[3]
    eps = exp_vec[4]
    zeta = exp_vec[5]    

    for mdot,mdot_num in mdot_index:
        # If we have more than two entries

        # Grab the correct data
        idx = (pdf.mdot[cat] == mdot) 
        
#        print(mdot/cs.sccm2eqA,mdot_num,pdf.P[cat][idx]*101325/760*1e-3)

        mdot_SI = pdf.mdot[cat][idx] * (pdf.mass[cat][idx] * cs.atomic_mass/cs.e) 
        do_SI = pdf.do[cat][idx] * 1e-3
        Lo_SI = pdf.Lo[cat][idx] * 1e-3
        M_SI = pdf.mass[cat][idx]*cs.atomic_mass
        P_SI = pdf.P[cat][idx] * 101325/760
        P_gd = mdot_SI * a[idx] / do_SI**2
        P_mag = mu0 * pdf.Id[cat][idx]**2 / do_SI**2     
        T_SI = (pdf.Tw[cat][idx] + 273.15)*3                  
        # x-correlation: six Pi prod   
        # P/(mdot * a / do**2)
        PI1 = P_SI / P_mag

        PI2 = pdf.do[cat][idx] / pdf.dc[cat][idx]
            
        PI3 = pdf.do[cat][idx] / pdf.Lo[cat][idx]

        PI4 = (pdf.mdot[cat][idx] / pdf.Id[cat][idx])**2
        PI4 *= M_SI * do_SI / (mu0 * cs.e**2)
        
        ## (mdot * do) / (M * a);
        PI5 = P_gd/P_mag

        
        ## P_iz / P_gd
        PI6 = cs.e*pdf.eiz[cat][idx] / (do_SI**2 * Lo_SI)
        PI6 /= P_mag
        PI6 *= 1/PI3
        
        # Re
        PI7 = reynolds_number(mdot_SI,do_SI,T_SI,pdf.mass[cat][idx],Tvec,mu_lj)

        X = C*PI2**alpha
        X *= PI3**beta
        X *= PI4**gamma
        X *= PI5**delta
        X *= PI6**eps
        X *= PI7**zeta

        Y = PI1
        
        # plot
        if(mdot_num > 2):
            # Plot the data
#            plt.loglog(X,Y,markerfacecolor=colors[cnt],marker=markers[cnt],markeredgecolor='k',linestyle='')
            plt.loglog(X,Y,markerfacecolor=colors[cnt],marker='o',markeredgecolor='k',linestyle='')
#            plt.plot(X,Y,markerfacecolor='k',marker='o',markeredgecolor='k',linestyle='')
            
            # Can also do a fit with more than 2 points
            Xvec = np.arange(np.min(np.log10(X)),np.max(np.log10(X)),0.01)
            p = np.polyfit(np.log10(X),np.log10(Y),1)
            
            # Plot the linear fit
#            plt.loglog(10**Xvec,10**np.polyval(p,Xvec),linestyle='--',color=colors[cnt])
            
            #print(mdot,p)
            
            # Plot an arbitrary X = Y line
            #plt.loglog(X,X)
            
            cnt = cnt + 1 
        
        else:       
            plt.loglog(X,Y,markerfacecolor=colors[0],marker='o',markeredgecolor='k',linestyle='') 
#            plt.loglog(X,Y,markerfacecolor='k',marker='o',markeredgecolor='k',linestyle='') 
#             plt.plot(X,Y,markerfacecolor='k',marker='o',markeredgecolor='k',linestyle='') 

    Xvec = np.logspace(np.log10(Ymin),np.log10(Ymax),100)
#    plt.loglog(Xvec,Xvec,'k-')
    plt.plot(Xvec,Xvec,'k-')

