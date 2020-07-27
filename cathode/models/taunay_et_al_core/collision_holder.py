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
# Copyright (C) 2019-2020 Pierre-Yves Taunay, Chris Wordingham
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

from cathode.collisions import cross_section as xsec
from cathode.collisions import reaction_rate as rr
import cathode.constants as cc

import numpy as np

class collision_holder():
    """ collision_holder

    A holder of all the electron-neutral collisions (ionization, elastic
    electron-neutral, and ground-state excitation).

    Attributes
    ----------
    xsec_dict: dictionary
    Dictionary of collisions cross sections. Each cross section is created from
    the cathode.collisions.cross_section "xsec" object from a specified
    collision file, which is assumed to be under "/data/species.dat" for a
    given species.

    The excitation cross section is special in that it has multiple levels. The
    'ex' keyword to the dictionary contains the total, lumped excited cross
    section but each individual level cross section and energies may be
    accessed through 'ex_list' and 'ex_levels' keywords, respectively.
    """

    # TODO: Add private variable for the species
    # TODO: Spline all the data beforehand so that it is faster
    def __init__(self, species):
        self.__xsec_dict = {}
        self.__xsec_dict['iz'] = None # Ionization xsec
        self.__xsec_dict['en'] = None # Electron-neutral xsec

        self.__xsec_dict['ex'] = None # Total 'lumped' excited xsec
        self.__xsec_dict['ex_list'] = None # List of all excitation levels
        self.__xsec_dict['ex_levels'] = None # Energy levels of the excited states

        if species == 'Xe' or species == 'Ar':
            file = 'data/' + str.lower(species) + '_all.dat'

            for kw in ['iz','ex','en']:
                if kw == 'iz':
                    lxsec = xsec.create_cross_section_spline(file,'ionization')

                elif kw == 'en':
                    lxsec = xsec.create_cross_section_spline(file,'elastic')

                elif kw == 'ex':
                    # Grab the lumped excitation cross section
                    lxsec = xsec.create_cross_section_spline(file,
                                                               'excitation',
                                                               'ALL')
                    # Get all the levels
                    ex_levels = lxsec.emins
                    nxsec = len(ex_levels)

                    # Get all excitation cross sections separately
                    lxsec_list = [xsec.create_cross_section_spline(file,
                                               'excitation',
                                               level)
                                    for level in range(1,nxsec+1)]

                    # Store the total excited cross section
                    self.__xsec_dict[kw + '_list'] = lxsec_list
                    self.__xsec_dict[kw + '_levels'] = ex_levels

                # Store the cross section spline or list of cross sections
                self.__xsec_dict[kw] = lxsec
    @property
    def xsec_dict(self):
        return self.__xsec_dict

    def ex_total_energy(self, Te):
        ''' Calculate the total excitation energy for all of the cross sections
        '''
        rr_all = np.array([rr.reaction_rate(x,Te) 
            for x in self.__xsec_dict['ex_list']])

        return np.dot(rr_all, self.__xsec_dict['ex_levels'])

    def ex_avg_energy(self, Te):
        '''
        Calculate the average excitation energy
        '''
        etot = self.ex_total_energy(Te)
        rrtot = rr.reaction_rate(self.__xsec_dict['ex'],Te)
        return etot/rrtot

    def rr(self,kw,Te):
        '''
        Calculate the reaction rate for a given reaction type
        '''
        if kw == 'ei':
            ret = 2.9e-12 * 10 * Te**(-3/2) # Lambda = 10
        else:
            # TODO: emins, emaxs do not need to be passed to the function
            # reaction_rate. It is dealt with in the function
            xsec = self.__xsec_dict[kw]
            emin = np.min(xsec.emins)
            emax = np.max(xsec.emaxs)
            ret = rr.reaction_rate(xsec,Te,Emin=emin,Emax=emax)
        return ret

    def xsec(self,kw,Te):
        '''
        Calculate the Maxwellian-averaged cross section for a given reaction 
        type
        '''
        # TODO: Maybe use the reaction_rate option to output xsec instead
        vm = np.sqrt(8*cc.e*Te/(np.pi*cc.me))
        ret = self.rr(kw,Te) / vm
        return ret

