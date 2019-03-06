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
This submodule contains functions related to the computation of collision
cross-sections.
"""
import re
import numpy as np
from scipy.interpolate import splev, splrep

import cathode.constants as cc

###############################################################################
#                             Cross Section Fits
###############################################################################
def charge_exchange(TiV, species='Xe'):
    """
    Returns charge exchange cross section for the specified species in m^2 for
    the specified ION temperature in eV.
    Applies only to ion - neutral collisions.
    Inputs:
        - TiV: Ion temperature/energy (eV)
        - species: Ion/Neutral species (e.g. 'Xe')
    Outputs:
        - Cross section (m2)

    References:
    - Xenon: Miller, J. S., et al, "Xenon charge exchange cross sections for
      electrostatic thruster models", Journal of Applied Physics, 91(3), 984–991
      2002.  https://doi.org/10.1063/1.1426246
    - Krypton: Hause M. L., et al. "Krypton charge exchange cross sections for
      Hall effect thruster models," Journal of Applied Physics, 113(16), 2013.
      https://doi.org/10.1063/1.48024322013
    - Argon and nitrogen: Nichols, B. J. and Witteborn, F. C., "Measurements of
      Resonant Charge Exchange Cross Sections in Nitrogen and Argon between 0.5
      and 17 eV,"  NASA TN-3625, 1966.
    """
    # Fit constants
    consts = {
        'Xe':[87.3, 13.6],
        'Xe2+':[45.7, 8.9],
        'Ar':[7.49, 0.73],
        'N2':[6.48, 0.24],
        'Kr':[80.7, 14.7],
        'Kr2+':[44.6, 9.8]}

    try:
        A, B = consts[species]
    except LookupError:
        msg = "ERROR ---"
        msg += "Charge-exchange cross sections: valid inputs are "
        msg += "'Xe', 'Xe2+', 'Ar', 'N2', 'Kr', or 'Kr2+'"
        print(msg)

    # Special case for Argon and N2 fits:
    if species == 'Ar' or species == 'N2':
        ret = (A*cc.angstrom - B*cc.angstrom*np.log(TiV))**2
    else:
        ret = (A - B*np.log(TiV))*cc.angstrom**2

    return ret

def ionization_xe_mk(TeV):
    """
    Electron-impact ionization cross-section for xenon. Initially proposed by
    Mandell and Katz and reused thereafter (see e.g. Goebel and Katz' textbook)
    Valid only for xenon and electron temperatures less than 5 eV.

    Inputs:
    - TeV: Electron temperature (eV)
    Output:
    - Cross section (m2)

    References:
    - Mandell, M. J. and Katz, I., "Theory of Hollow Cathode Operation in Spot
      and Plume Modes," 30th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &
      Exhibit, 1994.
    - Goebel, D. M. and Katz, I., "Fundamentals of Electric Propulsion,"
      Appendix D p.475, John Wiley and Sons, 2008.
    """
    ret = (3.97+0.643*TeV-0.0368*TeV**2)
    ret *= np.exp(-12.127/TeV)
    ret *= cc.angstrom**2

    return ret

def excitation_xe_mk(TeV):
    """
    Radiative excitation cross-section for xenon. Initially proposed by Mandell
    and Katz and reused thereafter (see e.g. Goebel and Katz' textbook)
    Valid only for xenon.

    Inputs:
    - TeV: Electron temperature (eV)
    Output:
    - Cross section (m2)

    References:
    - Mandell, M. J. and Katz, I., "Theory of Hollow Cathode Operation in Spot
      and Plume Modes," 30th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &
      Exhibit, 1994.
    - Goebel, D. M. and Katz, I., "Fundamentals of Electric Propulsion,"
      Appendix D p.475, John Wiley and Sons, 2008.
    """
    ret = 1.93e-19 * np.sqrt(TeV) * np.exp(-11.6 / TeV)

    return ret

def electron_neutral_xe_mk(TeV,xsec_type='variable'):
    """
    Electron-neutral cross-section for xenon. Initially proposed by Mandell and
    Katz, and reused thereafter (see e.g. Goebel and Katz' textbook). It is an
    improvement over the proposed model from 1994 which uses a constant cross
    section. Valid only for xenon.

    Inputs:
    - TeV: Electron temperature (eV)
    - xsec_type: The type of cross section model to use. Can either be
      "constant" or "variable". If "variable" uses the model from year 1999 and
      above. Otherwise uses a set value of 5 10^{-19} m2.
    Output:
    - Cross section (m2)

    References:
    - Mandell, M. J. and Katz, I., "Theory of Hollow Cathode Operation in Spot
      and Plume Modes," 30th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &
      Exhibit, 1994.
    - Katz, I., et al, "Sensitivity of Hollow Cathode Performance to Design and
      Operating Parameters," 35th AIAA/ASME/SAE/ASEE Joint Propulsion Conference
      & Exhibit, 1999. http://arc.aiaa.org/doi/pdf/10.2514/6.1999-2576
    - Goebel, D. M. and Katz, I., "Fundamentals of Electric Propulsion,"
      Appendix D p.475, John Wiley and Sons, 2008.
    """

    if xsec_type == 'constant':
        ret = 5e-19
    elif xsec_type == 'variable':
        num = (TeV/4)-0.1
        den = 1 + (TeV/4)**(1.6)
        ret = 6.6e-19 * num/den
    else:
        raise ValueError

    return ret

###############################################################################
#                           Cross Section Import
###############################################################################
class CrossSection:
    """
    Callable cross section spline object. Constructor stores the minimum and
    maximum energies for the spline data, the scaling to multiply by (in case
    the data is scaled prior to fitting), and the spline fit object itself.
    CrossSection objects can be added to one another to produce lumped cross
    sections and can be called using function syntax at an energy in eV.
    """

    def __init__(self, Emin, Emax, scaling, spline):
        # Create empty arrays for each parameter
        self.__emins = np.array([])
        self.__emaxs = np.array([])
        self.__scalings = np.array([])
        self.__splines = np.array([])

        # Fill with inputs
        self.__emins = np.append(self.__emins, Emin)
        self.__emaxs = np.append(self.__emaxs, Emax)
        self.__scalings = np.append(self.__scalings, scaling)
        self.__splines = np.append(self.__splines, spline)

    def __add__(self, other):
        if isinstance(other, CrossSection):
            return CrossSection(np.append(self.__emins,other.emins),
                                np.append(self.__emaxs,other.emaxs),
                                np.append(self.__scalings,other.scalings),
                                np.append(self.__splines,other.splines))
        else:
            print("ERROR --- cannot add CrossSection to different data type")
            return None

    def __call__(self, value):
        zvec = zip(self.__emins, self.__emaxs, self.__scalings,
                   range(len(self.__splines)//3))

        # Accumulator
        acc = 0.0
        for emin, emax, scaling, i in zvec:
            # Spline value
            spl_value = splev(value, self.__splines[3*i:3*i+3])

            # Boolean to restrict range
            bcond = (value >= emin) * (value <= emax)

            # Accumulate
            acc += bcond * scaling * spl_value

        return acc

    ### Getters
    @property
    def emins(self):
        return self.__emins
    @property
    def emaxs(self):
        return self.__emaxs
    @property
    def scalings(self):
        return self.__scalings
    @property
    def splines(self):
        return self.__splines


def _fetch_data(lines, match_index):
    """Private helper function for cross section creation"""
    separator = re.compile('-{5,}\n')

    # Join all lines following match index
    bulk = ''.join(lines[match_index:])

    # Split at separators, data is 1st grouping after match
    # Split again for import to numpy
    data = re.split(separator, bulk)[1]
    data = np.loadtxt(re.split('\n', data))
    print('\tImported level: '+str(data[0, 0])+' eV')
    emin = data[0, 0]
    emax = data[-1, 0]

    # Rescale and spline data
    _scaling = np.max(data[:, 1])
    data[:, 1] = data[:, 1]/_scaling

    _spline = splrep(data[:, 0], data[:, 1])

    return CrossSection(emin, emax, _scaling, _spline)


def create_cross_section_spline(filename, xsec_type, chosen=None):
    """
    Finds the string associated with xsec_type in filename and extracts the
    numeric cross-section data following it.
    If multiple matches are found, user is prompted to select one or all of
    them.
    ALL combines the cross sections into a lumped value by splining each
    individually and then comibining to make a lumped spline.

    Inputs:
            filename of lxcat data
            cross-section type (e.g. any of these: 'IONIZATION','Excitation','elastic')
    Optional Inputs:
            chosen (number corresponding to selected cross section or 'ALL')
    Output:
            spline representation function
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    _valid_types = ['EXCITATION', 'IONIZATION', 'ELASTIC', 'EFFECTIVE']
    if xsec_type.upper() not in _valid_types:
        print('ERROR --- Must select a valid cross section type.')
        return None

    pattern = re.compile(xsec_type.upper())
    match_num = 0
    matches = {}

    # Find each possible match and print a description
    for index, line in enumerate(lines):
        if re.match(pattern, line):
            match_num += 1
            print(str(match_num)+'.')
            print(lines[index+4].strip())
            print(lines[index+6].strip() + '\n')
            matches[match_num] = index

    # If no matches, print warning and return None
    # Note: an empty list tests "False"
    if not matches:
        print("WARNING --- No Cross Section match found.")
        return None

    # If only a single match, import it directly
    elif len(matches) == 1:
        print("Importing data...")
        return _fetch_data(lines, matches[1])

    else:
        print("Multiple matches found.")
        ########################## Input loop #################################
        failure = True
        while failure:
            if not chosen:
                input_txt = "Select cross section to import "
                input_txt += "(or enter ALL to lump)\n"
                chosen = input(input_txt).upper()

            if chosen == 'ALL':
                print("Importing and lumping all cross sections...")
                failure = False

            else:
                #Make sure the input number is valid
                try:
                    chosen = int(chosen)

                    if chosen not in matches.keys():
                        raise ValueError

                    failure = False
                    print("Importing cross section No." + str(chosen))

                except ValueError:
                    chosen = None
                    print("Please enter a cross section number or ALL.")
        #######################################################################

        #If a single cross section was selected, import it and return
        if chosen != 'ALL':
            return _fetch_data(lines, matches[chosen])
        #If ALL was selected, import each spline, then sum to lump
        else:
            splines = {}
            for m in matches.keys():
                splines[m] = _fetch_data(lines, matches[m])
                #create empty cross section object
                out = CrossSection([], [], [], [])

                #sum over all retrieved cross sections
                for sp in splines.values():
                    out += sp

            return out
