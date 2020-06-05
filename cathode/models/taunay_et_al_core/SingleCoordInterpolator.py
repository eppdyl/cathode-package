# "cathode" Python package
# Version: 1.0
# A package of various cathode models that have been published throughout the
# years. Associated publications:
# Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
# review of hollow cathode modeling: 0-D models," Journal of Propulsion and
# Power, in preparation.
# Taunay, P.-Y. C. R., Wordingham, C. J., and Choueiri, E.Y., "Physics of
# thermionic, orificed hollow cathodes," Plasma Sources Science and Technology,
# in preparation.
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
import numpy as np
import itertools
from scipy.interpolate import RegularGridInterpolator 

class SingleCoordInterpolator(RegularGridInterpolator): 
    ''' 
    A class that modifies RegularGridInterpolator so that it does
    not return a NaN in the case where the on of the dimensions on the grid has 
    a _single_ value.
    '''
    def __init__(self, points, values, method="linear", bounds_error=True,
            fill_value=np.nan):

        ### Extract only the arrays for which we have multiple points
        points_as_array = np.array(list(points))
        self.is_single = [] 
        self.single_value = []

        self.mask = np.zeros(len(points),dtype=bool)
        for idx,arr in enumerate(points):
            if len(arr) > 1:
                self.mask[idx] = True
                self.is_single.append(False)
                self.single_value.append(np.nan)
            else:
                self.is_single.append(True)
                self.single_value.append(arr[0])

        points_as_array = points_as_array[self.mask]
        
        extract_points = [elem for _,elem in enumerate(points_as_array)]
        extract_points = tuple(extract_points)

        if values.shape[0] == 1 and values.shape[1] == 1:
            values = values.reshape((values.shape[2],))
        elif values.shape[0] == 1 and values.shape[1] > 1:
            values = values.reshape((values.shape[1],values.shape[2]))
        elif values.shape[0] > 1 and values.shape[1] == 1:
            values = values.reshape((values.shape[0],values.shape[2]))


        ### Use the parent constructor with the smaller tuple
        super().__init__(extract_points,values,method,bounds_error,fill_value)

    def __call__(self, xi, method=None):
        ''' 
        Here xi is passed as the typical array: mdot, Id, ngo 
        '''
        for idx,val in enumerate(xi):
            # Sanity check: are we asking for a value which is equal to the
            # single value of that dimension?
            if self.is_single[idx]:
                sval = self.single_value[idx]
                b = np.isclose(val,sval)

                if not b:
                    raise ValueError("The input interpolation value of %d is " 
                    "not equal to the original single value of %d." % (val,sval))

        xi_extract = np.array(xi)[self.mask]

        result = super().__call__(xi_extract,method)
        return result
