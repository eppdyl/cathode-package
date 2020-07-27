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
import scipy.optimize as optim
import numpy as np

def bisect_next(f,a,b,min_spacing=1E-10,xtol=1E-20,atol=1E-15,diagnostics=False):
    '''
    Find the zeros of the function f(x) for x between a and b (a<b). The zeros
    are found within intervals of decreasing size until min_spacing is hit.
    bisect_next is called with a generator function:
        - list(bisect_next(...)) will return the list of ALL zeros between a
          and b
        - next(bisect_next(...)) will return the first zero
    Inputs:
        - f: the function for which a zero must be found
        - a: the minimum bound on x
        - b: the maximum bound on x
        - min_spacing: a
        - xtol: the tolerance on x
        - atol: the absolute tolerance
        - diagnostics: set to true for more information
    '''
    minbound = a
    maxbound = b
    done = False
    x0 = minbound
    x1 = minbound+min_spacing
    while not done:
        if x1 >= maxbound:
            done = True
        
        if np.isclose(f(minbound),0,atol=atol):
            if diagnostics:
                print('Boundary root found:',minbound)
            old_minbound = minbound
            minbound += min_spacing
            x0=minbound
            x1=minbound+min_spacing
            yield (old_minbound,f(old_minbound))
        elif np.isclose(f(maxbound),0,atol=atol):
            if diagnostics:
                print('Boundary root found:',maxbound)
            old_maxbound = maxbound
            maxbound -= min_spacing
            yield (old_maxbound,f(old_maxbound))
        else:
            root_found = False
            while not root_found:
                if diagnostics:
                    print('''[{},{}]'''.format(x0,x1))
                
                if f(x0)*f(x1) < 0:
                    x_out = optim.bisect(f,x0,x1,xtol=xtol,rtol=atol)
                    root_found = True
                    if diagnostics:
                        print('Root found:',x_out)
                
                x0 = x1
                
                if root_found and x_out <= maxbound:
                    yield (x_out,f(x_out))
                elif x1 > maxbound and root_found:
                    yield (x_out,f(x_out))
                elif x1 > maxbound:
                    break
                else:
                    x1 += min_spacing
