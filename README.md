# Cathode package
### Description
The ``cathode'' Python package contains various 0-D cathode models that have
been published throughout the years, starting with D. Siegfried and P. J. 
Wilbur in the late 1970s. 

We have re-implemented the models for a comprehensive review that was published
in the Joint Propulsion Conference proceedings:
```
Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
review of hollow cathode modeling: 0-D models," 53rd AIAA/ASME/SAE/ASEE Joint 
Propulsion Conference, 2017, AIAA-2017-4888. 
```
The re-implementation of each model is done based on the model description from 
the original publication.

We also have added necessary ``helper'' functions to compute flow quantities
such as the Reynolds or Knudsen number and interfaces for collision cross
sections and reaction rates. 

### How to use
Clone the repository and install the library:
```
sudo python setup.py build install
```

Then, in the Python code:
```python
import cathode
```

### Libraries required
The package requires Numpy, Scipy, and h5py.

### License
This work is licensed under the GNU Lesser General Public License (LGPL) v3.

### Contact
You can contact either Pierre-Yves Taunay (ptaunay@princeton.edu) or Chris
Wordingham (cjw4@alumni.princeton.edu).
