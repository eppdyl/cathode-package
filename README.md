# Cathode package
### Description
The "cathode" Python package contains various 0-D cathode models that have
been published throughout the years, starting with D. Siegfried and P. J. 
Wilbur in the late 1970s. 

We have re-implemented the models for a comprehensive review that was published
in the 2017 Joint Propulsion Conference proceedings:
```
Wordingham, C. J., Taunay, P.-Y. C. R., and Choueiri, E. Y., "A critical
review of hollow cathode modeling: 0-D models," 53rd AIAA/ASME/SAE/ASEE Joint 
Propulsion Conference, 2017, AIAA-2017-4888. 
```
The re-implementation of each model is done based on the model description from 
the original publication.

We also have implemented our own "helper" and "physics" functions that
were compiled for the review. These functions are necessary to compute collision cross
sections, reaction rates, and other flow quantities (e.g. Reynolds or Knudsen numbers).

Based on the results of the review, we have implemented our own 0-D model that
we presented at the 2019 AIAA Propulsion and Energy Forum:
```
Taunay, P.-Y. C. R., Wordingham, C. J., and Choueiri, E. Y., "A 0-D model for 
orificed hollow cathodes with application to the scaling of total pressure,"
AIAA Propulsion and Energy Forum, 2019, AIAA-2019-4246.
```

### Install 
Clone the repository and install the library:
```
sudo python setup.py build install
```

### How to use
In the Python code:
```python
import cathode
```

### Libraries required
The package requires Numpy, Scipy, and h5py.

### License
This work is licensed under the GNU Lesser General Public License (LGPL) v3.

### Citations
If you use any of the re-implemented 0-D models that are dated prior to 2017, 
please cite our 2017 review:
```
@inproceedings{Wordingham2017,
    author = {Wordingham, Christopher J. and Taunay, Pierre-Yves C. R. and Choueiri, Edgar Y.},
    booktitle= {53rd AIAA/ASME/SAE/ASEE Joint Propulsion Conference},
    doi= {10.2514/6.2017-4888},
    note= {AIAA-2017-4888},
    title= {A Critical Review of Orificed Hollow Cathode Modeling: 0-D Models},
    year= {2017}
}
```

If you use the 2019 0-D model, please cite the following paper:
```
@inproceedings{Taunay2019,
    author = {Taunay, Pierre-Yves C. R. and Wordingham, Christopher J. and Choueiri, Edgar Y.},
    booktitle= {55th AIAA/ASME/SAE/ASEE Joint Propulsion Conference},
    doi= {10.2514/6.2019-4246},
    note= {AIAA-2019-4246},
    title= {A 0-D model for orificed hollow cathodes with application to the scaling of total pressure},
    year= {2019}
}
```

If you use any other part of the source code that does not fall under either case above, please cite the package:
```
@misc{cathodePackage,
    author = {Wordingham, Christopher J. and Taunay, Pierre-Yves C. R.},
    howpublished = {\url{https://github.com/eppdyl/cathode-package}},
    year = {2017 -- },
    note = {Retrieved [insert date]}
}
```

### Contact
You can contact either Pierre-Yves Taunay (ptaunay@princeton.edu) or Chris
Wordingham (cjw4@alumni.princeton.edu).
