## About the JPL cathodes
Goebel and Katz (2008) mention both the NEXIS and the NSTAR cathodes.
Both cathodes use barium oxide for their insert.
- NEXIS: Nuclear Electric Xenon Ion Thruster System
- NSTAR: NASA Solar Electric Propulsion Technology Applications Readiness

The NSTAR ion engine features TWO different hollow cathodes:
1. The "Discharge" hollow cathode
2. The "Neutralizer" hollow cathode
Their dimensions are different! It is also difficult to find any good information about the size of these cathodes.
Nonetheless, here are some references that set some values of these dimensions...

Goebel also built several LaB6 cathodes, with various insert diameters.

---

## Literature review of dimensions 

### Mikellides et al. (2008)
- I. G. Mikellides, I. Katz, D. M. Goebel, K. K. Jameson, and J. E. Polk, ''Wear Mechanisms in Electron Sources for Ion Propulsion, I: Neutralizer Hollow Cathode,'' Journal of Propulsion and Power, vol. 24, no. 4, pp. 855–865, 2008.
- I. G. Mikellides, I. Katz, D. M. Goebel, K. K. Jameson, and J. E. Polk, ''Wear Mechanisms in Electron Sources for Ion Propulsion, II: Discharge Hollow Cathode,'' Journal of Propulsion and Power, vol. 24, no. 4, pp. 866–879, 2008.

Mikellides et al. have a table in Ref. [1] that compares the dimensions of the two cathodes. Ref. [2] contains the dimensions of the discharge hollow cathode (see Table 1).
From there, we have:

| Dimension | Neutralizer HC | Discharge HC | Ratio NHC / DHC |
|:---------:|:--------------:|:------------:|:---------------:|
| Orifice radius (mm) | 0.14 | 0.51 | 0.275 |
| Orifice length (mm) | 0.74 | 0.74 | 1 |
| Emitter insert length (cm) | 2.54 | 2.54 | 1 |
| Cathode tube diameter (cm) | 0.635 | 0.635 | 1 |

The cathode tube diameter is _not_ the insert diameter. To get the insert diameter, we have to dig down a bit deeper...

### Sengupta (2005), Katz (2005)
- A. Sengupta, “Destructive Physical Analysis of Hollow Cathodes from the Deep Space 1 Flight Spare Ion Engine 30,000 Hr Life Test,” 29th IEPC, 2005.
- I. Katz, J. E. Polk, I. G. Mikellides, D. M. Goebel, and S. E. Hornbeck, “Combined Plasma and Thermal Hollow Cathode Insert Model,” 29th IEPC, 2005.

The cathode insert thickness is reported to be 0.769 mm (or 769 micrometers) in Sengupta.
There is a gap between the insert and the cathode tube of about 100 micrometers, as mentioned in Katz.
The insert radius + thickness of the cathode tube would then be

6.35 / 2 - 0.760 - 0.1 = 2.315 mm

Note, however, that the insert radius and thickness reported in Katz paper is different! From the graphs of plasma density and potential, we can get an insert radius of 1.9 mm, as opposed to 2.315 mm. 
The insert thickness is also obtained as 0.295 mm, as opposed to 0.769 mm from Sengupta.

### Goebel and Katz (2008)
Goebel and Katz have some dimensions written down on various plots. 
There are obviously more references to the NSTAR than the NEXIS, as the latter is more recent with respect to the publication date.

- References to NSTAR:
..p.258: Fig 6.6: 0.38 cm-insert diameter for the NSTAR
..Fig 6.48: 0.38 cm-insert diameter for the NSTAR
.. p.249: 0.38 cm-insert diameter for the NSTAR (so the 2004 paper from JPC is for the NSTAR). 
.. p.274: 0.38 cm-insert diameter, orifice length is 0.75 mm for the NSTAR neutralizer

- References to NEXIS
.. p.246: NEXIS orifice diameter is 2.5 mm ; Goebel JAP 2005 paper suggests 2 to 3 mm, and shows results for 2 and 2.75 mm
.. p.258: Fig 6.6: 1.27 cm-insert diameter for the NEXIS
.. p.271: NEXIS orifice diameter is 2.5 mm

### Katz et al. (2003)
- I. Katz, J. R. Anderson, J. E. Polk, and J. R. Brophy, “One-Dimensional Hollow Cathode Model,” J. Propuls. Power, vol. 19, no. 4, pp. 595–600, 2003.

Katz et al. model the _neutralizer_ NSTAR hollow cathode. From their plots, we obtain the orifice length. With no chamfer, the orifice length is 0.74 mm.

### Mikellides et al. (2004)
- I. G. Mikellides, I. Katz, D. M. Goebel, and J. E. Polk, “Model of a Hollow Cathode Insert Plasma,” 40th AIAA/ASME/SAE/ASEE Joint Propulsion Conference and Exhibit, 2004.

Fig. 1 shows the basic dimensions of the NEXIS cathode
- Orifice diameter: 0.15 cm
- Insert length: 2.5 cm
- Insert inner diameter: 1.1 cm
- Insert outer diameter: 1.2 cm

### Mikellides et al. (2005)
- I. G. Mikellides, I. Katz, D. M. Goebel, and J. E. Polk, “Hollow cathode theory and experiment. II. A two-dimensional theoretical model of the emitter region,” J. Appl. Phys., vol. 98, 2005.

Fig. 1 shows the basic dimensions of the NEXIS cathode. 
- Orifice diameter: 0.15 cm
- Insert length: 2.5 cm
- Insert diameter: 1.2 cm

### Mikellides et al. (2006)
- I. G. Mikellides, I. Katz, D. M. Goebel, J. E. Polk, and K. K. Jameson, “Plasma processes inside dispenser hollow cathodes,” Phys. Plasmas, vol. 13, 2006.

Mikellides has a table with dimensions (Table 1). The actual orifice of the ``larger'' cathode (read: NEXIS) is 2.79 mm.
The emitter length for the NEXIS is also taken to be 2.54 cm.

The plots seem to suggest the following dimensions

| Cathode | Insert radius (mm) | Insert thickness (mm) |
|:-:|:-:|:-:|
|NSTAR| 1.9 | 0.22 |
|NEXIS| 6.0 | 0.52 |

---
## Average cathodes

Based on the data used above, we can get the following dimensions for each cathode. If a reference appears more than once, this means that the dimension has been experimented with multiple values.

#### NSTAR neutralizer cathode

| Reference | Orifice diameter (mm) | Orifice length (mm) | Insert inner diameter (mm) | Insert outer diameter (mm) | Insert thickness (mm) | Insert length (mm) | Cathode outer diameter (mm) | 
|:---------:|:---------------------:|:-------------------:|:--------------------------:|:--------------------------:|:---------------------:|:------------------:|:---------------------------:|
Katz (2003) 	       |      | 0.74 |      |       |       |      |      |
Goebel et al. (2004)   |      |      | 3.8  |       |       |      |      |
Katz (2005 IEPC)       |      |      | 3.8  | 4.39* | 0.295? |      |      | 
Sengupta (2005)        |      |      |      |       | 0.76  |      |      |
Mikellides et al (2006)|      |      | 3.8  | 4.24* | 0.22?  |      |      |
Goebel and Katz (2008) | 0.28 | 0.75 | 3.8  |       |       |      |      |
Mikellides (2008)      | 0.28 | 0.74 |      |       |       | 25.4 | 6.35 |

*: calculated OD from the thickness
?: these numbers are deduced from plots; not sure how accurate these are.


#### NSTAR discharge cathode

| Reference | Orifice diameter (mm) | Orifice length (mm) | Insert inner diameter (mm) | Insert outer diameter (mm) | Insert thickness (mm) | Insert length (mm) | Cathode outer diameter (mm) | 
|:---------:|:---------------------:|:-------------------:|:--------------------------:|:--------------------------:|:---------------------:|:------------------:|:---------------------------:|
Katz (2003) 	       |      | 0.74 |      |         |       |      |      |
Goebel et al. (2004)   |      |      | 3.8  |         |       |      |      |
Katz (2005 IEPC)       |      |      | 3.8  | 4.39*   | 0.295? |      |      | 
Sengupta (2005)        |      |      |      |         | 0.76  |      |      |
Mikellides et al (2006)|      |      | 3.8  | 4.24*   | 0.22?  |      |      |
Goebel and Katz (2008) |      | 0.75 | 3.8  |         |       |      |      |
Mikellides (2008)      | 1.02 | 0.74 |      |         |       | 25.4 | 6.35 |

*: calculated OD from the thickness
?: these numbers are deduced from plots; not sure how accurate these are.

A more accurate picture of the insert outer diameter would be:
OD = ID + 2*IThickness + 2*gap = 3.8 + 2*0.76 + 2*0.1 = 5.5 mm

#### NEXIS cathode
| Reference | Orifice diameter (mm) | Orifice length (mm) | Insert inner diameter (mm) | Insert outer diameter (mm) | Insert thickness (mm) | Insert length (mm) | Cathode outer diameter (mm) | 
|:---------:|:---------------------:|:-------------------:|:--------------------------:|:--------------------------:|:---------------------:|:------------------:|:---------------------------:|
Mikellides et al (2004)      | 1.5  |      | 11.0 | 12.0|       | 25.0 |  15.0  |
Mikellides et al (2005)      | 1.5  |      | 12.0 |     | 0.22  | 25.4 |  15.0  |
Goebel et al (JAP 2005)      | 2.0  |      |      |     |       |      |  15.0  |
Goebel et al (JAP 2005)      | 2.8  |      |      |     |       |      |  15.0  |
Mikellides et al (2006)      |      |      | 12.0 |     | 0.22  | 25.4 |  15.0  |
Goebel and Katz (2008)       | 2.5  |      | 12.7 |     |       |      |        |
Goebel and Katz (2008)       | 2.8  |      | 12.7 |     |       |      |        |

#### Goebel LaB6 cathodes 
##### 0.8 cm
| Reference | Orifice diameter (mm) | Orifice length (mm) | Insert inner diameter (cm) | Insert outer diameter (cm) | Insert thickness (cm) | Insert length (cm) | Cathode outer diameter (cm) | 
|:---------:|:---------------------:|:-------------------:|:--------------------------:|:--------------------------:|:---------------------:|:------------------:|:---------------------------:|
Goebel et al. (JPP 2007) | 3.8 |  |  0.38  | 0.64  |  |2.5  | 0.8 |
Goebel et al. (Rev. Sci. 2010) | 3.8 |  |  0.38  | 0.64 | 0.13 | 2.5 | 0.8 |

Thickness of Mo tube is deduced (ignoring graphite sleeves): 0.08 cm 



##### 1.5 cm
| Reference | Orifice diameter (mm) | Orifice length (mm) | Insert inner diameter (cm) | Insert outer diameter (cm) | Insert thickness (cm) | Insert length (cm) | Cathode outer diameter (cm) | 
|:---------:|:---------------------:|:-------------------:|:--------------------------:|:--------------------------:|:---------------------:|:------------------:|:---------------------------:|
Goebel et al. (JPP 2007) | 3.8 |  |  0.7  | 1.3 | 0.3 | 2.5 | 1.5 |
Goebel et al. (Rev. Sci. 2010) | 3.8 | | 0.7 |  | | 2.5 | 1.5 |

Thickness of cathode wall is 0.1 cm

##### 2.0 cm
| Reference | Orifice diameter (mm) | Orifice length (mm) | Insert inner diameter (cm) | Insert outer diameter (cm) | Insert thickness (cm) | Insert length (cm) | Cathode outer diameter (cm) | 
|:---------:|:---------------------:|:-------------------:|:--------------------------:|:--------------------------:|:---------------------:|:------------------:|:---------------------------:|
Goebel et al. (JPP 2007) | 3.8 |  |  1.2  | 1.8 | 0.3 | 2.5 | 2.0 |
Goebel et al. (Rev. Sci. 2010) | 3.8 | | 1.2 | 0.64 | | 2.5 | 2.0 |

Thickness of cathode wall is 0.1 cm
