"""
cathode.models.siegfried_wilbur

Defines the model equations and solution procedure for Siegfried and Wilbur's
0D hollow cathode model described in:
    - D. E.  Siegfried and P. J. Wilbur, "A model for mercury orificed hollow 
    cathodes - Theory and experiment," AIAA journal,Vol. 22, No. 10, 1984, 
    pp. 1405–1412.
    - D. E.  Siegfried and P. J. Wilbur, "Phenomenological Model Describing 
    Orificed, Hollow Cathode Operation," AIAA Journal, Vol. 21, No. 1, 1983, 
    pp. 5–6.
    - P. J. Wilbur, "Advanced Ion Thruster Research," Tech. Rep. CR-168340, 
    NASA, 1984.
    - D. E.  Siegfried and P. J. Wilbur, "Studies on an experimental quartz 
    tube hollow cathode," 14th International Electric Propulsion Conference, 
    1979
    - D. E.  Siegfried, "A Phenomenological Model for Orificed Hollow 
    Cathodes," Ph.D., Colorado State University, 1982.
"""
import cathode.constants as cc
import numpy as np

from scipy.optimize import root


def hg_lambda_pr(ne,ng,phi_p):
    '''
     Function: hg_lambda_pr
     Description: computes the mean free path for the primary electrons.
     Applicable to mercury only! Based on Siegfried and Wilbur's computer model
     
     TODO: Add citation for computer model
     Inputs: 
     - ne: Electron density (1/m3)
     - ng: Gas density (1/m3)
     - phi_p: Plasma potential (V)
    '''
    inv = 6.5e-17*ne/phi_p**2 + 1e3*ng*phi_p / (2.83e23 - 1.5*ng)
    
    return 1/inv


def jth_rd(DRD,Tc,phi_wf,schottky=False,ne=1,phi_p=1,TeV=1):
    '''
    Function: jth_rd
    Description: computers the thermionic current density, based on
    Richardson-Dushman's law, with an optional Schottky effect
    '''
    kb = cc.Boltzmann
    q = cc.elementary_charge
     
    fac = DRD*Tc**2

    # If we consider the Schottky effect, calculate the effective work function
    if(schottky):
        phi = phi_eff(ne,phi_p,TeV,phi_wf)

    jth = fac*np.exp(-q*phi/(kb*Tc))

    return jth


def phi_eff(ne,phi_p,TeV,phi_wf):
    '''
    Function: phi_eff
    Description: computes the effective work function, based on a model of a
    thermionic double sheath by Prewett and Allen    
    ''' 
    q = cc.elementary_charge
    eps0 = cc.epsilon0
    
    # Cathode surface field
    Ec = np.sqrt( ne*q*TeV / eps0 );

    tmp = 2*np.sqrt(1 + 2*phi_p/TeV);
    tmp = tmp - 4;

    Ec = Ec*np.sqrt(tmp);

    # Effective work function
    return phi_wf - np.sqrt(q/(4*np.pi*eps0)*Ec) 


def goal_function(X,args):
    
    q = cc.elementary_charge
    kb = cc.Boltzmann
    
    # Rescale
    ne = X[0]*1e20
    ng = X[1]*1e20
    phi_p = X[2]
    Tc = X[3]*1e3

    goal = np.zeros(4)
    
    # Unpack arguments    
    dc, eps_i, mass, phi_wf, DRD, TeV, P, Id, lambda_pr, qth = args
    
    # Effective length, areas
    rc = dc/2
    lpr = lambda_pr(ne,ng,phi_p)
    Leff = 2*lpr

    Ae = 2*np.pi*rc*Leff
    Ac = np.pi*rc**2
    As = 2*Ac + Ae

    # Power loss per unit length (W/m)
    qth_val = qth(Tc-273.15) # W/mm
    qth_val = qth_val*1e3 # W/m
    
    # Ion Current density, ion current
    ji = q*ne*np.sqrt(q*TeV/mass)
    Ii = ji*As

    # Electron current
    Ie = jth_rd(DRD,Tc,phi_wf,True,ne,phi_p,TeV)*Ae

    # Equations
    # Perfect gas law
    goal[0] = P - kb*(ne*(TeV*q/kb + Tc) + ng*Tc)

    # Current conservation
    goal[1] = Id-Ie-Ii

    # Insert surface power balance
    goal[2] = ji*Ae*(phi_p + eps_i - phi_wf) 
    goal[2] -= qth_val * Leff 
    goal[2] -= Ie*phi_eff(ne,phi_p,TeV)

    # Plasma power balance
    goal[3] = phi_p*Ie - eps_i * Ii - 5/2*TeV*Id

    return goal

def sw_pressure_correlation(mdot,Id,orifice_diameter,mass):
    '''
    Function: sw_pressure_correlation
    Calculates the pressure correlation proposed by Siegfried and Wilbur
    '''
    amu_mass = mass/cc.atomic_mass
    if amu_mass == cc.M.Hg:
        c1 = 13.7
        c2 = 7.82
    elif amu_mass == cc.M.Xe:
        c1 = 9.0
        c2 = 4.0
    elif amu_mass == cc.M.Ar:   
        c1 = 5.6
        c2 = 1.2

    Ptorr = mdot / (orifice_diameter*1e3)**2 * (c1 + c2*Id)*1e-3
    
    return Ptorr * cc.Torr

def sw_qth():
    return 1

def solve(cathode_diameter, 
          eps_i, mass,
          phi_wf, DRD,
          TeV, 
          P = None, mdot = None, orifice_diameter = None,
          Id = None, Pfunc = sw_pressure_correlation,
          lambda_pr = hg_lambda_pr, qth = sw_qth,
          solver_tol = 1E-8,solver_out = False):
    ### Sanity checks...
    if P == None and mdot == None and Id == None:
        raise ValueError("Either [P] or [mdot AND Id] must be specified")
        
    if (P != None) and (mdot != None or Id != None):
        raise ValueError("Only [P] or [mdot and Id] can be specified. Not both")
   
    ### Constants
    q = cc.elementary_charge # Electron charge 
    kb = cc.Boltzmann

    ### Id case
    if Id != None and mdot != None:
        cases = zip(Id,mdot)
        
        solvec = []
        for lId,lmdot in cases:
            lP = Pfunc(lmdot,lId,orifice_diameter,mass)
            
            # Initial guess: densities and temperature are scaled
            Tc0 = 1050 + 273.15 # Wall temperature, K
            ne0 = 1.5e20 # Electron density, 1/m3
            nc0 = 1./(kb*Tc0)*(lP - ne0*kb*(TeV*q/kb + Tc0)) # Neutral density 
            # from perfect gas law, 1/m3
            phi_p0 = 7 # Plasma potential, V
            
            x0 = [ne0*1e-20,nc0*1e-20,phi_p0,Tc0*1e-3]
    
            # Optimizer options
            root_options = {'maxiter':int(1e6),
                            'xtol':solver_tol,'ftol':solver_tol}

            # Arguments
            dc = cathode_diameter
            args = [dc, eps_i, mass, phi_wf, DRD, TeV, P, Id, lambda_pr, qth]
            
            # Solve!
            optimize_results = root(goal_function,x0,args=args,
                                    method='lm',options = root_options)
            
            # Extract and rescale results
            ne,nc,phi_p,Tc = optimize_results.x
            
            rescaled_results = [ne*1e20,nc*1e20,phi_p,Tc*1e3-273.15]
            
            solvec.append(rescaled_results)




#
#### Gas info 
#epsi = 10.437 # Ionization potential of Hg, V
#M = 208.59 * cc.atomic_mass # Mass of Hg, kg
#
#### Cathode info
#do = 0.96*1e-3 # Orifice diameter, m
#dc = 3.9*1e-3 # Cathode insert diameter, m
#rc = dc/2 # Cathode insert radius, m
#Li = 2.2*1e-3 # Insert length, m
#
#### Emitter
#phi_w = 1.94 # Work function, eV
#DRD = 120e4 # Richardson-Dushman constant, A/m2/K2
#
#### Experimental data
#TeV = 0.71 # Electron temperature, eV
#filename = 'data/siegfried-refined/qth.csv'
#qth_data = np.genfromtxt(filename,dtype=np.float64,delimiter=',',names=True)
#set_interp = interp1d(qth_data['Tc'],qth_data['qth'],kind='cubic')
#
#### Operating condition
#Idvec = np.arange(1.,5.,0.1)
#mdot = 100 # Equivalent mA
#
#### Storage for all solutions
#solvec = np.zeros(np.size(Idvec))
#solvec.resize((np.size(Idvec),4))
#
#### Let's solve for various currents...
#idx = 0
#for Id in np.nditer(Idvec):
#    # Get the pressure
#    PTorr = mdot / (do*1e3)**2 * (13.7 + 7.82*Id)*1e-3
#    P = PTorr*cs.torr
#    
#    # Initial guess: densities and temperature are scaled
#    Tc0 = 1050 + 273.15 # Wall temperature, K
#    ne0 = 1.5e20 # Electron density, 1/m3
#    nc0 = 1./(kb*Tc0)*(P - ne0*kb*(TeV*q/kb + Tc0)) # Neutral density from perfect gas law, 1/m3
#    phi_p0 = 7 # Plasma potential, V
#
#    x0 = [ne0*1e-20,nc0*1e-20,phi_p0,Tc0*1e-3]
#
#    # Arguments of the function
##    ne,nc,phi_p,Tc = fsolve(zerofunc,x0) 
#    optimize_results = root(zerofunc,x0,method='lm',options={'maxiter':1000000,'xtol':1e-8,'ftol':1e-8})
#    ne,nc,phi_p,Tc = optimize_results.x
#    solvec[idx,:] = [ne*1e20,nc*1e20,phi_p,Tc*1e3-273.15]
#    idx = idx + 1
#
##    print ne*1e20,nc*1e20,phi_p,Tc*1e3
#
#### Plots!
## First the theoretical data
#filename = 'data/siegfried-refined/ne_vs_Id_mdot-100mA_do-096mm.csv'
#ne_vs_Id = np.genfromtxt(filename,dtype=np.float64,delimiter=',',names=True,skip_header=12)
#
#filename = 'data/siegfried-refined/phip_vs_Id_mdot-100mA_do-096mm.csv'
#phip_vs_Id = np.genfromtxt(filename,dtype=np.float64,delimiter=',',names=True,skip_header=12)
#
#filename = 'data/siegfried-refined/Tc_vs_Id_mdot-100mA_do-096mm.csv'
#Tc_vs_Id = np.genfromtxt(filename,dtype=np.float64,delimiter=',',names=True,skip_header=12)
#
## Smooth the data taken from plots
#tck = splrep(ne_vs_Id['Id'],ne_vs_Id['ne'])
#ne_idnew = np.arange(ne_vs_Id['Id'][0],ne_vs_Id['Id'][-1],0.01)
#ne_sw = [ne_idnew,splev(ne_idnew,tck,der=0)]
#
#tck = splrep(phip_vs_Id['Id'],phip_vs_Id['phip'],s=0.01)
#phip_idnew = np.arange(phip_vs_Id['Id'][0],phip_vs_Id['Id'][-1],0.01)
#phip_sw = [phip_idnew,splev(phip_idnew,tck,der=0)]
#
#tck = splrep(Tc_vs_Id['Id'],Tc_vs_Id['Tc'],s=0.01)
#Tc_idnew = np.arange(Tc_vs_Id['Id'][0],Tc_vs_Id['Id'][-1],0.01)
#Tc_sw = [Tc_idnew,splev(Tc_idnew,tck,der=0)]
#
## Then some experimental data
#filename = 'data/siegfried-refined/ne_vs_Id_mdot-100mA_do-096mm_xp.csv'
#ne_vs_Id_xp = np.genfromtxt(filename,dtype=np.float64,delimiter=',',names=True,skip_header=12)
#
#filename = 'data/siegfried-refined/Tc_vs_Id_mdot-100mA_do-096mm_xp.csv'
#Tc_vs_Id_xp = np.genfromtxt(filename,dtype=np.float64,delimiter=',',names=True,skip_header=12)
#
#
## Actual plots
#f,axarr = plt.subplots(3,sharex=True)
#axarr[0].semilogy(Idvec,solvec[:,0],'k',ne_sw[0],ne_sw[1]*1e20,'k--',ne_vs_Id_xp['Id'],ne_vs_Id_xp['ne']*1e20,'k.')
#axarr[0].set_ylabel('Plasma density (m$^{-3}$)')
#axarr[0].grid(True)
#
#axarr[2].plot(Idvec,solvec[:,2],'k',phip_sw[0],phip_sw[1],'k--')
#axarr[2].set_ylabel('Plasma potential (V)')
#axarr[2].set_xlabel('Discharge current (A)')
#axarr[2].grid(True)
#
#axarr[1].plot(Idvec,solvec[:,3],'k',Tc_sw[0],Tc_sw[1],'k--',Tc_vs_Id_xp['Id'],Tc_vs_Id_xp['Tc'],'k.')
#axarr[1].set_ylabel('Insert temperature ($^\circ$C)')
#axarr[1].grid(True)
#
#plt.show()
