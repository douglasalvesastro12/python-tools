#Compilation of several functions
#=======================================================================
def bin_data(t, flux, flux_err, step):
    ''' Enter with t, flux and step for binning'''
    #bin the data, t_binned contains the boarders of bins
    t_binned = np.arange(t.min(), t.max() + step, step) #endpoint not included
    binned_time = [] #time to be plotted against avr flux within a bin
    binned_flux = []
    binned_flux_err = []
    for i in range(len(t_binned) -1):
        cond = (t >= t_binned[i]) & (t < t_binned[i+1])
        if len(flux[cond]) > 1:
            binned_time.append(t[cond].mean())
            binned_flux.append(np.mean(flux[cond]))
            binned_flux_err.append(np.sqrt((flux_err[cond]**2).mean()))
#             binned_flux_err.append(np.sqrt(np.var(flux[cond])/len(flux[cond]))) #photon noise
        elif len(flux[cond]) == 1: #if one datapoint, that value is taken
            binned_time.append(t[cond][0]) #t[cond] returns an array with 1 element [0] gets the element
            binned_flux.append(flux[cond][0])
            binned_flux_err.append(flux_err[cond][0])
            
    #step/2 makes the data point in the center of the bin
    return np.array(binned_time), np.array(binned_flux), np.array(binned_flux_err)
#========================================================================================
def phase_fold(epoch, period, time):
    # phase fold data:
    phase = (time - epoch) % period / period
    phase[np.where(phase>0.5)] -= 1
    
    return phase
#==========================================================================================
def f_batman(x, t0,per,rp,a,inc,baseline=0.0, ecc=0,w=90,  u=[0.34, 0.28],limb_dark ="quadratic"):
    """
    Function for computing transit models for the set of 8 free paramters
    x - time array
    """
    params = batman.TransitParams()
    params.t0 = t0                     #time of inferior conjunction
    params.per = per                  #orbital period
    params.rp =  rp         #planet radius (in units of stellar radii)
    params.a = a                      #semi-major axis (in units of stellar radii)
    params.inc = inc                     #orbital inclination (in degrees)
    params.ecc = ecc                     #eccentricity
    params.w = w                       #longitude of periastron (in degrees)
    params.u = u                #limb darkening coefficients [u1, u2]
    params.limb_dark = limb_dark       #limb darkening model

    m = batman.TransitModel(params, x)    #initializes model
    flux_m = m.light_curve(params)          #calculates light curve
    return np.array(flux_m)+baseline
#====================================================================================================
def linear_ephemeris(T0, T0err, P, Perr, Tdur, Tdurerr, Ntransits):
    '''
    Compute expected transit egress and ingress 
    '''
    N = np.arange(Ntransits)
    Ti = T0 + N*P - (N*Perr + T0err + 0.5 * Tdur + Tdurerr) #Ti = Ingress
    Te = T0 + N*P + (N*Perr + T0err + 0.5 * Tdur + Tdurerr) #Te = Egress
    
    return (Ti, Te)
#====================================================================================================
def select_full_transits(Ti, Te, P, npoints, *data):
    '''
    Select transits if there is npoints within transit Ti,Te window
    '''
    time, flux, flux_err = data
    N = np.arange( int((time.max() - time.min())/P) )
    t, f, ferr = {}, {}, {} 
    
    for ind in N:
        cond = (time >= Ti[ind]) & (time <= Te[ind]) 
        if len(time[cond]) > npoints: #make sure at least n datapoints are within the window 
            t[f'transit {ind}'] = time[cond]
            f[f'transit {ind}'] = flux[cond]
            ferr[f'transit {ind}'] = flux_err[cond]
        
    #unfold each array within dicts to put it into arrays
    transits_t, transits_f, transits_ferr = [], [], []
    for i in t.keys():
        for j,k,z in zip(t[f'{i}'], f[f'{i}'],ferr[f'{i}']):
            transits_t.append(j), transits_f.append(k), transits_ferr.append(z)
    
        
    return np.array(transits_t),np.array(transits_f),np.array(transits_ferr) 
#======================================================================================================
def transit_difference(T0, T0err, T0ref, T0ref_err, P=0.7920520, Perr=0.0000093):
    '''Computes Difference between two T0 central transits. Negative values mean T0ref is ahead (larger) of T0'''
    N = int(abs(T0 - T0ref) // P) #number of transits between T0 and T0ref
    if T0 > T0ref:
        diff = (T0 - (T0ref + N*P)) * 24 * 60 # Difference between T0s in minutes
        diff_err = (T0ref_err + T0err) * 24 * 60 #T0s errors are added to make T0 diff larger. diff_err = T0max - T0
    else:
        #eq [(T0ref- eT0ref) + NP - (T0+eT0)] - (T0ref + NP) - T0 = -eT0ref - eT0 
        diff = ((T0 + N*P) - T0ref) * 24 * 60 
        diff_err = -(T0ref_err + T0err) * 24 * 60
        
    return diff, diff_err
#============================================================================================================

