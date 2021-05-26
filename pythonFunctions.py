import numpy as np
import batman
import collections
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
from IPython.display import display, Math
import corner
#Compilation of several functions
#=======================================================================
def bin_data(t, flux, flux_err, step, avrg='mean'):
    '''
    Data binning by mean otherwise median. 
    Parameters:
    t: x axis
    flux: y axis
    step: size of bins
    
    '''
    #bin the data, t_binned contains the boarders of bins
    t_binned = np.arange(t.min(), t.max() + step, step) #endpoint not included
    binned_time = [] #time to be plotted against avr flux within a bin
    binned_flux = []
    binned_flux_err = []
    for i in range(len(t_binned) -1):
        cond = (t >= t_binned[i]) & (t < t_binned[i+1])
        if len(flux[cond]) > 1:
            binned_time.append(t[cond].mean() if avrg == 'mean' else np.median(t[cond]))
            binned_flux.append(np.mean(flux[cond]) if avrg == 'mean' else np.median(flux[cond]))
#            binned_flux_err.append(np.sqrt((flux_err[cond]**2).mean()))
            binned_flux_err.append(np.sqrt(np.var(flux[cond])/len(flux[cond]))) #photon noise
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
def f_batman(x, t0,per,rp,a,inc,baseline=0.0, ecc=0.1,w=90, u = [0.34, 0.28] ,limb_dark ="quadratic"):
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
    params.u = u               #limb darkening coefficients [u1, u2]
    params.limb_dark = limb_dark       #limb darkening model

    m = batman.TransitModel(params, x)    #initializes model
    flux_m = m.light_curve(params)          #calculates light curve
    return np.array(flux_m)+baseline
#====================================================================================================
def linear_ephemerides(T0, T0err, P, Perr, Tdur, Tdurerr, dt, time):
    '''
    Compute expected transit ingress and egress from entire given time
    dt: add a fractional oot wings. e.g, dt=0.2 means 20% of Tdur 
    Notice that it assumes continuous data, hence for ground-based some caught transits are empty
    '''
    init_time, end_time = time.min(), time.max()

    # Check if T0 is the 1st transit. If not shift T0 N transits backwards 
    N_init = int((T0 - init_time) / P) 
    if N_init != 0:
        print(f'Shifting T0 backwards {N_init} transits')
        T0 -= N_init * P
    #Sanity checks
    assert int((T0 - init_time) / P) == 0
    
    dt *= Tdur
    N = np.arange( int((end_time - init_time) / P) + 1) #arange doesnt include last point stop) added +1. Epochs from 1st transit
    Ti = T0 + N*P - (N*Perr + T0err + 0.5 * Tdur + Tdurerr) - dt #Ti = Beginning of window
    Te = T0 + N*P + (N*Perr + T0err + 0.5 * Tdur + Tdurerr) + dt #Te (late) = End of transit window
    
    return (Ti, Te)
#====================================================================================================
def transit_difference(T0, T0err, T0ref, T0ref_err, P=0.7920520, Perr=0.0000093):
    '''Computes Difference between two T0 central transits. Negative values mean T0ref is ahead (larger) of T0'''
    N = int(abs(T0 - T0ref) / P) #number of transits between T0 and T0ref
    if T0 > T0ref:
        diff = (T0 - (T0ref + N*P)) * 24 * 60 # Difference between T0s in minutes
        diff_err = (T0ref_err + T0err) * 24 * 60 #T0s errors are added to make T0 diff larger. diff_err = T0max - T0
    else:
        #eq [(T0ref- eT0ref) + NP - (T0+eT0)] - (T0ref + NP) - T0 = -eT0ref - eT0 
        diff = ((T0 + N*P) - T0ref) * 24 * 60 
        diff_err = -(T0ref_err + T0err) * 24 * 60
        
    return diff, diff_err
#============================================================================================================
def select_full_transits(Ti, Te, P, npoints, Ntransits, random_transits, *data, plot=True):
    '''
    Select all or a sample Ntransits of transits from LC for when npoints are within transit Ti,Te window.
    Need implementation for when Ntransits > Total transit. Do not exceed this constraint! Make a better while
    loop
    Parameters:
    Ti, Te: from linear_ephemerides function
    P: Planet period [days]
    npoints: Number of data points within transit windown
    random_transits: if True returns random transits
    Ntransits: Number of transits to get from LC. Nonsense if random_transit is False
    data: t, f, ferr 
    '''
    time, flux, flux_err = data
    N = np.arange( int((time.max() - time.min())/P)  + 1) #total possible N values
    t, f, ferr = {}, {}, {} #transits is stored here
    N_counts = [] 
    iterations = 0
    if random_transits == True:
        while len(N_counts) < Ntransits:
            if Ntransits > len(N):
                print('Ntransits to catch larger than total N transits in data')
                print(f'Total N transits from data: {len(N)}')
                break
                
            Nrand = np.random.choice(N) #pick transit number from all possible values 
            cond = (time >= Ti[Nrand]) & (time <= Te[Nrand])
            if len(time[cond]) == 0: # if data is discontinuous there may be no data for a given N then
                Nrand = np.random.choice(N) #pick a new N
                iterations +=1
            else:
                if len(time[cond]) >= npoints: #Check there's enough data points within transit windown (Ti, Te)
                    if len(N_counts) != 0:#Do not allow same transit windown to be picked 
                        if Nrand not in N_counts:
                            N_counts.append(Nrand)
                            t[f'transit {Nrand}'] = time[cond]
                            f[f'transit {Nrand}'] = flux[cond]
                            ferr[f'transit {Nrand}'] = flux_err[cond]
                        else:
                            Nrand = np.random.choice(N)
                            
                    else:#1st picked transit will be added here
                        N_counts.append(Nrand)
                        t[f'transit {Nrand}'] = time[cond]
                        f[f'transit {Nrand}'] = flux[cond]
                        ferr[f'transit {Nrand}'] = flux_err[cond]
                        Nrand = np.random.choice(N)

                else:# If 
                    Nrand = np.random.choice(N)
                    iterations +=1
                    if iterations== 1e4:
                        print(f'while loop iterations exceeded: value {iterations}. Investigate!')
                        print('Hints: No transits meet npoints criterium or Ntransits >>> Total N transits from data')
                        break
                
    else:
        N_delete = [] # if empty space in data or npoints is not met else is triggered. Delete indexes
        for ind in N:
            cond = (time >= Ti[ind]) & (time <= Te[ind])
            if (len(time[cond]) > npoints) & (len(time[cond]) != 0.): #make sure at least n datapoints are within the window 
                t[f'transit {ind}'] = time[cond]
                f[f'transit {ind}'] = flux[cond]
                ferr[f'transit {ind}'] = flux_err[cond]
            else:
                N_delete.append(ind)
    
        N = np.delete(N, N_delete)
    
    #unfold each array within dicts to put it into arrays
    transits_t, transits_f, transits_ferr = [], [], []
    #sort t dictionary
#    t = collections.OrderedDict(sorted(t.items()))
    #sort time to return sorted transits
    for i in sorted( N_counts if random_transits else N ): #N_counts if random_transits, if not N is used for catching all dips
        i = 'transit ' + str(i)
        for j,k,z in zip(t[f'{i}'], f[f'{i}'],ferr[f'{i}']):
            transits_t.append(j), transits_f.append(k), transits_ferr.append(z)
            
    N_ = (N_counts if random_transits else N)
    if plot:
        fig, ax = plt.subplots(len(N_),1, figsize = (6,len(N_)*2))
        for idx, ind in enumerate(N_): #do look in N_counts or N depending on random_transit value
            if len(N_) != 1:
                ax[idx].errorbar(t[f'transit {ind}'], f[f'transit {ind}'], ferr[f'transit {ind}'], fmt='g.', label = f'Transit {ind}')
                ax[idx].legend(loc='best')
            else:
                ax.errorbar(t[f'transit {ind}'], f[f'transit {ind}'], ferr[f'transit {ind}'], fmt='g.', label = f'Transit {ind}')
                ax.legend(loc='best')
        if len(N_) != 1:
            ax[-1].set_xlabel('t0 [days]')
        else:
            ax.set_xlabel('t0 [days]')
        plt.tight_layout()
        

    return np.array(transits_t),np.array(transits_f),np.array(transits_ferr), N_
#===================================================================================================================================
def draw_lc(time,cadence, sigma, pars, plot=True):
    '''
    Draw a sample LC from a normal distribution with mean = model, sdv = sigma.
    Parameters:
    time: initial and final time. Array-like. Cadence defines the time domain. 
    Cadence: in minutes  
    '''
    cadence /= (60 * 24)
    time = np.arange(time[0], time[-1], cadence)
    model = f_batman(time, *pars)
    
    flux, flux_err = np.random.normal(loc=model, scale = sigma), np.array([sigma] * model.size)
    if plot:
        fig, ax = plt.subplots(1,1, figsize = (9,6))

        ax.errorbar(time, flux, flux_err, fmt = 'k.', alpha = 0.1, label = 'mock data')
        ax.plot(time, model, 'r--', label = 'mock data model')
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Rel. Flux')
        ax.set_title(f'True pars: t0:{pars[0]:.4f}, per:{pars[1]:.4f}, rp:{pars[2]:.4f}, a:{pars[3]:.4f}, inc:{pars[4]:.4f}, bl:{pars[5]:.7f}')
        #, ecc:{ecc:.4f}, w:{w:.4f}, [u1, u2]: {u1, u2}')
        plt.legend()
    
    return time, flux, flux_err
#======================================================================================================================
def transit_dur(per,rp,a,inc):
    '''
    Compute transit duration from eq. ? from Haswell book transiting exoplanet.
    pars: a, rp, per is parameterised as batmam package
    '''
    
    b = a * np.cos(inc * np.pi/180)
    tdur = (per/np.pi) * np.arcsin( np.sqrt( (1 + rp)**2 - b**2) / a )
    
    #IMPLEMENT ERROR PROPAGATION HERE
    #https://physics.stackexchange.com/questions/503748/error-propagation-in-sine
    #
    
    return tdur
#=====================================================================================================================
def get_transit(Ti, Te, time, flux, flux_err):
    '''
    Pick individual transits from data within windown Ti (time of ingress) and Te (time of egress)
    '''
    
    cond = (time > Ti) & (time < Te)
    
    return time[cond], flux[cond], flux_err[cond] 
#=====================================================================================================================
def get_epochs(Ti, Te, N, time, flux, flux_err):
    '''
    Get minimum value of a selected transit as t0.
    Ti, Te: transit windows
    N: Transit epoch (integer)
    '''
    
    #get epochs
    epochs = []
    for i in N:

        t, f, ferr = get_transit(Ti[i], Te[i], time, flux, flux_err)
        epochs.append(t[f == f.min()])

    epochs = np.array(epochs).flatten()
    
    return epochs
