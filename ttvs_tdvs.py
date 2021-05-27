from mcmc_tdv import *
import random
import os

#TESS data TOI-193
#time, flux, flux_err = np.loadtxt('/home/astropc/allesfitter-master/ttvs_TOI-193/allesfit_linear_ephemerides/LTT9779b_alldata.csv', delimiter=',', skiprows=1).T

#TESS data TOI-195
time, flux, flux_err = np.loadtxt('/home/astropc/allesfitter-master/ttvs_TOI-175/allesfit_with_ttvs/TOI-175_S36.csv', delimiter=',', skiprows=1).T

#TOI-193 modeled pars (allesfitter)
t0, per, rp, a, inc = 1367.2755, 3.690621972, 0.039557499, 22.340000193, 89.3 #4915354.21679, 0.7920520, 0.0455, 3.877, 76.39
t0_err, per_err= 0.00035, 0.000012917#0.00025, 0.0000093

#Set path to save run
path = '/media/astropc/Data/Universidade/UniversidadChile/Clases/TallerIV/LTT9779/TTV/mcmc_plots/'
folder = 'run_TOI-195_TDV/'

#Create folder in which to store results
try:
    os.mkdir(path + folder)
except:
    print('Folder already exists!, Plots will be overwritten.')
savefiles = path + folder

#fig, ax = plt.subplots(1,2, figsize = (10, 7))

#ax[0].errorbar(time, flux, flux_err, fmt = '.', label = 'data')
#ax[0].set_xlabel('Time [days]')
#ax[0].set_ylabel('Rel. Flux')
#ax[0].legend()

#ax[1].errorbar(time, flux, flux_err, fmt = '.', label = 'zommed-in')
#ax[1].set_xlim(t0 - 0.1, t0 + 0.1)
#ax[1].set_xlabel('Time [days]')
#ax[1].set_ylabel('Rel. Flux')
#ax[1].legend()

#plt.tight_layout()

#plt.savefig(savefiles + 'LC.pdf')
#plt.savefig(savefiles + 'LC.png')
#plt.close(fig)

#Transit duration from allesfitter (uncomment function for an analytic approach)
Tdur = 1.24 #transit_dur(per, rp, a, inc) #0.4681 #
Tdur_err = 0.#0.0095
dt = -.3 #add/remove x% more of Tdur to the LC

#Get Ingress and Egress approx. time
Ti, Te, t0 = linear_ephemerides(t0, t0_err, per, per_err, Tdur, Tdur_err, dt, time) #errors are zero because data comes from model

#Select transit portions of LC
npoints = 1#200 #minimum npoints withing window Ti, Te
t_transit, f_transit, ferr_transit, N = select_full_transits(Ti, Te, per, npoints, 10, False, time, flux, flux_err, plot=True)

plt.tight_layout()

plt.savefig(savefiles + 'LC.pdf')
plt.savefig(savefiles + 'LC.png')


#TODO epochs vs lin_eph, for using lin_eph I need to make t0 always starting from beginning or make N in N * per 
#Linear ephemerides based on t0. If mock data, this is the true t0s.
linear_eph = t0 + N * per

#get epochs. Epochs are the time at minimum flux from individual transits
#(if mock data, E = T0 + N * per is prefered). Real data: use get_epochs
initial_epochs_guess = np.array([t_transit[i][np.argmin(f_transit[i])] for i in t_transit.keys()])
#use same a for all transits as guess
initial_a_guess = [a]*N.size
initial_pars = [*initial_epochs_guess, *initial_a_guess]

#labels
transit_labels = [f't0_{i}' for i in range(len(linear_eph))]
transit_labels_a = [f'a_{i}' for i in range(len(linear_eph))]
labels = [*transit_labels, *transit_labels_a]

#MCMC settings
ndim = len(labels)
nwalkers = ndim*10 # Number of walkers. It is recommended to be at least higher than twice the amount of parameters being fit

order_of_mag = 1e-3#np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3])

pos = initial_pars + order_of_mag * np.random.randn(nwalkers, ndim) # N(initial_pars, order_of_mag**2) 


# Set up the backend
filename = savefiles + "MCMC_run.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


chains = 15000
#Run MCMC with Multiple Cores
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(initial_epochs_guess, t_transit, f_transit, ferr_transit),pool=pool, backend=backend)
    sampler.run_mcmc(pos, chains, progress=True)

#Get autocorrelation time. This must be checked for all data. See trace_raw figures 
try:
    tau = sampler.get_autocorr_time()
    thinby = int(tau.max() / 2) # thin by half the max value. See emcee docs
    print(tau)
except:
    print('Probably no covergence! Check tau')
    
fig, axes = plt.subplots(ndim, sharex=True, figsize=(9.0, ndim * 2))
########################## NOTE! DRAINING RAM MEMORRY
for i in range(ndim):
    axes[i].plot(sampler.chain[:,:,i].T)
    axes[i].set_ylabel(labels[i])
    axes[i].axhline(y=initial_pars[i], linestyle='--', lw=1.5, color='k')
axes[-1].set_xlabel('Step')
plt.tight_layout()
plt.savefig(savefiles + 'trace_raw.pdf')
plt.savefig(savefiles + 'trace_raw.png')
plt.close(fig)
#####################
#Discard burn_in and get chains every thin value
burn_in = int(0.3 * chains)

try:
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinby, flat=True)
    print(f'Burn-in discarded {burn_in} and Thin by {thinby}')
except:
    print(f'Burn-in discarded {burn_in} and Thin by {int(burn_in*.1)}')
    flat_samples = sampler.get_chain(discard=burn_in, thin=int(burn_in*.01), flat=True)

f = open(savefiles + 'posterior.txt', 'w')
f.write('#pars,lower,upper' + "\n"); f.close()

fit = np.array([])

#TODO store all variables then save to file. More efficient.
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    
    with open(savefiles + 'posterior.txt', 'a+') as f:
        f.write(f"{labels[i]},{mcmc[1]},{q[0]},{q[1]}")
        f.write("\n")
    fit = np.append(fit, [mcmc[1]])
    
fig = corner.corner(flat_samples, labels=labels)

plt.savefig(savefiles + f'corner_Burn-in{burn_in}.pdf')
plt.savefig(savefiles + f'corner_Burn-in{burn_in}.png')
plt.close(fig)


fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
samples = sampler.get_chain(discard = burn_in, thin=int(burn_in*.1))
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
plt.tight_layout()
axes[-1].set_xlabel("step number");
plt.tight_layout()


plt.savefig(savefiles + f'trace_discarded-{burn_in}-chains.pdf')
plt.savefig(savefiles + f'trace_discarded-{burn_in}-chains.png')
plt.close(fig)

print('True/initialPars mcmc 50 percentile')
for idx in range(len(fit)):
    print("{} = {}\t{}".format(labels[idx],initial_pars[idx], fit[idx]))
        
inds = np.random.randint(len(flat_samples), size=50)

#Plot transits
fig, ax = plt.subplots(len(N), 1, figsize = (len(N)*2, len(N)*2.5))
for i,tr in enumerate(N):    
    for ind in inds:
        ax_ = ax[i]
        sample = flat_samples[ind] #get samples parameters
        pars = [sample[:len(transit_labels)][i], sample[len(transit_labels):][i]] #from samples get t0 and other pars to plot individual transits.
        sample_flux = f_batman(t_transit[f'transit {tr}'], *pars) # apply pars to model
        #t, f, ferr = get_transit(Ti[tr], Te[tr], t_transit, f_transit, ferr_transit) # get individual transits
        #t_model, f_model, ferr_model = get_transit(Ti[tr], Te[tr], t_transit, sample_flux, np.zeros(t_transit.size))  #get individual trasits from model
        ax_.errorbar(t_transit[f'transit {tr}'], f_transit[f'transit {tr}'], ferr_transit[f'transit {tr}'], fmt='.', alpha=0.05)
        ax_.plot(t_transit[f'transit {tr}'], sample_flux, color='k', linestyle='dashed')
        ax_.set_xlabel(labels[i] + ' [days]')
plt.tight_layout()
plt.savefig(savefiles + 'transits.pdf')
plt.savefig(savefiles + 'transits.png')
plt.close(fig)


#plot TTVs 

ttvs = []
ttvs_err_l = []
ttvs_err_u = []
for i in range(len(fit[:len(transit_labels)])):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    errs = np.diff(mcmc)
    ttvs.append(mcmc[1]) 
    ttvs_err_l.append(errs[0])
    ttvs_err_u.append(errs[-1])
    
ttvs = np.array(ttvs)
ttvs_err = np.array([ttvs_err_l, ttvs_err_u])
std_ttvs = np.std(linear_eph - ttvs) * 24 * 60
fig, ax = plt.subplots(2,1, sharex=True)

#plot1
ax[0].plot(linear_eph, linear_eph, 'k')
ax[0].errorbar(linear_eph, ttvs, ttvs_err, fmt = 'o', label = "modeled t0s")
#ax[0].set_xlabel('Epochs [days]')
ax[0].set_ylabel('Linear Ephemerides [days]')
ax[0].legend()

#plot2
ax[1].errorbar(linear_eph, -1*(linear_eph - ttvs) * 24 * 60, ttvs_err * 24 * 60, fmt = 'k.', label = f"std {std_ttvs:.4f} min")
ax[1].set_xlabel('Epochs [days]')
ax[1].set_ylabel('O-C [min]')
ax[1].legend()

plt.tight_layout()
#plt.savefig(savefiles + 'ttvs.pdf')
plt.savefig(savefiles + 'ttvs.png')
plt.close(fig)


#Compute Tdur from a values
a_mcmc = []
a_mcmc_err_l = []
a_mcmc_err_u = []
for i in range(len(fit[:len(transit_labels)]), len(fit)):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    errs = np.diff(mcmc)
    a_mcmc.append(mcmc[1]) 
    a_mcmc_err_l.append(errs[0])
    a_mcmc_err_u.append(errs[-1])
    
a_mcmc = np.array(a_mcmc)
a_err = np.array([a_mcmc_err_l, a_mcmc_err_u])

    
Tdur = transit_dur(per, rp, a_mcmc, inc) #in hrs
#Tdur +- error
Tdur_l = transit_dur(per, rp, a_mcmc - a_err[0], inc) 
Tdur_u = transit_dur(per, rp, a_mcmc + a_err[1], inc)

Tdur_err = (Tdur_u - Tdur_l)/2 


fig, ax = plt.subplots(1, 1, figsize = (9, 5))

ax.errorbar(linear_eph, Tdur, Tdur_err, fmt = '.', label = 'TOI-193 TDV')
ax.set_xlabel('Central Transit Time [days]')
ax.set_ylabel('TDV [days]')

plt.savefig(savefiles + 'tdv.png')


#store variables
#Initial parameters (for comparison). t0s aren't true vals but approximated from get_epochs. 
np.savetxt(savefiles + 'initial_guess.txt', np.column_stack((labels, initial_pars)), delimiter=',', fmt='%s')
#Save linear_ephemeris, t0s and t0s errors from mcmc
header = 'linEph,ttvs,ttvs_err_l,ttvs_err_u,Tdur,Tdur_err'
X = np.column_stack((linear_eph, ttvs,ttvs_err_l, ttvs_err_u, Tdur, Tdur_err))
np.savetxt(savefiles + 'ttvs_days.txt',X,delimiter=',',header=header)

plt.close('all')
