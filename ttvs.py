from mcmc import *
import random
import os

#random.seed(10)
#Create fake data to MCMC
#True parameters for LC
#t0, per, rp, a, inc = 0.1, 2.3, 0.01, 12.2, 87.
#pars = [t0, per, rp, a, inc]
#select mock data
cadence=2. #min
sigma=1e-3 #Gaussian noise scale
#t = np.array([0.,30])
#time, flux, flux_err = draw_lc(t, cadence, sigma, pars, False)

#TESS data TOI-193
time, flux, flux_err = np.loadtxt('/home/astropc/allesfitter-master/ttvs_TOI-193/allesfit_linear_ephemerides/LTT9779b_alldata.csv', delimiter=',', skiprows=1).T

#TOI-193 modeled pars (allesfitter)
t0, per, rp, a, inc = 4915354.21679, 0.7920520, 0.0455, 3.877, 76.39
t0_err, per_err= 0.00025, 0.0000093

#Set path to save run
path = '/media/astropc/Data/Universidade/UniversidadChile/Clases/TallerIV/LTT9779/TTV/mcmc_plots/'
folder = 'run1_TOI-193/'

#Create folder in which to store results
try:
    os.mkdir(path + folder)
except:
    print('Folder already exists!, Plots will be overwritten.')
savefiles = path + folder

fig, ax = plt.subplots(1,2, figsize = (10, 7))

ax[0].errorbar(time, flux, flux_err, fmt = '.', label = 'data')
ax[0].set_xlabel('Time [days]')
ax[0].set_ylabel('Rel. Flux')
ax[0].legend()

ax[1].errorbar(time, flux, flux_err, fmt = '.', label = 'zommed-in')
ax[1].set_xlim(t0 - 0.1, t0 + 0.1)
ax[1].set_xlabel('Time [days]')
ax[1].set_ylabel('Rel. Flux')
ax[1].legend()

plt.tight_layout()

#plt.savefig(savefiles + 'LC.pdf')
plt.savefig(savefiles + 'LC.png')
plt.close(fig)

#Transit duration from allesfitter (uncomment function for an analytic approach)
Tdur = 0.4681 #transit_dur(per, rp, a, inc) 
Tdur_err = 0.0095
dt = 0.0 #add 50% more of Tdur to the LC

#Get Ingress and Egress approx. time
Ti, Te = linear_ephemerides(t0, t0_err, per, per_err, Tdur, Tdur_err, dt, time) #errors are zero because data comes from model

#Select transit portions of LC
npoints = 200 #minimum npoints withing window Ti, Te
t_transit, f_transit, ferr_transit, N = select_full_transits(Ti, Te, per, npoints, 4, True, time, flux, flux_err, plot=False)

#Linear ephemerides based on t0.
linear_eph = t0 + N * per

#get epochs. Epochs are the time at minimum flux from individual transits
#initial_epochs_guess = get_epochs(Ti, Te, N, t_transit, f_transit, ferr_transit)
initial_epochs_guess = np.array([t_transit[i][np.argmin(f_transit[i])] for i in t_transit.keys()])
transit_labels = [f't0_{i}' for i in range(len(linear_eph))]

initial_pars = [*initial_epochs_guess, per, rp, a, inc]
labels = [*transit_labels, 'per', 'rp', 'a', 'inc']

#MCMC settings
ndim = len(labels)
nwalkers = ndim*10 # Number of walkers. It is recommended to be at least higher than twice the amount of parameters being fit

order_of_mag_t0 = [1e-1]*len(transit_labels)

order_of_mag = 1e-1#np.array([*order_of_mag_t0, 1e-3, 1e-3, 1e-3, 1e-3])

pos = initial_pars + order_of_mag * np.random.randn(nwalkers, ndim) # N(initial_pars, order_of_mag**2) 


# Set up the backend
filename = savefiles + "MCMC_run.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


chains = 10000
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
#plt.savefig(savefiles + 'trace_raw.pdf')
plt.savefig(savefiles + 'trace_raw.png')
plt.close(fig)
#############################
#Discard burn_in and get chains every thin value
burn_in = int(0.3 * chains)

try:
    flat_samples = sampler.get_chain(discard=burn_in, thin=thinby, flat=True)
    print(f'Burn-in discarded {burn_in} and Thin by {thinby}')
except:
    print(f'Burn-in discarded {burn_in} and Thin by {int(burn_in*.1)}')
    flat_samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)

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

#plt.savefig(savefiles + f'corner_Burn-in{burn_in}.pdf')
plt.savefig(savefiles + f'corner_Burn-in{burn_in}.png')
plt.close(fig)

#flat_samples_lastportion = sampler.get_chain(discard=int(chains*0.8), thin=10, flat=True)
#fig = corner.corner(flat_samples_lastportion, labels=labels)

#plt.savefig(savefiles + f'corner_Burn-in{int(chains*0.8)}.pdf')
#plt.savefig(savefiles + f'corner_Burn-in{int(chains*0.8)}.png')
#plt.close(fig)


#fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
#samples = sampler.get_chain(discard = burn_in, thin=int(burn_in*.1))
#for i in range(ndim):
    #ax = axes[i]
    #ax.plot(samples[:, :, i], "k", alpha=0.3)
    #ax.set_xlim(0, len(samples))
    #ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)
#plt.tight_layout()
#axes[-1].set_xlabel("step number");
#plt.tight_layout()


#plt.savefig(savefiles + f'trace_discarded-{burn_in}-chains.pdf')
#plt.savefig(savefiles + f'trace_discarded-{burn_in}-chains.png')
#plt.close(fig)

print('True/initialPars mcmc 50 percentile')
for idx in range(len(fit)):
    print("{} = {}\t{}".format(labels[idx],initial_pars[idx], fit[idx]))
        
inds = np.random.randint(len(flat_samples), size=50)

#TODO sample_flux.size differes from t_transit.size
#plot individual transits and random parameters samples from chain 
fig, ax = plt.subplots(len(N), 1, figsize = (10, len(N)*2.5))
for i,tr in enumerate(N):    
    for ind in inds:
        ax_ = ax[i]
        sample = flat_samples[ind] #get samples parameters
        pars = [sample[:-4][i], *sample[-4:]] #from samples get t0 and other pars to plot individual transits.
        sample_flux = f_batman(t_transit[f'transit {tr}'], *pars) # apply pars to model
        #t, f, ferr = get_transit(Ti[tr], Te[tr], t_transit, f_transit, ferr_transit) # get individual transits
        #t_model, f_model, ferr_model = get_transit(Ti[tr], Te[tr], t_transit, sample_flux, np.zeros(t_transit.size))  #get individual trasits from model
        ax_.plot(t_transit[f'transit {tr}'], sample_flux, color='k', alpha=0.05, zorder=10)
        ax_.errorbar(t_transit[f'transit {tr}'], f_transit[f'transit {tr}'], ferr_transit[f'transit {tr}'], fmt='.', alpha=0.5, zorder=10)
        ax_.set_xlabel(labels[i] + ' [days]')
plt.tight_layout()
#plt.savefig(savefiles + 'transits.pdf')
plt.savefig(savefiles + 'transits.png')
plt.close(fig)


#plot TTVs 

ttvs = []
ttvs_err_l = []
ttvs_err_u = []
for i in range(len(fit[:-4])):
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
ax[1].set_title(r'$\sigma_{t_{0}} = (\frac{Tdur}{1/cadence})^{1/2} * \sigma_{ph} * \frac{r_{p}}{r_{s}}$ = ' + f'{(Tdur*24*60/(2*(1/cadence)))**(.5) * sigma * rp**(-2) :.4f} min')
ax[1].legend()

plt.tight_layout()
#plt.savefig(savefiles + 'ttvs.pdf')
plt.savefig(savefiles + 'ttvs.png')
plt.close(fig)

#store variables
#Initial parameters (for comparison). t0s aren't true vals but approximated from get_epochs. 
np.savetxt(savefiles + 'initial_guess.txt', np.column_stack((labels, initial_pars)), delimiter=',', fmt='%s')
#Save linear_ephemeris, t0s and t0s errors from mcmc
header = 'linEph,ttvs,ttvs_err_l,ttvs_err_u'
X = np.column_stack((linear_eph, ttvs,ttvs_err_l, ttvs_err_u))
np.savetxt(savefiles + 'ttvs_days.txt',X,delimiter=',',header=header)

plt.close('all')
