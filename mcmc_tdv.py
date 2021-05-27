from pythonFunctions import *

#Priors
def log_prior(theta, epochs):
    # Uninformative uniform prior 
    # The parameters are stored as a vector of values, so unpack them
    t0s, a = theta[:len(epochs)], theta[len(epochs):]
    dt0 = 1.
    t0_probs = [] #store priors on all t0s
    a_probs = []
    for i, t0 in enumerate(t0s):
        #gaussian prior on t0s
        t0_mu = epochs[i]
        t0_sigma = 1e-3
        t0_prob = 1 / t0_sigma / (2 * np.pi) ** 0.5 * np.exp(-0.5 * ((t0 - t0_mu) / (t0_sigma)) ** 2)
        t0_probs.append(t0_prob)
        #if t0 < epochs[i] - dt0 or t0 > epochs[i] + dt0:
            #return -np.inf
    # A Gaussian prior on the a/rs
    for i, a_ in enumerate(a):
        a_mu = 22.340000193#a[i]#3.877#2.3
        a_sigma = 1e-3
        a_prob = 1 / a_sigma / (2 * np.pi) ** 0.5 * np.exp(-0.5 * ((a_ - a_mu) / (a_sigma)) ** 2)
        a_probs.append(a_prob)
    return np.sum(np.log(t0_probs)) + np.sum(np.log(a_probs)) 

def lc_model(theta, x):
    '''
    Model = transit1(t0_0, *pars) + transit2(t0_1, *pars) + .... 
    x: time domain
    '''
    t0, a = theta[:int(len(theta)/2)], theta[int(len(theta)/2):]
    #Each transit time with same other parameters but t0 create model = LC1 + LC2 + ...
    model = np.concatenate([list(f_batman(x[i], t0[j], a[j])) for j,i in enumerate(x.keys())]) #pars contains t0 to create fake data
    
    return model

#model = LC1(t0_0, pars) + LC1(t0_1, pars) + ....
def log_likelihood(theta, x, y, yerr):
    flux, sigma2 =  np.concatenate([y[i] ** 2 for i in y.keys()]), np.concatenate([yerr[i] ** 2 for i in yerr.keys()])
    model = lc_model(theta, x)
    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(2. * np.pi*sigma2))

def log_probability(theta, linear_eph, x, y, yerr):
    lp = log_prior(theta, linear_eph)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


