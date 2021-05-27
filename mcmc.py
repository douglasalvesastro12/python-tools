from pythonFunctions import *

#Priors
def log_prior(theta, epochs):
    # Uninformative uniform prior 
    # The parameters are stored as a vector of values, so unpack them
    t0s, per, rp, a, inc = theta[:-4], *theta[-4:]
    dt0 = 0.1
    if a < 3. or a>4.5 or rp < 0.035 or rp > .046:
        return -np.inf
    # inclination should be less than 90
    if inc > 70. or inc < 75.:
        return -np.inf
    for i, t0 in enumerate(t0s):
        if t0 < epochs[i] - dt0 or t0 > epochs[i] + dt0:
            return -np.inf
    # A Gaussian prior on the orbital period
    period_mu = 0.792#2.3
    period_sigma = 1e-1
    period_prob = 1 / period_sigma / (2 * np.pi) ** 0.5 * np.exp(-0.5 * ((per - period_mu) / (period_sigma)) ** 2)
    return np.log(period_prob)

def lc_model(theta, x):
    '''
    Model = transit1(t0_0, *pars) + transit2(t0_1, *pars) + .... 
    x: time domain
    '''
    t0, per, rp, a, inc = theta[:-4], *theta[-4:]
    #get each transits epoch within window of +- per/2
    #x_model = [x[(x >= t0[i] - per/2) & (x <= t0[i] + per/2)] for i in range(len(t0))]
    #Each transit time with same other parameters but t0 create model = LC1 + LC2 + ...
    model = np.concatenate([list(f_batman(x[i], t0[j], *theta[-4:])) for j,i in enumerate(x.keys())]) #pars contains t0 to create fake data
    
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


