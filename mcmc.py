from pythonFunctions import *

#Priors
def log_prior(theta):
    
    # Uninformative uniform prior 
    # The parameters are stored as a vector of values, so unpack them
    t0, per, rp, a, inc = theta[:-4], *theta[-4:]
    if a < 0 or rp < 0 or rp > 1.:
        return -np.inf
    # inclination should be less than 90
    if inc > 90.:
        return -np.inf
    #if t0_0 < epochs[0] - 0.01 or t0_0 > epochs[0] + 0.01:
        #return -np.inf
    #if t0_1 < epochs[1] - 0.01 or t0_1 > epochs[1] + 0.01:
        #return -np.inf
    #if t0_2 < epochs[2] - 0.01 or t0_2 > epochs[2] + 0.01:
        #return -np.inf
    #if t0_3 < epochs[3] - 0.01 or t0_3 > epochs[3] + 0.01:
        #return -np.inf
    #if t0_4 < epochs[4] - 0.01 or t0_4 > epochs[4] + 0.01:
        #return -np.inf
        
    # A Gaussian prior on the orbital period
    period_mu = 2.3
    period_sigma = 1e-3
    period_prob = 1 / period_sigma / (2 * np.pi) ** 0.5 * np.exp(-0.5 * ((per - period_mu) / (period_sigma)) ** 2)
    return np.log(period_prob)

def lc_model(theta, x):
    '''
    Model = transit1(t0_0, *pars) + transit2(t0_1, *pars) + .... 
    x: time domain
    '''
    t0, per, rp, a, inc = theta[:-4], *theta[-4:]
    #get each transits epoch within window of +- per/2
    x_model = [x[(x > t0[i] - per/2) & (x < t0[i] + per/2)] for i in range(len(t0))]
    #Each transit time with same other parameters but t0 create model = LC1 + LC2 + ...
    model = np.concatenate([list(f_batman(x_model[i], t0[i], *theta[-4:])) for i in range(len(t0))]) #pars contains t0 to create fake data
    
    return model

#model = LC1(t0_0, pars) + LC1(t0_1, pars) + ....
def log_likelihood(theta, x, y, yerr):

    model = lc_model(theta, x)
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2. * np.pi*sigma2))

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


