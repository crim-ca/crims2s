# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rv_continuous, gamma
from scipy.optimize import minimize
from scipy.special import gamma as Gamma
from scipy.special import beta as Beta


class csgamma_gen(rv_continuous):
    r"""A censored-shifted gamma (CSGamma) random variable, censored numerically below :math:`0`.

    Notes
    -----
    The probability density function (PDF) for a CSGamma random variable is:
    
        .. math::            
           \tilde{f}_{k,\theta,s}(x) =  F_{k,\theta,s}(0)\delta(x) + f_{k,\theta,s}(x) 1_{x>0}(x),
    
    where,
    
        .. math::
            F_{k,\theta,s}(x) = \int_{-\infty}^{x} f_{k,\theta,s}(t) dt
            
    is the cumulative distribution function (CDF) for a gamma distribution with shape parameter 
    :math:`k>0`, scale parameter :math:`\theta>0` and location parameter :math:`s\leq0`, and

        .. math::
            f_{k,\theta,s}(x) = \frac{(x-s)^{k-1} e^{\frac{-(x-s)}{\theta}}}{\theta^{k}\Gamma(k)}
         
    is its PDF. The support for a CSGamma random variable is :math:`x \geq 0`. The location parameter :math:`s<0`
    shifts the gamma distribution to the left below 0, allowing for a portion of the distribution to be censored and become
    a point mass at :math:`x=0`.


    :class:`csgamma_gen` inherets all of the available methods from :py:class:`~scipy.stats.rv_continuous`.
    Those that have been subclassed are:

    ``_pdf``
        
    ``_cdf``    

    ``_ppf``

    
    Additional methods added to :class:`csgamma_gen` are:       
        
    ``ecdf``
        Function for evaluating the empirical distribution function for some sample of data.

    ``crps_csgamma``
        Function for computing the continuous rank probability score when one or both of 
        the CDF's in that score are represented by a censored shifted gamma distribution.
    
    """

    #####################################################################
    ######################### Subclassed Methods ########################
    #####################################################################
    def gamma_sss(self, k, theta, s):
        """
        Returns a shape-shift-scale paramaterization of the gamma distribution using :class:`scipy.stats.gamma`.
        Parameterized so that when :math:`s<=0`, the shifting is to the left below zero.
        """
        return gamma(k, loc=s, scale=theta)

    def _pdf(self, x, k, theta, s):
        """
        Subclass the _pdf method (returns the pdf of the 
        CSgamma distribution at x)
        """

        gs = self.gamma_sss(k, theta, s)
        # the ranges of x that break up the piecewise
        # pdf
        condlist = [x < 0.0, x == 0.0, x > 0]
        # the piecewise pdf associated with the entries in condlist
        choicelist = [0.0, gs.cdf(0.0), gs.pdf(x)]

        return np.select(condlist, choicelist)

    def _cdf(self, x, k, theta, s):
        """
        Subclass the _cdf method (returns the cdf of the 
        CSgamma distribution at x)
        """
        gs = self.gamma_sss(k, theta, s)

        # the ranges of x that break up the piecewise
        # cdf
        condlist = [x < 0.0, x >= 0.0]
        # the piecewise pdf associated with the entries in condlist
        choicelist = [0.0, gs.cdf(x)]

        return np.select(condlist, choicelist)

    def _ppf(self, rho, k, theta, s):
        """
        Subclass the _ppf method (returns the inverse of the cumulative distribution function for
        the CSGamma distribution at probabilities rho).
        """
        gs = self.gamma_sss(k, theta, s)
        condlist = [np.logical_and(rho >= 0, rho <= gs.cdf(0.0)), rho > gs.cdf(0.0)]
        # the piecewise pdf associated with the entries in condlist
        choicelist = [0.0, gs.ppf(rho)]
        return np.select(condlist, choicelist)

    def fit(self, x):
        ###########################################################################################
        ################# Performs MLE on the data using the CSGamma distribution #################
        ################# and minimizing the negative of its log likelihood       #################
        ################# when no 0's are present in the data                   ###################
        ################# this problem reduces to the MLE of the gamma distribution ###############
        ################# with loc=0                                                ###############
        ###########################################################################################

        if np.any(x == 0):

            def loglikelihood(params, y):
                k, theta, s = params.T

                x_sub = x[x != 0.0]
                m = len(x_sub)
                n0 = len(x) - m

                # transform variables
                y0 = (0.0 - s) / theta
                y_sub = (x_sub - s) / theta

                # first term based on CDF at 0
                T1 = n0 * np.log(gamma.cdf(y0, k))
                # Second term based on PDF of gamma
                T2 = (
                    (k - 1) * np.sum(np.log(y_sub))
                    - np.sum(y_sub)
                    - m * (np.log(theta) + np.log(Gamma(k)))
                )
                return -(T1 + T2)

            # initial guesses for k0 and theta0 are from regular gamma distribution fit to sample>0
            k0, s0, theta0 = gamma.fit(
                x[x > 0], floc=0.0
            )  # force the location parameter s to be 0.0
            s0 = -1.0  # set it to an arbitrary negative number close to 0
            params0 = [k0, theta0, s0]  # initial guesses

            # minimize the negative of the log-likelihood of the GSGamma distribution
            res = minimize(
                loglikelihood,
                params0,
                bounds=((0.0, np.inf), (0.0, np.inf), (-np.inf, np.inf)),
                args=(x,),
            )
            k, theta, s = res.x
        else:
            # fit to regular gamma, but force location parameter (s) to be zero
            k, s, theta = gamma.fit(x, floc=0)

        return k, theta, s

    def _argcheck(self, k, theta, s):
        # subclass the argcheck method to ensure parameters
        # are constrained to their bounds
        check = (k > 0.0) & (theta > 0.0)
        if check == True:
            return True
        else:
            return False

    def ecdf(self, x, data):
        r"""
        For computing the empirical cumulative distribution function (ecdf) of a
        given sample.
        
        Args:
            x (float or ndarray):
                The value(s) at which the ecdf is evaluated
               
            data (float or ndarray):
                A sample for which to compute the ecdf.
                
        Returns: ecdf_vals (ndarray):            
            The ecdf for X_samp, evaluated at x.
            
        """

        if isinstance(x, np.float):
            # if x comes in as float, turn it into a numpy array
            x = np.array([x])

        if isinstance(data, np.float):
            # if X_samp comes in as float, turn it into a numpy array
            data = np.array([data])

        # sort the values of X_samp from smallest to largest
        xs = np.sort(data)

        # get the sample size of xs satisfying xs<=x for each x
        def func(vals):
            return len(xs[xs <= vals])

        ys = [len(xs[xs <= vals]) for vals in x]

        return np.array(ys) / float(len(xs))

    def crps_csgamma(self, params_fcst, params_obs=None, x=None, y=None):
        """
        A general function for computing the CRPS for a single forecast/observation pair.
        
        * If the observation is a single value (i.e. not described by a CSGamma distribution),
        the CRPS is evaluated using the closed-form solution provided in Eq. 10 in 
        Scheuerer and Hamill 2015.
        
        * If the observation is described by a censored-shifted gamma distribution,
        the CRPS is solved for numerically. 
        
        Args:
            params_fcst (list or array), shape=(3,):
                The `k, theta, s` parameters of a CSGamma distribution for the forecast 

            params_obs (list or array, optional), shape=(3,):
                The `k, theta, s` parameters of a CSGamma distribution for the observation(s)
            
            x (array, optional):
                If `params_obs` is NOT provided, this argument must be provided. It is the discretized
                range of the independent variable. For instance, for precipitation this may be created using
                `x = np.arange(0, 50+1, 1)` to discretize precipitation from 0 mm to 50 mm by 1 mm step size. This
                variable is used to evaluate the CDF's of the forecast and observation.
                
            y (float, optional):
                If `params_obs` is NOT provided, this argument must be provided. It is the observation.
                            
        """

        if params_obs is None:
            assert y is not None

        if params_obs is not None:
            assert x is not None

        k, theta, s = params_fcst
        if params_obs is None:
            # use the closed-form solution
            c_tilde = -s / theta
            y_tilde = (y - s) / theta

            T1 = theta * y_tilde * (2.0 * gamma.cdf(y_tilde, k) - 1.0)
            T2 = -theta * c_tilde * gamma.cdf(c_tilde, k) ** 2.0
            T3 = (
                theta
                * k
                * (
                    1.0
                    + 2 * gamma.cdf(c_tilde, k) * gamma.cdf(c_tilde, k + 1)
                    - gamma.cdf(c_tilde, k) ** 2.0
                    - 2.0 * gamma.cdf(y_tilde, k + 1)
                )
            )
            T4 = (
                -theta
                * k
                / np.pi
                * Beta(0.5, k + 0.5)
                * (1.0 - gamma.cdf(2 * c_tilde, 2.0 * k))
            )

            return T1 + T2 + T3 + T4

        if params_obs is not None:
            cdf_fcst = self.cdf(x, k, theta, s)
            cdf_obs = self.cdf(x, params_obs[0], params_obs[1], params_obs[2])

            return np.trapz((cdf_fcst - cdf_obs) ** 2.0, x)


csgamma = csgamma_gen(name="csgamma", shapes="k,theta,s")

