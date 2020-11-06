
from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,HUGE_VAL
cimport cython
from scipy.special import logsumexp
from scipy.optimize import newton
from scipy.integrate import quad
from cosmolisa.cosmology cimport CosmologicalParameters, _StarFormationDensity, _IntegrateRateWeightedComovingVolumeDensity
from libc.math cimport isfinite
from define_galaxy cimport Galaxy

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def logLikelihood_single_event(const list hosts,
                                const list catalog,
                                const double m_th,
                                const double number_density,
                                object event,
                                CosmologicalParameters omega,
                                const double zmin = 0.0,
                                const double zmax = 1.0,
                                const double M_max = -6.,
                                const double M_min = -23.,
                                const double M_cutoff = -15.):
    """
    Likelihood function for a single GW event.
    Loops over all possible hosts to accumulate the likelihood
    Parameters:
    ===============
    hosts: :obj:'list': list of potential hosts
    catalog: :obj:'list': complete galaxy catalog
    m_th: :obj:'numpy.double': threshold magnitude
    number_density: :obj:'numpy.double': galaxy number density
    event: :obj: : gravitational event (includes posterior distributions)
    omega: :obj:'lal.CosmologicalParameter': cosmological parameter structure
    zmin: :obj:'numpy.double': minimum redshift
    zmax: :obj:'numpy.double': maximum redshift
    M_max: :obj:'numpy.double': maximum absolute magnitude
    M_min: :obj:'numpy.double': minimum absolute magnitude
    M_cutoff: :obj: 'numpy.double': cutoff magnitude
    """
    return _logLikelihood_single_event(hosts, catalog, m_th, number_density, event, omega, zmin, zmax)

cdef _logLikelihood_single_event(const list hosts,
                                const list catalog,
                                const double m_th,
                                object event,
                                CosmologicalParameters omega,
                                const double zmin,
                                const double zmax,
                                const double M_max,
                                const double M_min,
                                const double M_cutoff):

    cdef unsigned int i
    '''
    N_h: galaxies within 95% CR
    N_obs: observed galaxies
    N_tot: total number of galaxies (nV)
    N_em: potential emitters
    N_b: expected bright galaxies
    N_dark: dark galaxies (N_tot - N_obs)
    N_dark_em: emitting dark galaxies
    N_noem: non-emitting galaxies
    '''
#   Galaxy numbers
    cdef unsigned int N_h   = len(hosts)
    cdef unsigned int N_obs = len(catalog)
    cdef unsigned int N_tot, N_dark, N_dark_em, N_noem, N_b, N_em
#   Probabilities
    cdef double p_dark_with_post, p_dark_no_post
    cdef double p_noem
    cdef np.ndarray[double, ndim = 1, mode = "c"] p_with_post = np.zeros(N_obs, dtype = np.float64)
    cdef np.ndarray[double, ndim = 1, mode = "c"] p_no_post = np.zeros(N_obs, dtype = np.float64)
    cdef double[::1] p_with_post_view = p_with_post
    cdef double[::1] p_no_post_view   = p_no_post
#   Computing
    cdef Galaxy dark_galaxy = Galaxy(-1,0,0,0, False, weight = 1.)
    
