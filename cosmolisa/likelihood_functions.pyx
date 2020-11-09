import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,INFINITY,log10
cimport cython
from define_galaxy cimport Galaxy
from scipy.integrate import quad, dblquad
from cosmolisa.cosmology cimport CosmologicalParameters

cdef double ComputeLogLhWithPost()

cdef double ComputeLogLhNoPost()

cdef double ComputeLogLhNoEmission()

cdef double prob_Nobs(unsigned int N_obs, unsigned int N_b, string distribution = 'poisson'):
    '''
    Computes the probability of observing N_obs galaxies given the expected number N_b
    Stirling approximation for log factorial:
        log(x!) ~ x log(x) - x
    =====================
    distribution: :obj::'string': probability distribution for the number of bright galaxies. 'poission' or 'binomial'.
    '''
    
    if distribution == 'poisson':
        return N_obs*(log(N_b/N_obs)+1)-N_b
    if distribution == 'binomial':
        print('WARNING: binomial not implemented yet')
        return 0

cdef unsigned int ComputeEmitters(unsigned int N_tot, object Schechter, double M_cutoff, double M_min):
    '''
    Eq. 8
    Computes the expected number of emitting galaxies
    '''
    return int(N_tot*(quad(Schechter, M_min, M_cutoff)[0]))

cdef unsigned int ComputeBright(double n, CosmologicalParameters omega, object Schechter, double m_th, double zmin, double zmax, double M_min):
    '''
    Eq. 9
    Computes the expected number of bright galaxies
    '''
    cdef object integrand(double M, double z, CosmologicalParameters omega, object Schechter):
        return omega.ComovingVolumeElement(z)*Schechter(M))
        
    cdef object upperbound(double z, double m_th, CosmologicalParameters omega):
        return lambda z: absM(z,m_th,omega)
        
    return int(n*dblquad(integrand, zmin, zmax, M_min, upperbound, args = (omega, Schechter))[0])
    

cdef inline double absM(double z, double m, CosmologicalParameters omega):
    return m - 5.0*log10(1e5*omega.LuminosityDistance(z)) + 5.*log10(omega.h)
