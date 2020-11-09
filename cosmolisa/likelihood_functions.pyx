import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,HUGE_VAL,INFINITY,log10
cimport cython
from define_galaxy cimport Galaxy
from scipy.integrate import quad, dblquad
from cosmolisa.cosmology cimport CosmologicalParameters

cdef double ComputeLogLhWithPost()

cdef double ComputeLogLhNoPost()

cdef double ComputeLogLhNoEmission()

cdef double prob_Nobs(unsigned int N_obs, unsigned int N_b):

cdef unsigned int ComputeEmitters(unsigned int N_tot, object Schechter, double M_cutoff, double M_min):
    '''
    Eq. 8
    '''
    return int(N_tot*(quad(Schechter, M_min, M_cutoff)[0])

cdef unsigned int ComputeBright(double n, CosmologicalParameters omega, object Schechter, double m_th, double zmin, double zmax, double M_min):
    '''
    Eq. 9
    '''
    cdef object integrand(double M, double z, CosmologicalParameters omega, object Schechter):
        return omega.ComovingVolumeElement(z)*Schechter(M))
        
    cdef object upperbound(double z, double m_th, CosmologicalParameters omega):
        return lambda z: absM(z,m_th,omega)
        
    return int(n*dblquad(integrand, zmin, zmax, M_min, upperbound, args = (omega, Schechter))[0])
    

cdef inline double absM(double z, double m, CosmologicalParameters omega):
    return m - 5.0*log10(1e5*omega.LuminosityDistance(z)) + 5.*log10(omega.h)
