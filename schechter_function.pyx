"""
Module with Schechter magnitude function:
(C) Walter Del Pozzo (2014)
Modified by SR
"""
from scipy.integrate import quad
from scipy.special import gammainc, gamma
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp, pow

# format (Mstar_obs, alpha, mmin, mmax)
cpdef dict schechter_function_params = { 'B':(-20.457,-1.07),
                                         'K':(-23.55,-1.02)}

cdef class SchechterMagFunctionInternal:
    """
    Returns a Schechter magnitude function for a given set of parameters

    Parameters
    ----------
    Mstar_obs : observed characteristic magnitude used to define
                Mstar = Mstar_obs + 5.*np.log10(h)
    alpha : observed characteristic slope.
    phistar : density (can be set to unity)
    """
    cdef public double Mstar
    cdef public double phistar
    cdef public double alpha
    cdef public double mmin
    cdef public double mmax
    cdef public double norm
    def __cinit__(self, double Mstar, double alpha, double mmin, double mmax, double phistar=1.):
        self.Mstar = Mstar
        self.phistar = phistar
        self.alpha = alpha
        self.mmin = mmin
        self.mmax = mmax
        self.norm = -1

    def __call__(self, double m):
        return self._evaluate(m)

    cpdef double _evaluate(self, double m):
        return 0.4*log(10.0)*self.phistar \
               * pow(10.0, -0.4*(self.alpha+1.0)*(m-self.Mstar)) \
               * exp(-pow(10, -0.4*(m-self.Mstar)))

    cdef double normalise(self):
        cdef double lowbound
        cdef double hibound
        if self.norm == -1:

            lowbound = pow(10, -0.4*(self.mmax-self.Mstar))
            hibound  = pow(10, -0.4*(self.mmin-self.Mstar))

            # self.norm = quad(self._evaluate, self.mmin, self.mmax)[0]
            self.norm = (gammainc(self.alpha+2, hibound)-gammainc(self.alpha+2, lowbound))*gamma(self.alpha+2)
        return self.norm

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cpdef double pdf(self, double m):
        self.normalise()
        return self._evaluate(m)/self.norm

cpdef tuple SchechterMagFunction(double mmin, double mmax, double h=0.7, str band='B'):
    """
    Returns a Schechter magnitude function for a given set of parameters

    Parameters
    ----------
     : Hubble parameter in (km/s/Mpc)/100 (default=0.7)
    band : Either B or K band magnitude to define SF params (default='B').

    Example usage
    -------------

    smf = SchechterMagFunction(h=0.7, band='B')
    (integral, error) = scipy.integrate.quad(smf)
    """
    if band == 'constant': # Perform incompleteness correction using B-band SF for constant luminosity weights
        band = 'B'
    cdef double Mstar_obs, alpha
    
    Mstar_obs, alpha = schechter_function_params[band]
    cdef double Mstar = Mstar_obs + 5.*np.log10(h)
    cdef object smf = SchechterMagFunctionInternal(Mstar, alpha, mmin, mmax)
    return smf.pdf, alpha, Mstar

cdef inline double M_Mobs(double h, double M_obs):
    """
    Given an observed absolute magnitude, returns absolute magnitude
    """
    return M_obs + 5.*np.log10(h)
