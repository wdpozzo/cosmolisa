from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,pow,log10,INFINITY
cimport cython
from scipy.integrate import quad
from scipy.special.cython_special cimport erf
from scipy.special import logsumexp
from scipy.optimize import newton
from schechter import *
from scipy.integrate import quad
from galaxy cimport Galaxy
from cosmology cimport CosmologicalParameters

cdef inline double log_add(double x, double y): return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))
cdef inline double linear_density(double x, double a, double b): return a+log(x)*b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cpdef double logLikelihood_single_event(list hosts, object event, CosmologicalParameters omega, double m_th, int Ntot, int em_selection = 0, double zmin = 0.0, double zmax = 1.0):
    """
    Likelihood function for a single GW event.
    Loops over all possible hosts to accumulate the likelihood
    Parameters:
    ===============
    hosts: :obj: 'numpy.array' with shape Nx3. The columns are redshift, redshift_error, angular_weight
    event: :obj: 'something.Event'. Stores posterior distributions for GW event.
    x: :obj: 'numpy.double' cpnest sampling array.
    meandl: :obj: 'numpy.double': mean of the DL marginal likelihood
    sigma: :obj:'numpy.double': standard deviation of the DL marginal likelihood
    omega: :obj:'lal.CosmologicalParameter': cosmological parameter structure
    Ntot: :obj: 'numpy.int': Total number of galaxies in the considered volume (seen and unseen)
    event_redshift: :obj:'numpy.double': redshift for the the GW event
    em_selection :obj:'numpy.int': apply em selection function. optional. default = 0
    """
    cdef unsigned int i
    cdef unsigned int N = len(hosts)
    cdef unsigned int M = Ntot-N
    cdef double logTwoPiByTwo = 0.5*log(2.0*np.pi)
    cdef double logL_galaxy
    cdef double dl
    cdef double score_z, sigma_z
    cdef double logL_sum = -INFINITY
    cdef double logL_prod = 0
    cdef double p_no_post_dark, p_with_post_dark

    cdef Galaxy mockgalaxy = Galaxy(-1, 0,0,0,False, weight = 1./Ntot)

    cdef np.ndarray[double, ndim=1, mode="c"] p_with_post = np.zeros(N, dtype=np.float64)
    cdef double[::1] p_with_post_view = p_with_post
    cdef np.ndarray[double, ndim=1, mode="c"] p_no_post = np.zeros(N, dtype=np.float64)
    cdef double[::1] p_no_post_view = p_no_post
    # Attenzione: non sono ancora stati sistemati i prior sulle posizioni per la dark galaxy
    for i in range(N):
        # Voglio calcolare, per ogni galassia, le due
        # quantità rilevanti descritte in CosmoInfer.
        p_no_post_view[i]   = ComputeLogLhNoPost(hosts[i], omega, zmin, zmax)
        p_with_post_view[i] = ComputeLogLhWithPost(hosts[i], event, omega, zmin, zmax)

    # Calcolo le likelihood anche per una singola dark galaxy
    p_no_post_dark   = ComputeLogLhNoPost(mockgalaxy, omega, zmin, zmax)
    p_with_post_dark = ComputeLogLhWithPost(mockgalaxy, event, omega, zmin, zmax)
    # Calcolo i termini che andranno sommati tra loro (logaritmi)
    cdef np.ndarray[double, ndim=1, mode="c"] addends = np.zeros(N, dtype=np.float64)
    cdef double[::1] addends_view = addends
    cdef double sum = np.sum(p_no_post)

    for i in range(N):
        addends_view[i] = sum - p_no_post_view[i] + p_with_post_view[i] + M*p_no_post_dark
    cdef double dark_term = sum + (M-1)*p_no_post_dark + p_with_post_dark

    # Manca da fare la somma finale

    cdef double logL = -INFINITY
    for i in range(N):
        logL = log_add(addends_view[i], logL)

    if np.isfinite(dark_term):
        for i in range(M):
            logL = log_add(dark_term, logL)

    return logL

cdef inline double absM(double z, double m, CosmologicalParameters omega):
    '''
    Magnitudine assoluta di soglia
    '''
    return m - 5.0*log10(1e5*omega.LuminosityDistance(z)) # promemoria: 10^5 è Mpc/10pc

cdef inline double SchVar(double M, double Mstar) nogil:
    return pow(10,0.4*(Mstar-M))

cdef double myERF(double x) nogil:
    return (1+erf(x))/2.

cdef inline double gaussian(double x, double x0, double sigma) nogil:
    return exp(-(x-x0)**2/(2*sigma**2))/(sigma*sqrt(2*M_PI))

cdef double Integrand_dark(double z, CosmologicalParameters omega, double alpha, double Mstar, double Mmin, double Mmax, double CoVol):
    return -(gammainc(alpha+2,SchVar(Mmax, Mstar))-gammainc(alpha+2,SchVar(Mmin, Mstar)))*omega.ComovingVolumeElement(z)/CoVol

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ComputeLogLhWithPost(Galaxy gal, object event, CosmologicalParameters omega, double zmin, double zmax, double m_th = 18, double M_max = 0, double M_min = -27):
    '''
    Attenzione: controllare i nomi al momento di definire la classe Event
    '''

    cdef unsigned int i, n = 1000
    cdef double mag_int
    cdef double LD_i

    cdef double I = 0.0
    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef double[::1] z_view = z

    cdef double dz = (zmax - zmin)/n

    cdef object post_RA  = event.post_RA
    cdef object post_DEC = event.post_DEC
    cdef object post_LD  = event.post_LD

    cdef double CoVol
    cdef object Schechter
    cdef double alpha, Mstar, Mth

    if gal.is_detected:
        # if absM(gal.z, gal.app_magnitude, omega) > absM(gal.z, m_th, omega):
        #     return -INFINITY
        # else:
        mag_int = myERF(m_th) # Integrale distribuzione in magnitudine (Analitico, vedi pdf.)
        for i in range(n):
            LD_i = omega.LuminosityDistance(z_view[i])
            I += event.post_LD(LD_i)*gaussian(z_view[i], gal.z, gal.dz) # Attenzione! GLADE non ha l'info sul dz. Va deciso "a mano"
        return log(dz*I*mag_int*gal.weight*event.post_RA(gal.RA)*event.post_DEC(gal.DEC))
    else:
        Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h) # Modo semplice per tirare fuori i parametri di Schechter
        CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))
        for i in range(n):
            LD_i = omega.LuminosityDistance(z_view[i])
            Mth = absM(z_view[i], m_th, omega)
            I += post_LD(LD_i)*Integrand_dark(z_view[i], omega, alpha, Mstar, Mth, M_max, CoVol)

        return log(dz*I*gal.weight)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ComputeLogLhNoPost(Galaxy gal, CosmologicalParameters omega, double zmin, double zmax, double m_th = 17, double M_max = 0, double M_min = -27):
    '''
    Calcolo probabilità di osservare la galassia considerata.
    Si considera, nel caso di galassia osservata, la densità di probabilità dovuta alla misura (gaussiane con errore da determinarsi)
    Nel caso di galassia non osservata invece è necessario integrare sulle distribuzioni di probabilità (Schechter e dV/dz).

    m_th è la threshold dello strumento, Mmax è la massima magnitudine assoluta si ipotizza una galassia possa avere.

    Nota: Il prior in posizione per le galassie che non ho visto è 1/4pi (ovvero tutto il cielo) oppure una porzione corrispondente
    alla regione al 95%?
    '''
    cdef unsigned int i, n = 1000
    cdef double mag_int
    cdef double LD_i

    cdef double I = 0.0
    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef double[::1] z_view = z

    cdef double dz = (zmax - zmin)/n

    cdef object Schechter
    cdef double alpha, Mstar, CoVol, Mth

    if (gal.is_detected and M_min is None) or (m_th is None and not gal.is_detected):
        raise SystemExit('No M or m threshold provided.')

    if gal.is_detected:
        # if absM(gal.z, gal.app_magnitude, omega) > absM(gal.z, m_th, omega):
        #     print('invalid cosmology')
        #     return -INFINITY
        # else:
        return log(myERF(m_th))

    else:
        Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h) # Modo semplice per tirare fuori i parametri di Schechter
        CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

        for i in range(n):
            Mth = absM(z_view[i], m_th, omega)
            I += Integrand_dark(z_view[i], omega, alpha, Mstar,Mth, M_max, CoVol)

        return log(dz*I)
