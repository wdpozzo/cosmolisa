import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,erf,erfc,INFINITY,log10
cimport cython
from cosmolisa.galaxy cimport Galaxy
from scipy.integrate import quad, dblquad
from cosmolisa.cosmology cimport CosmologicalParameters

cdef inline double log_add(double x, double y): return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

cdef double ComputeLogLhWithPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    int N_em,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z = 0.0001):
    '''
    Computes the probability for a single galaxy of being the host AND observing/not observing that particular object.
    Parameters:
    ================
    gal: :obj:'Galaxy: Potential host galaxy
    event: :obj: :'object': gravitational event (includes posterior distributions)
    omega: :obj:'lal.CosmologicalParameters': cosmological parameters structure
    zmin: :obj:'numpy.double': minimum redshift
    zmax: :obj:'numpy.double': maximum redshift
    M_cutoff: :obj:'numpy.double': cutoff magnitude for potential emitters
    N_em: :obj:'int': total number of potential emitters
    m_th: :obj:'numpy.double': threshold apparent magnitude
    M_max: :obj:'numpy.double': Schechter function upper limit
    M_min: :obj:'numpy.double': Schechter function lower limit
    sigma_z: :obj:'numpy.double': redshift uncertainty (proper motion)
    '''
    
    
    return _ComputeLogLhWithPost(gal, event, omega, zmin, zmax, M_cutoff, N_em, m_th, M_max, M_min, sigma_z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _ComputeLogLhWithPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    int N_em,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z):

    cdef unsigned int i, j, n = 100
    cdef double dI_z, I_z = -INFINITY
    cdef double dI_M, I_M = -INFINITY

    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] z_view = z
    cdef double[::1] M_view = M
    cdef double dz = (zmax - zmin)/n
    cdef double dm = (M_max - M_min)/n
    
    cdef double LD
    cdef double M_th
    cdef double p_emission, p_post, p_mag, p_propmotion, p_z_cosmo, prior_wave, prior_galaxy
    
    cdef double CoVol = omega.ComovingVolume(zmax) - omega.ComovingVolume(zmin)
    
    if gal.is_detected:
        #redshift integral
        for i in range(n):
            LD = omega.LuminosityDistance(z_view[i])
            p_emission = log(1./N_em)
            p_post = event.logP([LD, gal.DEC, gal.RA])
            p_mag = log(Schechter(absM(gal.z, gal.app_magnitude, omega)))
            p_propmotion = log_gaussian(gal.z, z_view[i], sigma_z))
            p_z_cosmo = log(omega.ComovingVolumeElement(z_view[i]))-log(CoVol)
            prior_wave = -log(4*M_PI)+2*log(LD)-log((omega.LuminosityDistance(zmax)-omega.LuminosityDistance(zmin))/3.)
            prior_galaxy = -log(4*M_PI)
            dI_z = p_post+p_emission+p_mag+prior_galaxy+p_propmotion+p_z_cosmo-prior_wave+log(dz)
            I_z = log_add(I_z, dI_z)
        return I_z

    else:
        #redshift integral
        for i in range(n):
            LD = omega.LuminosityDistance(z_view[i])
            p_emission = log(1./N_em)
            p_post = event.marg_logP(LD)
            p_z_cosmo = log(omega.ComovingVolumeElement(z_view[i]))-log(CoVol)
            prior_wave = -log(4*M_PI)+2*log(LD)-log((omega.LuminosityDistance(zmax)-omega.LuminosityDistance(zmin))/3.)
            prior_galaxy = -log(4*M_PI)
            
            #magnitude integral
            M_th = absM(z_view[i], m_th, omega)
            I_M = -INFINITY
            for j in range(n):
                if (M_view[j] > M_th) and (M_view[j] < M_cutoff):
                    dI_M = log(Schechter(M_view[j], z_view[i], omega)) + log(dM)
                I_M = log_add(I_M, dI_M)
            
            dI_z = p_post+p_emission+I_M+prior_galaxy+p_propmotion+p_z_cosmo-prior_wave+log(dz)
            I_z = log_add(I_z, dI_z)
        return I_z

cdef double ComputeLogLhNoPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z = 0.0001):
    '''
    Computes the probability for a single galaxy of observing/not observing that particular object.
    Parameters:
    ================
    gal: :obj:'Galaxy: Potential host galaxy
    event: :obj: :'object': gravitational event (includes posterior distributions)
    omega: :obj:'lal.CosmologicalParameters': cosmological parameters structure
    zmin: :obj:'numpy.double': minimum redshift
    zmax: :obj:'numpy.double': maximum redshift
    M_cutoff: :obj:'numpy.double': cutoff magnitude for potential emitters
    m_th: :obj:'numpy.double': threshold apparent magnitude
    M_max: :obj:'numpy.double': Schechter function upper limit
    M_min: :obj:'numpy.double': Schechter function lower limit
    sigma_z: :obj:'numpy.double': redshift uncertainty (proper motion)
    '''
    
    return _ComputeLogLhNoPost(gal, event, omega, zmin, zmax, M_cutoff, m_th, M_max, M_min, sigma_z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _ComputeLogLhNoPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z):
                                    
    cdef unsigned int i, j, n = 100
    cdef double dI_z, I_z = -INFINITY
    cdef double dI_M, I_M = -INFINITY

    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] z_view = z
    cdef double[::1] M_view = M
    cdef double dz = (zmax - zmin)/n
    cdef double dm = (M_max - M_min)/n
    
    cdef double LD
    cdef double M_th
    cdef double p_mag, p_propmotion, p_z_cosmo, prior_galaxy
    
    cdef double CoVol = omega.ComovingVolume(zmax) - omega.ComovingVolume(zmin)
    
    if gal.is_detected:
        #redshift integral
        for i in range(n):
            LD = omega.LuminosityDistance(z_view[i])
            p_mag = log(Schechter(absM(gal.z, gal.app_magnitude, omega)))
            p_propmotion = log_gaussian(gal.z, z_view[i], sigma_z))
            p_z_cosmo = log(omega.ComovingVolumeElement(z_view[i]))-log(CoVol)
            prior_galaxy = -log(4*M_PI)
            dI_z = p_mag+p_propmotion+prior_galaxy+p_z_cosmo+log(dz)
            I_z = log_add(I_z, dI_z)
        return I_z

    else:
        #redshift integral
        for i in range(n):
            LD = omega.LuminosityDistance(z_view[i]
            p_z_cosmo = log(omega.ComovingVolumeElement(z_view[i]))-log(CoVol)
            prior_wave = -log(4*M_PI)+2*log(LD)-log((omega.LuminosityDistance(zmax)-omega.LuminosityDistance(zmin))/3.)
            
            #magnitude integral
            M_th = absM(z_view[i], m_th, omega)
            I_M = -INFINITY
            for j in range(n):
                if (M_view[j] > M_th) and (M_view[j] < M_cutoff):
                    dI_M = log(Schechter(M_view[j], z_view[i], omega)) + log(dM)
                I_M = log_add(I_M, dI_M)
            
            dI_z = I_M+p_z_cosmo+log(dz)
            I_z = log_add(I_z, dI_z)
        return I_z
        
cdef double ComputeLogLhNoEmission(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double M_max,
                                    double M_min):
    '''
    Computes the probability for a single galaxy of not observing that particular object assuming that this galaxy is not a potential emitter.
    Parameters:
    ================
    gal: :obj:'Galaxy: Potential host galaxy
    event: :obj: :'object': gravitational event (includes posterior distributions)
    omega: :obj:'lal.CosmologicalParameters': cosmological parameters structure
    zmin: :obj:'numpy.double': minimum redshift
    zmax: :obj:'numpy.double': maximum redshift
    M_cutoff: :obj:'numpy.double': cutoff magnitude for potential emitters
    M_max: :obj:'numpy.double': Schechter function upper limit
    M_min: :obj:'numpy.double': Schechter function lower limit
    '''

    return _ComputeLogLhNoEmission(gal, event, omega, zmin, zmax, M_cutoff, M_max, M_min)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _ComputeLogLhNoEmission(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double M_max,
                                    double M_min):
                                    
    cdef unsigned int i, j, n = 100
    cdef double dI_z, I_z = -INFINITY
    cdef double dI_M, I_M = -INFINITY

    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] z_view = z
    cdef double[::1] M_view = M
    cdef double dz = (zmax - zmin)/n
    cdef double dm = (M_max - M_min)/n
    
    cdef double LD
    cdef double M_th
    cdef double p_z_cosmo
    
    cdef double CoVol = omega.ComovingVolume(zmax) - omega.ComovingVolume(zmin)
    
    #redshift integral
    for i in range(n):
        LD = omega.LuminosityDistance(z_view[i]
        p_z_cosmo = log(omega.ComovingVolumeElement(z_view[i]))-log(CoVol)
        #magnitude integral
        M_th = absM(z_view[i], m_th, omega)
        I_M = -INFINITY
        for j in range(n):
            if (M_view[j] > M_cutoff):
                dI_M = log(Schechter(M_view[j], z_view[i], omega)) + log(dM)
            I_M = log_add(I_M, dI_M)
            
        dI_z = I_M+p_z_cosmo+log(dz)
        I_z = log_add(I_z, dI_z)
    return I_z

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

cdef inline double gaussian(double x, double x0, double sigma) nogil:
    return exp(-(x-x0)**2/(2*sigma**2))/(sigma*sqrt(2*M_PI))
    
cdef inline double log_gaussian(double x, double x0, double sigma) nogil:
    return (-(x-x0)**2/(2*sigma**2))-log(sigma*sqrt(2*M_PI))
