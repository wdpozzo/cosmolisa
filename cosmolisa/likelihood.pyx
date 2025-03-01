"""
# cython: profile=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
cimport cython
from libc.math cimport log, exp, sqrt, cos, fabs, sin, sinh, M_PI, \
    erf, erfc, HUGE_VAL, log1p, M_SQRT1_2, M_2_SQRTPI
from scipy.optimize import newton

from cosmolisa.cosmology cimport CosmologicalParameters

#######################################################################
#                          DARK SIREN
#######################################################################


def lk_dark_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax,
                            const double com_vol):
    return _lk_dark_single_event_trap(hosts, meandl, sigmadl, omega,
                                      model, zmin, zmax, com_vol)


@cython.boundscheck(False) # Disable bounds checking for array accesses
@cython.wraparound(False) # Disable negative indexing for arrays
@cython.nonecheck(False) # Disables automatic checking for None values
@cython.cdivision(True) # Enables C-style division
cdef double _lk_dark_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax,
                            const double com_vol) nogil:

    cdef int i
    cdef int N = 100
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin + dz
    cdef double I = (0.5
        * (_lk_dark_single_event_integrand_trap(zmin, hosts, meandl,
                                                sigmadl, omega, model,
                                                zmin, zmax, com_vol)
        + _lk_dark_single_event_integrand_trap(zmax, hosts, meandl,
                                               sigmadl, omega, model,
                                               zmin, zmax, com_vol)))
    for i in range(1, N):
        I += _lk_dark_single_event_integrand_trap(z, hosts, meandl,
                                                  sigmadl, omega, model,
                                                  zmin, zmax, com_vol)
        z += dz
    return I*dz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_dark_single_event_integrand_trap(const double event_redshift,
                                        const double[:,::1] hosts, #(Nx4) matrix
                                        const double meandl,
                                        const double sigmadl,
                                        CosmologicalParameters omega,
                                        str model,
                                        const double zmin,
                                        const double zmax,
                                        const double com_vol) nogil:

    cdef unsigned int j
    cdef double dl
    cdef double L_gal = 0.0
    cdef double L_galaxy = 0.0
    cdef double L_detector = 0.0
    cdef double sigma_z, score_z
    cdef unsigned int Ng = hosts.shape[0]
    cdef double OneOverSqrtTwoPi = M_SQRT1_2*0.5*M_2_SQRTPI # 1/sqrt(2*pi)

    # GW likelihood: N(dl - meandl; sigmadl^2)
    dl = omega._LuminosityDistance(event_redshift) # dl given Omega and z
    cdef double weak_lensing_error = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared = sigmadl**2 + weak_lensing_error**2
    cdef double SigmaNorm = OneOverSqrtTwoPi * 1/sqrt(SigmaSquared)
    L_detector = (SigmaNorm * exp(-0.5*(dl-meandl)*(dl-meandl)
                  / SigmaSquared))

    # Redshift prior: sum_j^Ng w_j*N(z - zj; sigmaz^2)
    for j in range(Ng):

        # Estimate sig_z_j ~ (z_jobs-z_jcos) = (v_pec/c)*(1+z_j).
        sigma_z = hosts[j,1] * (1 + hosts[j,0])

        score_z = (event_redshift - hosts[j,0])/sigma_z
        L_gal = (hosts[j,2] * OneOverSqrtTwoPi * (1/sigma_z)
                 * exp(-0.5*score_z*score_z))
        L_galaxy += L_gal
  
    # Additional dV/dz factor
    if com_vol == 1:
        dVdz = omega._ComovingVolumeElement(event_redshift)
    else:
        dVdz = 1.0

    return L_detector * L_galaxy * dVdz


#######################################################################
#                          BRIGHT SIREN
#######################################################################

def lk_bright_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax):
    """Likelihood function p( Di | O, M, I) for a single bright GW 
    event of data Di assuming cosmological model M and parameters O.
    Following the formalism of <arXiv:2102.01708>.
    Use EM host data to compute the likelihood.
    Parameters:
    ===============
    hosts: :obj: 'numpy.array' with shape Nx4. The columns are
        redshift, redshift_error, angular_weight, magnitude
    meandl: :obj: 'numpy.double': LISA mean of the luminosity distance dL
    sigmadl: :obj: 'numpy.double': LISA standard deviation of dL
    omega: :obj: 'lal.CosmologicalParameter': cosmological parameter
        structure O
    event_redshift: :obj: 'numpy.double': redshift for the GW event
    zmin: :obj: 'numpy.double': minimum GW redshift
    zmax: :obj: 'numpy.double': maximum GW redshift
    """

    return _lk_bright_single_event_trap(hosts, meandl, sigmadl, omega,
                                      model, zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_bright_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax):

    cdef int i
    cdef int N = 100
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin + dz
    cdef double I = (0.5
        * (_lk_bright_single_event_integrand_trap(zmin, hosts, meandl,
                                                sigmadl, omega, model)
        + _lk_bright_single_event_integrand_trap(zmax, hosts, meandl,
                                               sigmadl, omega, model)))
    for i in range(1, N):
        I += _lk_bright_single_event_integrand_trap(z, hosts, meandl,
                                                  sigmadl, omega, model)
        z += dz
    return I*dz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_bright_single_event_integrand_trap(
                                        const double event_redshift,
                                        const double[:,::1] hosts,
                                        const double meandl,
                                        const double sigmadl,
                                        CosmologicalParameters omega,
                                        str model) nogil:

    cdef double dl
    cdef double L_EM = 0.0
    cdef double L_detector = 0.0
    cdef double sigma_z, score_z
    cdef double OneOverSqrtTwoPi = M_SQRT1_2*0.5*M_2_SQRTPI

    # GW likelihood: N(dl - meandl; sigmadl^2)
    dl = omega._LuminosityDistance(event_redshift)
    cdef double weak_lensing_error = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared = sigmadl**2 + weak_lensing_error**2
    cdef double SigmaNorm = OneOverSqrtTwoPi * 1/sqrt(SigmaSquared)
    L_detector = (SigmaNorm * exp(-0.5*(dl-meandl)*(dl-meandl)
                  / SigmaSquared))

    # Redshift prior: N(z - zEM; sigmaz^2)
    # Read sig_z_EM from EM data.
    sigma_z = hosts[0,1]

    score_z = (event_redshift - hosts[0,0])/sigma_z
    L_EM = (OneOverSqrtTwoPi * (1/sigma_z)
                * exp(-0.5*score_z*score_z))
    
    return L_detector * L_EM


##########################################################
#                                                        #
#                   Other functions                      #
#                                                        #
##########################################################


def sigma_weak_lensing(const double z, const double dl):
    return _sigma_weak_lensing(z, dl)

cdef inline double _sigma_weak_lensing(const double z, 
                                       const double dl) nogil:
    """Weak lensing error. From <arXiv:1601.07112v3>,
    Eq. (7.3) corrected by a factor 0.5 
    to match <arXiv:1004.3988v2>.
    Parameters:
    ===============
    z: :obj:'numpy.double': redshift
    dl: :obj:'numpy.double': luminosity distance
    """
    return 0.5*0.066*dl*((1.0-(1.0+z)**(-0.25))/0.25)**1.8


cpdef double find_redshift(CosmologicalParameters omega, double dl):
    return newton(objective, 1.0, args=(omega,dl))

cdef double objective(double z, CosmologicalParameters omega, double dl):
    return dl - omega._LuminosityDistance(z)
