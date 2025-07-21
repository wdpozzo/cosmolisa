"""
# cython: profile=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
cimport cython
from libc.math cimport exp, sqrt, M_SQRT1_2, M_2_SQRTPI, pi
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
                            gal_interp,
                            const double com_vol):
    return _lk_dark_single_event_trap(hosts, meandl, sigmadl, omega,
                                      model, zmin, zmax, gal_interp, com_vol)


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
                            gal_interp,
                            const double com_vol):

    cdef int i
    cdef int N = 100
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin + dz
    cdef double z_prior_norm = 0.0
    cdef double I = (0.5
        * (_lk_dark_single_event_integrand_trap(zmin, hosts, meandl,
                                                sigmadl, omega, model,
                                                zmin, zmax, gal_interp,
                                                com_vol)
        + _lk_dark_single_event_integrand_trap(zmax, hosts, meandl,
                                               sigmadl, omega, model,
                                               zmin, zmax, gal_interp,
                                               com_vol)))
    if not com_vol == 1:
        z_prior_norm = (0.5 * (gal_interp(zmin) + gal_interp(zmax)))
    else:
        z_prior_norm = (0.5 * (gal_interp(zmin)
                * omega._ComovingVolumeElement(zmin) 
                + gal_interp(zmax)*omega._ComovingVolumeElement(zmax)))

    for i in range(1, N):
        I += _lk_dark_single_event_integrand_trap(z, hosts, meandl,
                                                  sigmadl, omega, model,
                                                  zmin, zmax, gal_interp,
                                                  com_vol)
        if not com_vol == 1:
            z_prior_norm += gal_interp(z)
        else:
            z_prior_norm += gal_interp(z)*omega._ComovingVolumeElement(z) 

        z += dz
    
    return I/z_prior_norm


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
                                        gal_interp,
                                        const double com_vol):

    cdef double dl
    cdef double L_galaxy = 0.0
    cdef double L_detector = 0.0
    cdef double OneOverSqrtTwoPi = M_SQRT1_2*0.5*M_2_SQRTPI # 1/sqrt(2*pi)

    # GW likelihood: N(dl - meandl; sigmadl^2)
    dl = omega._LuminosityDistance(event_redshift) # dl given Omega and z
    cdef double weak_lensing_error = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared = sigmadl**2 + weak_lensing_error**2
    cdef double SigmaNorm = OneOverSqrtTwoPi * 1/sqrt(SigmaSquared)
    L_detector = (SigmaNorm * exp(-0.5*(dl-meandl)*(dl-meandl)
                  / SigmaSquared))

    L_galaxy = gal_interp(event_redshift)

    if com_vol == 1:
        L_galaxy *= omega._ComovingVolumeElement(event_redshift) 

    return L_detector * L_galaxy


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


cpdef build_interpolant(event):
    """Build an interpolant for the galaxy redshift prior of the event.
    Parameters
    ----------
    event: Event
        The event for which to build the interpolant.
    Returns
    -------
    interpolant: function
        A function of the galaxy weighted mixture model.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    cdef int i
    cdef double[:] z_range = np.linspace(event.zmin, event.zmax, 1000)
    # cdef cnp.ndarray[double, ndim=1] z_range = np.linspace(event.zmin, event.zmax, 1000)
    cdef double[:] mixture = np.zeros(z_range.shape[0])

    for gal in event.potential_galaxy_hosts:
        sigma_z = gal.dredshift * (1 + gal.redshift)
        for i in range(len(z_range)):
            score_z = (z_range[i] - gal.redshift) / sigma_z
            p_gal = (gal.weight / (sqrt(2 * pi) * sigma_z)
                     * exp(-0.5 * score_z**2))
            mixture[i] += p_gal

    # NO DIFFERENCE IF THE FOLLOWING TWO LINES ARE INCLUDED 
    # (AS EXPECTED SINCE THIS NORMALIZATION DOES NOT DEPEND ON 
    # THE COSMOLOGY)
    # cdef double normalization_factor = np.trapz(mixture, z_range)
    # mixture = np.asarray(mixture) / normalization_factor

    # Create the interpolant
    interpolant = interp1d(z_range, mixture, bounds_error=False,
                           fill_value=0.0)

    # # CHECK THE INTERPOLANT
    # import matplotlib.pyplot as plt

    # true_values = np.array([mixture[i] for i in range(len(z_range))])
    # interpolant_values = np.array([interpolant(z) for z in z_range])

    # plt.plot(z_range, true_values, label="True Function", linestyle="--")
    # plt.plot(z_range, interpolant_values, label="Interpolant", linestyle="-")
    # plt.xlabel("Redshift (z)")
    # plt.ylabel("Galaxy Weighted Mixture")
    # plt.legend()
    # plt.title("Comparison of True Function vs Interpolant")
    # plt.show()
    print(f"Computed galaxy mixture model for event {event.ID}.")
    return interpolant


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
