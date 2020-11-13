from libc.math cimport INFINITY, isnan
import numpy as np
cimport numpy as np
import pickle

cdef class Galaxy:
    """
    Galaxy class:
    initialise a galaxy defined by its redshift, redshift error
    and weight determined by its angular position
    relative to the LISA posterior
    """
    def __init__(self, int ID,
                       double right_ascension,
                       double declination,
                       double redshift,
                       bint is_detected,
                       double z_error = 0.01,
                       double abs_magnitude=INFINITY,
                       double app_magnitude=INFINITY,
                       double dapp_magnitude=INFINITY,
                       double weight=0.0):

        self.ID             = ID
        self.RA             = right_ascension
        self.DEC            = declination
        self.z              = redshift
        self.dz             = redshift*z_error
        self.weight         = weight
        self.is_detected    = is_detected # Boolean value
        self.abs_magnitude  = abs_magnitude
        self.app_magnitude  = app_magnitude
        self.dapp_magnitude = dapp_magnitude


cpdef object read_galaxy_catalog(dict limits, str catalog_file = '', double n_tot = 1.):
    '''
    The catalog can be passed either as a path or, if precedently loaded, as np.array.
    In case both data and path are provided, already loaded data are used.

    GLADE flag description:

    flag1:  Q: the source is from the SDSS-DR12 QSO catalog
            C: the source is a globular cluster
            G: the source is from another catalog and not identified as a globular cluster

    flag2:  0: no z nor LD
            1: measured z
            2: measured LD
            3: measured spectroscopic z

    flag3:  0: velocity field correction has not been applied to the object
            1: we have subtracted the radial velocity of the object
    '''

#    if catalog_file is None:
#        raise SystemExit('No catalog data nor file provided.')
#
#
#    if catalog_file is not None:
#    '''
#        glade_names = "PGCname, GWGCname, HyperLedaname, 2MASSname, SDSS-DR12name,\
#                    flag1, RA, DEC, dist, dist_err, z, B, B_err, B_abs, J, J_err,\
#                    H, H_err, K, K_err, flag2, flag3"
#    '''
    catalog_data = np.atleast_1d(np.genfromtxt(catalog_file, names=True)) # Troubles with single row files.

    catalog = []

    for i in range(catalog_data.shape[0]):
        # Check the entries: B-band mag (abs and apparent), redshift and proximity to GW position posteriors
        temp_gal = Galaxy(i, np.deg2rad(catalog_data['ra'][i]), np.deg2rad(catalog_data['dec'][i]), catalog_data['z'][i], True, z_error = z_error, app_magnitude = catalog_data['B'][i], dapp_magnitude = catalog_data['B_err'][i] , abs_magnitude = catalog_data['B_abs'][i])
        if isinbound(temp_gal, limits):
            z_error = 0.1  # assumo un 10% conservativo oltre i 300 Mpc
            catalog.append(temp_gal) # Controlla nomi con catalogo!
            # Warning: GLADE stores no information on dz. 2B corrected.

    catalog = catalog_weight(catalog, weight = 'uniform', ngal = n_tot) # Implementare meglio la selezione del peso delle galassie.

    return catalog

cdef bint isinbound(Galaxy galaxy, dict limits):
    if (limits['RA'][0] <= np.deg2rad(galaxy.RA) <= limits['RA'][1]) and (limits['DEC'][0] <= np.deg2rad(galaxy.DEC) <= limits['DEC'][1]) and (limits['z'][0] <= galaxy.z <= limits['z'][1]) :
        return True
    return False


cpdef object catalog_weight(object catalog, str weight = 'uniform', double ngal = 1.):
    '''
    Method:
    Assign a weight for each galaxy in catalog according to the emission probability
    of the galaxy.
    Please note that this has to be a relative probability rather than an absolute one.

        - Uniform weighting: 1/N ('uniform')
    '''
    if weight == 'uniform':
        for galaxy in catalog:
            if ngal == 1:
                galaxy.weight = 1./len(catalog)
            else:
                galaxy.weight = 1./ngal

    return catalog
