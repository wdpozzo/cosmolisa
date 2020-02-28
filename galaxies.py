import numpy
import pickle

class Galaxy(object):
    """
    Galaxy class:
    initialise a galaxy defined by its redshift, redshift error
    and weight determined by its angular position
    relative to the LISA posterior
    """
    def __init__(self, ID, right_ascension, declination, redshift, is_detected, abs_magnitude=None, app_magnitude=None, weight=None):

        self.ID                 = ID
        self.right_ascension    = right_ascension
        self.declination        = declination
        self.abs_magnitude      = abs_magnitude
        self.app_magnitude      = app_magnitude
        self.redshift    = redshift
        # self.dredshift   = dredshift
        self.weight      = weight
        self.is_detected = is_detected # Boolean value


def read_galaxy_catalog(catalog_file, limits):
    '''
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
    glade_names = "PGCname, GWGCname, HyperLedaname, 2MASSname, SDSS-DR12name,\
                    flag1, RA, DEC, dist, dist_err, z, B, B_err, B_abs, J, J_err,\
                    H, H_err, K, K_err, flag2, flag3"
    data = np.genfromtxt(catalog_file, names=glade_names)
    catalog = []
    for i in range(data.shape[0]):
        # Check the entries: B-band mag (abs and apparent), redshift and proximity to GW position posteriors
        if (~np.isnan(data['B'][i])) and (~np.isnan(data['B_abs'][i])) and (data['flag3'][i] == 1 or data['flag3'][i] == 3) and isinbound(data[i], limits):
            catalog.append(Galaxy(i, data['RA'][i], data['DEC'][i], data['z'][i], 1, app_magnitude = data['band'][i])) # Controlla nomi con catalogo!
            # Warning: GLADE stores no information on dz. 2B corrected.

    return catalog

def isinbound(galaxy, limits):
    if (limits['RA'][0] <= galaxy['RA'] <= limits['RA'][1]) and (limits['DEC'][0] <= galaxy['DEC'] <= limits['DEC'][1]) and (limits['z'][0] <= galaxy['z'] <= limits['z'][1]) :
        return 1
    return 0
