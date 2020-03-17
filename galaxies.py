#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pickle

class Galaxy(object):
    """
    Galaxy class:
    initialise a galaxy defined by its redshift, redshift error
    and weight determined by its angular position
    relative to the LISA posterior
    """
    def __init__(self, ID, right_ascension, declination, redshift, is_detected, abs_magnitude=None, app_magnitude=None, weight=None):

        self.ID             = ID
        self.RA             = right_ascension
        self.DEC            = declination
        self.abs_magnitude  = abs_magnitude
        self.app_magnitude  = app_magnitude
        self.z              = redshift
        self.dz             = redshift*0.1
        self.weight         = weight
        self.is_detected    = is_detected # Boolean value




def read_galaxy_catalog(limits, catalog_data = None, catalog_file = None):
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

    if catalog_data is None and catalog_file is None:
        raise SystemExit('No catalog data nor file provided.')

    if catalog_data is not None and catalog_file is not None:
        print('Both data and path provided. Loaded data will be used.')

    if catalog_file is not None and catalog_data is None:
        glade_names = "PGCname, GWGCname, HyperLedaname, 2MASSname, SDSS-DR12name,\
                    flag1, RA, DEC, dist, dist_err, z, B, B_err, B_abs, J, J_err,\
                    H, H_err, K, K_err, flag2, flag3"
        catalog_data = np.atleast_1d(np.genfromtxt(catalog_file, names=glade_names)) # Troubles with single row files.

    catalog = []

    for i in range(catalog_data.shape[0]):
        # Check the entries: B-band mag (abs and apparent), redshift and proximity to GW position posteriors
        if (~np.isnan(catalog_data['B'][i])) and (~np.isnan(catalog_data['B_abs'][i])) and (catalog_data['flag2'][i] == 1 or catalog_data['flag2'][i] == 3) and isinbound(catalog_data[i], limits):
            catalog.append(Galaxy(i, np.deg2rad(catalog_data['RA'][i]), np.deg2rad(catalog_data['DEC'][i]), catalog_data['z'][i], True, app_magnitude = catalog_data['B'][i], abs_magnitude = catalog_data['B_abs'][i])) # Controlla nomi con catalogo!
            # Warning: GLADE stores no information on dz. 2B corrected.

    catalog = catalog_weight(catalog) # Implementare meglio la selezione del peso delle galassie.

    return catalog

def isinbound(galaxy, limits):
    if (limits['RA'][0] <= np.deg2rad(galaxy['RA']) <= limits['RA'][1]) and (limits['DEC'][0] <= np.deg2rad(galaxy['DEC']) <= limits['DEC'][1]) and (limits['z'][0] <= galaxy['z'] <= limits['z'][1]) :
        return True
    return True


def catalog_weight(catalog, weight = 'uniform'):
    '''
    Method:
    Assign a weight for each galaxy in catalog according to the emission probability
    of the galaxy.
    Please note that this has to be a relative probability rather than an absolute one.

        - Uniform weighting: 1/N ('uniform')
    '''
    if weight == 'uniform':
        for galaxy in catalog:
            galaxy.weight = 1./len(catalog)

    return catalog
