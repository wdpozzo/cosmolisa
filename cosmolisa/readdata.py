import numpy as np
import sys
import os
from galaxies import *
import lal
from volume_reconstruction.dpgmm.dpgmm import *
from volume_reconstruction.utils.utils import *
import dill as pickle
from scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
# from numba import jit
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def logPosterior(args):
    density,celestial_coordinates = args
    cartesian_vect = celestial_to_cartesian(celestial_coordinates)
    logPs = [np.log(density[0][ind])+prob.logProb(cartesian_vect) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)+np.log(Jacobian(cartesian_vect))

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))


def LumDist(z, omega):
    return 3e3*(z + (1-omega.om +omega.ol)*z**2/2.)/omega.h

def dLumDist(z, omega):
    return 3e3*(1+(1-omega.om+omega.ol)*z)/omega.h

def RedshiftCalculation(LD, omega, zinit=0.3, limit = 0.001):
    '''
    Redshift given a certain luminosity, calculated by recursion.
    Limit is the less significative digit.
    '''
    LD_test = LumDist(zinit, omega)
    if abs(LD-LD_test) < limit :
        return zinit
    znew = zinit - (LD_test - LD)/dLumDist(zinit,omega)
    return RedshiftCalculation(LD, omega, zinit = znew)



class Event_test(object):
    """
    Event class:
    initialise a GW event based on its distance and potential
    galaxy hosts
    """
    def __init__(self,
                 ID,
                 catalog_file,
                 event_file,
                 levels_file,
                 EMcp         = 0,
                 n_tot        = None,
                 gal_density  = 0.6675): # galaxies/Mpc^3 (from Conselice et al., 2016)

        if catalog_file is None:
            raise SystemExit('No catalog provided')

        self.ID                     = ID
        self.potential_galaxy_hosts = read_galaxy_catalog({'RA':[0., 360.], 'DEC':[-90., 90.], 'z':[0., 4.]}, catalog_file = catalog_file, n_tot = None)
        self.n_hosts                = len(self.potential_galaxy_hosts)

        self.cl      = np.genfromtxt(levels_file, names = True)
        self.vol_90  = self.cl['volume']*0.6
        self.area_90 = self.cl['area']
        self.LDmin   = self.cl['LD_min']
        self.LDmax   = self.cl['LD_max']
        self.ramin   = self.cl['ra_min']
        self.ramax   = self.cl['ra_max']
        self.decmin  = self.cl['dec_min']
        self.decmax  = self.cl['dec_max']
        self.zmin    = self.cl['z_min']
        self.zmax    = self.cl['z_max']

        self.posterior = np.genfromtxt(event_file, names = True)

        self.LD   = self.posterior['LD']
        self.dLD  = self.posterior['dLD']
        self.ra   = self.posterior['ra']
        self.dra  = self.posterior['dra']
        self.dec  = self.posterior['dec']
        self.ddec = self.posterior['ddec']

        if n_tot is not None:
            self.n_tot = n_tot
        else:
            if EMcp:
                self.n_tot = 1.
            elif gal_density is not None:
                self.n_tot = int(gal_density*self.vol_90)
        print('Total number of galaxies in the considered volume ({0} Mpc^3): {1}'.format(self.vol_90, self.n_tot))
        self.potential_galaxy_hosts = catalog_weight(self.potential_galaxy_hosts, weight = 'uniform', ngal = self.n_tot)

    def logP(self, galaxy):
        '''
        galaxy must be a list with [LD, dec, ra]
        '''
        try:
            gauss_LD = gaussian(galaxy[0], self.LD, self.dLD)
            if gauss_LD == 0:
                return -np.inf
            pLD   = np.log(gauss_LD)
            pra   = np.log(gaussian(galaxy[2], self.ra, self.dra))
            pdec  = np.log(gaussian(galaxy[1], self.dec, self.ddec))
            logpost = pLD+pra+pdec
        except:
            logpost = -np.inf
        return logpost

    def marg_logP(self, LD):
        try:
            gauss_LD = gaussian(LD, self.LD, self.dLD)
            if gauss_LD == 0:
                return -np.inf
            logpost = np.log(gauss_LD)
        except:
            logpost = -np.inf
        return logpost


class Event_CBC(object):

    def __init__(self,
                 ID,
                 catalog_file,
                 density,
                 levels_file,
                 distance_file,
                 area_file,
                 EMcp         = 0,
                 n_tot        = None,
                 gal_density  = 0.6675): # galaxies/Mpc^3 (from Conselice et al., 2016)

        if catalog_file is None:
            raise SystemExit('No catalog provided')

        self.ID                     = ID
        self.potential_galaxy_hosts = read_galaxy_catalog({'RA':[0., 360.], 'DEC':[-90., 90.], 'z':[0., 4.]}, catalog_file = catalog_file, n_tot = None)
        self.n_hosts                = len(self.potential_galaxy_hosts)
        self.density_model          = pickle.load(open(density, 'rb'))

        self.cl      = np.genfromtxt(levels_file, names = ['CL','vol','area','LD', 'ramin', 'ramax', 'decmin', 'decmax'])
        self.vol_90  = self.cl['vol'][np.where(self.cl['CL']==0.90)[0][0]]#-self.cl['vol'][np.where(self.cl['CL']==0.05)[0][0]]
        self.area_90 = self.cl['area'][np.where(self.cl['CL']==0.90)[0][0]]
        self.LDmin   = self.cl['LD'][np.where(self.cl['CL']==0.05)[0][0]]
        self.LDmax   = self.cl['LD'][np.where(self.cl['CL']==0.95)[0][0]]
        self.LDmean  = self.cl['LD'][np.where(self.cl['CL']==0.5)[0][0]]
        # self.vol_90  = 4*np.pi*(self.LDmax**3-self.LDmin**3)*self.area_90/(180.*360.*3)
        self.ramin   = self.cl['ramin'][np.where(self.cl['CL']==0.9)[0][0]]
        self.ramax   = self.cl['ramax'][np.where(self.cl['CL']==0.9)[0][0]]
        self.decmin  = self.cl['decmin'][np.where(self.cl['CL']==0.9)[0][0]]
        self.decmax  = self.cl['decmax'][np.where(self.cl['CL']==0.9)[0][0]]

        marginalized_post = np.genfromtxt(distance_file, names = True)
        self.interpolant = interp1d(marginalized_post['dist'], marginalized_post['post'], 'linear', fill_value = 0., bounds_error=False)

        areas   = np.genfromtxt(area_file, names = True)
        self.area = interp1d(areas['dist'], areas['area'], 'linear', fill_value = 0., bounds_error = False)




        if n_tot is not None:
            self.n_tot = n_tot
        else:
            if EMcp:
                self.n_tot = 1.
            elif self.LDmean < 0: # se l'oggetto è vicino non posso assumere omogeneità
                self.n_tot = len(self.potential_galaxy_hosts)
            elif gal_density is not None:
                self.n_tot = int(gal_density*self.vol_90)
        print('Total number of galaxies in the considered volume ({0} Mpc^3): {1}'.format(self.vol_90, self.n_tot))
        self.potential_galaxy_hosts = catalog_weight(self.potential_galaxy_hosts, weight = 'uniform', ngal = self.n_tot)
        self.zmin = RedshiftCalculation(self.LDmin, lal.CreateCosmologicalParameters(0.3,0.7,0.3,-1,0,0))
        self.zmax = RedshiftCalculation(self.LDmax, lal.CreateCosmologicalParameters(1,0.7,0.3,-1,0,0))

    def logP(self, galaxy):
        '''
        galaxy must be a list with [LD, dec, ra]
        '''
        try:
            logpost = logPosterior((self.density_model, np.array(galaxy)))-2.*np.log(galaxy[0])+np.log(self.LDmax**3-self.LDmin**3)-np.log(3)
        except:
            logpost = -np.inf
        return logpost

    def marg_logP(self, LD):
            marg_post    = self.interpolant(LD)
            if marg_post > 0:
                logpost = np.log(marg_post)-2.*np.log(LD)+np.log(self.LDmax**3-self.LDmin**3)-np.log(3)
            else:
                logpost = -np.inf
            return logpost

class Event_CBC_EM(object):

    def __init__(self,
                 ID,
                 catalog_file,
                 event_file,
                 levels_file,
                 distance_file,
                 EMcp         = 0,
                 n_tot        = None,
                 gal_density  = 0.6675): # galaxies/Mpc^3 (from Conselice et al., 2016)

        if catalog_file is None:
            raise SystemExit('No catalog provided')

        self.ID                     = ID
        self.potential_galaxy_hosts = read_galaxy_catalog({'RA':[0., 360.], 'DEC':[-90., 90.], 'z':[0., 4.]}, catalog_file = catalog_file, n_tot = None)
        self.n_hosts                = len(self.potential_galaxy_hosts)
        self.samples_DL             = np.genfromtxt(event_file)
        self.pdf                    = gaussian_kde(self.samples_DL)

        self.cl      = np.genfromtxt(levels_file, names = ['CL','vol','area','LD', 'ramin', 'ramax', 'decmin', 'decmax'])
        self.vol_90  = self.cl['vol'][np.where(self.cl['CL']==0.90)[0][0]]#-self.cl['vol'][np.where(self.cl['CL']==0.05)[0][0]]
        self.area_90 = self.cl['area'][np.where(self.cl['CL']==0.90)[0][0]]
        self.LDmin   = self.cl['LD'][np.where(self.cl['CL']==0.05)[0][0]]
        self.LDmax   = self.cl['LD'][np.where(self.cl['CL']==0.95)[0][0]]
        self.LDmean  = self.cl['LD'][np.where(self.cl['CL']==0.5)[0][0]]
        # self.vol_90  = 4*np.pi*(self.LDmax**3-self.LDmin**3)*self.area_90/(180.*360.*3)
        self.ramin   = self.cl['ramin'][np.where(self.cl['CL']==0.9)[0][0]]
        self.ramax   = self.cl['ramax'][np.where(self.cl['CL']==0.9)[0][0]]
        self.decmin  = self.cl['decmin'][np.where(self.cl['CL']==0.9)[0][0]]
        self.decmax  = self.cl['decmax'][np.where(self.cl['CL']==0.9)[0][0]]

        marginalized_post = np.genfromtxt(distance_file, names = True)
        self.interpolant = interp1d(marginalized_post['dist'], marginalized_post['post'], 'linear')



        if n_tot is not None:
            self.n_tot = n_tot
        else:
            if EMcp:
                self.n_tot = 1.
            elif self.LDmean < 0: # se l'oggetto è vicino non posso assumere omogeneità
                self.n_tot = len(self.potential_galaxy_hosts)
            elif gal_density is not None:
                self.n_tot = int(gal_density*self.vol_90)
        print('Total number of galaxies in the considered volume ({0} Mpc^3): {1}'.format(self.vol_90, self.n_tot))
        self.potential_galaxy_hosts = catalog_weight(self.potential_galaxy_hosts, weight = 'uniform', ngal = self.n_tot)
        self.zmin = RedshiftCalculation(self.LDmin, lal.CreateCosmologicalParameters(0.3,0.7,0.3,-1,0,0))
        self.zmax = RedshiftCalculation(self.LDmax, lal.CreateCosmologicalParameters(2,0.7,0.3,-1,0,0))

    def logP(self, galaxy):
        '''
        galaxy must be a list with [LD, dec, ra]
        '''
        try:
            logpost = np.log(self.pdf(galaxy[0]))-2.*np.log(galaxy[0])+np.log(self.LDmax**3-self.LDmin**3)-np.log(3)
        except:
            logpost = -np.inf
        return logpost


def read_TEST_event(input_folder, emcp = 0, n_tot = None, gal_density = 0.066, nevmax = None):
    '''
    Classe di evento costruita per finalità di test. Le distribuzioni di probabilità sono gaussiane e centrate su una galassia a scelta.
    '''
    all_files     = os.listdir(input_folder)
    event_folders = []
    for file in all_files:
        if not '.' in file and 'event' in file:
            event_folders.append(file)
    event_folders.sort(key=natural_keys)
    events = []
    ID = 0.

    if nevmax is not None:
        event_folders = event_folders[:nevmax:]
    print(event_folders)

    for evfold in event_folders:
        ID +=1
        catalog_file  = input_folder+evfold+'/galaxy_0.9.txt'
        event_file    = input_folder+evfold+'/posterior.txt'
        levels_file   = input_folder+evfold+'/confidence_region.txt'
        events.append(Event_test(ID, catalog_file, event_file, levels_file, EMcp = emcp, gal_density=gal_density))
    return np.array(events)

def read_CBC_event(input_folder, emcp = 0, n_tot = None, gal_density = 0.6675, nevmax = None):
    all_files     = os.listdir(input_folder)
    event_folders = []
    for file in all_files:
        if not '.' in file and 'event' in file:
            event_folders.append(file)
    events = []
    ID = 0.
    for evfold in event_folders:
        ID +=1
        catalog_file  = input_folder+evfold+'/galaxy_0.9_-1.000.txt'
        event_file    = input_folder+evfold+'/dpgmm_density.p'
        levels_file   = input_folder+evfold+'/confidence_levels.txt'
        distance_file = input_folder+evfold+'/distance_map.txt'
        area_file     = input_folder+evfold+'/diff_area.txt'
        events.append(Event_CBC(ID, catalog_file, event_file, levels_file, distance_file, area_file, EMcp = emcp, gal_density=gal_density))
    return np.array(events)

def read_CBC_EM_event(input_folder, emcp = 0, n_tot = None, gal_density = 0.6675, nevmax = None):
    all_files     = os.listdir(input_folder)
    event_folders = []
    for file in all_files:
        if not '.' in file and 'event' in file:
            event_folders.append(file)
    events = []
    ID = 0.
    for evfold in event_folders:
        ID +=1
        catalog_file  = input_folder+evfold+'/galaxy_0.9.txt'
        event_file    = input_folder+evfold+'/samples_DL.txt'
        levels_file   = input_folder+evfold+'/confidence_levels.txt'
        distance_file = input_folder+evfold+'/distance_map.txt'
        events.append(Event_CBC_EM(ID, catalog_file, event_file, levels_file, distance_file, EMcp = emcp, gal_density=gal_density))
    return np.array(events)





def read_event(event_class,*args,**kwargs):

    if event_class == "TEST": return read_TEST_event(*args, **kwargs)
    if event_class == "CBC": return read_CBC_event(*args, **kwargs)
    if event_class == "CBC_EM": return read_CBC_EM_event(*args, **kwargs)
    else:
        print("I do not know the class %s, exiting\n"%event_class)
        exit(-1)
