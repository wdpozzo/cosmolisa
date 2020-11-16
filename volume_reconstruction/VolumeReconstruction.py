#! /usr/bin/env python

from __future__ import division
import os, sys, numpy as np
import dill as  pickle
from volume_reconstruction.dpgmm.dpgmm import *
import copy
import healpy as hp
from scipy.special import logsumexp
import optparse as op
import lal
from lal import LIGOTimeGPS
import multiprocessing as mp
try:
    import copy_reg
except:
    import copyreg as copy_reg
import types
from volume_reconstruction.utils.cumulative import *
from volume_reconstruction.utils.utils import *
import matplotlib
from matplotlib.ticker import MultipleLocator
import time

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

# ---------------------
# DPGMM posterior class
# ---------------------

class DPGMMSkyPosterior(object):
    """
    Dirichlet Process Gaussian Mixture model class
    input parameters:

    posterior_samples: posterior samples for which the density estimate needs to be calculated

    dimension: the dimensionality of the problem. default = 3

    max_stick: maximum number of mixture components. default = 16

    bins: number of bins in the d,ra,dec directions. default = [10,10,10]

    dist_max: maximum radial distance to consider. default = 218 Mpc

    nthreads: number of multiprocessing pool workers to use. default = multiprocessing.cpu_count()

    injection: the injection file.

    catalog: the galaxy catalog for the ranked list of galaxies
    """
    def __init__(self,
                 posterior_samples,
                 dimension          = 3,
                 max_sticks         = 16,
                 bins               = [10,10,10],
                 dist_max           = 218,
                 output             = None,
                 nthreads           = None,
                 injection          = None,
                 catalog            = None,
                 standard_cosmology = True,
                 h                  = -1):

        self.posterior_samples  = np.array(posterior_samples)
        self.dims               = dimension
        self.max_sticks         = max_sticks
        if nthreads == None:
            self.nthreads       = mp.cpu_count()//2
        else:
            self.nthreads       = nthreads
        self.bins               = bins
        self.dist_max           = dist_max
        self.pool               = mp.Pool(self.nthreads)
        self.injection          = injection
        self.catalog            = None
        self.output             = output
        self.h                  = h
        self._initialise_grid()
        if catalog is not None:
            self.catalog = readGC(catalog,self,standard_cosmology=standard_cosmology, h = h)

    def _initialise_dpgmm(self):
        self.model = DPGMM(self.dims)
        for point in self.posterior_samples:
            self.model.add(celestial_to_cartesian(point))

        self.model.setPrior(mean = celestial_to_cartesian(np.mean(self.posterior_samples,axis=1)))
        self.model.setThreshold(1e-4)
        self.model.setConcGamma(0.1,1)

    def _initialise_grid(self):
        self.grid   = []
        a           = 0.9*self.posterior_samples[:,0].min()
        b           = 1.1*self.posterior_samples[:,0].max()
        self.grid.append(np.linspace(a,b,self.bins[0]))
        a           = -np.pi/2.0
        b           = np.pi/2.0
        if self.posterior_samples[:,1].min() < 0.0:
            a = 1.1*self.posterior_samples[:,1].min()
        else:
            a = 0.9*self.posterior_samples[:,1].min()
        if self.posterior_samples[:,1].max() < 0.0:
            b = 0.9*self.posterior_samples[:,1].max()
        else:
            b = 1.1*self.posterior_samples[:,1].max()

        self.grid.append(np.linspace(a,b,self.bins[1]))
        a   = 0.0
        b   = 2.0*np.pi
        a   = 0.9*self.posterior_samples[:,2].min()
        b   = 1.1*self.posterior_samples[:,2].max()
        self.grid.append(np.linspace(a,b,self.bins[2]))

        self.dD     = np.diff(self.grid[0])[0]
        self.dDEC   = np.diff(self.grid[1])[0]
        self.dRA    = np.diff(self.grid[2])[0]
        print('The number of grid points in the sky is :',self.bins[1]*self.bins[2],'resolution = ',np.degrees(self.dRA)*np.degrees(self.dDEC), ' deg^2')
        print('The number of grid points in distance is :',self.bins[0],'minimum resolution = ',self.dD,' Mpc')
        print('Total grid size is :',self.bins[0]*self.bins[1]*self.bins[2])
        print('Minimum Volume resolution is :',self.dD*self.dDEC*self.dRA,' Mpc^3')

    def compute_dpgmm(self):
        self._initialise_dpgmm()
        try:
            sys.stderr.write("Loading dpgmm model\n")
            self.model = pickle.load(open(os.path.join(self.output,'dpgmm_model.p'), 'rb'))
        except:
            sys.stderr.write("Model not found, recomputing\n")
            solve_args      = [(nc, self.model) for nc in range(1, self.max_sticks+1)]
            solve_results   = self.pool.map(solve_dpgmm, solve_args)
            self.scores     = np.array([r[1] for r in solve_results])
            self.model      = (solve_results[self.scores.argmax()][-1])
            # pickle dump the dpgmm model
            pickle.dump(self.model, open(os.path.join(self.output,'dpgmm_model.p'), 'wb'))
            print("best model has ",self.scores.argmax()+1,"components")
        try:
            sys.stderr.write("Loading density model\n")
            self.density = pickle.load(open(os.path.join(self.output,'dpgmm_density.p'), 'rb'))
        except:
            sys.stderr.write("Model density not found, recomputing\n")
            self.density = self.model.intMixture()
            # pickle dump the dpgmm model density
            pickle.dump(self.density, open(os.path.join(self.output,'dpgmm_density.p'), 'wb'))

    def rank_galaxies(self):

        sys.stderr.write("Ranking the galaxies: computing log posterior for %d galaxies\n"%(self.catalog.shape[0]))
        jobs        = ((self.density,np.array((d,dec,ra))) for d, dec, ra in zip(self.catalog[:,2],self.catalog[:,1],self.catalog[:,0]))
        results     = self.pool.imap(logPosterior, jobs, chunksize = np.int(self.catalog.shape[0]/ (self.nthreads * 16)))
        logProbs    = np.array([r for r in results])

        idx         = ~np.isnan(logProbs)
        self.ranked_probability = logProbs[idx]
        self.ranked_ra          = self.catalog[idx,0]
        self.ranked_dec         = self.catalog[idx,1]
        self.ranked_dl          = self.catalog[idx,2]
        self.ranked_z           = self.catalog[idx,3]
        self.ranked_B           = self.catalog[idx,4]
        self.ranked_dB          = self.catalog[idx,5]
        self.ranked_Babs        = self.catalog[idx,6]
        self.peculiarmotion     = self.catalog[idx,7]

        order                   = self.ranked_probability.argsort()[::-1]

        self.ranked_probability = self.ranked_probability[order]
        self.ranked_ra          = self.ranked_ra[order]
        self.ranked_dec         = self.ranked_dec[order]
        self.ranked_dl          = self.ranked_dl[order]
        self.ranked_z           = self.ranked_z[order]
        self.ranked_B           = self.ranked_B[order]
        self.ranked_dB          = self.ranked_dB[order]
        self.ranked_Babs        = self.ranked_Babs[order]
        self.peculiarmotion     = self.peculiarmotion[order]

    def evaluate_volume_map(self):
        N = self.bins[0]*self.bins[1]*self.bins[2]
        sys.stderr.write("computing log posterior for %d grid points\n"%N)
        sample_args         = ((self.density,np.array((d,dec,ra))) for d in self.grid[0] for dec in self.grid[1] for ra in self.grid[2])
        results             = self.pool.imap(logPosterior, sample_args, chunksize = N//(self.nthreads * 32))
        self.log_volume_map = np.array([r for r in results]).reshape(self.bins[0],self.bins[1],self.bins[2])
        self.volume_map     = np.exp(self.log_volume_map)
        # normalise
        dsquared         = self.grid[0]**2
        cosdec           = np.cos(self.grid[1])
        self.volume_map /= np.sum(self.volume_map*dsquared[:,None,None]*cosdec[None,:,None]*self.dD*self.dRA*self.dDEC)

    def evaluate_sky_map(self):
        dsquared        = self.grid[0]**2
        self.skymap     = np.trapz(dsquared[:,None,None]*self.volume_map, x=self.grid[0], axis=0)
        self.log_skymap = np.log(self.skymap)

    def evaluate_distance_map(self):
        cosdec                     = np.cos(self.grid[1])
        intermediate               = np.trapz(self.volume_map, x=self.grid[2], axis=2)
        self.distance_map          = np.trapz(cosdec*intermediate, x=self.grid[1], axis=1)
        self.log_distance_map      = np.log(self.distance_map)
        self.unnormed_distance_map = self.distance_map
        self.distance_map         /= (self.distance_map*np.diff(self.grid[0])[0]).sum()
        print('Normalization constant for distance map: {0}'.format((self.distance_map*np.diff(self.grid[0])[0]).sum()))

    def ConfidenceVolume(self, adLevels):
        # create a normalized cumulative distribution
        self.log_volume_map_sorted  = np.sort(self.log_volume_map.flatten())[::-1]
        self.log_volume_map_cum     = fast_log_cumulative(self.log_volume_map_sorted)

        # find the indeces  corresponding to the given CLs
        adLevels        = np.ravel([adLevels])
        args            = [(self.log_volume_map_sorted,self.log_volume_map_cum,level) for level in adLevels]
        adHeights       = self.pool.map(FindHeights,args)
        self.heights    = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}
        volumes         = []

        for height in adHeights:

            (index_d, index_dec, index_ra,) = np.where(self.log_volume_map>=height)
            volumes.append(np.sum([self.grid[0][i_d]**2. *np.cos(self.grid[1][i_dec]) * self.dD * self.dRA * self.dDEC for i_d,i_dec in zip(index_d,index_dec)]))

        self.volume_confidence = np.array(volumes)

        if self.injection is not None:
            ra,dec           = self.injection.get_ra_dec()
            distance         = self.injection.distance
            logPval          = logPosterior((self.density,np.array((distance,dec,ra))))
            confidence_level = np.exp(self.log_volume_map_cum[np.abs(self.log_volume_map_sorted-logPval).argmin()])
            height           = FindHeights((self.log_volume_map_sorted,self.log_volume_map_cum,confidence_level))
            (index_d, index_dec, index_ra,) = np.where(self.log_volume_map>=height)
            searched_volume  = np.sum([self.grid[0][i_d]**2. *np.cos(self.grid[1][i_dec]) * self.dD * self.dRA * self.dDEC for i_d,i_dec in zip(index_d,index_dec)])
            self.injection_volume_confidence    = confidence_level
            self.injection_volume_height        = height
            return self.volume_confidence,(confidence_level,searched_volume)

        del self.log_volume_map_sorted
        del self.log_volume_map_cum
        return self.volume_confidence,None



    def DifferentialVolume(self):
        # create a normalized cumulative distribution
        self.log_volume_map_sorted  = np.sort(self.log_volume_map.flatten())[::-1]
        self.log_volume_map_cum     = fast_log_cumulative(self.log_volume_map_sorted)

        # find the indeces  corresponding to the given CLs
        adLevels        = np.ravel([0.9])
        args            = [(self.log_volume_map_sorted,self.log_volume_map_cum,level) for level in adLevels]
        adHeights       = self.pool.map(FindHeights,args)
        heights         = {str(lev):hei for lev,hei in zip(adLevels,adHeights)}

        bins_path = '/Users/stefanorinaldi/Documents/Sim/bins.txt'


        out_bins = open(bins_path, 'w')
        out_bins.write('ra dec dist\n')


        for height in adHeights:
            surfaces = {}
            distance_index = []
            diffvol        = []
            diffarea       = []
            diffdist       = []

            (index_d, index_dec, index_ra,) = np.where(self.log_volume_map>=height)
            for d, dec, ra in zip(index_d, index_dec, index_ra):
                out_bins.write('{} {} {}\n'.format(self.grid[2][ra],self.grid[1][dec],self.grid[0][d]))
                if not np.isin(str(d), list(surfaces.keys())):
                    surfaces[str(d)] = []
                    distance_index.append(d)
                surfaces[str(d)].append(dec)
            out_bins.close()
            for d in distance_index:
                diffvol.append(np.sum([self.grid[0][d]**2. *np.cos(self.grid[1][i_dec]) * self.dRA * self.dDEC for i_dec in surfaces[str(d)]]))
                diffarea.append(np.sum([np.cos(self.grid[1][i_dec]) * self.dRA * self.dDEC for i_dec in surfaces[str(d)]]))#*(180.0/np.pi)**2.0)
                diffdist.append(self.grid[0][d])

        self.differential_volume    = np.array(diffvol)
        self.differential_area      = np.array(diffarea)
        self.differential_distances = np.array(diffdist)

        del self.log_volume_map_sorted
        del self.log_volume_map_cum
        return self.differential_volume,None


    def ConfidenceArea(self, adLevels):

        # create a normalized cumulative distribution
        self.log_skymap_sorted  = np.sort(self.log_skymap.flatten())[::-1]
        self.log_skymap_cum     = fast_log_cumulative(self.log_skymap_sorted)
        # find the indeces  corresponding to the given CLs
        adLevels                = np.ravel([adLevels])
        args                    = [(self.log_skymap_sorted,self.log_skymap_cum,level) for level in adLevels]
        adHeights               = self.pool.map(FindHeights,args)
        areas                   = []

        for height in adHeights:
            (index_dec,index_ra,) = np.where(self.log_skymap>=height)
            areas.append(np.sum([self.dRA*np.cos(self.grid[1][i_dec])*self.dDEC for i_dec in index_dec])*(180.0/np.pi)**2.0)

        self.area_confidence = np.array(areas)

        if self.injection is not None:
            ra,dec                  = self.injection.get_ra_dec()
            id_ra                   = np.abs(self.grid[2]-ra).argmin()
            id_dec                  = np.abs(self.grid[1]-dec).argmin()
            logPval                 = self.log_skymap[id_dec,id_ra]
            confidence_level        = np.exp(self.log_skymap_cum[np.abs(self.log_skymap_sorted-logPval).argmin()])
            height                  = FindHeights((self.log_skymap_sorted,self.log_skymap_cum,confidence_level))
            (index_dec,index_ra,)   = np.where(self.log_skymap >= height)
            searched_area           = np.sum([self.dRA*np.cos(self.grid[1][i_dec])*self.dDEC for i_dec in index_dec])*(180.0/np.pi)**2.0

            return self.area_confidence,(confidence_level,searched_area)

        del self.log_skymap_sorted
        del self.log_skymap_cum
        return self.area_confidence,None

    def ConfidenceCoordinates(self, adLevels):
        # create a normalized cumulative distribution
        self.log_skymap_sorted  = np.sort(self.log_skymap.flatten())[::-1]
        self.log_skymap_cum     = fast_log_cumulative(self.log_skymap_sorted)
        # find the indeces  corresponding to the given CLs
        adLevels                = np.ravel([adLevels])
        args                    = [(self.log_skymap_sorted,self.log_skymap_cum,level) for level in adLevels]
        adHeights               = self.pool.map(FindHeights,args)
        ramin                   = []
        ramax                   = []
        decmin                  = []
        decmax                  = []

        for height in adHeights:
            (index_dec,index_ra,) = np.where(self.log_skymap>=height)
            ra     = self.grid[2][index_ra]
            dec    = self.grid[1][index_dec]
            ramin.append(ra.min())
            ramax.append(ra.max())
            decmin.append(dec.min())
            decmax.append(dec.max())

        return ramin, ramax, decmin, decmax

    def ConfidenceDistance(self, adLevels):
        cumulative_distribution     = np.cumsum(self.distance_map*self.dD)
        distances                   = []

        for cl in adLevels:
            idx = np.abs(cumulative_distribution-cl).argmin()
            distances.append(self.grid[0][idx])

        self.distance_confidence = np.array(distances)

        if self.injection!=None:
            idx                 = np.abs(self.injection.distance-self.grid[0]).argmin()
            confidence_level    = cumulative_distribution[idx]
            searched_distance   = self.grid[0][idx]
            return self.distance_confidence,(confidence_level,searched_distance)

        return self.distance_confidence,None




# ---------------
# DPGMM functions
# ---------------

def logPosterior(args):
    density,celestial_coordinates = args
    cartesian_vect = celestial_to_cartesian(celestial_coordinates)
    logPs = [np.log(density[0][ind])+prob.logProb(cartesian_vect) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)+np.log(Jacobian(cartesian_vect))

def logPosteriorCartesian(args):
    density,cartesian_coordinates = args
    logPs = [np.log(density[0][ind])+prob.logProb(cartesian_coordinates) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)

def Posterior(args):
    density,celestial_coordinates = args
    cartesian_vect = celestial_to_cartesian(celestial_coordinates)
    Ps = [density[0][ind]*prob.prob(cartesian_vect) for ind,prob in enumerate(density[1])]
    return reduce(np.sum,Ps)*np.abs(np.cos(celestial_coordinates[2]))*celestial_coordinates[0]**2

def solve_dpgmm(args):
    (nc, model_in) = args
    model          = DPGMM(model_in)
    for _ in range(nc-1): model.incStickCap()
#    try:
    it = model.solve(iterCap=1024)
    return (model.stickCap, model.nllData(), model)
#    except:
#        return (model.stickCap, -np.inf, model)

# --------
# jacobian
# --------

def log_jacobian(dgrid, nside):
  # get the number of pixels for the healpy nside
  npix = np.int(hp.nside2npix(nside))
  # calculate the jacobian on the d_grid, copy over for the required number of pixels, appropriately reshape the array and return
  return np.array([2.*np.log(d) for d in dgrid]*npix).reshape(npix,-1).T

# -----------------------
# confidence calculations
# -----------------------

def FindHeights(args):
    (sortarr,cumarr,level) = args
    return sortarr[np.abs(cumarr-np.log(level)).argmin()]

def FindHeightForLevel(inLogArr, adLevels):
    # flatten and create reversed sorted list
    adSorted = np.sort(inLogArr.flatten())[::-1]
    # create a normalized cumulative distribution
    adCum = np.array([logsumexp(adSorted[:i+1]) for i in range(len(adSorted))])
    adCum -= adCum[-1]
    # find values closest to levels
    adHeights = []
    adLevels = np.ravel([adLevels])
    for level in adLevels:
        idx = (np.abs(adCum-np.log(level))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)
    return adHeights

def FindLevelForHeight(inLogArr, logvalue):
    # flatten and create reversed sorted list
    adSorted = np.sort(inLogArr.flatten())[::-1]
    # create a normalized cumulative distribution
    adCum = np.array([logsumexp(adSorted[:i+1]) for i in range(len(adSorted))])
    adCum -= adCum[-1]
    # find index closest to value
    idx = (np.abs(adSorted-logvalue)).argmin()
    return np.exp(adCum[idx])

#---------
# utilities
#---------

def readGC(file,dpgmm,standard_cosmology=True, h = -1):
    ra, dec, z, dl, B, dB, B_abs, pecmot = [], [], [], [], [], [], [], []

    '''
    Glade 2.3
    '''
    glade_names = "PGCname, GWGCname, HyperLedaname, 2MASSname, SDSS-DR12name,\
                flag1, RA, DEC, dist, dist_err, z, B, B_err, B_abs, J, J_err,\
                H, H_err, K, K_err, flag2, flag3"
    cat = np.genfromtxt(file, names = True) #glade_names)
    if standard_cosmology:
        omega       = lal.CreateCosmologicalParameters(0.7, 0.3, 0.7, -1.0, 0.0, 0.0)
        zmin, zmax  = find_redshift_limits([0.69,0.71], [0.29,0.31], dpgmm.grid[0][0], dpgmm.grid[0][-1])
    elif not h == -1:
         omega       = lal.CreateCosmologicalParameters(h, 0.3, 0.7, -1.0, 0.0, 0.0)
         zmin, zmax  = find_redshift_limits([h-0.01,h+0.01], [0.29,0.31], dpgmm.grid[0][0], dpgmm.grid[0][-1])
    else:
        zmin,zmax   = find_redshift_limits([0.3,1],[0.0,1.0],dpgmm.grid[0][0],dpgmm.grid[0][-1])
    sys.stderr.write("selecting galaxies within redshift %f and %f from distances in %f and %f\n"%(zmin,zmax,dpgmm.grid[0][0],dpgmm.grid[0][-1]))

    for gal in cat:
        # Flag2 = 0: no distance/redshift measurement.
        if np.float(gal['z']) > 0.0:
            if not(standard_cosmology) and h == -1:
                h_random       = np.random.uniform(0.3,1)
                om      = np.random.uniform(0.0,1.0)
                ol      = 1.0-om
                omega   = lal.CreateCosmologicalParameters(h_random,om,ol,-1.0,0.0,0.0)

            ra.append(np.float(gal['ra']))
            dec.append(np.float(gal['dec']))
            z.append(np.float(gal['z']))
            B.append(np.float(gal['B']))
            if np.isfinite(gal['B_err']):
                dB.append(np.float(gal['B_err']))
            else:
                dB.append(0.5)
            B_abs.append(np.float(gal['B_abs']))
            #pecmot.append(np.float(gal['flag3']))
            pecmot.append(1)

            if not(np.isnan(z[-1])) and (zmin < z[-1] < zmax):
                dl.append(lal.LuminosityDistance(omega,z[-1]))
            else:
                dl.append(-1)
    return np.column_stack((np.radians(np.array(ra)),np.radians(np.array(dec)),np.array(dl),np.array(z), np.array(B), np.array(dB), np.array(B_abs), np.array(pecmot)))

def find_redshift_limits(h, om, dmin, dmax):

    from scipy.optimize import newton

    def my_target(z,omega,d):
        return d - lal.LuminosityDistance(omega,z)

    zu = []
    zl = []

    for hi in np.linspace(h[0],h[1],10):
        for omi in np.linspace(om[0],om[1],10):
            omega = lal.CreateCosmologicalParameters(hi,omi,1.0-omi,-1.0,0.0,0.0)
            zu.append(newton(my_target,np.random.uniform(0.0,1.0),args=(omega,dmax)))
            zl.append(newton(my_target,np.random.uniform(0.0,1.0),args=(omega,dmin)))

    return np.min(zl), np.max(zu)

#---------
# plotting
#---------

def parse_to_list(option, opt, value, parser):
    """
    parse a comma separated string into a list
    """
    setattr(parser.values, option.dest, value.split(','))


#-------------------
# Main for scripts
#-------------------


def VolRec(input, output, bins, dmax, event_id, nthreads = None, maxstick = 16, catalog = None, plots = 0, ranks = 1000, nsamps = None, cosmology = 1, threed = 0, tfile = None, hubble = -1, injfile = None):

    np.random.seed(1)
    CLs                 = [0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.68,0.7,0.75,0.8,0.9,0.95] # add options?
    input_file          = input
    injFile             = injfile
    eventID             = event_id
    out_dir             = output
    bins                = np.array(bins,dtype=np.int)
    h                   = float(hubble)
    os.system('mkdir -p %s'%(out_dir))

    if injFile is not None:
        injections          = SimInspiralUtils.ReadSimInspiralFromFiles([injFile])
        injection           = injections[0] # pass event id from options
        (ra_inj, dec_inj)   = injection.get_ra_dec()
        tc                  = injection.get_time_geocent()
        GPSTime             = lal.LIGOTimeGPS(str(tc))
        gmst_rad_inj        = lal.GreenwichMeanSiderealTime(GPSTime)
        dist_inj            = injection.distance
        print('injection values -->',dist_inj,ra_inj,dec_inj,tc)
    else:
        injection           = None

    samples = np.genfromtxt(input_file,names=True)

    # we are going to normalise the distance between 0 and 1
    if 'time' in samples.dtype.names:
        time_name = 'time'
    elif 'tc' in samples.dtype.names:
        time_name = 'tc'
    elif 't0' in samples.dtype.names:
        time_name = 't0'

    if "dist" in samples.dtype.names:
        samples = np.column_stack((samples["dist"],samples["dec"],samples["ra"],samples[time_name]))
    elif "distance" in samples.dtype.names:
        samples = np.column_stack((samples["distance"],samples["dec"],samples["ra"],samples[time_name]))
    elif "logdistance" in samples.dtype.names:
        samples = np.column_stack((np.exp(samples["logdistance"]),samples["dec"],samples["ra"],samples[time_name]))



    samps       = []
    gmst_rad    = []

    if nsamps is not None:
        idx = np.random.choice(range(0,len(samples[:,0])),size=nsamps)
    else:
        idx = range(0,len(samples[:,0]))

    for k in range(len(samples[idx,0])):
        GPSTime = lal.LIGOTimeGPS(samples[k,3])
        gmst_rad.append(lal.GreenwichMeanSiderealTime(GPSTime))
        samps.append((samples[k,0],samples[k,1],samples[k,2]))

    dpgmm = DPGMMSkyPosterior(samps,
                              dimension          = 3,
                              max_sticks         = maxstick,
                              bins               = bins,
                              dist_max           = dmax,
                              nthreads           = nthreads,
                              injection          = injection,
                              catalog            = catalog,
                              output             = output,
                              standard_cosmology = cosmology,
                              h                  = hubble)

    dpgmm.compute_dpgmm()

    if dpgmm.catalog is not None:

        dpgmm.rank_galaxies()

        np.savetxt(os.path.join(output,'galaxy_ranks.txt'),
                   np.array([np.degrees(dpgmm.ranked_ra[:ranks]),
                             np.degrees(dpgmm.ranked_dec[:ranks]),
                             dpgmm.ranked_dl[:ranks],
                             dpgmm.ranked_z[:ranks],
                             dpgmm.ranked_B[:ranks],
                             dpgmm.ranked_dB[:ranks],
                             dpgmm.ranked_Babs[:ranks],
                             dpgmm.peculiarmotion[:ranks],
                             dpgmm.ranked_probability[:ranks]]).T,
                   fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                   header='ra\tdec\tDL\tz\tB\tB_err\tB_abs\tpec.mot.corr.\tlogposterior')

    dpgmm.evaluate_volume_map()
    volumes, searched_volume          = dpgmm.ConfidenceVolume(CLs)
    dpgmm.evaluate_sky_map()
    areas, searched_area              = dpgmm.ConfidenceArea(CLs)
    ramin, ramax, decmin, decmax      = dpgmm.ConfidenceCoordinates(CLs)
    dpgmm.evaluate_distance_map()
    distances, searched_distance      = dpgmm.ConfidenceDistance(CLs)
    differ_volume, searched_volume    = dpgmm.DifferentialVolume()

    if dpgmm.catalog is not None:
        number_of_galaxies = np.zeros(len(CLs),dtype=np.int)

        for i,CL in enumerate(CLs):
            threshold = dpgmm.heights[str(CL)]
            (k,) = np.where(dpgmm.ranked_probability>threshold)
            number_of_galaxies[i] = len(k)

        np.savetxt(os.path.join(output,'galaxy_in_confidence_regions.txt'), np.array([CLs,number_of_galaxies]).T, fmt='%.2f\t%d')

        if dpgmm.injection is not None:
            threshold = dpgmm.injection_volume_height
            (k,) = np.where(dpgmm.ranked_probability>threshold)
            number_of_galaxies = len(k)
            with open(os.path.join(output,'searched_galaxies.txt'),'w') as f:
                f.write('%.5f\t%d\n'%(dpgmm.injection_volume_confidence,number_of_galaxies))
                f.close()

    if plots:
        import matplotlib.pyplot as plt
        plt.plot(dpgmm.grid[0],dpgmm.distance_map,color="k",linewidth=2.0)
        plt.hist(samples[:,0],bins=dpgmm.grid[0],density=True,facecolor="0.9")
        if injFile!=None: plt.axvline(dist_inj,color="k",linestyle="dashed")
        plt.xlabel(r"$\mathrm{Distance/Mpc}$")
        plt.ylabel(r"$\mathrm{probability}$ $\mathrm{density}$")
        plt.savefig(os.path.join(output,'distance_posterior.pdf'),bbox_inches='tight')
    path = os.path.join(output,'confidence_levels.txt')
    np.savetxt(path, np.array([CLs, volumes, areas, distances, ramin, ramax, decmin, decmax]).T, fmt='%.2f\t%f\t%f\t%f\t%f\t%f\t%f\t%f')
    if dpgmm.injection is not None: np.savetxt(os.path.join(output,'searched_quantities.txt'), np.array([searched_volume,searched_area,searched_distance]), fmt='%s\t%s\t%s')

    np.savetxt(os.path.join(output,'distance_map.txt'), np.array([dpgmm.grid[0], dpgmm.unnormed_distance_map]).T, fmt='%f\t%f', header='dist\tpost')
    np.savetxt(os.path.join(output,'diff_volume.txt'), np.array([dpgmm.differential_distances, dpgmm.differential_volume]).T, fmt='%f\t%f', header='dist\tvolume')
    np.savetxt(os.path.join(output,'diff_area.txt'), np.array([dpgmm.differential_distances, dpgmm.differential_area]).T, fmt='%f\t%f', header='dist\tarea')


    # dist_inj,ra_inj,dec_inj,tc
    if injFile is not None:
        gmst_deg = np.mod(np.degrees(gmst_rad_inj), 360)
        lon_cen = lon_inj = np.degrees(ra_inj) - gmst_deg
        lat_cen = lat_inj = np.degrees(dec_inj)
    else:
        gmst_deg = np.mod(np.degrees(np.array(gmst_rad)), 360)
        lon_cen = np.degrees(np.mean(samples[idx,2])) - np.mean(gmst_deg)
        lat_cen = np.degrees(np.mean(samples[idx,1]))

    lon_samp = np.degrees(samples[idx,2]) - gmst_deg
    lat_samp = np.degrees(samples[idx,1])

    ra_map,dec_map = dpgmm.grid[2],dpgmm.grid[1]
    lon_map = np.degrees(ra_map) - np.mean(gmst_deg)
    lat_map = np.degrees(dec_map)

    if plots:
        sys.stderr.write("producing sky maps \n")
        try:
            plt.figure()
            plt.plot(np.arange(1,dpgmm.max_sticks+1),dpgmm.scores,'.')
            plt.xlabel(r"$\mathrm{number}$ $\mathrm{of}$ $\mathrm{components}$")
            plt.ylabel(r"$\mathrm{marginal}$ $\mathrm{likelihood}$")
            plt.savefig(os.path.join(out_dir, 'scores.pdf'))
        except:
            pass

    if catalog:
        out_dir = os.path.join(output, 'galaxies_scatter')
        os.system("mkdir -p %s"%out_dir)
        sys.stderr.write("producing 3 dimensional maps\n")
        # Create a sphere
        x = dpgmm.ranked_dl*np.cos(dpgmm.ranked_dec)*np.cos(dpgmm.ranked_ra)
        y = dpgmm.ranked_dl*np.cos(dpgmm.ranked_dec)*np.sin(dpgmm.ranked_ra)
        z = dpgmm.ranked_dl*np.sin(dpgmm.ranked_dec)

        threshold = dpgmm.heights['0.9']
        (k,) = np.where(dpgmm.ranked_probability>threshold)
        path = os.path.join(output,'galaxy_0.9_%.3f.txt') %(dpgmm.h)
        np.savetxt(path,
                   np.array([np.degrees(dpgmm.ranked_ra[k]),np.degrees(dpgmm.ranked_dec[k]),dpgmm.ranked_dl[k],dpgmm.ranked_z[k],dpgmm.ranked_B[k], dpgmm.ranked_dB[k], dpgmm.ranked_Babs[k], dpgmm.peculiarmotion[k],dpgmm.ranked_probability[k]]).T,
                   fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                   header='ra\tdec\tDL\tz\tB\tB_err\tB_abs\tpec.mot.corr.\tlogposterior')

        # saving non-host catalog
        threshold = dpgmm.heights['0.9']
        (k,) = np.where(dpgmm.ranked_probability < threshold)
        path = os.path.join(output,'catalog_%.3f.txt') %(dpgmm.h)
        np.savetxt(path,
                   np.array([np.degrees(dpgmm.ranked_ra[:]),np.degrees(dpgmm.ranked_dec[:]),dpgmm.ranked_dl[:],dpgmm.ranked_z[:],dpgmm.ranked_B[:], dpgmm.ranked_dB[:], dpgmm.ranked_Babs[:], dpgmm.peculiarmotion[:],dpgmm.ranked_probability[:]]).T,
                   fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                   header='ra\tdec\tDL\tz\tB\tB_err\tB_abs\tpec.mot.corr.\tlogposterior')

        imax = dpgmm.ranked_probability.argmax()
        threshold = dpgmm.heights['0.5']
        (k,) = np.where(dpgmm.ranked_probability>threshold)
        MIN = dpgmm.grid[0][0]
        MAX = dpgmm.grid[0][-1]
        sys.stderr.write("%d galaxies above threshold, plotting\n"%(len(k)))
        np.savetxt(os.path.join(output,'galaxy_0.5.txt'),
                   np.array([np.degrees(dpgmm.ranked_ra[k]),np.degrees(dpgmm.ranked_dec[k]),dpgmm.ranked_dl[k],dpgmm.ranked_z[k],dpgmm.ranked_B[k], dpgmm.ranked_dB[k], dpgmm.ranked_Babs[k], dpgmm.peculiarmotion[k],dpgmm.ranked_probability[k]]).T,
                   fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                   header='ra\tdec\tDL\tz\tB\tB_err\tB_abs\tpec.mot.corr.\tlogposterior')

    if threed:
        # produce a volume plot
        from volume_reconstruction.plotting.plot import volume_rendering
        volume_rendering()


    sys.stderr.write("\n")


#-------------------
# start the program
#-------------------

def main():
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type="string", dest="input", help="Input file")
    parser.add_option("--inj",type="string",dest="injfile",help="injection file",default=None)
    parser.add_option("-o", "--output", type="string", dest="output", help="Output file")
    parser.add_option("--bins", type="string", dest="bins", help="number of bins in d,dec,ra", action='callback',
                      callback=parse_to_list)
    parser.add_option("--dmax", type="float", dest="dmax", help="maximum distance (Mpc)")
    parser.add_option("--max-stick", type="int", dest="max_stick", help="maximum number of gaussian components", default=16)
    parser.add_option("-e", type="int", dest="event_id", help="event ID")
    parser.add_option("--threads", type="int", dest="nthreads", help="number of threads to spawn", default=None)
    parser.add_option("--catalog", type="string", dest="catalog", help="galaxy catalog to use", default=None)
    parser.add_option("--plots", type="string", dest="plots", help="produce plots", default=False)
    parser.add_option("-N", type="int", dest="ranks", help="number of ranked galaxies to list in output", default=1000)
    parser.add_option("--nsamps", type="int", dest="nsamps", help="number of posterior samples to utilise", default=None)
    parser.add_option("--cosmology", type="int", dest="cosmology", help="assume a lambda CDM cosmology?", default=True)
    parser.add_option("--3d", type="int", dest="threed", help="3d volume map", default=0)
    parser.add_option("--tfile", type="string", dest="tfile", help="coalescence time file", default=None)
    parser.add_option("--hubble", type = "string", dest="h", help="reduced hubble constant value", default='-1')
    (options, args)     = parser.parse_args()

    print(options)
    np.random.seed(1)
    CLs                 = [0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.68,0.7,0.75,0.8,0.9,0.95] # add options?
    input_file          = options.input
    injFile             = options.injfile
    eventID             = options.event_id
    out_dir             = options.output
    options.bins        = np.array(options.bins,dtype=np.int)
    options.h           = float(options.h)
    os.system('mkdir -p %s'%(out_dir))

    if injFile is not None:
        injections          = SimInspiralUtils.ReadSimInspiralFromFiles([injFile])
        injection           = injections[0] # pass event id from options
        (ra_inj, dec_inj)   = injection.get_ra_dec()
        tc                  = injection.get_time_geocent()
        GPSTime             = lal.LIGOTimeGPS(str(tc))
        gmst_rad_inj        = lal.GreenwichMeanSiderealTime(GPSTime)
        dist_inj            = injection.distance
        print('injection values -->',dist_inj,ra_inj,dec_inj,tc)
    else:
        injection           = None

    samples = np.genfromtxt(input_file,names=True)

    # we are going to normalise the distance between 0 and 1
    if 'time' in samples.dtype.names:
        time_name = 'time'
    elif 'tc' in samples.dtype.names:
        time_name = 'tc'
    elif 't0' in samples.dtype.names:
        time_name = 't0'

    if "dist" in samples.dtype.names:
        samples = np.column_stack((samples["dist"],samples["dec"],samples["ra"],samples[time_name]))
    elif "distance" in samples.dtype.names:
        samples = np.column_stack((samples["distance"],samples["dec"],samples["ra"],samples[time_name]))
    elif "logdistance" in samples.dtype.names:
        samples = np.column_stack((np.exp(samples["logdistance"]),samples["dec"],samples["ra"],samples[time_name]))



    samps       = []
    gmst_rad    = []

    if options.nsamps is not None:
        idx = np.random.choice(range(0,len(samples[:,0])),size=options.nsamps)
    else:
        idx = range(0,len(samples[:,0]))

    for k in range(len(samples[idx,0])):
        GPSTime = lal.LIGOTimeGPS(samples[k,3])
        gmst_rad.append(lal.GreenwichMeanSiderealTime(GPSTime))
        samps.append((samples[k,0],samples[k,1],samples[k,2]))

    dpgmm = DPGMMSkyPosterior(samps,
                              dimension          = 3,
                              max_sticks         = options.max_stick,
                              bins               = options.bins,
                              dist_max           = options.dmax,
                              nthreads           = options.nthreads,
                              injection          = injection,
                              catalog            = options.catalog,
                              output             = options.output,
                              standard_cosmology = options.cosmology,
                              h                  = options.h)

    dpgmm.compute_dpgmm()

    if dpgmm.catalog is not None:

        dpgmm.rank_galaxies()

        np.savetxt(os.path.join(options.output,'galaxy_ranks.txt'),
                   np.array([np.degrees(dpgmm.ranked_ra[:options.ranks]),
                             np.degrees(dpgmm.ranked_dec[:options.ranks]),
                             dpgmm.ranked_dl[:options.ranks],
                             dpgmm.ranked_z[:options.ranks],
                             dpgmm.ranked_B[:options.ranks],
                             dpgmm.ranked_dB[:options.ranks],
                             dpgmm.ranked_Babs[:options.ranks],
                             dpgmm.peculiarmotion[:options.ranks],
                             dpgmm.ranked_probability[:options.ranks]]).T,
                   fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                   header='ra\tdec\tDL\tz\tB\tB_err\tB_abs\tpec.mot.corr.\tlogposterior')

    dpgmm.evaluate_volume_map()
    volumes, searched_volume          = dpgmm.ConfidenceVolume(CLs)
    dpgmm.evaluate_sky_map()
    areas, searched_area              = dpgmm.ConfidenceArea(CLs)
    ramin, ramax, decmin, decmax      = dpgmm.ConfidenceCoordinates(CLs)
    dpgmm.evaluate_distance_map()
    distances, searched_distance      = dpgmm.ConfidenceDistance(CLs)
    surfaces, searched_surface        = dpgmm.DifferentialVolume()

    if dpgmm.catalog is not None:
        number_of_galaxies = np.zeros(len(CLs),dtype=np.int)

        for i,CL in enumerate(CLs):
            threshold = dpgmm.heights[str(CL)]
            (k,) = np.where(dpgmm.ranked_probability>threshold)
            number_of_galaxies[i] = len(k)

        np.savetxt(os.path.join(options.output,'galaxy_in_confidence_regions.txt'), np.array([CLs,number_of_galaxies]).T, fmt='%.2f\t%d')

        if dpgmm.injection is not None:
            threshold = dpgmm.injection_volume_height
            (k,) = np.where(dpgmm.ranked_probability>threshold)
            number_of_galaxies = len(k)
            with open(os.path.join(options.output,'searched_galaxies.txt'),'w') as f:
                f.write('%.5f\t%d\n'%(dpgmm.injection_volume_confidence,number_of_galaxies))
                f.close()

    if options.plots:
        import matplotlib.pyplot as plt
        plt.plot(dpgmm.grid[0],dpgmm.distance_map,color="k",linewidth=2.0)
        plt.hist(samples[:,0],bins=dpgmm.grid[0],density=True,facecolor="0.9")
        if injFile!=None: plt.axvline(dist_inj,color="k",linestyle="dashed")
        plt.xlabel(r"$\mathrm{Distance/Mpc}$")
        plt.ylabel(r"$\mathrm{probability}$ $\mathrm{density}$")
        plt.savefig(os.path.join(options.output,'distance_posterior.pdf'),bbox_inches='tight')
    path = os.path.join(options.output,'confidence_levels.txt')
    np.savetxt(path, np.array([CLs, volumes, areas, distances, ramin, ramax, decmin, decmax]).T, fmt='%.2f\t%f\t%f\t%f\t%f\t%f\t%f\t%f')
    if dpgmm.injection is not None: np.savetxt(os.path.join(options.output,'searched_quantities.txt'), np.array([searched_volume,searched_area,searched_distance]), fmt='%s\t%s\t%s')

    np.savetxt(os.path.join(options.output,'distance_map.txt'), np.array([dpgmm.grid[0], dpgmm.unnormed_distance_map]).T, fmt='%f\t%f', header='dist\tpost')
    np.savetxt(os.path.join(options.output,'diff_volume.txt'), np.array([dpgmm.differential_distances, dpgmm.differential_volume]).T, fmt='%f\t%f', header='dist\tvolume')
    np.savetxt(os.path.join(options.output,'diff_area.txt'), np.array([dpgmm.differential_distances, dpgmm.differential_area]).T, fmt='%f\t%f', header='dist\tarea')


    # dist_inj,ra_inj,dec_inj,tc
    if injFile is not None:
        gmst_deg = np.mod(np.degrees(gmst_rad_inj), 360)
        lon_cen = lon_inj = np.degrees(ra_inj) - gmst_deg
        lat_cen = lat_inj = np.degrees(dec_inj)
    else:
        gmst_deg = np.mod(np.degrees(np.array(gmst_rad)), 360)
        lon_cen = np.degrees(np.mean(samples[idx,2])) - np.mean(gmst_deg)
        lat_cen = np.degrees(np.mean(samples[idx,1]))

    lon_samp = np.degrees(samples[idx,2]) - gmst_deg
    lat_samp = np.degrees(samples[idx,1])

    ra_map,dec_map = dpgmm.grid[2],dpgmm.grid[1]
    lon_map = np.degrees(ra_map) - np.mean(gmst_deg)
    lat_map = np.degrees(dec_map)

    if options.plots:
        sys.stderr.write("producing sky maps \n")
        try:
            plt.figure()
            plt.plot(np.arange(1,dpgmm.max_sticks+1),dpgmm.scores,'.')
            plt.xlabel(r"$\mathrm{number}$ $\mathrm{of}$ $\mathrm{components}$")
            plt.ylabel(r"$\mathrm{marginal}$ $\mathrm{likelihood}$")
            plt.savefig(os.path.join(out_dir, 'scores.pdf'))
        except:
            pass
#         from mpl_toolkits.basemap import Basemap,shiftgrid
#         # make an orthographic projection map
#         plt.figure()
#         m = Basemap(projection='ortho', lon_0=round(lon_cen, 2), lat_0=lat_cen, resolution='c')
#         m.drawcoastlines(linewidth=0.5, color='0.5')
#         m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
#         m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
#         m.drawmapboundary(linewidth=0.5, fill_color='white')
#         X,Y = m(*np.meshgrid(lon_map, lat_map))
# #        plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
#         S = m.contourf(X,Y,dpgmm.skymap,100,linestyles='-', hold='on', origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = 0.0)
#         if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
#         plt.savefig(os.path.join(out_dir, 'marg_sky.pdf'))
#
#         plt.figure()
#         m = Basemap(projection='hammer', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
#         m.drawcoastlines(linewidth=0.5, color='0.5')
#         m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
#         m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
#         m.drawmapboundary(linewidth=0.5, fill_color='white')
#         X,Y = m(*np.meshgrid(lon_map, lat_map))
#         plt.scatter(*m(lon_samp, lat_samp), color='k', s=0.1, lw=0)
#         S = m.contourf(X,Y,dpgmm.skymap,100,linestyles='-', hold='on',origin='lower', cmap='YlOrRd', s=2, lw=0, vmin = 0.0)
#         if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='r', s=500, marker='+')
#         plt.savefig(os.path.join(out_dir, 'marg_sky_hammer.pdf'))

        if options.plots:
            if options.catalog:
                out_dir = os.path.join(out_dir, 'galaxies_scatter')
                os.system("mkdir -p %s"%out_dir)
                sys.stderr.write("producing 3 dimensional maps\n")
                # Create a sphere
                x = dpgmm.ranked_dl*np.cos(dpgmm.ranked_dec)*np.cos(dpgmm.ranked_ra)
                y = dpgmm.ranked_dl*np.cos(dpgmm.ranked_dec)*np.sin(dpgmm.ranked_ra)
                z = dpgmm.ranked_dl*np.sin(dpgmm.ranked_dec)

                threshold = dpgmm.heights['0.9']
                (k,) = np.where(dpgmm.ranked_probability>threshold)
                path = os.path.join(options.output,'galaxy_0.9_%.3f.txt') %(dpgmm.h)
                np.savetxt(path,
                           np.array([np.degrees(dpgmm.ranked_ra[k]),np.degrees(dpgmm.ranked_dec[k]),dpgmm.ranked_dl[k],dpgmm.ranked_z[k],dpgmm.ranked_B[k], dpgmm.ranked_dB[k], dpgmm.ranked_Babs[k], dpgmm.peculiarmotion[k],dpgmm.ranked_probability[k]]).T,
                           fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                           header='ra\tdec\tDL\tz\tB\tB_err\tB_abs\tpec.mot.corr.\tlogposterior')

                imax = dpgmm.ranked_probability.argmax()
                threshold = dpgmm.heights['0.5']
                (k,) = np.where(dpgmm.ranked_probability>threshold)
                MIN = dpgmm.grid[0][0]
                MAX = dpgmm.grid[0][-1]
                sys.stderr.write("%d galaxies above threshold, plotting\n"%(len(k)))
                np.savetxt(os.path.join(options.output,'galaxy_0.5.txt'),
                           np.array([np.degrees(dpgmm.ranked_ra[k]),np.degrees(dpgmm.ranked_dec[k]),dpgmm.ranked_dl[k],dpgmm.ranked_z[k],dpgmm.ranked_B[k], dpgmm.ranked_dB[k], dpgmm.ranked_Babs[k], dpgmm.peculiarmotion[k],dpgmm.ranked_probability[k]]).T,
                           fmt='%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t',
                           header='ra\tdec\tDL\tz\tB\tB_err\tB_abs\tpec.mot.corr.\tlogposterior')
                from mpl_toolkits.mplot3d import Axes3D
#                fig = plt.figure(figsize=(13.5,8))  # PRL default width
                fig = plt.figure(figsize=(13.5,9))
                ax = fig.add_subplot(111, projection='3d')
                # draw sphere
                u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
                xs = MAX*np.cos(u)*np.sin(v)
                ys = MAX*np.sin(u)*np.sin(v)
                zs = MAX*np.cos(v)
                ax.plot_wireframe(xs, ys, zs, color="0.95", alpha=0.5, lw=0.5)
                ax.scatter([0.0],[0.0],[0.0],c='k',s=200,marker=r'$\bigoplus$',edgecolors='none')
                S = ax.scatter(x[k],y[k],z[k],c=dpgmm.ranked_probability[k],s=1000*(dpgmm.ranked_dl[k]/options.dmax)**2,marker='o',edgecolors='None', cmap = plt.get_cmap("viridis"))#,norm=matplotlib.colors.LogNorm()
#                ax.scatter(x[imax],y[imax],z[imax],c=dpgmm.ranked_probability[imax],s=128,marker='+')#,norm=matplotlib.colors.LogNorm()
                C = fig.colorbar(S)
                C.set_label(r"$\mathrm{probability}$ $\mathrm{density}$")

                ax.plot(np.linspace(-MAX,MAX,100),np.zeros(100),color='0.5', lw=0.7, zdir='y', zs=0.0)
                ax.plot(np.linspace(-MAX,MAX,100),np.zeros(100),color='0.5', lw=0.7, zdir='x', zs=0.0)
                ax.plot(np.zeros(100),np.linspace(-MAX,MAX,100),color='0.5', lw=0.7, zdir='y', zs=0.0)

                ax.set_xlim([-MAX, MAX])
                ax.set_ylim([-MAX, MAX])
                ax.set_zlim([-MAX, MAX])

                ax.set_xlabel(r"$D_L/\mathrm{Mpc}$")
                ax.set_ylabel(r"$D_L/\mathrm{Mpc}$")
                ax.set_zlabel(r"$D_L/\mathrm{Mpc}$")

                ax.view_init(elev=10, azim=135)
                ax.grid(False)
                ax.xaxis.pane.set_edgecolor('black')
                ax.yaxis.pane.set_edgecolor('black')
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                [t.set_va('center') for t in ax.get_yticklabels()]
                [t.set_ha('left') for t in ax.get_yticklabels()]
                [t.set_va('center') for t in ax.get_xticklabels()]
                [t.set_ha('right') for t in ax.get_xticklabels()]
                [t.set_va('center') for t in ax.get_zticklabels()]
                [t.set_ha('left') for t in ax.get_zticklabels()]
                ax.xaxis._axinfo['tick']['inward_factor'] = 0
                ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
                ax.yaxis._axinfo['tick']['inward_factor'] = 0
                ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
                ax.zaxis._axinfo['tick']['inward_factor'] = 0
                ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
                ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

                for ii in range(0,360,1):
                    sys.stderr.write("producing frame %03d\r"%ii)
                    ax.view_init(elev=10., azim=ii)
                    #plt.savefig(os.path.join(out_dir, 'galaxies_3d_scatter_%03d.png'%ii),dpi=300)
                sys.stderr.write("\n")

                # make an animation
                # os.system("ffmpeg -f image2 -r 10 -i %s -vcodec mpeg4 -y %s"%(os.path.join(out_dir, 'galaxies_3d_scatter_%03d.png'),os.path.join(out_dir, 'movie.mp4')))

                # plt.figure()
                # lon_gals = np.degrees(dpgmm.ranked_ra[k][::-1]) - np.mean(gmst_deg)
                # lat_gals = np.degrees(dpgmm.ranked_dec[k][::-1])
                # dl_gals = dpgmm.ranked_dl[k][::-1]
                # logProbability = dpgmm.ranked_probability[k][::-1]
                # m = Basemap(projection='moll', lon_0=round(lon_cen, 2), lat_0=0, resolution='c')
                # m.drawcoastlines(linewidth=0.5, color='0.5')
                # m.drawparallels(np.arange(-90,90,30), labels=[1,0,0,0], labelstyle='+/-', linewidth=0.1, dashes=[1,1], alpha=0.5)
                # m.drawmeridians(np.arange(0,360,60), linewidth=0.1, dashes=[1,1], alpha=0.5)
                # m.drawmapboundary(linewidth=0.5, fill_color='white')
                #
                # S = plt.scatter(*m(lon_gals, lat_gals), s=10, c=dl_gals, lw=0, marker='o')
                #
                # if injFile is not None: plt.scatter(*m(lon_inj, lat_inj), color='k', s=500, marker='+')
                # cbar = m.colorbar(S,location='bottom',pad="5%")
                # cbar.set_label(r"$\log(\mathrm{Probability})$")
                # plt.savefig(os.path.join(out_dir, 'galaxies_marg_sky.pdf'))

    if options.threed:
        # produce a volume plot
        from volume_reconstruction.plotting.plot import volume_rendering
        volume_rendering()


    sys.stderr.write("\n")

if __name__=='__main__':
    main()
