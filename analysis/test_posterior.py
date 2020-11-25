#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cosmolisa.cosmology as cs
from cosmolisa.volumereconstruction import VolRec
from cosmolisa.readdata import *
import cosmolisa.likelihood as lk
import lal
import sys
import os
from time import perf_counter
import ray
import configparser

def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

@ray.remote
def computeloglikelihood(e, hi, opts):
    VolRec(input = opts['post'], output = opts['data'], bins = [60,180,360], dmax = 600, event_id = e.ID, catalog = opts['cat'], cosmology = 0, hubble = hi)
    omega = cs.CosmologicalParameters(hi, 0.3,0.7,-1,0)
    hosts_file = opts['data']+'galaxy_0.9_%.3f.txt' %(hi)
    cat_file   = opts['data']+'catalog_%.3f.txt' %(hi)
    hosts = read_galaxy_catalog({'RA':[0., 360.], 'DEC':[-90., 90.], 'z':[0., 4.]}, catalog_file = hosts_file, n_tot = 1.)
    cat   = read_galaxy_catalog({'RA':[0., 360.], 'DEC':[-90., 90.], 'z':[0., 4.]}, catalog_file = cat_file, n_tot = 1.)
    m_th = float(opts['m_th'])
    n = float(opts['n'])
    logL = lk.logLikelihood_single_event(hosts, cat, m_th, n, e, omega, zmin = 0.0005, zmax = 0.22, DL_max = 700)
    omega.DestroyCosmologicalParameters()
    return logL

if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    opts = config['DEFAULT']
    ray.init()
    
    init_time = perf_counter()

    events = read_event(opts['event_class'], input_folder = opts['data'], emcp = int(opts['EMcp']), nevmax = opts['nevmax'])

    if opts['out'] == None:
        opts['out'] = opts['data'] + 'output/'
        if not os.path.exists(opts['out']):
            os.mkdir(opts['out'])
            
            
    hmin = float(opts['hmin'])
    hmax = float(opts['hmax'])
    npoints = int(opts['npoints'])
    h = np.linspace(hmin, hmax, npoints)
    dh = (hmax - hmin)/npoints

    evcounter = 0
    likelihood_list = []
    likelihood_unnormed = []
    
    # loop over the events
    for e in events:
        likelihood_tasks = []
        evcounter += 1
        for hi in h:
            logL = 0.
            sys.stdout.write('Event %d of %d, h = %.3f, hmax = %.3f\n' % (evcounter, len(events), hi, h.max()))
            likelihood_tasks.append(computeloglikelihood.remote(e, hi, opts))
        
        likelihood = np.array(ray.get(likelihood_tasks))
        likelihood_unnormed.append(likelihood)
        # save unnormed loglikelihood
        np.savetxt(opts['out']+'likelihood_unnormed_'+str(e.ID)+'.txt', np.array([h, likelihood]).T, header = 'h\t\tlogL')
        
        # likelihood normalization
        likelihood = np.exp(likelihood - np.log(np.sum(np.exp(likelihood - likelihood.max()))*dh) - likelihood.max())
        likelihood_list.append(likelihood)
        # save likelihood
        np.savetxt(opts['out']+'likelihood_'+str(e.ID)+'.txt', np.array([h, likelihood]).T, header = 'h\t\tlogL')
        
    # compute unnormed joint likelihood
    joint_likelihood = np.zeros(len(likelihood))
    for like in likelihood_list:
        if np.isfinite(like.sum()):
            joint_likelihood += like
    # save joint likelihood
    joint_likelihood = np.exp(joint_likelihood - np.log(np.sum(np.exp(joint_likelihood - joint_likelihood.max()))*dh) - joint_likelihood.max())
    np.savetxt(opts['out']+'joint_likelihood.txt', np.array([h, joint_likelihood]).T, header = 'h\t\tlogL')
    
    # relevant quantities (68 & 95% credbile intervals and maximum a posteriori)
    percentiles = weighted_quantile(h*100, [0.05, 0.16, 0.50, 0.84, 0.95], sample_weight = np.exp(joint_likelihood))
    hmax = 100*h[np.where(joint_likelihood == joint_likelihood.max())]
    percentiles[2] = hmax
    
    # maquillage stuff
    thickness   = [0.4,0.5,1,0.5,0.4]
    styles      = ['dotted', 'dashed', 'solid', 'dashed','dotted']
    results     = str(percentiles[2])+'^{+'+str(percentiles[3]-percentiles[2])+'}_{-'+str(percentiles[2]-percentiles[1])+'}'
    title       = '$H_0 = '+results+'\ km\\cdot s^{-1}\\cdot Mpc^{-1}$'
    
    # plotting
    
    # Likelihood
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    fig.suptitle(title)
    # single event likelihood
    for l in likelihood_list:
        ax.plot(h*100., l/100., linewidth = 0.3)
    # values
    ax.axvline(float(opts['trueh'])*100, linewidth = 0.5, color = 'r')
    for p, t, s in zip(percentiles, thickness, styles):
        ax.axvline(p, ls = s, linewidth = t, color = 'darkblue')
    # joint likelihood
    ax.plot(h*100., joint_likelihood/100.)
    # axes
    ax.set_xlabel('$H_0\ [km\\cdot s^{-1}\\cdot Mpc^{-1}]$')
    ax.set_ylabel('$p(H_0)$')
    fig.savefig(opts['out']+'H0_posterior.pdf', bbox_inches='tight')
    
    # Log likelihood
    
    fig_log = plt.figure()
    ax_log  = fig_log.add_subplot(111)
    fig_log.suptitle(title)
    # single event likelihood
    for l in likelihood_list:
        ax_log.plot(h*100., l/100., linewidth = 0.3)
    # values
    ax_log.axvline(float(opts['trueh'])*100, linewidth = 0.5, color = 'r')
    for p, t, s in zip(percentiles, thickness, styles):
        ax_log.axvline(p, ls = s, linewidth = t, color = 'darkblue')
    # joint likelihood
    ax_log.plot(h*100., np.array(likelihood_unnormed))
    # axes
    ax_log.set_xlabel('$H_0\ [km\\cdot s^{-1}\\cdot Mpc^{-1}]$')
    ax_log.set_ylabel('$p(H_0)$')
    fig_log.savefig(opts['out']+'H0_logposterior.pdf', bbox_inches='tight')
    
    # duration
    end_time = perf_counter()
    print('Elapsed time: %f s' %(end_time - init_time))
    print('Elapsed time: %f m' %((end_time - init_time)/60.))
