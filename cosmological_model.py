#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
from scipy.special import logsumexp
from functools import reduce
from scipy.stats import norm
import unittest
import lal
import cpnest.model
import sys
import os
import readdata
import matplotlib
import corner
import itertools as it
import cosmology as cs
import numpy as np
import likelihood as lk
import matplotlib.pyplot as plt
from displaypost import plot_post
import math


"""
G = the GW is in a galaxy that I see
N = the GW is in a galaxy that I do not see
D = a GW
I = I see only GW with SNR > 20

p(H|D(G+N)I) propto p(H|I)p(D(G+N)|HI)
p(D(G+N)|HI) = p(DG+DN|HI) = p(DG|HI)+p(DN|HI) = p(D|GHI)p(G|HI)+p(D|NHI)p(N|HI) = p(D|HI)(p(G|HI)+p(N|HI))
"""

class CosmologicalModel(cpnest.model.Model):

    names  = [] #'h','om','ol','w0','w1']
    bounds = [] #[0.5,1.0],[0.04,1.0],[0.0,1.0],[-2.0,0.0],[-3.0,3.0]]

    def __init__(self, model, data, *args, **kwargs):

        super(CosmologicalModel,self).__init__()
        # Set up the data
        self.data           = data
        self.N              = len(self.data)
        self.model          = model
        self.em_selection   = kwargs['em_selection']
        self.z_threshold    = kwargs['z_threshold']
        self.snr_threshold  = kwargs['snr_threshold']
        self.event_class    = kwargs['event_class']
        self.O              = None

        if self.model == "LambdaCDM":

            self.names  = ['h','om']
            self.bounds = [[0.5,1.0],[0.04,0.5]]

        elif self.model == "LambdaCDMDE":

            self.names  = ['h','om','ol','w0','w1']
            self.bounds = [[0.5,1.0],[0.04,0.5],[0.0,1.0],[-2.0,0.0],[-3.0,3.0]]

        elif self.model == "CLambdaCDM":

            self.names  = ['h','om','ol']
            self.bounds = [[0.5,1.0],[0.04,0.5],[0.0,1.0]]

        elif self.model == "DE":

            self.names  = ['w0','w1']
            self.bounds = [[-3.0,-0.3],[-1.0,1.0]]

        else:

            print("Cosmological model %s not supported. Exiting...\n"%self.model)
            exit()
        #
        # for e in self.data:
        #     self.bounds.append([e.zmin,e.zmax])
        #     self.names.append('z%d'%e.ID)

        #  self._initialise_galaxy_hosts()

        print("==================================================")
        print("cpnest model initialised with:")
        print("Cosmological model: {0}".format(self.model))
        print("Number of events: {0}".format(len(self.data)))
        print("EM correction: {0}".format(self.em_selection))
        print("==================================================")

    # def _initialise_galaxy_hosts(self):
    #     self.hosts = {e.ID:np.array([(g.redshift,g.dredshift,g.weight) for g in e.potential_galaxy_hosts]) for e in self.data}

    def log_prior(self,x):
        logP = super(CosmologicalModel,self).log_prior(x)

        if np.isfinite(logP):
            """
            apply a uniform in comoving volume density redshift prior
            """
            if self.model == "LambdaCDM":

                z_idx = 2
                self.O = cs.CosmologicalParameters(x['h'],x['om'],1.0-x['om'],-1.0,0.0)

            elif self.model == "CLambdaCDM":

                z_idx = 3
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],-1.0,0.0)

            elif self.model == "LambdaCDMDE":

                z_idx = 5
                self.O = cs.CosmologicalParameters(x['h'],x['om'],x['ol'],x['w0'],x['w1'])

            elif self.model == "DE":

                z_idx = 2
                self.O = cs.CosmologicalParameters(0.73,0.25,0.75,x['w0'],x['w1'])

#            if self.event_class == "EMRI" or self.event_class == "sBH":
#                for j,e in enumerate(self.data):
#                    #log_norm = np.log(self.O.IntegrateComovingVolumeDensity(self.bounds[z_idx+j][1]))
#                    logP += np.log(self.O.UniformComovingVolumeDensity(x['z%d'%e.ID]))#-log_norm

        return logP

    def log_likelihood(self,x):

        # compute the p(GW|G\Omega)p(G|\Omega)+p(GW|~G\Omega)p(~G|\Omega)
        # logL = np.sum([lk.logLikelihood_single_event(self.hosts[e.ID], e.dl, e.sigma, self.O,
        #                         em_selection = self.em_selection, zmin = self.bounds[2+j][0], zmax = self.bounds[2+j][1]) for j,e in enumerate(self.data)])
        logL = 0.
        for e in self.data:
            logL += lk.logLikelihood_single_event(e.potential_galaxy_hosts, e, self.O, 18., Ntot = e.n_tot, zmin = e.zmin, zmax = e.zmax)
        self.O.DestroyCosmologicalParameters()
        if math.isinf(logL):
            return -np.inf

        return logL

truths = {'h':0.73,'om':0.25,'ol':0.75,'w0':-1.0,'w1':0.0}
usage=""" %prog (options)"""

if __name__=='__main__':

    parser = OptionParser(usage)
    parser.add_option('-d', '--data',        default=None, type='string', metavar='data', help='Galaxy data location')
    parser.add_option('-o', '--out-dir',     default=None, type='string', metavar='DIR', help='Directory for output')
    parser.add_option('-c', '--event-class', default=None, type='string', metavar='event_class', help='Class of the event(s) [MBH, EMRI, sBH]')
    parser.add_option('-e', '--event',       default=None, type='int', metavar='event', help='Event number')
    parser.add_option('-m', '--model',       default='LambdaCDM', type='string', metavar='model', help='Cosmological model to assume for the analysis (default LambdaCDM). Supports LambdaCDM, CLambdaCDM, LambdaCDMDE, and DE.')
    parser.add_option('-j', '--joint',       default=0, type='int', metavar='joint', help='Run a joint analysis for N events, randomly selected (EMRI only).')
    parser.add_option('-z', '--zhorizon',    default=1000.0, type='float', metavar='zhorizon', help='Horizon redshift corresponding to the SNR threshold')
    parser.add_option('--snr_threshold',     default=0.0, type='float', metavar='snr_threshold', help='SNR detection threshold')
    parser.add_option('--em_selection',      default=0, type='int', metavar='em_selection', help='Use EM selection function')
    parser.add_option('--reduced_catalog',   default=0, type='int', metavar='reduced_catalog', help='Select randomly only a fraction of the catalog')
    parser.add_option('-t', '--threads',     default=None, type='int', metavar='threads', help='Number of threads (default = 1/core)')
    parser.add_option('-s', '--seed',        default=0, type='int', metavar='seed', help='Random seed initialisation')
    parser.add_option('--nlive',             default=1000, type='int', metavar='nlive', help='Number of live points')
    parser.add_option('--poolsize',          default=100, type='int', metavar='poolsize', help='Poolsize for the samplers')
    parser.add_option('--maxmcmc',           default=1000, type='int', metavar='maxmcmc', help='Maximum number of mcmc steps')
    parser.add_option('--postprocess',       default=0, type='int', metavar='postprocess', help='Run only the postprocessing')
    parser.add_option('-n', '--nevmax',      default=None, type='int', metavar='nevmax', help='Maximum number of considered events')
    parser.add_option('-u', '--uncert',      default='0.1', type='float', metavar='uncert', help='Relative uncertainty on z of each galaxy (peculiar motion)')
    parser.add_option('-a', '--hosts',       default=None, type='int', metavar='hosts', help='Total number of galaxies in considered volume')
    (opts,args)=parser.parse_args()

    em_selection = opts.em_selection

    if opts.event_class == 'TEST':
        errors = {'z':0.001, 'RA':0.01, 'DEC':0.01}
        omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.,0.,0.) # True cosmology
        rel_z_error = opts.uncert # errore relativo sullo z della galassia (moto proprio + errore sperimentale)
        events = readdata.read_event(opts.event_class, errors = errors, omega = omega, input_folder = opts.data, N_ev_max = opts.nevmax, rel_z_error = rel_z_error, n_tot = opts.hosts)

    else:
        events = readdata.read_event(opts.event_class, opts.data, opts.event)

    model = opts.model
    if opts.event is None:
        opts.event = 0

    if opts.out_dir is None:
        output = opts.data+"/EVENT_1%03d/"%(opts.event+1)
    else:
        output = opts.out_dir

    C = CosmologicalModel(model,
                          events,
                          em_selection  = em_selection,
                          snr_threshold = opts.snr_threshold,
                          z_threshold   = opts.zhorizon,
                          event_class   = opts.event_class)

    if opts.postprocess == 0:
        work=cpnest.CPNest(C,
                           verbose      = 3,
                           poolsize     = opts.poolsize,
                           nthreads     = opts.threads,
                           nlive        = opts.nlive,
                           maxmcmc      = opts.maxmcmc,
                           output       = output,
                           nhamiltonian = 0)
        work.run()
        print('log Evidence {0}'.format(work.NS.logZ))
        x = work.posterior_samples.ravel()
    else:
        x = np.genfromtxt(os.path.join(output,"chain_"+str(opts.nlive)+"_1234.txt"), names=True)
        from cpnest import nest2pos
        x = nest2pos.draw_posterior_many([x], [opts.nlive], verbose=False)


    if model == "LambdaCDM":
        samps = np.column_stack((x['h'],x['om']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_kwargs={"fontsize": 12},
               use_math_text=True, truths=[0.7,0.3],
               filename=os.path.join(output,'joint_posterior.pdf'))

    if model == "CLambdaCDM":
        samps = np.column_stack((x['h'],x['om'],x['ol'],1.0-x['om']-x['ol']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$',
                        r'$\Omega_\Lambda$',
                        r'$\Omega_k$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_kwargs={"fontsize": 12},
               use_math_text=True, truths=[0.73,0.25,0.75,0.0],
               filename=os.path.join(output,'joint_posterior.pdf'))

    if model == "LambdaCDMDE":
        samps = np.column_stack((x['h'],x['om'],x['ol'],x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$h$',
                                 r'$\Omega_m$',
                                 r'$\Omega_\Lambda$',
                                 r'$w_0$',
                                 r'$w_1$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[0.73,0.25,0.75,-1.0,0.0],
                        filename=os.path.join(output,'joint_posterior.pdf'))

    if model == "DE":
        samps = np.column_stack((x['w0'],x['w1']))
        fig = corner.corner(samps,
                        labels= [r'$w_0$',
                                 r'$w_1$'],
                        quantiles=[0.05, 0.5, 0.95],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        use_math_text=True, truths=[-1.0,0.0],
                        filename=os.path.join(output,'joint_posterior.pdf'))

    fig.savefig(os.path.join(output,'joint_posterior.pdf'), bbox_inches='tight')
    plot_post(work, 'h', output, 0.7)
