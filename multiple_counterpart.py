#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cosmological_model as CM
import numpy as np
import matplotlib.pyplot as plt
import lal
import readdata
import cpnest.model

class options(object):

    def __init__(self, event_class, event, data, out_dir = None,model = 'LambdaCDM', nevmax = None):

        self.event_class = event_class
        self.out_dir     = out_dir
        self.event       = event
        self.model       = model
        self.nevmax      = nevmax
        self.data        = data

if __name__ == '__main__':

    Ntot_events = 10

    for i in range(Ntot_events):
        opts = options('TEST', i, './MDC', nevmax = i+1)

        errors = {'z':0.001, 'RA':0.01, 'DEC':0.01}
        omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,0,0,0) # True cosmology
        events = readdata.read_event(opts.event_class, errors = errors, omega = omega, input_folder = opts.data, N_ev_max = opts.nevmax)

        model = opts.model
        output = opts.data+"/MULTIPLEEVENT_1%03d/"%(opts.event+1)

        print('Working on run {0} of {1}'.format(i, Ntot_events))

        C = CM.CosmologicalModel(model,
                              events,
                              em_selection  = 0,
                              snr_threshold = 0.0,
                              z_threshold   = 1000.,
                              event_class   = opts.event_class)

        work=cpnest.CPNest(C,
                           verbose      = 1,
                           poolsize     = 100,
                           nthreads     = 4,
                           nlive        = 1000,
                           maxmcmc      = 100,
                           output       = output,
                           nhamiltonian = 0)
        work.run()
        print('log Evidence {0}'.format(work.NS.logZ))
        x = work.posterior_samples.ravel()

        samps = np.column_stack((x['h'],x['om']))
        fig = corner.corner(samps,
               labels= [r'$h$',
                        r'$\Omega_m$'],
               quantiles=[0.05, 0.5, 0.95],
               show_titles=True, title_kwargs={"fontsize": 12},
               use_math_text=True, truths=[0.7,0.3],
               filename=os.path.join(output,'joint_posterior.pdf'))

        fig.savefig(os.path.join(output,'joint_posterior.pdf'), bbox_inches='tight')
