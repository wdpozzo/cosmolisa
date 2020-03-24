#!/usr/bin/env python
# -*- coding: utf-8 -*-

from likelihood import *
import readdata
from galaxies import *
import lal
import cosmology as cs

catalog_file = 'MDC/catalog_1.txt'
# skypos = {'z':0.07, 'RA':np.deg2rad(182.656296), 'DEC':np.deg2rad(16.032934)}
errors = {'z':0.001, 'RA':0.01, 'DEC':0.01}
omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.,0,0) # True cosmology
events = readdata.read_event('TEST', errors = errors, omega = omega, input_folder = './darksiren')
print(events)
omega = cs.CosmologicalParameters(0.7,0.3,0.7,-1.0,0.0)
for event in events:
    logL = logLikelihood_single_event(event.potential_galaxy_hosts, event, omega, m_th=17., Ntot = event.n_hosts ,zmin = event.zmin, zmax = event.zmax)
    print(logL)
