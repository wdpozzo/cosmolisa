import numpy as np
import sys
import os
from galaxies import *
import lal


def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

# class Event(object):
#     """
#     Event class:
#     initialise a GW event based on its distance and potential
#     galaxy hosts
#     """
#     def __init__(self,
#                  ID,
#                  dl,
#                  sigma,
#                  redshifts,
#                  dredshifts,
#                  weights,
#                  zmin,
#                  zmax,
#                  snr,
#                  z_true,
#                  snr_threshold = 8.0,
#                  VC = None,
#                  catalog_file = None,
#                  catalog_data = None):
#
#         # self.potential_galaxy_hosts = [Galaxy(r,dr,w) for r,dr,w in zip(redshifts,dredshifts,weights)]
#         self.potential_galaxy_hosts = read_galaxy_catalog({'RA':[ramin, ramax], 'DEC':[decmin, decmax], 'z':[zmin, zmax]}, catalog_data, catalog_file)
#         self.n_hosts                = len(self.potential_galaxy_hosts)
#         self.ID                     = ID
#         self.dl                     = dl
#         self.sigma                  = sigma
#         self.dmax                   = (self.dl+3.0*self.sigma)
#         self.dmin                   = self.dl-3.0*self.sigma
#         self.zmin                   = zmin
#         self.zmax                   = zmax
#         self.snr                    = snr
#         self.VC                     = VC
#         self.z_true                 = z_true
#         if self.dmin < 0.0: self.dmin = 0.0

class Event_test(object):
    """
    Event class:
    initialise a GW event based on its distance and potential
    galaxy hosts
    """
    def __init__(self,
                 ID,
                 dz,
                 dRA,
                 dDEC,
                 z_true,
                 RA_true,
                 DEC_true,
                 omega,
                 catalog_file = None,
                 catalog_data = None):

        self.ramin   = RA_true-3*dRA
        self.ramax   = RA_true+3*dRA
        self.decmin  = DEC_true-3*dDEC
        self.decmax  = DEC_true+3*dDEC
        self.zmin    = z_true-3*dz
        self.zmax    = z_true+3*dz

        if catalog_file is None and catalog_data is None:
            raise SystemExit('No catalog provided')

        catalog = read_galaxy_catalog({'RA':[self.ramin, self.ramax], 'DEC':[self.decmin, self.decmax], 'z':[self.zmin, self.zmax]}, catalog_data, catalog_file)

        self.potential_galaxy_hosts = catalog
        self.n_hosts                = len(self.potential_galaxy_hosts)
        self.ID                     = ID
        self.LD                     = lal.LuminosityDistance(omega, z_true)
        self.dLD                    = self.LD-lal.LuminosityDistance(omega, z_true-dz)
        self.dz                     = dz
        self.dRA                    = dRA
        self.dDEC                   = dDEC
        self.z_true                 = z_true
        self.RA_true                = RA_true
        self.DEC_true               = DEC_true

    def post_LD(self, LD):
        app = gaussian(LD, self.LD, self.dLD)
        return app

    def post_RA(self, RA):
        app = gaussian(RA, self.RA_true, self.dRA)
        return app

    def post_DEC(self, DEC):
        app = gaussian(DEC, self.DEC_true, self.dDEC)
        return app


def read_TEST_event(skypos = None, errors = None, omega = None, catalog_file = None, input_folder = None, catalog_data = None):
    '''
    Classe di evento costruita per finalità di test. Le distribuzioni di probabilità sono gaussiane e centrate su una galassia a scelta.
    '''
    all_files   = os.listdir(input_folder)
    events_list = [f for f in all_files if 'catalog' in f]
    print(events_list)
    events = []

    for ev in events_list:
        catalog_file        = input_folder+"/"+ev
        event_file          = open(catalog_file,"r")
        data                = event_file.readline().split(' ')
        events.append(Event_test(0, errors['z'], errors['RA'], errors['DEC'], float(data[10]), np.deg2rad(float(data[6])), np.deg2rad(float(data[7])), omega, catalog_file, catalog_data))
        event_file.close()



    return np.array(events)


def read_event(event_class,*args,**kwargs):

    if event_class == "TEST": return read_TEST_event(*args, **kwargs)
    else:
        print("I do not know the class %s, exiting\n"%event_class)
        exit(-1)

if __name__=="__main__":
    input_folder = '/Users/wdp/repositories/LISA/LISA_BHB/errorbox_data/EMRI_data/EMRI_M1_GAUSS'
    event_number = None
    e = read_event("EMRI",input_folder, event_number)
    print(e)
