#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import lal
import sys
import os
import readdata
import cosmolisa.cosmology as cs
import numpy as np
import matplotlib.pyplot as plt
from displaypost import plot_post
import math
from schechter import *
import random as rd
from scipy.stats import poisson

def appM(z, M, omega):
    return M + 5.0*np.log10(1e5*lal.LuminosityDistance(omega,z)) - 5.*np.log10(omega.h)

def gaussian(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def LumDist(z, omega):
    return 3e3*(z + (1-omega.om +omega.ol)*z**2/2.)/omega.h

def dLumDist(z, omega):
    return 3e3*(1+(1-omega.om+omega.ol)*z)/omega.h

def RedshiftCalculation(LD, omega, zinit=0.3, limit = 0.0001):
    '''
    Redshift given a certain luminosity, calculated by recursion.
    Limit is the less significative digit.
    '''
    LD_test = LumDist(zinit, omega)
    if abs(LD-LD_test) < limit :
        return zinit
    znew = zinit - (LD_test - LD)/dLumDist(zinit,omega)
    return RedshiftCalculation(LD, omega, zinit = znew)

def generate_galaxy(ipar,Schechter, omega, i, ID,ra,dec,z,z_cosmo,DL,absB,dB,appB,host):
    ID_try = i
    ra_try = rd.uniform(0, 2*np.pi)
    dec_try = np.arcsin(rd.uniform(np.sin(-np.pi/2.), np.sin(np.pi/2.)))
    while 1:
        z_temp = rd.uniform(ipar['z_min'], ipar['z_max'])
        if rd.random()*ipar['dCoVolMax'] < lal.ComovingVolumeElement(z_temp,omega):
            z_cosmo_try = z_temp
            break
    z_pec = rd.gauss(0, 0.0001)
    z_try = z_cosmo_try+z_pec
    DL_try = lal.LuminosityDistance(omega,z_cosmo_try)
    while 1:
        B_temp = rd.uniform(ipar['M_min'], ipar['M_max'])
        if rd.random()*ipar['pM_max'] < Schechter(B_temp):
            absB_try = B_temp
            break

    if appM(z_cosmo_try, absB_try, omega) < ipar['m_th']:
        ID.append(ID_try)
        ra.append(np.rad2deg(ra_try))
        dec.append(np.rad2deg(dec_try))
        z.append(z_try)
        z_cosmo.append(z_cosmo_try)
        DL.append(DL_try)
        absB.append(absB_try)
        dB.append(0.5)
        appB.append(appM(z_cosmo_try, absB_try, omega))
        host.append(0)

    return

if __name__ == '__main__':

    n_ev = 0
    omega = lal.CreateCosmologicalParameters(0.697, 0.306, 0.694, -1, 0, 0)
    M_max    = -6.
    M_min    = -23.

    m_th = 18.

    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, omega.h)
    # output = '/path/to/folder/'
    output = '/Users/stefanorinaldi/Desktop/cat_test/'
    if not os.path.exists(output):
        os.mkdir(output)
    numberdensity = 0.066

    z_min = 0
    z_max = 0.001

    dCoVolMax = lal.ComovingVolumeElement(z_max,omega)
    pM_max    = Schechter(M_max)
    CoVol = lal.ComovingVolume(omega, z_max) - lal.ComovingVolume(omega, z_min)
    print('CoVol=', CoVol)
    ev_density = n_ev/CoVol
    np.savetxt(output+'evdensity.txt', np.array([ev_density]).T, header = 'evdensity')
    N_tot = int(CoVol*numberdensity)

    ipar = {
            'z_min': z_min,
            'z_max': z_max,
            'M_min': M_min,
            'M_max': M_max,
            'm_th': m_th,
            'dCoVolMax': dCoVolMax,
            'pM_max': pM_max
            }

    ID      = []
    ra      = []
    dec     = []
    z_cosmo = []
    z       = []
    appB    = []
    absB    = []
    dB      = []
    DL      = []
    host    = []

    ID_h      = []
    ra_h      = []
    dec_h     = []
    z_cosmo_h = []
    z_h       = []
    appB_h    = []
    absB_h    = []
    dB_h      = []
    DL_h      = []
    host_h    = []

    for i in range(N_tot):

        sys.stdout.write('{0} out of {1}\r'.format(i+1, N_tot))
        sys.stdout.flush()
        generate_galaxy(ipar, Schechter, omega, i, ID,ra,dec,z,z_cosmo,DL,absB,dB,appB,host)

    for i in range(n_ev):
        while 1:
            index = rd.randint(0,N_tot-1)
            if absB[index] < -15.:
                #new_B = rd.gauss(M_mean,sigma)
                #new_Bapp = appM(z_cosmo[index], new_B, omega)
                host[index] = 1
                #absB[index] = new_B
                #appB[index] = new_Bapp
                ID_h.append(ID[index])
                ra_h.append(ra[index])
                dec_h.append(dec[index])
                z_cosmo_h.append(z_cosmo[index])
                z_h.append(z[index])
                appB_h.append(appB[index])
                absB_h.append(absB[index])
                dB_h.append(dB[index])
                DL_h.append(DL[index])
                host_h.append(host[index])
                break

    header = 'ID\tra\t\tdec\t\tz\t\tz_cosmo\t\tDL\t\tB_abs\t\tB\t\tB_err\t\thost'
    fmt = '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d'
    np.savetxt(output+'mockcatalog.txt', np.array([ID, ra, dec, z, z_cosmo, DL, absB, appB, dB, host]).T, fmt = fmt, header = header)
    if n_ev > 0:
        np.savetxt(output+'hosts.txt', np.array([ID_h, ra_h, dec_h, z_h, z_cosmo_h, DL_h, absB_h, appB_h, dB_h, host_h]).T, fmt = fmt, header = header)

    fig_z_cosmo = plt.figure()
    fig_z_pm    = plt.figure()
    fig_M       = plt.figure()
    fig_M_hosts = plt.figure()

    ax_z_cosmo  = fig_z_cosmo.add_subplot(111)
    ax_z_pm     = fig_z_pm.add_subplot(111)
    ax_M        = fig_M.add_subplot(111)
    ax_M_hosts  = fig_M_hosts.add_subplot(111)


    app_z = np.linspace(z_min, z_max, 1000)
    app_CoVol = []


    for zi in app_z:
        app_CoVol.append(lal.ComovingVolumeElement(zi, omega)/CoVol)


    app_z_pm    = np.linspace(-5*0.0001, 5*0.0001, 1000)
    app_M       = np.linspace(M_min, M_max, 1000)
    app_M_hosts = np.linspace(M_mean-4*sigma, M_mean+4*sigma, 1000)
    app_pM      = []

    for Mi in app_M:
        ratio = float(n_ev)/float(N_tot)
        app_pM.append((1-ratio)*Schechter(Mi)+ratio*gaussian(Mi, M_mean, sigma))

    ax_z_cosmo.hist(z_cosmo, bins = int(np.sqrt(len(z_cosmo))), density = True, color='lightblue', label = '$a_{cosmo}$')
    ax_z_cosmo.plot(app_z, app_CoVol, color = 'red', linewidth = 0.5, label = '$\\propto dV_{cov}/dz$')
    ax_z_cosmo.set_xlabel('$z_{cosmo}$')
    ax_z_cosmo.set_ylabel('$p(z_{cosmo})$')
    ax_z_cosmo.legend(loc = 0)
    fig_z_cosmo.savefig(output+'z_cosmo.pdf', bbox_inches='tight')

    ax_z_pm.hist(np.array(z)-np.array(z_cosmo), bins = int(np.sqrt(len(z_cosmo))), density = True, color='lightblue', label ='z_{pm}')
    ax_z_pm.plot(app_z_pm, gaussian(app_z_pm, 0, 0.001), color = 'red', linewidth = 0.5, label = '$\\propto exp((z_{pm})^2/2\\sigma^2)$')
    ax_z_pm.set_xlabel('$z_{pm}$')
    ax_z_pm.set_ylabel('$p(z_{pm})$')
    ax_z_pm.legend(loc=0)
    fig_z_pm.savefig(output+'z_pm.pdf', bbox_inches='tight')

    ax_M.hist(absB, bins = int(np.sqrt(len(absB))), density = True, color='lightblue', label = '$M$')
    ax_M.plot(app_M, app_pM, color = 'red', linewidth = 0.5, label = '$\\propto Sch(M)$')
    ax_M.set_xlabel('$M\ (B\ band)$')
    ax_M.set_ylabel('$p(M)$')
    ax_M.legend(loc = 0)
    fig_M.savefig(output+'M.pdf', bbox_inches='tight')

    if n_ev > 0:
        ax_M_hosts.hist(absB_h, bins = int(np.sqrt(len(absB_h))), density = True, color='lightblue', label = '$M$')
        ax_M_hosts.plot(app_M_hosts, gaussian(app_M_hosts, M_mean, sigma), color = 'red', linewidth = 0.5, label = '$f(M)$')
        ax_M_hosts.set_xlabel('$M\ (B\ band, hosts)$')
        ax_M_hosts.set_ylabel('$p(M)$')
        ax_M_hosts.legend(loc=0)
        fig_M_hosts.savefig(output+'M_hosts.pdf', bbox_inches='tight')

    z_em = []
    z_det = []
    for zi, Bi, absBi in zip(z,appB, absB):
        if absBi < -15.:
            z_em.append(zi)
            if Bi < 18.:
                z_det.append(zi)

    N_em, bins, other = plt.hist(z_em, bins = 100)# int(np.sqrt(len(z_em))))
    N_det, bins, other = plt.hist(z_det, bins = bins)

    # reds = []
    # for i in range(len(bins)-1):
    #     reds.append((bins[i]+bins[i+1])/2.)
    # reds = np.array(reds)
    # gamma = np.array(N_det)/np.array(N_em)
    #
    # np.savetxt(output+'completeness_z.txt', np.array([reds, gamma]).T, header = 'z\tgamma')
    #
    # fig_completeness = plt.figure()
    # ax_compl = fig_completeness.add_subplot(111)
    # ax_compl.plot(reds,gamma)
    # ax_compl.set_xlabel('$z$')
    # ax_compl.set_ylabel('$\\gamma(z)$')
    # ax_compl.set_xlim(0,0.04)
    # fig_completeness.savefig(output+'completeness_z.pdf', bbox_inches='tight')
