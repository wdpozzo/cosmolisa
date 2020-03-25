#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np

def get_pdf(CPjob, label):
    '''
    Given a CPNest object and the name (string) of the required variable, computes the pdf
    '''
    samples = CPjob.get_posterior_samples()[label]
    pdf     = gaussian_kde(samples)
    return pdf, samples

def plotting(pdf, samples, label, inj_val = None, directory = './'):

    median = np.percentile(samples, 50, axis = 0)
    sigma  = median - np.percentile(samples, 16, axis = 0)
    # x  = np.linspace(pdf.dataset.min(),pdf.dataset.max(),1000)
    x  = np.linspace(median-4*sigma, median+4*sigma, 1000)
    px = pdf(x)
    plt.figure()
    plt.plot(x, px, c = 'k')
    plt.suptitle(label+'$=%.3f_{-%.3f}^{+%.3f}$' % (median, median-np.percentile(samples, 5, axis = 0), np.percentile(samples, 95, axis = 0)-median))
    plt.axvline(median, ls = '--', c = 'k')
    plt.axvline( np.percentile(samples, 16, axis = 0), ls = '--', linewidth = 1, c = 'k')
    plt.axvline( np.percentile(samples, 84, axis = 0), ls = '--', linewidth = 1, c = 'k')
    plt.axvline( np.percentile(samples, 5, axis = 0), ls = ':', linewidth = 1, c = 'k')
    plt.axvline( np.percentile(samples, 95, axis = 0), ls = ':', linewidth = 1, c = 'k')
    if inj_val is not None:
        plt.axvline(inj_val, c = 'r')
    plt.xlabel(label)
    plt.ylabel('p('+label+')')
    plt.savefig(directory+'/'+label+'.pdf')

def plot_post(CPjob, label, out_dir = './', inj_val = None):

    pdf, samples = get_pdf(CPjob, label)
    plotting(pdf, samples, label, inj_val, out_dir)
