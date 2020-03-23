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
    return pdf

def find16(pdf):
    x = np.linspace(pdf.dataset.min(),pdf.dataset.max(),1000)
    for ext in x:
        if pdf.integrate_box(x[0], ext) > 0.16:
            return ext

def find84(pdf):
    x = np.linspace(pdf.dataset.min(),pdf.dataset.max(),1000)
    for ext in x:
        if pdf.integrate_box(x[0], ext) > 0.84:
            return ext

def median65(pdf):
    up = find84(pdf)
    down = find16(pdf)
    sigma = (up-down)/2.
    median = down+sigma
    return median, sigma

def plotting(pdf, label, directory = './'):

    median, sigma = median65(pdf)
    x  = np.linspace(pdf.dataset.min(),pdf.dataset.max(),1000)
    px = pdf(x)
    plt.figure()
    plt.plot(x, px)
    plt.suptitle(label+'=%.3f$\\pm$%.3f' %(median, sigma))
    plt.axvline(median, ls = '--', c = 'g')
    plt.axvline(find16(pdf), ls = '--', c = 'g')
    plt.axvline(find84(pdf), ls = '--', c = 'g')
    plt.xlabel(label)
    plt.ylabel('p('+label+')')
    plt.savefig(directory+'/'+label+'.pdf')

def plot_post(CPjob, label, out_dir = './'):

    pdf = get_pdf(CPjob, label)
    plotting(pdf, label, out_dir)
