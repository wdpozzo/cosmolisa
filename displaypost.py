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

def findpercent(pdf, percent):
    x = np.linspace(pdf.dataset.min(),pdf.dataset.max(),1000)
    for ext in x:
        if pdf.integrate_box(x[0], ext) > percent:
            return ext

def sigma(pdf):
    up = findpercent(pdf, 0.16)
    down = findpercent(pdf, 0.84)
    sigma = (up-down)/2.
    return sigma

def plotting(pdf, label, directory = './'):

    median = findpercent(pdf, 0.50)
    sigma  = sigma(pdf)
    # x  = np.linspace(pdf.dataset.min(),pdf.dataset.max(),1000)
    x  = np.linspace(median-4*sigma, median+4*sigma, 1000)
    px = pdf(x)
    plt.figure()
    plt.plot(x, px, c = 'g')
    plt.suptitle(label+'=%.3f$\\pm$%.3f' %(median, sigma))
    plt.axvline(median, ls = '--', c = 'g')
    plt.axvline(findpercent(pdf, 0.16), ls = '--', linewidth = 1, c = 'g')
    plt.axvline(findpercent(pdf, 0.84), ls = '--', linewidth = 1, c = 'g')
    plt.axvline(findpercent(pdf, 0.05), ls = ':', linewidth = 1, c = 'g')
    plt.axvline(findpercent(pdf, 0.95), ls = ':', linewidth = 1, c = 'g')
    plt.xlabel(label)
    plt.ylabel('p('+label+')')
    plt.savefig(directory+'/'+label+'.pdf')

def plot_post(CPjob, label, out_dir = './'):

    pdf = get_pdf(CPjob, label)
    plotting(pdf, label, out_dir)
