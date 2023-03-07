import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import h5py

COSMOLISA_PATH = os.getcwd()
sys.path.insert(1, os.path.join(COSMOLISA_PATH,'DPGMM'))
from dpgmm import *
sys.path.insert(1, COSMOLISA_PATH)

import multiprocessing as mp
from scipy.special import logsumexp
from scipy.stats import rv_discrete
from scipy.interpolate import interp1d

def FindHeightForLevel(inArr, adLevels):
    # flatten the array
    oldshape = np.shape(inArr)
    adInput = np.reshape(inArr, oldshape[0]*oldshape[1])
    # GET ARRAY SPECIFICS
    nLength = np.size(adInput)

    # CREATE REVERSED SORTED LIST
    adTemp = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted

    # CREATE NORMALISED CUMULATIVE DISTRIBUTION
    adCum = np.zeros(nLength)
    adCum[0] = adSorted[0]
    for i in range(1, nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
    adCum = adCum - adCum[-1]

    # FIND VALUE CLOSEST TO LEVELS
    adHeights = []
    for item in adLevels:
        idx=(np.abs(adCum - np.log(item))).argmin()
        adHeights.append(adSorted[idx])

    adHeights = np.array(adHeights)

    return adHeights

def initialise_dpgmm(dims, posterior_samples):
    model = DPGMM(dims)
    for point in posterior_samples:
        model.add(point)

    model.setPrior()
    model.setThreshold(1e-4)
    model.setConcGamma(1.0,1.0)
    return model

def compute_dpgmm(model,max_sticks=16):
    solve_args = [(nc, model) for nc in range(1, max_sticks+1)]
    solve_results = pool.map(solve_dpgmm, solve_args)
    scores = np.array([r[1] for r in solve_results])
    model = (solve_results[scores.argmax()][-1])
    print("best model has ", scores.argmax()+1, "components")
    return model.intMixture()

def evaluate_grid(density,x,y):
    sys.stderr.write("computing log posterior for "
                     "{} grid points\n".format(len(x)*len(y)))
    sample_args = ((density,xi,yi) for xi in x for yi in y)
    results = pool.map(sample_dpgmm, sample_args)
    return np.array([r for r in results]).reshape(len(x),len(y))

def sample_dpgmm(args):
    (dpgmm,x,y) = args
    logPs = [prob.logProb([x,y]) for ind, prob in enumerate(dpgmm[1])]
    return logsumexp(logPs, b=dpgmm[0])

def solve_dpgmm(args):
    (nc, model) = args
    for _ in range(nc-1): model.incStickCap()
    try:
        it = model.solve(iterCap=1024)
        return (model.stickCap, model.nllData(), model)
    except:
        return (model.stickCap, -np.inf, model)

def marginalise(pdf, dx, axis):
    return np.sum(pdf*dx, axis=axis)

def renormalise(pdf, dx):
    return pdf / (pdf*dx).sum()

def par_dic(a1, a2):
    return {'p1': a1, 'p2': a2}

# Used to sample the distribution 
def rejection_sampling(ndraws, grid, interp, interp_min, interp_max):
    samples = []
    naccept, ntrial = 0, 0
    while naccept < ndraws:
        p = np.random.uniform(grid[0], grid[-1])
        q = np.random.uniform(interp_min, interp_max)
        if q < interp(p):
            samples.append(p)
            naccept = naccept + 1
        ntrial = ntrial + 1
    return samples



def merge_split_catalogs(realisations=None, base_path=None, 
                            runtag=None, chunks=None, 
                            chunk_realis_name=None, 
                            model='LambdaCDM', Nbins=1024):

    fmt = "{{0:{0}}}".format('.3f').format
    colors_dict = {
        "chunk1_samp": 'green', 
        "chunk2_samp": 'blueviolet', 
        "chunk3_samp": 'red', 
        "chunk4_samp": 'orange',
        "chunk5_samp": 'lightpink',
        "chunk6_samp": 'maroon',
        "chunk7_samp": 'gray',
        "chunk8_samp": 'cyan',
        "combined": 'black'
                }
    # Create folder where to create sub-directores for each realisation
    # the chunks of each realisation will be merged and saved in these
    # sub-directories.
    # Create parent directory which contains all merged realisations
    joint_folder = base_path
    # os.system("mkdir -p {}".format(joint_folder))

    # Read the different parts into which each realisation has been divided.
    # Do it for each realisation.
    for k in realisations:
        # Create sub-directory for each realisation 
        out_folder = os.path.join(joint_folder, f"Realisation_Seed7_{k}_{chunks}_split_data_merged")
        os.system("mkdir -p {}".format(out_folder))

        print(f"Reading data chunks of realisation {k}")

        chunk_paths = {f"chunk{i}_samp": os.path.join(base_path, 
                    f"Realisation_Seed7_{k}_{chunks}_{i}", 
                    # chunk_realis_name.format(k)+f"_split_data_chunk_{i}"
                    )
                    for i in range(1, chunks+1)}

        # Collect posterior samples from each chunk (different events)
        # to be merged later.
        all_events = {}
        for path_name, path in chunk_paths.items():                      
            filename = os.path.join(path, "CPNest", "cpnest.h5")
            print("\nReading posterior samples "+ 
                "stored in {}".format(filename))
            h5_file = h5py.File(filename, 'r')
            posteriors = h5_file['combined'].get('posterior_samples')
            if model == "LambdaCDM":
                p1 = posteriors['h']
                p2 = posteriors['om']
                all_events[f"{path_name}"] = dict(p1=p1, p2=p2)
            elif model == "DE":
                p1 = posteriors['w0']
                p2 = posteriors['w1']
                all_events[f"{path_name}"] = dict(p1=p1, p2=p2)

        joint_posterior = np.zeros((Nbins, Nbins), dtype=np.float64)

        # Plot - Compute single and joint posteriors
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(111)
        ax.grid(alpha = 0.5, linestyle = 'dotted')

        if model == "LambdaCDM":
            x_flat = np.linspace(0.6, 0.86, Nbins)
            y_flat = np.linspace(0.04, 0.5, Nbins)
        elif model == "DE":
            x_flat = np.linspace(-1.5, -0.3, Nbins)
            y_flat = np.linspace(-1.0, 1.0, Nbins)
        else:
            sys.exit("DPGMM only accepts 2D models (LambdaCDM, DE). "
                    "Exiting.")

        dx = np.diff(x_flat)[0]
        dy = np.diff(y_flat)[0]
        X, Y = np.meshgrid(x_flat, y_flat)

        # Print statistics and prepare samples to be combined
        file_path = os.path.join(out_folder, "quantiles.txt")
        sys.stdout = open(file_path, "w+")
        print("{} {}".format(model, 'Realisation_{}'.format(k)))
        print("Will save .5, .16, .50, .84, .95 quantiles in {}".format(
            out_folder))

        hhs, llgs = [], []
        for chunk, chunk_dict in all_events.items():
            p1 = chunk_dict['p1']#[::10]
            p2 = chunk_dict['p2']#[::10]
            # Compute and save .05, .16, .5, .84, .95 quantiles for each chunk.
            if model == "LambdaCDM":
                p1_name, p2_name = 'h', 'om'
                p1_name_string, p2_name_string = r"$h$", r"$\Omega_m$"
            elif model == "DE":
                p1_name, p2_name = 'w0', 'wa'
                p1_name_string, p2_name_string = r"$w_0$", r"$w_a$"

            print("\n\n"+chunk)

            p1_ll, p1_l, p1_median, p1_h, p1_hh = np.percentile(
                p1, [5.0, 16.0, 50.0, 84.0, 95.0], axis=0)
            p2_ll, p2_l, p2_median, p2_h, p2_hh = np.percentile(
                p2, [5.0, 16.0, 50.0, 84.0, 95.0], axis=0)

            p1_inf_68, p1_sup_68 = p1_median - p1_l,  p1_h - p1_median
            p1_inf_90, p1_sup_90 = p1_median - p1_ll, p1_hh - p1_median
            p2_inf_68, p2_sup_68 = p2_median - p2_l,  p2_h - p2_median
            p2_inf_90, p2_sup_90 = p2_median - p2_ll, p2_hh - p2_median

            p1_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(
                p1_inf_68, p1_median, p1_sup_68)
            p1_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(
                p1_inf_90, p1_median, p1_sup_90)
            p2_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(
                p2_inf_68, p2_median, p2_sup_68)
            p2_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(
                p2_inf_90, p2_median, p2_sup_90)

            print("{} 68% CI:".format(p1_name))
            print(p1_credible68)
            print("{} 90% CI:".format(p1_name))
            print(p1_credible90)
            print("\n{} 68% CI:".format(p2_name))
            print(p2_credible68)
            print("{} 90% CI:".format(p2_name))
            print(p2_credible90, "\n")

            model_dp = initialise_dpgmm(2, np.column_stack((p1,p2)))
            logdensity = compute_dpgmm(model_dp, max_sticks=4)
            single_posterior = evaluate_grid(logdensity, x_flat, y_flat)
            joint_posterior += single_posterior
            levs = np.sort(FindHeightForLevel(single_posterior.T, [0.0,0.68,0.90]))
            # C = ax.contourf(X, Y, single_posterior.T, levs[:-1], colors=colors_dict[chunk], zorder=5, linestyles='solid', alpha=0.1) # fill regions 68-90
            # C = ax.contourf(X, Y, single_posterior.T, levs[1:], colors=colors_dict[chunk], zorder=5, linestyles='solid', alpha=0.3) # fill regions 0-68
            # C = ax.contour(X, Y, single_posterior.T, levs[:-1], linewidths=1., colors=colors_dict[chunk], zorder=6, linestyles='solid') # show contours 68 and 90
            C = ax.contour(X, Y, single_posterior.T, levs[:-2], linewidths=1.5, colors=colors_dict[chunk], zorder=6, linestyles='solid') # show contours 90
            h,_ = C.legend_elements()
            hhs.append(h[0])
            llgs.append("{}".format(chunk))

        levs = np.sort(FindHeightForLevel(joint_posterior.T,[0.0,0.68,0.90]))
        C = ax.contourf(X, Y, joint_posterior.T, levs[:-1], colors='whitesmoke', zorder=10, linestyles='solid', alpha=0.85)
        C = ax.contourf(X, Y, joint_posterior.T, levs[1:], colors=colors_dict['combined'], zorder=10, linestyles='solid', alpha=0.3)
        C = ax.contour(X, Y, joint_posterior.T, levs[:-1], linewidths=2.0, colors=colors_dict['combined'], zorder=11, linestyles='solid')
        hh,_ = C.legend_elements()
        hhs.append(hh[0])
        llgs.append("{}".format('combined'))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend([x_h for x_h in hhs], [lg for lg in llgs], fontsize=18)

        if model == "LambdaCDM":
            ax.axvline(truths['h'], color='k', linestyle='dashed', lw=0.5)
            ax.axhline(truths['om'], color='k', linestyle='dashed', lw=0.5)
            ax.set_xlabel(p1_name_string, fontsize=18)
            ax.set_ylabel(p2_name_string, fontsize=18)
            # if cosmo3G:
            xlimits = [0.6, 0.75]
            # else:
                # xlimits = [0.65, 0.8]
            ylimits = [0.04, 0.50]
            plt.xlim(xlimits[0], xlimits[1])
            plt.ylim(ylimits[0], ylimits[1])

        elif model == "DE":
            ax.axvline(truths['w0'], color='k', linestyle='dashed', lw=0.5)
            ax.axhline(truths['wa'], color='k', linestyle='dashed', lw=0.5)
            ax.set_xlabel(p1_name_string, fontsize=18)
            ax.set_ylabel(p2_name_string, fontsize=18)
            xlimits = [-1.3, -0.7]
            ylimits = [-1., 1.]
        plt.savefig(os.path.join(out_folder, "jp_combined.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(out_folder, "jp_combined.png"), bbox_inches='tight')

        # Plot - Level contours of joint distribution
        fig_j = plt.figure(figsize=(10,8))
        ax_j = plt.axes()
        hs, lgs = [], []
        levs_long = np.concatenate(([0.05], [k for k in np.arange(0.1, 1.0, 0.1)]))
        colors_long = np.flip(['k', 'darkgrey', 'crimson', 'darkorange', 
                            'gold', 'limegreen', 'darkturquoise', 
                            'royalblue', 'mediumorchid', 'magenta'])
        for l, c in zip(levs_long, colors_long):
            cntr = ax_j.contour(X, Y, joint_posterior.T, 
                levels=np.sort(FindHeightForLevel(joint_posterior.T, [l])),
                colors=c, linewidths=1.2)
            h,_ = cntr.legend_elements()
            hs.append(h[0])
            lgs.append(r'${0} \% \, CR$'.format(int(l*100.)))
        if model == "LambdaCDM":
            ax_j.axvline(truths['h'], color='k', linestyle='dashed', lw=0.5)
            ax_j.axhline(truths['om'], color='k', linestyle='dashed', lw=0.5)
            ax_j.set_xlabel(p1_name_string, fontsize=18)
            ax_j.set_ylabel(p2_name_string, fontsize=18)
            leg_loc = 'upper right'
        elif model == "DE":
            ax_j.axvline(truths['w0'], color='k', linestyle='dashed', lw=0.5)
            ax_j.axhline(truths['wa'], color='k', linestyle='dashed', lw=0.5)
            ax_j.set_xlabel(p1_name_string, fontsize=18)
            ax_j.set_ylabel(p2_name_string, fontsize=18)
            leg_loc = 'lower left'
        plt.legend([x_h for x_h in hs], [lg for lg in lgs], loc=leg_loc, fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(os.path.join(out_folder,
                                "jp_contour_levels.pdf"),
                                bbox_inches='tight')

        # Compute marginalised 1D PDFs statistics
        pdf_p1 = renormalise(marginalise(np.exp(joint_posterior.T), dy, axis=0), dx)
        pdf_p2 = renormalise(marginalise(np.exp(joint_posterior.T), dx, axis=1), dy)
        print("\n1D PDF normalisation {}: ".format(p1_name), marginalise(pdf_p1, dx, axis=0))
        print("1D PDF normalisation {}: ".format(p2_name), marginalise(pdf_p2, dy, axis=0))

        custm_p1 = rv_discrete(name='cutsm', values=(x_flat, pdf_p1 * dx))
        custm_p2 = rv_discrete(name='cutsm', values=(y_flat, pdf_p2 * dy))

        custm_p1_ll, custm_p1_l, custm_p1_median, custm_p1_h, custm_p1_hh = custm_p1.interval(.90)[0], custm_p1.interval(.68)[0], custm_p1.median(), custm_p1.interval(.68)[1], custm_p1.interval(.90)[1]
        custm_p2_ll, custm_p2_l, custm_p2_median, custm_p2_h, custm_p2_hh = custm_p2.interval(.90)[0], custm_p2.interval(.68)[0], custm_p2.median(), custm_p2.interval(.68)[1], custm_p2.interval(.90)[1]

        custm_p1_inf_68, custm_p1_sup_68 = custm_p1_median - custm_p1_l, custm_p1_h - custm_p1_median
        custm_p1_inf_90, custm_p1_sup_90 = custm_p1_median - custm_p1_ll, custm_p1_hh - custm_p1_median
        custm_p2_inf_68, custm_p2_sup_68 = custm_p2_median - custm_p2_l, custm_p2_h - custm_p2_median
        custm_p2_inf_90, custm_p2_sup_90 = custm_p2_median - custm_p2_ll, custm_p2_hh - custm_p2_median

        custm_p1_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p1_inf_68, custm_p1_median, custm_p1_sup_68)
        custm_p1_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p1_inf_90, custm_p1_median, custm_p1_sup_90)
        custm_p2_credible68 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p2_inf_68, custm_p2_median, custm_p2_sup_68)
        custm_p2_credible90 = '[-{:.3f}, {:.3f}, +{:.3f}]'.format(custm_p2_inf_90, custm_p2_median, custm_p2_sup_90)

        print("\n\njoint posterior")
        print("{} 68% CI:".format(p1_name))
        print(custm_p1_credible68)
        print("{} 90% CI:".format(p1_name))
        print(custm_p1_credible90)
        print("\n{} 68% CI:".format(p2_name))
        print(custm_p2_credible68)
        print("{} 90% CI:".format(p2_name))
        print(custm_p2_credible90)

        # Plot - 1D PDFs
        plt.figure(figsize=(8,8))
        plt.plot(x_flat, pdf_p1, c='k', linewidth=2.0)
        plt.axvline(custm_p1_median, linestyle='--', color='k', zorder=-1)
        plt.axvline(custm_p1_l, linestyle='--', color='k', zorder=-1)
        plt.axvline(custm_p1_h, linestyle='--', color='k', zorder=-1)
        plt.axvline(truths[p1_name], color='dodgerblue', linestyle='-', zorder=-1)
        plt.xlabel(p1_name_string, fontsize=18)
        plt.ylabel(r"$PDF$", fontsize=18)
        plt.xlim(xlimits[0], xlimits[1])
        plt.ylim(bottom=0.0)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        par_n = 'h' if model == 'LambdaCDM' else 'w_0' 
        plt.title(r'${0} = {{{1}}}_{{-{2}}}^{{+{3}}} \,\, (68 \% \, CI)$'.format(par_n, fmt(custm_p1_median), fmt(custm_p1_inf_68), fmt(custm_p1_sup_68)), fontsize=18)
        plt.savefig(os.path.join(out_folder,'{}.png'.format(p1_name)), bbox_inches='tight')

        plt.figure(figsize=(8,8))
        plt.plot(y_flat, pdf_p2, c='k', linewidth=2.0)
        plt.axvline(custm_p2_median, linestyle='--', color='k', zorder=-1)
        plt.axvline(custm_p2_l, linestyle='--', color='k', zorder=-1)
        plt.axvline(custm_p2_h, linestyle='--', color='k', zorder=-1)
        plt.axvline(truths[p2_name], color='dodgerblue', linestyle='-', zorder=-1)
        plt.xlabel(p2_name_string, fontsize=18)
        plt.ylabel(r"$PDF$", fontsize=18)
        plt.xlim(ylimits[0], ylimits[1])
        plt.ylim(bottom=0.0)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        par_n = '\Omega_m' if model == 'LambdaCDM' else 'w_a'
        plt.title(r'${0} = {{{1}}}_{{-{2}}}^{{+{3}}} \, (68 \% \, CI)$'.format(par_n, fmt(custm_p2_median), fmt(custm_p2_inf_68), fmt(custm_p2_sup_68)), fontsize=18)
        plt.savefig(os.path.join(out_folder,'{}.png'.format(p2_name)), bbox_inches='tight')

        plt.close()

        # Sample the 1D PDFs with rejection sampling
        interp_p1 = interp1d(x_flat, pdf_p1, kind='cubic')
        interp_p2 = interp1d(y_flat, pdf_p2, kind='cubic')

        interp_p1_values = interp_p1(x_flat)
        interp_p2_values = interp_p2(y_flat)

        interp_p1_min = interp_p1_values.min()
        interp_p1_max = interp_p1_values.max()
        interp_p2_min = interp_p2_values.min()
        interp_p2_max = interp_p2_values.max()

        p1_sampled = rejection_sampling(ndraws=15000, 
                                        grid=x_flat, 
                                        interp=interp_p1, 
                                        interp_min=interp_p1_min, 
                                        interp_max=interp_p1_max)

        p2_sampled = rejection_sampling(ndraws=15000, 
                                        grid=y_flat, 
                                        interp=interp_p2, 
                                        interp_min=interp_p2_min, 
                                        interp_max=interp_p2_max)

        np.savetxt(os.path.join(out_folder,'samples.dat'), 
                np.column_stack((p1_sampled, p2_sampled)), 
                header='{}\t{}'.format(p1_name, p2_name))

        plt.figure(figsize=(8,8))
        plt.plot(x_flat, pdf_p1, c='k', linewidth=2.0)
        plt.plot(x_flat, interp_p1(x_flat), '--', c='red')
        plt.hist(p1_sampled, bins=40, density=True, label="p1 drawn")
        plt.legend()
        plt.savefig(os.path.join(out_folder,'{}_samples.png'.format(p1_name)),
                    bbox_inches='tight')

        plt.figure(figsize=(8,8))
        plt.plot(y_flat, pdf_p2, c='k', linewidth=2.0)
        plt.plot(y_flat, interp_p2(y_flat), '--', c="red")
        plt.hist(p2_sampled, bins=40, density=True, label="p2 drawn")
        plt.legend()
        plt.savefig(os.path.join(out_folder,'{}_samples.png'.format(p2_name)),
                    bbox_inches='tight')

        plt.close()


if __name__=="__main__":

    truths = {'h': 0.673, 'om': 0.315, 'ol': 0.685}

    print(mp.cpu_count())

    pool = mp.Pool(mp.cpu_count()) #mp.cpu_count()

    # /sps/lisaf/cliu/results/EMRI_M1/3-sigma_zcut10_WKLS_SNR100/Error_boxes_LambdaCDM/Realisation_1_3_1
    # /sps/lisaf/cliu/results/EMRI_M1/3-sigma_zcut10_WKLS_SNR50/Error_boxes_LambdaCDM/Realisation_Seed7_1_6_1

    base_path = '/sps/lisaf/cliu/results/EMRI_M1/3-sigma_zcut10_WKLS_SNR100/Error_boxes_LambdaCDM'
    model = 'LambdaCDM'
    runtag = ''
    realisations = ["1","2","3","4","5","6","7","8","9","10"]  #,
    chunks = 2
    chunk_realis_name = ''
    nBins = 1024

    merge_split_catalogs(realisations=realisations,
                        base_path=base_path,
                        runtag=runtag,
                        chunks=chunks,
                        chunk_realis_name=chunk_realis_name,
                        model=model,
                        Nbins=nBins)