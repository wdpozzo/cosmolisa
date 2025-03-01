#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import time
import configparser
import subprocess
import numpy as np
import json
from optparse import OptionParser
from configparser import ConfigParser

# Import internal and external modules.
from cosmolisa import readdata
from cosmolisa import plots
from cosmolisa import cosmology as cs
from cosmolisa import likelihood as lk
# import nessai
from nessai.model import Model
from nessai.flowsampler import FlowSampler


class CosmologicalModel(Model):
    """CosmologicalModel class:
    Data, likelihood, prior, and settings of the analysis
    are specified here. The abstract modules 'log_prior' and
    'log_likelihood', as well as the attributes 'names' and
    'bounds', are inherited from nessai.model.Model and
    have to be explicitly defined inside this class.
    """

    def __init__(self, model, data, *args, **kwargs):

        self.data = data
        self.N = len(self.data)
        self.event_class = kwargs['event_class']
        self.model_str = model
        self.model = model.split("+")
        self.truths = kwargs['truths']
        self.z_threshold = kwargs['z_threshold']
        self.snr_threshold = kwargs['snr_threshold']
        self.com_vol = kwargs['com_vol']
        self.O = None

        self.gw = 0
        self.names_list = []
        self.bounds_dict = dict()

        if ('h' in self.model):
            self.names_list.append('h')
            self.bounds_dict['h'] = kwargs['prior_bounds']['h']

        if ('om' in self.model):
            self.names_list.append('om')
            self.bounds_dict['om'] = kwargs['prior_bounds']['om']

        if ('ol' in self.model):
            self.names_list.append('ol')
            self.bounds_dict['ol'] = kwargs['prior_bounds']['ol']

        if ('w0' in self.model):
            self.names_list.append('w0')
            self.bounds_dict['w0'] = kwargs['prior_bounds']['w0']

        if ('w1' in self.model):
            self.names_list.append('w1')
            self.bounds_dict['w1'] = kwargs['prior_bounds']['w1']

        if ('Xi0' in self.model):
            self.names_list.append('Xi0')
            self.bounds_dict['Xi0'] = kwargs['prior_bounds']['Xi0']

        if ('n1' in self.model):
            self.names_list.append('n1')
            self.bounds_dict['n1'] = kwargs['prior_bounds']['n1']

        if ('b' in self.model):
            self.names_list.append('b')
            self.bounds_dict['b'] = kwargs['prior_bounds']['b']

        if ('n2' in self.model):
            self.names_list.append('n2')
            self.bounds_dict['n2'] = kwargs['prior_bounds']['n2']
        # Some consistency checks.
        for par in self.names_list:
            assert kwargs['prior_bounds'][par][0] <= self.truths[par], (
             f"{par}: your lower prior bound excludes the true value!")
            assert kwargs['prior_bounds'][par][1] >= self.truths[par], (
             f"{par}: your upper prior bound excludes the true value!")
        if 'Xi0' in self.names_list or 'n1' in self.names_list:
            if 'b' in self.names_list or 'n2' in self.names_list:
                print("The chosen beyondGR parameters are not consistent. "
                      "Exiting.")
                exit() 

        self.names = self.names_list
        self.bounds = self.bounds_dict

        if ('GW' in self.model):
            self.gw = 1
        else:
            self.gw = 0
        

        assert len(self.names) != 0, ("Undefined parameter space!"
        "Please check that the model exists.")

        # If we are using GWs, add the relevant redshift parameters.
        if (self.gw == 1):
            pass
        else:
            self.gw_redshifts = np.array([e.z_true for e in self.data])
        
        self._initialise_galaxy_hosts()
            

        print("\n"+5*"===================="+"\n")
        print("CosmologicalModel model initialised with:")
        print(f"Event class: {self.event_class}")
        print(f"Analysis model: {self.model}")
        print(f"Number of events: {len(self.data)}")
        print(f"Free parameters: {self.names}")
        print("\n"+5*"===================="+"\n")
        print("Prior bounds:")
        for name in self.names:
            print(f"{str(name).ljust(17)}: {self.bounds[name]}")
        print("\n"+5*"===================="+"\n")

    def _initialise_galaxy_hosts(self):
        self.hosts = {
            e.ID: np.array([(g.redshift, g.dredshift, g.weight, g.magnitude)
            for g in e.potential_galaxy_hosts]) for e in self.data
            }
        self.galaxy_redshifts = np.hstack([self.hosts[e.ID][:,0] 
            for e in self.data]).copy(order='C')
        self.galaxy_magnitudes = np.hstack([self.hosts[e.ID][:,3] 
            for e in self.data]).copy(order='C')
        self.areas = {
            e.ID: 0.000405736691211125 * (87./e.snr)**2 for e in self.data
            }
        
    def log_prior(self, x):
        """
        Returns natural-log of prior given a live point assuming
        uniform priors on each parameter.        
        """
        logP = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            logP -= np.log(self.bounds[n][1] - self.bounds[n][0])

        return logP

    def log_likelihood(self, x):
        """Natural-log-likelihood assumed in the inference.
        It implements the inference model settings according
        to the options specified by the user.
        """
        logL_GW = np.zeros(x.size)
        cosmo_par = [self.truths['h'], self.truths['om'],
                     self.truths['ol'], self.truths['w0'],
                     self.truths['w1'], self.truths['Xi0'],
                     self.truths['n1'], self.truths['b'],
                     self.truths['n2']]
        if ('h' in self.model):
            cosmo_par[0] = x['h']
        if ('om' in self.model):
            cosmo_par[1:3] = x['om'], 1.0 - x['om']
        if ('ol' in self.model):
            cosmo_par[2] = x['ol']
        if ('w0' in self.model):
            cosmo_par[3] = x['w0']
        if ('w1' in self.model):
            cosmo_par[4] = x['w1']
        if ('Xi0' in self.model):
            cosmo_par[5] = x['Xi0']
        if ('n1' in self.model):
            cosmo_par[6] = x['n1']
        if ('b' in self.model):
            cosmo_par[7] = x['b']                
        if ('n2' in self.model):
            cosmo_par[8] = x['n2']                
        else:
            pass                
        self.O = cs.CosmologicalParameters(*cosmo_par)

        if (self.event_class == 'dark_siren'):
            logL_GW += np.sum([np.log(
                    lk.lk_dark_single_event_trap(
                    self.hosts[e.ID], e.dl, e.sigmadl, self.O,
                    self.model_str, zmin=e.zmin, zmax=e.zmax,
                    com_vol=self.com_vol))
                    for _, e in enumerate(self.data)])
        elif (self.event_class == 'MBHB'):
            logL_GW += np.sum([np.log(
                    lk.lk_bright_single_event_trap(
                    self.hosts[e.ID], e.dl, e.sigmadl, self.O,
                    self.model_str, zmin=e.zmin, zmax=e.zmax))
                    for _, e in enumerate(self.data)])
                    
        # IMPROVEME
        # Same results are obtained without destroying self.O.
        self.O.DestroyCosmologicalParameters()

        return logL_GW

usage="""\n\n %prog --config-file config.ini\n
    ######################################################################################################################################################
    IMPORTANT: This code requires the installation of the 'nessai' package: https://github.com/mj-will/nessai
               See the instructions in cosmolisa/README.md.
    ######################################################################################################################################################

    #=======================#
    # Input parameters      #
    #=======================#

    'data'                 Default: ''.                                      Data location.
    'outdir'               Default: './default_dir'.                         Directory for output.
    'event_class'          Default: ''.                                      Class of the event(s) ['dark_siren', 'MBHB'].
    'model'                Default: ''.                                      Specify the cosmological parameters to sample over ['h', 'om', 'ol', 'w0', 'wa', 'Xi0', 'n1', 'b', 'n2'] and the type of analysis ['GW'] separated by a '+'.
    'truths'               Default: {"h": 0.673, "om": 0.315, "ol": 0.685}.  Cosmology truths values. If not specified, default values are used.
    'prior_bounds'         Default: {"h": [0.6, 0.86], "om": [0.04, 0.5]}.   Prior bounds specified by the user. Must contain all the parameters specified in 'model'.
    'random'               Default: 0.                                       Run a joint analysis with N events, randomly selected.
    'zhorizon'             Default: '1000.0'.                                Impose low-high cutoffs in redshift. It can be a single number (upper limit) or a string with z_min and z_max separated by a comma.
    'z_event_sel'          Default: 0.                                       Select N events ordered by redshift. If positive (negative), choose the X nearest (farthest) events.
    'one_host_sel'         Default: 0.                                       For each event, associate only the nearest-in-redshift host (between z_gal and event z_true).
    'single_z_from_GW'     Default: 0.                                       Impose a single host for each GW having redshift equal to z_true. It works only if one_host_sel = 1.
    'equal_wj'             Default: 0.                                       Impose all galaxy angular weights equal to 1.
    'event_ID_list'        Default: ''.                                      String of specific ID events to be read (separated by commas and without single/double quotation marks).
    'max_hosts'            Default: 0.                                       Select events according to the allowed maximum number of hosts.
    'z_gal_cosmo'          Default: 0.                                       If set to 1, read and use the cosmological redshift of the galaxies instead of the observed one.
    'snr_selection'        Default: 0.                                       Select in SNR the N loudest/faintest (N<0/N>0) events, where N=snr_selection.
    'snr_threshold'        Default: 0.0.                                     Impose an SNR detection threshold X>0 (X<0) and select the events above (belove) X.
    'sigma_pv'             Default: 0.0023.                                  Uncertainty associated to peculiar velocity value, equal to (vp / c), used in the computation of the GW redshift uncertainty (0.0015 in https://arxiv.org/abs/1703.01300).
    'split_data_num'       Default: 1.                                       Choose the number of parts into which to divide the list of events. Values: any integer number equal or greater than 2.
    'split_data_chunk'     Default: 0.                                       Choose which chunk of events to analyse. Only works if split_data_num > 1. Values: 1 up to split_data_num.
    'reduced_catalog'      Default: 0.                                       Select randomly only a fraction of the catalog (4 yrs of observation, hardcoded).
    'postprocess'          Default: 0.                                       Run only the postprocessing. It works only with reduced_catalog=0.
    'screen_output'        Default: 0.                                       Print the output on screen or save it into a file.
    'nlive'                Default: 1000.                                    Number of live points.
    'seed'                 Default: 0.                                       Random seed initialisation.
    'pytorch_threads'      Default: 1.                                       Number of threads that pytorch can use.
    'n_pool'               Default: None.                                    Threads for evaluating the likelihood.
    'checkpoint_int'       Default: 21600.                                   Time interval between sampler periodic checkpoint in seconds. Defaut: 21600 (6h).

"""

def main():
    """Main function to be called when cosmoLISA is executed."""
    run_time = time.perf_counter()
    parser = OptionParser(usage)
    parser.add_option('--config-file', type='string', metavar='config_file',
                      default=None)

    (opts,args) = parser.parse_args()
    config_file = opts.config_file

    if not(config_file):
        parser.print_help()
        parser.error("Please specify a config file.")
    if not(os.path.exists(config_file)):
        parser.error("Config file {} not found.".format(config_file))
    Config = configparser.ConfigParser()
    Config.read(config_file)

    config_par = {
        'data': '',
        'outdir': "./default_dir",
        'event_class': '',
        'model': '',
        'truth_par': {"h": 0.673, "om": 0.315, "ol": 0.685},
        'prior_bounds': {"h": [0.6, 0.86], "om": [0.04, 0.5]},
        'com_vol': 0,
        'random': 0,
        'zhorizon': "1000.0",
        'rel_LISAsigmadl': 0.0,
        'dl_scat': 0,
        'z_event_sel': 0,
        'one_host_sel': 0,
        'single_z_from_GW': 0,
        'equal_wj': 0,
        'gals_dVdz': 0,
        'gals_uniform': 0,
        'event_ID_list': '',
        'snr_range': '',
        'max_hosts': 0,
        'z_gal_cosmo': 0,
        'snr_selection': 0,
        'snr_threshold': 0.0,
        'sigma_pv': 0.0023,
        'split_data_num': 1,
        'split_data_chunk': 0,
        'reduced_catalog': 0,
        'postprocess': 0,
        'screen_output': 0,    
        'nlive': 1000,
        'seed': 1234,
        'pytorch_threads': 1,
        'n_pool': 1,
        'checkpoint_int': 10800,
        }

    for key in config_par:
        keytype = type(config_par[key])
        try: 
            if ('truth_par' in key) or ('prior_bounds' in key):
                config_par[key] = json.loads(
                    Config.get('input parameters', '{}'.format(key)))
            else:
                config_par[key] = keytype(Config.get('input parameters',
                                                     key))
        except (KeyError, configparser.NoOptionError, TypeError):
            pass

    try:
        outdir = str(config_par['outdir'])
    except(KeyError, ValueError):
        outdir = "default_dir"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    os.system("mkdir -p {}/nessai".format(outdir))
    os.system("mkdir -p {}/Plots".format(outdir))
    #FIXME: avoid cp command when reading the config file from the 
    # outdir directory to avoid the 'same file' cp error
    os.system("cp {} {}/.".format(opts.config_file, outdir))
    output_sampler = os.path.join(outdir, "nessai")

    if not(config_par['screen_output']):
        if not(config_par['postprocess']):
            sys.stdout = open(os.path.join(outdir, "stdout.txt"), 'w')
            sys.stderr = open(os.path.join(outdir, "stderr.txt"), 'w')

    formatting_string = 6*"===================="
    max_len_keyword = len('split_data_chunk')

    print("\n"+formatting_string)
    print("\n"+"Running cosmoLISA")
    # FIXME: The code doesn't like the following line:
    # NameError: name 'nessai' is not defined
    # print(f"nessai installation version: {nessai.__version__}")
    print(f"cosmolisa likelihood version: {lk.__file__}")
    print("\n"+formatting_string)

    print((f"\nReading config file: {config_file}\n"))
    for key in config_par:
        print(("{name} : {value}".format(name=key.ljust(max_len_keyword),
                                         value=config_par[key])))

    truths = {
        'h': 0.673,
        'om': 0.315,
        'ol': 0.685,
        'w0': -1.0,
        'w1': 0.0,
        'Xi0': 1.0,
        'n1': 1.5,
        'b': 0.0,
        'n2': 1.0,
        }

    for par in truths.keys():
        if par in config_par['truth_par'].keys():
            truths[par] = config_par['truth_par'][par]

    print("\n"+formatting_string+"\nTruths:")
    for key in truths:
        print(("{name} : {value}".format(name=key.ljust(max_len_keyword),
                                         value=truths[key])))
    print(formatting_string+"\n")

    omega_true = cs.CosmologicalParameters(truths['h'], truths['om'],
                                           truths['ol'], truths['w0'],
                                           truths['w1'], truths['Xi0'],
                                           truths['n1'], truths['b'],
                                           truths['n2'])


    ###################################################################
    ### Reading the catalog according to the user's options.
    ###################################################################

    # Choose between dark or bright siren analysis
    if (config_par['event_class'] == "dark_siren"):
        events = readdata.read_dark_siren_event(
            config_par['data'],
            max_hosts=config_par['max_hosts'],
            z_event_sel=config_par['z_event_sel'],
            snr_selection=config_par['snr_selection'],
            sigma_pv=config_par['sigma_pv'],
            zhorizon=config_par['zhorizon'],
            one_host_selection=config_par['one_host_sel'],
            z_gal_cosmo=config_par['z_gal_cosmo'],
            dl_scat=config_par['dl_scat'],
            rel_LISAsigmadl=config_par['rel_LISAsigmadl'],
            event_ID_list=config_par['event_ID_list'],
            snr_range=config_par['snr_range'],
            snr_threshold=config_par['snr_threshold'],
            reduced_cat=config_par['reduced_catalog'],
            single_z_from_GW=config_par['single_z_from_GW'],
            equal_wj=config_par['equal_wj'],
            gals_dVdz=config_par['gals_dVdz'],
            gals_uniform=config_par['gals_uniform'],
            omega_true=omega_true)
    elif (config_par['event_class'] == "MBHB"):
        events = readdata.read_MBHB_event(config_par['data'])
    else:
        print(f"Unknown event_class '{config_par['event_class']}'."
              " Exiting.\n")
        exit()

    if (len(events) == 0):
        print("The passed catalog is empty. Exiting.\n")
        exit()

    if (config_par['random'] != 0):
        events = readdata.pick_random_events(events, config_par['random'])

    ###################################################################
    ### Modifying the event properties according to the user's options.
    ###################################################################

    if not (config_par['split_data_num'] <= 1):
        assert \
            config_par['split_data_chunk'] <= config_par['split_data_num'], \
            "Data split in {} chunks; chunk number {} has been chosen".format(
                config_par['split_data_num'], config_par['split_data_chunk'])
        events = sorted(events, key=lambda x: getattr(x, 'ID'))
        q, r = divmod(len(events), config_par['split_data_num'])
        split_events = list([events[i*q + min(i, r):(i+1)*q + min(i+1, r)] 
                             for i in range(config_par['split_data_num'])])
        print(f"\nInitial list of {len(events)} events split into"
              f" {len(split_events)} chunks." 
              f"\nChunk number {config_par['split_data_chunk']} is chosen.")
        events = split_events[config_par['split_data_chunk']-1]


    print(f"\nDetailed list of the {len(events)} selected event(s):")
    print("\n"+formatting_string)
    events = sorted(events, key=lambda x: getattr(x, 'ID'))
    for e in events:
        print("ID: {}  |  ".format(str(e.ID).ljust(3))
                +"SNR: {} |  ".format(str(e.snr).ljust(9))
                +"z_true: {} |  ".format(str(e.z_true).ljust(7))
                +"dl: {} Mpc  |  ".format(str(e.dl).ljust(9))
                +"sigmadl: {} Mpc  | ".format(str(e.sigmadl)[:6].ljust(7))
                +"hosts: {}".format(str(len(e.potential_galaxy_hosts))
                                        .ljust(8))
                +"zmin: {}".format(str(e.zmin).ljust(8))
                +"zmax: {}".format(str(e.zmax).ljust(8)))
        if config_par['event_class'] == "MBHB":
            +"z_host: {} |  ".format(
            str(e.potential_galaxy_hosts[0].redshift).ljust(8))

    print(formatting_string+"\n")
    print("nessai will be initialised with:")
    print(f"nlive:                   {config_par['nlive']}")
    print(f"pytorch_threads:         {config_par['pytorch_threads']}")
    print(f"n_pool:                  {config_par['n_pool']}")
    print(f"periodic_checkpoint_int: {config_par['checkpoint_int']}")

    C = CosmologicalModel(
        model=config_par['model'],
        data=events,
        truths=truths,
        prior_bounds=config_par['prior_bounds'],
        snr_threshold=config_par['snr_threshold'],
        z_threshold=float(config_par['zhorizon']),
        event_class=config_par['event_class'],
        com_vol=config_par['com_vol'])

    # FIXME: add all the settings options of nessai.
    # IMPROVEME: postprocess doesn't work when events are 
    # randomly selected, since 'events' in C are different 
    # from the ones read from chain.txt.
    if (config_par['postprocess'] == 0):
        sampler = FlowSampler(
            C,
            nlive=config_par['nlive'],
            pytorch_threads=config_par['pytorch_threads'],
            n_pool=config_par['n_pool'],
            seed=config_par['seed'],
            output=output_sampler,
            checkpoint_interval=config_par['checkpoint_int'],
            )

        sampler.run()
        print("\n"+formatting_string+"\n")

        x = sampler.posterior_samples.ravel()

        # Save git info.
        with open("{}/git_info.txt".format(outdir), 'w+') as fileout:
            subprocess.call(['git', 'diff'], stdout=fileout)
 
        # Save content of installed files.
        files_to_save = []
        files_path = lk.__file__.replace(lk.__file__.split("/")[-1], "")
        for te in os.listdir(files_path):
            if not ".so" in te and not "__" in te:
                files_to_save.append(te)
        files_to_save.sort()
        output_file = open(os.path.join(outdir, "installed_files.txt"), 'w')
        for fi in files_to_save:
            f = open(f"{files_path}/{fi}", 'r')
            output_file.write("____________________\n")
            output_file.write(f"{fi}\n____________________\n")
            output_file.write(f.read())
            output_file.write(10*"\n")
    else:
        print(f"Reading the .h5 file... from {outdir}")
        import h5py
        filename = os.path.join(outdir,"raynest", "results.json")
        h5_file = h5py.File(filename, 'r')
        x = h5_file['combined'].get('posterior_samples')

    ###################################################################
    ###################          MAKE PLOTS         ###################
    ###################################################################

    params = [m for m in C.model if m not in ['GW']]

    if (len(params) == 1):
        plots.histogram(x, par=params[0],
                        truths=truths, outdir=outdir)
    else:
        plots.corner_plot(x, pars=params,
                          truths=truths, outdir=outdir)

    # Compute the run-time.
    if (config_par['postprocess'] == 0):
        run_time = (time.perf_counter() - run_time)/60.0
        print("\nRun-time (min): {:.2f}\n".format(run_time))


if __name__=='__main__':
    main()
