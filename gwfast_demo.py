import sys

def print_usage():
    print("################################ Usage ################################")
    print("# Usage:                                                              #")
    print("#    -test argv: evaluate the time for single event run               #")
    print("#        argv is the number of events to test                         #")
    print("#    -plot argv: plot the distribution of measurement error           #")
    print("#        argv is the number of fixed parameters                       #")
    print("#    -run argv: simulate N (sample_size_need=200 default) events      #")
    print("#        argv is the work_id (should be >=1) of multiprocessing       #")
    print("#    -hdf argv: convert the txt files to hdf5 file                    #")
    print("#        argv means txt files with work_id<=argv will be converted    #")
    print("#    -h or --help: show this message                                  #")
    print("# Note: 1. please modify the h5_res_dir to your path                  #")
    print("#       2. you may need to install bilby package via:                 #")
    print("#          conda install -c conda-forge bilby                         #")
    print("#       3. for -run option, use sh work_submit.sh for multiprocessing #")
    print("#######################################################################")
    exit()
try:
    if sys.argv[1]=='-h' or sys.argv[1]=='--help':
        print_usage()
    elif sys.argv[1]=='-test':
        run_fisher, save_to_h5, plot_fisher = 1, 0, 0
        test_it, test_events = 1, int(sys.argv[2])
    elif sys.argv[1]=='-plot':
        run_fisher, save_to_h5, plot_fisher = 0, 0, 1
        plot_idx = int(sys.argv[2])
    elif sys.argv[1]=='-run':
        run_fisher, save_to_h5, plot_fisher = 1, 0, 0
        test_it, work_id = 0, int(sys.argv[2])
    elif sys.argv[1]=='-hdf':
        run_fisher, save_to_h5, plot_fisher = 0, 1, 0
        total_files = int(sys.argv[2])
    else:
        print("Invalid arguments! Please use -h for more information!")
        exit()
except IndexError:
    print("Invalid arguments! Please use -h for more information!")
    exit()

sample_size_need = 200
h5_res_dir = '/Users/tangsp/Code/Result/fisher/'
meta_data_fname = 'GWFAST_results.h5'
group_name = 'ET+2CE/NSBH'
subgroup_name = 'CovInfo:IMRPhenomNSBH'

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

load_population = 1
if load_population:
    from astropy import units
    from astropy.cosmology import FlatLambdaCDM
    from bilby.core.prior import PriorDict, Uniform, Sine, DeltaFunction, TruncatedGaussian, Interped
    cosmo_model = FlatLambdaCDM(H0=67.66, Om0=0.3097)
    psi_z = lambda z,alpha_z,z_p,beta_z: (1+z)**alpha_z/(1+((1+z)/(1+z_p))**(alpha_z+beta_z))
    func_p_z = lambda z,alpha_z,z_p,beta_z: psi_z(z,alpha_z,z_p,beta_z)/(1+z)*\
        4*np.pi*(cosmo_model.differential_comoving_volume(z).to(units.Gpc**3/units.sr).value)
    z_merger_array = np.linspace(0, 10, 1000)

    BH_mass_model = lambda m,a1=1.04e11,b1=2.1489,a2=799.1,b2=0.2904,a3=2.845e-3,b3=1.686:\
        1/(1/(a1*np.exp(-b1*m)+a2*np.exp(-b2*m)) + 1/a3*np.exp(-b3*m))
    BH_mass_array = np.linspace(2.5, 30, 1000)

    injection_model = PriorDict()
    injection_model['m1'] = Interped(xx=BH_mass_array, yy=BH_mass_model(BH_mass_array), minimum=2.5, maximum=30)
    injection_model['m2'] = TruncatedGaussian(mu=1.33, sigma=0.09, minimum=1.0, maximum=2.5)
    injection_model['z'] = Interped(xx=z_merger_array, yy=func_p_z(z_merger_array, 1.42, 1.84, 4.62), minimum=0, maximum=10)
    injection_model['theta'] = Sine()
    injection_model['phi'] = Uniform(minimum=0, maximum=2*np.pi)
    injection_model['iota'] = Sine()
    injection_model['psi'] = Uniform(minimum=0, maximum=np.pi)
    injection_model['Phicoal'] = Uniform(minimum=0, maximum=2*np.pi)
    injection_model['chi1z'] = TruncatedGaussian(mu=0, sigma=0.15, minimum=-1, maximum=1)
    injection_model['chi2z'] = Uniform(minimum=-0.05, maximum=0.05)
    injection_model['Lambda1'] = DeltaFunction(peak=0)
    injection_model['Lambda2'] = Uniform(minimum=0, maximum=2000)
    injection_model['tGPS'] = Uniform(minimum=0, maximum=10*units.year.to(units.s))
    total_par_size = len(injection_model)

    fixed_parname_list = [['deltaLambda'],['tcoal','deltaLambda'],['Phicoal','deltaLambda'],\
        ['psi','deltaLambda'],['psi','Phicoal','deltaLambda']]
    shape1_list = [total_par_size, total_par_size+2]+[total_par_size+1\
        -len(fixed_parname) for fixed_parname in fixed_parname_list]

if run_fisher:
    import os, copy
    import gwfast.gwfastGlobals as glob
    AllDetectors = copy.deepcopy(glob.detectors)
    ET2CEd = {det: AllDetectors[det] for det in ['ETS', 'CE1Id', 'CE2NM']}
    ET2CEd['ETS'].update({'psd_path':os.path.join(glob.detPath, 'ET-0000A-18.txt'), 'fmin':2})
    ET2CEd['CE1Id']['psd_path'] = os.path.join(glob.detPath, 'ce_strain', 'cosmic_explorer.txt')
    ET2CEd['CE1Id'].update({'lat':46.5, 'long':-119.4, 'xax':171, 'fmin':5})
    ET2CEd['CE2NM']['psd_path'] = os.path.join(glob.detPath, 'ce_strain', 'cosmic_explorer_20km.txt')
    ET2CEd['CE2NM'].update({'lat':30.6, 'long':-90.8, 'xax':242.7, 'fmin':5})

    from gwfast.signal import GWSignal
    from gwfast.network import DetNet
    from gwfast.waveforms import IMRPhenomNSBH
    ParNums = IMRPhenomNSBH().ParNums
    myETNet = DetNet({d:GWSignal(IMRPhenomNSBH(), psd_path=ET2CEd[d]['psd_path'], detector_shape=ET2CEd[d]['shape'], \
        det_lat=ET2CEd[d]['lat'], det_long=ET2CEd[d]['long'], det_xax=ET2CEd[d]['xax'], verbose=False, \
        useEarthMotion=True, fmin=ET2CEd[d]['fmin'], IntTablePath=None) for d in ET2CEd.keys()}, verbose=False)

    from gwfast.fisherTools import CovMatr, fixParams
    from gwfast.gwfastUtils import GPSt_to_LMST, Mceta_from_m1m2
    def multi_run_func(total_task, work_id):
        np.random.seed()
        data_stored_in_pool = [np.empty(shape=(0,shape1)) for shape1 in shape1_list]
        q = 0
        while q < total_task:
            injection_parameters = injection_model.sample(1)
            data_stored_in_pool[0] = np.vstack((data_stored_in_pool[0], \
                np.array(list(injection_parameters.values())).flatten()))
            injection_parameters['dL'] = cosmo_model.luminosity_distance(injection_parameters['z']).to(units.Gpc).value
            injection_parameters['tcoal'] = GPSt_to_LMST(injection_parameters.pop('tGPS'), lat=0., long=0.)
            injection_parameters['Mc'], injection_parameters['eta'] = \
                Mceta_from_m1m2(injection_parameters.pop('m1'), injection_parameters.pop('m2'))
            injection_parameters['Mc'] *= (1+injection_parameters.pop('z'))
            
            opt_snr = myETNet.SNR(injection_parameters)[0]
            if opt_snr<12:
                data_stored_in_pool[1] = np.vstack((data_stored_in_pool[1], \
                    np.append(-1*np.ones(shape1_list[1]-1), opt_snr)))
                for j in range(len(shape1_list)-2):
                    data_stored_in_pool[j+2] = np.vstack(\
                        (data_stored_in_pool[j+2], -1*np.ones(shape1_list[j+2])))
            else:
                totF = myETNet.FisherMatr(injection_parameters, use_chi1chi2=True)
                if test_it:
                    print('Injection parameters: ', injection_parameters)
                    print('Fisher matrix: ', totF[:,:,0])
                totCov, invErr = CovMatr(totF)
                totCov = totCov[:,:,0]
                data_stored_in_pool[1] = np.vstack((data_stored_in_pool[1], \
                    np.append(totCov.diagonal(), [invErr[0], opt_snr])))

                for j,fixed_parname in enumerate(fixed_parname_list):
                    newFish, _ = fixParams(totF, ParNums, fixed_parname)
                    newCov, invErr = CovMatr(newFish)
                    newCov = newCov[:,:,0]
                    data_stored_in_pool[j+2] = np.vstack((data_stored_in_pool[j+2], \
                        np.append(newCov.diagonal(), invErr[0])))
                q += 1
                message = 'work-id: {}, status: {}/{}'.format(work_id, q, total_task)
                end = '' if q<total_task else '\n'
                print('\r'+message, end=end, flush=True)
        return data_stored_in_pool

    if test_it:
        import time
        start_time = time.time()
        multi_run_func(test_events, 0)
        single_time = (time.time() - start_time)/test_events
        print('single run takes {:.3f}s'.format(single_time))
    else:
        data_stored_in_pool = multi_run_func(sample_size_need, work_id)
        for j,res in enumerate(data_stored_in_pool):
            np.savetxt(h5_res_dir+'fisher_metadata_{}_{}.txt'.format(j, work_id), res)

if save_to_h5:
    data_stored = [np.empty(shape=(0,shape1)) for shape1 in shape1_list]
    exit_state = 0
    for j in range(len(shape1_list)):
        for i in range(total_files):
            if not os.path.exists(h5_res_dir+'fisher_metadata_{}_{}.txt'.format(j,i+1)):
                if exit_state==0:
                    print('The following files do not exist: ')
                print('    fisher_metadata_{}_{}.txt'.format(j,i+1))
                exit_state = 1
    if exit_state==1:
        exit() 
    for j in range(len(shape1_list)):
        for i in range(total_files):
            res = np.loadtxt(h5_res_dir+'fisher_metadata_{}_{}.txt'.format(j,i+1))
            data_stored[j] = np.vstack((data_stored[j], res))
            #os.system('rm '+h5_res_dir+'fisher_metadata_{}_{}.txt'.format(j,i+1))

    def convert_to_structured_data(initial_data, column_names):
        new_dtype = list(zip(column_names, [initial_data.dtype]*len(column_names)))
        initial_data = initial_data.T
        structured_data = np.empty(initial_data.shape[1], dtype=new_dtype)
        for name in column_names:
            structured_data[name] = initial_data[column_names.index(name),:]
        return structured_data

    import h5py
    if os.path.exists(h5_res_dir+meta_data_fname):
        fp = h5py.File(h5_res_dir+meta_data_fname, "a")
    else:
        fp = h5py.File(h5_res_dir+meta_data_fname, "w")
    if group_name in fp.keys():
        group = fp[group_name]
    else:
        group = fp.create_group(group_name)

    column_names = list(injection_model.keys())
    column_names[0] = 'm1_src'
    column_names[1] = 'm2_src'
    structured_data = convert_to_structured_data(data_stored[0], column_names)
    if 'injections' in group.keys():
        del group['injections']
    group.create_dataset('injections', data=structured_data)

    if subgroup_name in group.keys():
        subgroup = group[subgroup_name]
    else:
        subgroup = group.create_group(subgroup_name)

    from gwfast.waveforms import IMRPhenomNSBH
    ParNums = IMRPhenomNSBH().ParNums
    column_names = list(ParNums.keys())+['invError', 'optimalSNR']
    from functools import reduce
    fixed_index_list = [[str(ParNums[name]) for name in fixed_parname] for fixed_parname in fixed_parname_list]
    dataset_name_list = ['fiducial']+['fix '+reduce(lambda x,y: x+','+y, fixed_index) for fixed_index in fixed_index_list]
    for i,dname in enumerate(dataset_name_list):
        column_names_i = column_names.copy()
        if i!=0:
            for key in fixed_parname_list[i-1]+['optimalSNR']:
                column_names_i.remove(key)
        structured_data = convert_to_structured_data(data_stored[i+1], column_names_i)
        if dname in subgroup.keys():
            del subgroup[dname]
        subgroup.create_dataset(dname, data=structured_data)
    fp.close()
    print('Save data to {}, successful!'.format(meta_data_fname))

if plot_fisher:
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    def plot_multi_p1d(source_data, colors, ls_list=None, addtext=None, pcts=[5, 50, 95],\
        xlim=None, ylim=None, xlabel='x', ylabel='PDF', axis=None, hist_bins=None, log_step=False):
        total_groups = len(source_data)
        if ls_list is None:
            ls_list = ['-']*total_groups
        if total_groups%2==0:
            locs = ['left', 'right']
        else:
            locs = [['left', 'center', 'right'], \
                ['center', 'left', 'right'], \
                ['left', 'right', 'center']]
            locs = locs[total_groups%3]
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
        if xlim is None:
            xmin = min([np.min(data) for data in source_data])
            xmax = max([np.max(data) for data in source_data])
            xlim = (xmin, xmax)
        maxxpdf = []
        for j,data in enumerate(source_data):
            if hist_bins is not None:
                if addtext is None:
                    axis.hist(data, color=colors[j], density=1, histtype='step', bins=hist_bins)
                else:
                    axis.hist(data, color=colors[j], density=1, histtype='step', bins=hist_bins, label=addtext[j])
            x_kde = gaussian_kde(data, bw_method='scott')
            if log_step:
                xx = np.logspace(np.log10(max(xlim[0], np.min(data))),\
                    np.log10(min(xlim[1], np.max(data))), 100, base=10)
            else:
                xx = np.linspace(max(xlim[0], np.min(data)), min(xlim[1], np.max(data)), 100)
            pdf_x = x_kde(xx)
            maxxpdf.append(max(pdf_x))
            if False:
                if addtext is None:
                    axis.plot(xx, pdf_x, color=colors[j], ls=ls_list[j])
                else:
                    axis.plot(xx, pdf_x, color=colors[j], ls=ls_list[j], label=addtext[j])
            if pcts is not None:
                percs = np.percentile(data, pcts)
                if False:
                    for p in [percs[0],percs[2]]:
                        axis.axvline(p, linestyle=':', color=colors[j])
                if total_groups==1 or total_groups%2==0:
                    tsize, tpad, tloc = 12, 5+int(j/2)*17, locs[j%2]
                else:
                    tsize, tpad, tloc = 12, 5+int(j/3)*15, locs[j%3]
                temp_axis_x = axis.twiny()
                temp_axis_x.tick_params(labelleft=False, labeltop=False, labelright=False, \
                    labelbottom=False, top=False, bottom=False, left=False, right=False)
                temp_axis_x.set_xlim(*xlim)
                temp_axis_x.set_title(r"${:.3f}_{{-{:.3f}}}^{{+{:.3f}}}$".format(percs[1], \
                    percs[1]-percs[0], percs[2]-percs[1]), fontsize=tsize, color=colors[j], \
                    pad=tpad, loc=tloc)
        axis.set_xlim(*xlim)
        if ylim is None:
            axis.set_ylim(0, 1.1*max(maxxpdf))
        else:
            axis.set_ylim(*ylim)
        axis.set_xlabel(xlabel, fontsize=12)
        axis.set_ylabel(ylabel, fontsize=12)
        return fig, axis
    
    import h5py
    from astropy import units
    from astropy.cosmology import FlatLambdaCDM
    cosmo_model = FlatLambdaCDM(H0=67.66, Om0=0.3097)

    from gwfast.gwfastUtils import Mceta_from_m1m2
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18,9))
    gs_grid = GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1,1], wspace=0.15, hspace=0.30)
    axes = [plt.subplot(gs) for gs in gs_grid]
    plot_which_pars = ['Mc', 'eta', 'dL', 'chi1z']
    xlim_list = [(1e-6, 1e-2), (1e-5, 1e-1), (1e-3,10), (1e-3,2)]
    show_logspace = [1, 0, 1, 0]
    correspond_xlabel = [r'$\Delta$'+lb for lb in \
        [r'$\mathcal{M}/\mathcal{M}~(\times 100)$', r'$\eta$', r'$d_L/d_L$', r'$\chi_{\rm BH}$']]
    if plot_idx==2:
        plot_which_dset = ['fix 12', 'fix 6,12', 'fix 7,12', 'fix 8,12']
        addtext_list = ['fix '+r'$\Delta\Lambda$', 'fix '+r'$\psi,\Delta\Lambda$', \
            'fix '+r'$t_c,\Delta\Lambda$', 'fix '+r'$\Phi_c,\Delta\Lambda$']
    else:
        plot_which_dset = ['fix 12', 'fix 6,8,12']
        addtext_list = ['fix '+r'$\Delta\Lambda$', 'fix '+r'$\psi,\Phi_c,\Delta\Lambda$']
    for j,(par,xlabel,axis,xlim) in enumerate(zip(plot_which_pars,correspond_xlabel,axes,xlim_list)):
        source_data_to_plot = []
        with h5py.File(h5_res_dir+meta_data_fname, "r") as fp:
            #print(fp[group_name][subgroup_name].keys())
            opt_snr = fp[group_name][subgroup_name]['fiducial']['optimalSNR']
            valid_index = opt_snr>12
            Mc_vals, _ = Mceta_from_m1m2(fp[group_name]['injections']['m1_src'], fp[group_name]['injections']['m2_src'])
            Mc_vals *= (1+fp[group_name]['injections']['z'])
            dL_vals = cosmo_model.luminosity_distance(fp[group_name]['injections']['z']).to(units.Gpc).value
            par_vals_dict = {'Mc':Mc_vals, 'dL':dL_vals}
            for dset_name in plot_which_dset:
                sample_data = fp[group_name][subgroup_name][dset_name][par]
                inv_err = fp[group_name][subgroup_name][dset_name]['invError']
                if dset_name=='fiducial':
                    valid_index = valid_index*(sample_data>0)
                else:
                    valid_index = valid_index*(inv_err<0.05)*(sample_data>0)
                if par in ['Mc', 'dL']:
                    relative_err = sample_data[valid_index]**0.5/par_vals_dict[par][valid_index]
                    trunc_index = (relative_err>xlim[0])*(relative_err<xlim[1])
                    if par=='Mc':
                        relative_err *= 100
                    source_data_to_plot.append(relative_err[trunc_index])
                else:
                    absolute_err = sample_data[valid_index]**0.5
                    trunc_index = (absolute_err>xlim[0])*(absolute_err<xlim[1])
                    source_data_to_plot.append(absolute_err[trunc_index])
            fp.close()
        if par=='Mc':
            xlim = [xlim[0]*100, xlim[1]*100]
        if show_logspace[j]:
            hist_bins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100, base=10)
        else:
            hist_bins = np.linspace(xlim[0], xlim[1], 100)
        _, axis = plot_multi_p1d(source_data_to_plot, ['k', 'blue', 'red', 'orange', 'green'], ls_list=['--',':','-', '-.', '--'], \
            addtext=addtext_list, pcts=[15.865, 50, 84.135], xlim=xlim, ylim=None, xlabel=xlabel, \
            ylabel='PDF', axis=axis, hist_bins=hist_bins, log_step=show_logspace[j])
        if True:
            xx, frac = np.loadtxt(h5_res_dir+par+'.csv', skiprows=1, delimiter=',').T
            if par=='Mc':
                xx *= 100
            pdf_x = frac[1:]/np.diff(xx)
            axis.set_ylim(0, max(axis.get_ylim()[1], 1.1*max(pdf_x)))
            axis.stairs(pdf_x, xx, color='purple', label='Iacovelli+')
        axis.tick_params(direction='in', which='both', right=True, top=True, labelsize=12)
        if j==0:
            axis.legend(fontsize=12)
        if show_logspace[j]:
            axis.set_xscale('log')
            axis.set_yscale('log')
            axis.set_ylim(1e-3, 10*axis.get_ylim()[1])
    plt.savefig(h5_res_dir+'std_dist_{}.pdf'.format(plot_idx))
