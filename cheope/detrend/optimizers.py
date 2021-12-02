import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import json
from pathlib import Path
from uncertainties import ufloat
import h5py
import shutil
from mpi4py import MPI
import copyreg

import cheope.pyconstants as cst
import cheope.pycheops_analysis as pyca
import cheope.linear_ephemeris as lep

# MPI._p_pickle.dumps = dill.dumps
# MPI._p_pickle.loads = dill.loads

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


printlog = pyca.printlog
fig_ext = ["png", "pdf"]


class Optimizers:
    """
    Optimizers class:
    it contains al list of all the possible optimizers used for Bayesian analysis.

    The Emcee function is used for MCMC analysis and t prints the results in the most
    appropriate format for emcee.

    The Ultranest function uses Nested sampling and returns also the characteristic plots for this
    alsorithm.
    """

    def __init__(self):
        self.optimizers_list = ["emcee", "ultranest"]

    def emcee(
        self,
        inpars=None,
        dataset=None,
        olog=None,
        params_lm_loop=None,
        star=None,
    ):

        (
            visit_args,
            star_args,
            planet_args,
            emcee_args,
            ultranest_args,
            read_file_status,
        ) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.ultranest_args,
            inpars.read_file_status,
        )

        aperture = visit_args["aperture"]

        # =====================
        # name of the file in pycheops_data
        file_key = visit_args["file_key"]

        # visit_folder
        main_folder = visit_args["main_folder"]
        visit_number = visit_args["visit_number"]
        shape = visit_args["shape"]

        # visit_folder = Path('/home/borsato/Dropbox/Research/exoplanets/objects/KELT/KELT-6/data/CHEOPS_DATA/pycheops_analysis/visit_01/')
        visit_name = "visit_{:02d}_{:s}_{:s}_shape_ap{:s}_BF_{:s}".format(
            visit_number,
            file_key,
            shape.lower(),
            aperture.upper(),
            visit_args["optimizer"],
        )
        visit_folder = Path(os.path.join(main_folder, visit_name))

        logs_folder = os.path.join(main_folder, "logs")

        star_name = star_args["star_name"]

        nwalkers = emcee_args["nwalkers"]
        nprerun = emcee_args["nprerun"]
        nsteps = emcee_args["nsteps"]
        nburn = emcee_args["nburn"]
        nthin = emcee_args["nthin"]
        nthreads = emcee_args["nthreads"]
        progress = emcee_args["progress"]

        # Run emcee from last best fit
        printlog("\n-Run emcee from last best fit with:", olog=olog)
        printlog(" nwalkers = {}".format(nwalkers), olog=olog)
        printlog(" nprerun  = {}".format(nprerun), olog=olog)
        printlog(" nsteps   = {}".format(nsteps), olog=olog)
        printlog(" nburn    = {}".format(nburn), olog=olog)
        printlog(" nthreads = {}".format(nthreads), olog=olog)
        printlog(" nthin    = {}".format(nthin), olog=olog)
        printlog("", olog=olog)
        # Run default optimizer EMCEE
        result = dataset.emcee_sampler(
            params=params_lm_loop,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=False,
            progress=progress,
        )

        printlog(dataset.emcee_report(min_correl=0.5), olog=olog)

        printlog("\n-Plot trace of the chains", olog=olog)
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_trace_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_corner_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        params_med, _, params_mle, stats_mle = pyca.get_best_parameters(
            result, dataset, nburn=nburn, dataset_type="visit", update_dataset=True
        )
        # update emcee.params -> median and emcee.params_mle -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med[p]
            dataset.emcee.params_best[p] = params_mle[p]

        printlog("MEDIAN PARAMETERS", olog=olog)
        for p in params_med:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_med[p].value, params_med[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "02_params_emcee_median.dat"),
            params_med,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(dataset, params_best=params_med, glint=False, olog=olog)
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "02_lc_emcee_median.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_lc_emcee_median.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_fft_emcee_median.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("MLE PARAMETERS", olog=olog)
        for p in params_mle:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_mle[p].value, params_mle[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "02_params_emcee_mle.dat"),
            params_mle,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(dataset, params_best=params_mle, glint=False, olog=olog)
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(visit_folder.resolve(), "02_lc_emcee_mle.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(visit_folder.resolve(), "02_lc_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(visit_folder.resolve(), "02_fft_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        file_emcee = pyca.save_dataset(
            dataset, visit_folder.resolve(), star_name, file_key, gp=False
        )
        printlog("-Dumped dataset into file {}".format(file_emcee), olog=olog)

        ### *** ==============================================================
        ### *** ===== TRAIN GP ===============================================
        printlog("", olog=olog)
        printlog("TRAIN GP HYPERPARAMETERS FIXING PARAMETERS", olog=olog)

        params_fixed = pyca.copy_parameters(params_mle)
        # for p in ['T_0','D','W','b']: # only transit shape
        for p in params_mle:  # fixing all transit and detrending parameters
            params_fixed[p].set(vary=False)
        params_fixed["log_sigma"].set(vary=True)

        result_gp_train = dataset.emcee_sampler(
            params=params_fixed,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=True,
            progress=progress,
        )

        printlog(dataset.emcee_report(min_correl=0.5), olog=olog)

        printlog("\n-Plot trace of the chains of GP training", olog=olog)
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "03_trace_emcee_gp_train.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        (
            params_med_gp_train,
            _,
            params_mle_gp_train,
            stats_mle,
        ) = pyca.get_best_parameters(
            result_gp_train,
            dataset,
            nburn=nburn,
            dataset_type="visit",
            update_dataset=False,
        )

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med_gp_train,
            par_type="median-GPtrain",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "03_lc_emcee_median_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "03_lc_emcee_median_gp_train.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle_gp_train,
            par_type="mle-GPtrain",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "03_lc_emcee_mle_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "03_lc_emcee_mle_gp_train.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        ### *** =======================================++=====================
        ### *** ===== FIT TRANSIT + DETRENDING + GP =++=======================
        printlog("\nRUN FULL FIT TRANSIT+DETRENDING+GP W/ EMCEE", olog=olog)

        params_fit_gp = pyca.copy_parameters(params_mle)
        for p in ["log_S0", "log_omega0", "log_sigma"]:
            # params_fit_gp[p] = params_mle_gp_train[p]
            # printlog("{} = {} user_data = {}".format(p, params_fit_gp[p], params_fit_gp[p].user_data), olog=olog)
            # params_fit_gp[p].user_data = ufloat(params_mle_gp_train[p].value, 2*params_mle_gp_train[p].stderr)
            # printlog("{} = {} user_data = {}".format(p, params_fit_gp[p], params_fit_gp[p].user_data), olog=olog)
            params_fit_gp.add(
                p,
                value=params_mle_gp_train[p].value,
                vary=True,
                min=params_mle_gp_train[p].min,
                max=params_mle_gp_train[p].max,
            )
            params_fit_gp[p].user_data = ufloat(
                params_mle_gp_train[p].value, 2 * params_mle_gp_train[p].stderr
            )

        # log_Q = 1/sqrt(2)
        params_fit_gp.add("log_Q", value=np.log(1 / np.sqrt(2)), vary=False)

        result_gp = dataset.emcee_sampler(
            params=params_fit_gp,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            # add_shoterm = True, # not needed the second time
            progress=progress,
        )

        printlog(dataset.emcee_report(min_correl=0.5), olog=olog)

        printlog("\n-Plot trace of the chains", olog=olog)
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_trace_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_corner_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        params_med_gp, _, params_mle_gp, _ = pyca.get_best_parameters(
            result_gp, dataset, nburn=nburn, dataset_type="visit", update_dataset=True
        )
        # update emcee.params -> median and emcee.params_mle -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med_gp[p]
            dataset.emcee.params_best[p] = params_mle_gp[p]

        printlog("MEDIAN PARAMETERS w/ GP", olog=olog)
        for p in params_med_gp:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_med_gp[p].value, params_med_gp[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "04_params_emcee_median_gp.dat"),
            params_med_gp,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(
            dataset, params_best=params_med_gp, glint=False, olog=olog
        )
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med_gp,
            par_type="median w/ GP",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "04_lc_emcee_median_gp.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_lc_emcee_median_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_fft_emcee_median_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("MLE PARAMETERS w/ GP", olog=olog)
        for p in params_mle_gp:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_mle_gp[p].value, params_mle_gp[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "04_params_emcee_mle_gp.dat"),
            params_mle_gp,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(
            dataset, params_best=params_mle_gp, glint=False, olog=olog
        )
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle_gp,
            par_type="mle w/ GP",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "04_lc_emcee_mle_gp.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_lc_emcee_mle_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_fft_emcee_mle_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        file_emcee = pyca.save_dataset(
            dataset, visit_folder.resolve(), star_name, file_key, gp=True
        )
        printlog("-Dumped dataset into file {}".format(file_emcee), olog=olog)

    def ultranest(
        self,
        inpars=None,
        dataset=None,
        olog=None,
        params_lm_loop=None,
        star=None,
    ):

        (
            visit_args,
            star_args,
            planet_args,
            emcee_args,
            ultranest_args,
            read_file_status,
        ) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.ultranest_args,
            inpars.read_file_status,
        )

        aperture = visit_args["aperture"]

        # =====================
        # name of the file in pycheops_data
        file_key = visit_args["file_key"]

        # visit_folder
        main_folder = visit_args["main_folder"]
        visit_number = visit_args["visit_number"]
        shape = visit_args["shape"]

        # visit_folder = Path('/home/borsato/Dropbox/Research/exoplanets/objects/KELT/KELT-6/data/CHEOPS_DATA/pycheops_analysis/visit_01/')
        visit_name = "visit_{:02d}_{:s}_{:s}_shape_ap{:s}_BF_{:s}".format(
            visit_number,
            file_key,
            shape.lower(),
            aperture.upper(),
            visit_args["optimizer"],
        )
        visit_folder = Path(os.path.join(main_folder, visit_name))

        logs_folder = os.path.join(main_folder, "logs")

        star_name = star_args["star_name"]

        live_points = ultranest_args["live_points"]
        tolerance = ultranest_args["tol"]
        cluster_num_live_points = ultranest_args["cluster_num_live_points"]
        logdir = os.path.join(visit_folder.resolve(), "02_ultranest")
        resume = ultranest_args["resume"]
        adaptive_nsteps = ultranest_args["adaptive_nsteps"]

        # Run emcee from last best fit
        printlog("\n-Run Ultranest from last best fit with:", olog=olog)
        printlog(" live_points              = {}".format(live_points), olog=olog)
        printlog(" tolerance                = {}".format(tolerance), olog=olog)
        printlog(
            " cluster_num_live_points  = {}".format(cluster_num_live_points),
            olog=olog,
        )
        printlog(
            " logdir                   = {}".format(logdir),
            olog=olog,
        )
        printlog(
            " resume                   = {}".format(resume),
            olog=olog,
        )
        printlog("", olog=olog)
        sampler = dataset.ultranest_sampler(
            params=params_lm_loop,
            live_points=live_points,
            tol=tolerance,
            cluster_num_live_points=cluster_num_live_points,
            logdir=logdir,
            resume=resume,
            adaptive_nsteps=adaptive_nsteps,
            add_shoterm=False,
        )
        """
        dir(sampler) = ['Lmin', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', 
        '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', 
        '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', 
        '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_adaptive_strategy_advice', '_check_likelihood_function', 
        '_create_point', '_expand_nodes_before', '_find_strategy', '_refill_samples', '_set_likelihood_function', '_setup_distributed_seeds', 
        '_should_node_be_expanded', '_update_region', '_update_results', '_widen_nodes', '_widen_roots', 'build_tregion', 'cluster_num_live_points', 
        'comm', 'derivedparamnames', 'draw_multiple', 'ib', 'likes', 'live_points_healthy', 'log', 'log_to_disk', 'log_to_pointstore', 'loglike', 
        'min_num_live_points', 'mpi_rank', 'mpi_size', 'ncall', 'ncall_region', 'ndraw_max', 'ndraw_min', 'num_bootstraps', 'num_params', 'paramnames', 
        'plot', 'plot_corner', 'plot_run', 'plot_trace', 'pointpile', 'pointstore', 'print_results', 'region', 'region_class', 'region_nodes', 'results', 
        'root', 'run', 'run_iter', 'run_sequence', 'sampler', 'samples', 'samplesv', 'sampling_slow_warned', 'stepsampler', 'store_tree', 'transform', 
        'transformLayer', 'transform_limits', 'tregion', 'use_mpi', 'use_point_stack', 'volfactor', 'wrapped_axes', 'x_dim']
        """

        result_json_path = os.path.join(logdir, "info/results.json")
        results = json.load(open(result_json_path))

        flatchain = np.array(sampler.results["samples"])

        # for obj in (MPI.COMM_NULL, MPI.COMM_SELF, MPI.COMM_WORLD):
        # assert pickle.loads(pickle.dumps(obj)) == obj

        # comm = MPI.COMM_WORLD
        # size = comm.Get_size()
        # rank = comm.Get_rank()
        # if rank == 0:
        #     rt_folder = (
        #         "/data2/zingales/cheops_analysis-package_testold/notebook/run_example/"
        #     )
        #     with open(os.path.join(rt_folder, "sampler.pickle"), "wb") as sampler_file:
        #         pickle.dump(sampler, sampler_file, pickle.HIGHEST_PROTOCOL)

        #     with open(os.path.join(rt_folder, "results.pickle"), "wb") as results_file:
        #         pickle.dump(results, results_file, pickle.HIGHEST_PROTOCOL)

        params_med, params_mle = pyca.get_best_parameters_ultranest(
            results, params_lm_loop, sampler, dataset_type="visit"
        )

        printlog("Plotting run", olog=olog)
        sampler.plot_run()

        printlog("Plotting traces", olog=olog)
        sampler.plot_trace()

        printlog("Generating corner plot", olog=olog)
        sampler.plot_corner()

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )

        printlog("MEDIAN PARAMETERS", olog=olog)
        for p in params_med:
            printlog(
                "{} = {} +/- {}".format(p, params_med[p].value, params_med[p].stderr),
                olog=olog,
            )

        pyca.quick_save_params_ultranest(
            os.path.join(visit_folder.resolve(), "02_params_ultranest_median.dat"),
            params_med,
            dataset.lc["bjd_ref"],
        )

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med,
            par_type="median",
            nsamples=300,
            flatchains=flatchain,
            model_filename=os.path.join(
                visit_folder.resolve(), "02_lc_ultranest_median.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_lc_ultranest_median.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_fft_ultranest_median.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("MLE PARAMETERS", olog=olog)
        for p in params_mle:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_mle[p].value, params_mle[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params_ultranest(
            os.path.join(visit_folder.resolve(), "02_params_ultranest_mle.dat"),
            params_mle,
            dataset.lc["bjd_ref"],
        )

        # _ = pyca.computes_rms(dataset, params_best=params_mle, glint=False, olog=olog)
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle,
            par_type="mle",
            nsamples=300,
            flatchains=flatchain,
            model_filename=os.path.join(
                visit_folder.resolve(), "02_lc_ultranest_mle.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_lc_ultranest_mle.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_fft_ultranest_mle.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        file_ultranest = pyca.save_dataset(
            dataset, visit_folder.resolve(), star_name, file_key, gp=False
        )
        printlog("-Dumped dataset into file {}".format(file_ultranest), olog=olog)

        ### *** ==============================================================
        ### *** ===== TRAIN GP ===============================================
        printlog("", olog=olog)
        printlog("TRAIN GP HYPERPARAMETERS FIXING PARAMETERS", olog=olog)

        params_fixed = pyca.copy_parameters(params_mle)
        # for p in ['T_0','D','W','b']: # only transit shape
        for p in params_mle:  # fixing all transit and detrending parameters
            params_fixed[p].set(vary=False)
        params_fixed["log_sigma"].set(vary=True)

        logdir = os.path.join(visit_folder.resolve(), "03_ultranest")
        sampler_gp = dataset.ultranest_sampler(
            params=params_fixed,
            live_points=live_points,
            tol=tolerance,
            cluster_num_live_points=cluster_num_live_points,
            logdir=logdir,
            resume=resume,
            adaptive_nsteps=adaptive_nsteps,
            add_shoterm=True,
        )

        result_json_path = os.path.join(logdir, "info/results.json")
        results_gp = json.load(open(result_json_path))

        flatchain = np.array(sampler_gp.results["samples"])

        printlog("Plotting run", olog=olog)
        sampler_gp.plot_run()

        printlog("Plotting traces", olog=olog)
        sampler_gp.plot_trace()

        printlog("Generating corner plot", olog=olog)
        sampler_gp.plot_corner()

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )

        params_med_gp_train, params_mle_gp_train = pyca.get_best_parameters_ultranest(
            results_gp, params_fixed, sampler_gp, dataset_type="visit"
        )

        pyca.quick_save_params_ultranest(
            os.path.join(visit_folder.resolve(), "03_params_ultranest_median.dat"),
            params_med_gp_train,
            dataset.lc["bjd_ref"],
        )

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med_gp_train,
            par_type="median-GPtrain",
            nsamples=300,
            flatchains=flatchain,
            model_filename=os.path.join(
                visit_folder.resolve(), "03_lc_ultranest_median_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(),
                    "03_lc_ultranest_median_gp_train.{}".format(ext),
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle_gp_train,
            par_type="mle-GPtrain",
            nsamples=300,
            flatchains=flatchain,
            model_filename=os.path.join(
                visit_folder.resolve(), "03_lc_ultranest_mle_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(),
                    "03_lc_ultranest_mle_gp_train.{}".format(ext),
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        pyca.quick_save_params_ultranest(
            os.path.join(visit_folder.resolve(), "03_params_ultranest_mle.dat"),
            params_mle_gp_train,
            dataset.lc["bjd_ref"],
        )

        ### *** =======================================++=====================
        ### *** ===== FIT TRANSIT + DETRENDING + GP =++=======================
        printlog("\nRUN FULL FIT TRANSIT+DETRENDING+GP W/ ULTRANEST", olog=olog)

        params_fit_gp = pyca.copy_parameters(params_mle)
        for p in ["log_S0", "log_omega0", "log_sigma"]:
            # params_fit_gp[p] = params_mle_gp_train[p]
            # printlog("{} = {} user_data = {}".format(p, params_fit_gp[p], params_fit_gp[p].user_data), olog=olog)
            # params_fit_gp[p].user_data = ufloat(params_mle_gp_train[p].value, 2*params_mle_gp_train[p].stderr)
            # printlog("{} = {} user_data = {}".format(p, params_fit_gp[p], params_fit_gp[p].user_data), olog=olog)
            params_fit_gp.add(
                p,
                value=params_mle_gp_train[p].value,
                vary=True,
                min=params_mle_gp_train[p].min,
                max=params_mle_gp_train[p].max,
            )
            params_fit_gp[p].user_data = ufloat(
                params_mle_gp_train[p].value, 2 * params_mle_gp_train[p].stderr
            )

        # log_Q = 1/sqrt(2)
        params_fit_gp.add("log_Q", value=np.log(1 / np.sqrt(2)), vary=False)

        logdir = os.path.join(visit_folder.resolve(), "04_ultranest")
        sampler_gp = dataset.ultranest_sampler(
            params=params_fit_gp,
            live_points=live_points,
            tol=tolerance,
            cluster_num_live_points=cluster_num_live_points,
            logdir=logdir,
            resume=resume,
            adaptive_nsteps=adaptive_nsteps,
            add_shoterm=False,
        )

        result_json_path = os.path.join(logdir, "info/results.json")
        results_gp = json.load(open(result_json_path))

        flatchain = np.array(sampler_gp.results["samples"])

        printlog("Plotting run", olog=olog)
        sampler_gp.plot_run()

        printlog("Plotting traces", olog=olog)
        sampler_gp.plot_trace()

        printlog("Generating corner plot", olog=olog)
        sampler_gp.plot_corner()

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )

        params_med_gp, params_mle_gp = pyca.get_best_parameters_ultranest(
            results_gp, params_fit_gp, sampler_gp, dataset_type="visit"
        )

        printlog("MEDIAN PARAMETERS w/ GP", olog=olog)
        for p in params_med_gp:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_med_gp[p].value, params_med_gp[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params_ultranest(
            os.path.join(visit_folder.resolve(), "04_params_ultranest_median_gp.dat"),
            params_med_gp,
            dataset.lc["bjd_ref"],
        )

        # _ = pyca.computes_rms(
        #     dataset, params_best=params_med_gp, glint=False, olog=olog
        # )
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med_gp,
            par_type="median w/ GP",
            nsamples=300,
            flatchains=flatchain,
            model_filename=os.path.join(
                visit_folder.resolve(), "04_lc_ultranest_median_gp.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_lc_ultranest_median_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_fft_ultranest_median_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("MLE PARAMETERS w/ GP", olog=olog)
        for p in params_mle_gp:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_mle_gp[p].value, params_mle_gp[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params_ultranest(
            os.path.join(visit_folder.resolve(), "04_params_ultranest_mle_gp.dat"),
            params_mle_gp,
            dataset.lc["bjd_ref"],
        )

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle_gp,
            par_type="mle w/ GP",
            nsamples=300,
            flatchains=flatchain,
            model_filename=os.path.join(
                visit_folder.resolve(), "04_lc_ultranest_mle_gp.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_lc_ultranest_mle_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_fft_ultranest_mle_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        file_ultranest = pyca.save_dataset(
            dataset, visit_folder.resolve(), star_name, file_key, gp=True
        )
        printlog("-Dumped dataset into file {}".format(file_ultranest), olog=olog)


class OptimizersKeplerTESS:
    def __init__(self):
        self.optimizers_list = ["emcee", "ultranest"]


class OptimizersMultivisit:
    def __init__(self):
        self.optimizers_list = ["emcee", "ultranest"]

    def emcee(
        self,
        inpars=None,
        M=None,
        olog=None,
        new_params=None,
        T_0=None,
        T_ref=None,
        P_ref=None,
        log_omega0=None,
        log_S0=None,
        extra_priors=None,
    ):

        (
            visit_args,
            star_args,
            planet_args,
            emcee_args,
            ultranest_args,
            read_file_status,
        ) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.ultranest_args,
            inpars.read_file_status,
        )

        aperture = visit_args["aperture"]

        # =====================
        # name of the file in pycheops_data
        file_key = visit_args["file_key"]

        # visit_folder
        main_folder = visit_args["main_folder"]
        visit_number = visit_args["visit_number"]
        shape = visit_args["shape"]

        logs_folder = os.path.join(main_folder, "logs")

        star_name = star_args["star_name"]

        nwalkers = emcee_args["nwalkers"]
        nprerun = emcee_args["nprerun"]
        nsteps = emcee_args["nsteps"]
        nburn = emcee_args["nburn"]
        nthin = emcee_args["nthin"]
        nthreads = emcee_args["nthreads"]
        progress = emcee_args["progress"]

        result_lin = M.fit_transit(
            T_0=T_0,
            P=P_ref,
            # TTV yes or not, default: ttv=False
            ttv=False,
            # default for decorrelation:
            unroll=visit_args["unroll"],
            nroll=visit_args["nroll"],
            # if you want to check the effect of roll angle or how it does without
            # unroll=False,
            unwrap=visit_args["unwrap"],  # defaul False
            D=new_params["D"],
            W=new_params["W"],
            b=new_params["b"],
            f_c=new_params["f_c"],
            f_s=new_params["f_s"],
            h_1=new_params["h_1"],
            h_2=new_params["h_2"],
            log_omega0=log_omega0,
            log_S0=log_S0,
            extra_priors=extra_priors,
            burn=emcee_args["nprerun"],
            steps=emcee_args["nsteps"],
            nwalkers=emcee_args["nwalkers"],
            progress=emcee_args["progress"],
            n_threads=emcee_args["nthreads"],
        )
        # WARNING: better have priors on shape and gp hyperparameters,
        # otherwise gp will try to fit also the transit!

        printlog("REPORT", olog=olog)
        printlog("{}".format(M.fit_report(min_correl=0.8)), olog=olog)

        printlog("PARAMETERS MULTIVISIT - LIN", olog=olog)
        par_med, _, par_mle, _ = pyca.get_best_parameters(
            result_lin, M, nburn=0, dataset_type="multivisit"
        )
        # updates params/parbest in result and M.result
        result_lin.params = par_med.copy()
        M.result.params = par_med.copy()
        result_lin.parbest = par_mle.copy()
        M.result.parbest = par_mle.copy()
        pyca.quick_save_params(
            os.path.join(main_folder, "params_med_lin.dat"), par_med, bjd_ref=cst.btjd
        )
        pyca.quick_save_params(
            os.path.join(main_folder, "params_mle_lin.dat"), par_mle, bjd_ref=cst.btjd
        )

        # bin30m_ph = bin30m/result_lin.params['P'].value
        bin30m_ph = False

        printlog("LC no-detrend plot", olog=olog)
        fig = M.plot_fit(
            title="Not detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=False,
        )
        for ext in fig_ext:
            plt_file = os.path.join(
                main_folder, "lcs_nodetrend_plot_lin.{}".format(ext)
            )
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("LC detrended plot", olog=olog)
        fig = M.plot_fit(
            title="Detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=True,
        )
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_detrend_plot_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        fig, out_lin = pyca.custom_plot_phase(M, result_lin, title="Fit lin. ephem.")
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_phased_plot_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)
        out_file = plt_file.replace(".png", ".dat")
        out = np.column_stack([v for v in out_lin.values()])
        head = "".join(["{:s} ".format(k) for k in out_lin.keys()])
        fmt = "%23.16e " * (len(out_lin) - 1) + "%03.0f"
        np.savetxt(out_file, out, header=head, fmt=fmt)

        printlog("Trace plot", olog=olog)
        fig = M.trail_plot(plotkeys="all", plot_kws={"alpha": 0.1})
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "trace_plot_all_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("Corner plot", olog=olog)
        gnames = pyca.global_names.copy()
        pk = []
        for n in gnames:
            if n in result_lin.params:
                if result_lin.params[n].vary:
                    pk.append(n)
        pk.append("T_0")
        printlog("{}".format(pk), olog=olog)
        fig = M.corner_plot(plotkeys=pk)
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "corner_plot_all_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        try:
            printlog("MASSRADIUS MULTIVISIT", olog=olog)
            _, fig = M.massradius(
                m_star=star_args["Mstar"],
                r_star=star_args["Rstar"],
                K=planet_args["Kms"],
                jovian=True,
                verbose=True,
            )
            for ext in fig_ext:
                plt_file = os.path.join(main_folder, "massradius_lin.{}".format(ext))
                fig.savefig(plt_file, bbox_inches="tight")
            plt.close(fig)
        except:
            printlog("massradius error: to be investigated", olog=olog)

        sys.stdout.flush()

        params_fit = result_lin.parbest.copy()
        printlog("\nRUNNING EMCEE - FIT TTV", olog=olog)
        sys.stdout.flush()

        if "log_S0" in params_fit:
            log_S0, log_omega0 = params_fit["log_S0"], params_fit["log_omega0"]
        else:
            log_S0, log_omega0 = None, None

        result_fit = M.fit_transit(
            T_0=params_fit["T_0"].value,
            P=params_fit["P"].value,
            # TTV yes or not, default: ttv=False
            ttv=True,
            # default for decorrelation:
            unroll=visit_args["unroll"],
            nroll=visit_args["nroll"],
            # if you want to check the effect of roll angle or how it does without
            # unroll=False,
            unwrap=visit_args["unwrap"],  # defaul False
            D=params_fit["D"],
            W=params_fit["W"],
            b=params_fit["b"],
            f_c=params_fit["f_c"],
            f_s=params_fit["f_s"],
            h_1=params_fit["h_1"],
            h_2=params_fit["h_2"],
            log_omega0=log_omega0,
            log_S0=log_S0,
            extra_priors=extra_priors,
            burn=emcee_args["nprerun"],
            steps=emcee_args["nsteps"],
            nwalkers=emcee_args["nwalkers"],
            progress=emcee_args["progress"],
        )

        printlog("REPORT", olog=olog)
        printlog("{}".format(M.fit_report(min_correl=0.8)), olog=olog)

        printlog("PARAMETERS MULTIVISIT - FIT", olog=olog)
        par_med, stats_med, par_mle, stats_mle = pyca.get_best_parameters(
            result_fit, M, nburn=0, dataset_type="multivisit"
        )
        # updates params/parbest in result and M.result
        result_fit.params = par_med.copy()
        M.result.params = par_med.copy()
        result_fit.parbest = par_mle.copy()
        M.result.parbest = par_mle.copy()
        pyca.quick_save_params(
            os.path.join(main_folder, "params_med_fit.dat"), par_med, bjd_ref=cst.btjd
        )
        pyca.quick_save_params(
            os.path.join(main_folder, "params_mle_fit.dat"), par_mle, bjd_ref=cst.btjd
        )

        printlog("LC no-detrend plot", olog=olog)
        fig = M.plot_fit(
            title="Not detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=False,
        )
        for ext in fig_ext:
            plt_file = os.path.join(
                main_folder, "lcs_nodetrend_plot_fit.{}".format(ext)
            )
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("LC detrended plot", olog=olog)
        fig = M.plot_fit(
            title="Detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=True,
        )
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_detrend_plot_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        fig, out_fit = pyca.custom_plot_phase(M, result_fit, title="Fit TTV")
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_phased_plot_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)
        printlog("{}".format(plt_file), olog=olog)
        printlog("{}".format(os.path.splitext(plt_file)), olog=olog)
        out_file = "{}.dat".format(os.path.splitext(plt_file)[0])
        out = np.column_stack([v for v in out_fit.values()])
        head = "".join(["{:s} ".format(k) for k in out_fit.keys()])
        fmt = "%23.16e " * (len(out_fit) - 1) + "%03.0f"
        np.savetxt(out_file, out, header=head, fmt=fmt)

        printlog("Trace plot", olog=olog)
        fig = M.trail_plot(plotkeys="all", plot_kws={"alpha": 0.1})
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "trace_plot_all_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("Corner plot", olog=olog)
        pk = []
        for n in result_fit.params:
            if result_fit.params[n].vary:
                if "ttv" in n or n in gnames:
                    pk.append(n)
        printlog("{}".format(pk), olog=olog)
        fig = M.corner_plot(plotkeys=pk)
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "corner_plot_all_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("MASSRADIUS MULTIVISIT", olog=olog)
        try:
            _, fig = M.massradius(
                m_star=star_args["Mstar"],
                r_star=star_args["Rstar"],
                K=planet_args["Kms"],
                jovian=True,
                verbose=True,
            )
            for ext in fig_ext:
                plt_file = os.path.join(main_folder, "massradius_fit.{}".format(ext))
                fig.savefig(plt_file, bbox_inches="tight")
            plt.close(fig)
        except:
            printlog("massradius error: to be investigated", olog=olog)
        sys.stdout.flush()

        bjdc = cst.btjd
        printlog("\nTTV SUMMARY", olog=olog)
        # extract single visit T_0
        printlog("Input linear ephem: {} + N x {}".format(T_ref, P_ref), olog=olog)
        Tr = T_ref - bjdc
        T0s, err_T0s = [], []
        epo_1, Tlin_1, oc_1 = [], [], []

        for m in M.datasets:
            tt = m.emcee.params["T_0"].value  # + m.bjd_ref - bjdc
            ett = m.emcee.params["T_0"].stderr
            T0s.append(tt)
            err_T0s.append(ett)

            epo = np.rint((tt - Tr.n) / P_ref.n)
            tl = Tr + epo * P_ref  # lep.linear_transit_time(Tr.n, P_ref.n, epo)
            epo_1.append(epo)
            Tlin_1.append(tl)
            # oc_1.append(ufloat(tt, ett) - tl)
            oc_1.append(tt - tl.n)

        T0s, err_T0s = np.array(T0s), np.array(err_T0s)
        epo_1, Tlin_1, oc_1 = (
            np.array(epo_1),
            np.array(Tlin_1),
            np.array(oc_1) * cst.day2sec,
        )

        # recompute new T_ref e P_ref from T0s
        epo_a, Tr_x, Pr_x, err_lin_x = lep.compute_lin_ephem(
            T0s, eT0=err_T0s, epoin=epo_1, modefit="wls"
        )

        Tr_a = ufloat(Tr_x, err_lin_x[0])
        Pr_a = ufloat(Pr_x, err_lin_x[1])
        printlog(
            "Updated with fitted T0s the input linear ephem: {} + N x {}".format(
                Tr_a, Pr_a
            ),
            olog=olog,
        )
        Tlin_a, oc_a = [], []

        Tr_b = ufloat(result_lin.params["T_0"].value, result_lin.params["T_0"].stderr)
        Pr_b = ufloat(result_lin.params["P"].value, result_lin.params["P"].stderr)
        printlog("Fitted linear ephem: {} + N x {}".format(Tr_b, Pr_b), olog=olog)
        epo_b, Tlin_b, oc_b = [], [], []

        oc_c = []

        for i, m in enumerate(M.datasets):
            tl_a = Tr_a + epo_a[i] * Pr_a
            Tlin_a.append(tl_a)
            # oc_a.append(ufloat(T0s[i], err_T0s[i]) - tl_a)
            oc_a.append(T0s[i] - tl_a.n)

            epox = lep.calculate_epoch(T0s[i], Tr_b.n, Pr_b.n)
            tl_b = Tr_b + epox * Pr_b
            epo_b.append(epox)
            Tlin_b.append(tl_b)
            # oc_b.append(ufloat(T0s[i], err_T0s[i]) - tl_b)
            oc_b.append(T0s[i] - tl_b.n)

            k = "ttv_{:02d}".format(i + 1)
            oc_c.append(ufloat(result_fit.params[k].value, result_fit.params[k].stderr))

        printlog("\nT0s {}".format(T0s), olog=olog)
        printlog("err_T0s {}".format(err_T0s), olog=olog)
        printlog("\nepo {}".format(epo_1), olog=olog)
        printlog("Tlin_1 {}".format(Tlin_1), olog=olog)
        printlog("oc_1 (s) {}".format(oc_1), olog=olog)
        epo_a, Tlin_a, oc_a = (
            np.array(epo_a),
            np.array(Tlin_a),
            np.array(oc_a) * cst.day2sec,
        )
        printlog("\nepo_a {}".format(epo_a), olog=olog)
        printlog("Tlin_a {}".format(Tlin_a), olog=olog)
        printlog("oc_a  (s) {}".format(oc_a), olog=olog)
        epo_b, Tlin_b, oc_b = (
            np.array(epo_b),
            np.array(Tlin_b),
            np.array(oc_b) * cst.day2sec,
        )
        printlog("\nepo_b {}".format(epo_b), olog=olog)
        printlog("Tlin_b {}".format(Tlin_b), olog=olog)
        printlog("oc_b  (s) {}".format(oc_b), olog=olog)
        oc_c = np.array(oc_c)
        printlog("\noc_c  (s) {}".format(oc_c), olog=olog)

        printlog("", olog=olog)
        sys.stdout.flush()

        t0_file = os.path.join(main_folder, "T0s_summary.dat")
        with open(t0_file, "w") as f:
            l = "# BJD_TDB - {}".format(bjdc)
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _1: input linear ephem = {:.6f} (+/- {:.6f}) + N x {:.6f} (+/- {:.6f})".format(
                Tr.n, Tr.s, P_ref.n, P_ref.s
            )
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _a linear ephem = {:.6f} (+/- {:.6f}) + N x {:.6f} (+/- {:.6f})".format(
                Tr_a.n, Tr_a.s, Pr_a.n, Pr_a.s
            )
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _b linear ephem = {:.6f} (+/- {:.6f}) + N x {:.6f} (+/- {:.6f})".format(
                Tr_b.n, Tr_b.s, Pr_b.n, Pr_b.s
            )
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _c fitted TTV (or O-C)"
            printlog(l, olog=olog)
            f.write(l + "\n")
            head = "# 0 epo_1 1 T_0_1 2 err_T_0_1 3 Tlin_1 4 unc_Tlin_1 5 oc_s_1"
            head += " 6 epo_a 7 Tlin_a 8 unc_Tlin_a 9 oc_s_a"
            head += " 10 epo_b 11 Tlin_b 12 unc_Tlin_b 13 oc_s_b"
            head += " 14 oc_s_c 15 err_oc_s_c"
            printlog(head, olog=olog)
            f.write(head + "\n")

            for i, e1 in enumerate(epo_1):

                l1 = "{:+05.0f} {:13.6f} {:+13.6f} {:13.6f} {:+13.6f} {:+13.6f}".format(
                    e1,
                    T0s[i],
                    err_T0s[i],
                    Tlin_1[i].n,
                    Tlin_1[i].s,
                    oc_1[i],
                )
                la = "{:+05.0f} {:13.6f} {:+13.6f} {:+13.6f}".format(
                    epo_a[i],
                    Tlin_a[i].n,
                    Tlin_a[i].s,
                    oc_a[i],
                )
                lb = "{:+05.0f} {:13.6f} {:+13.6f} {:+13.6f}".format(
                    epo_b[i],
                    Tlin_b[i].n,
                    Tlin_b[i].s,
                    oc_b[i],
                )
                lc = "{:+13.6f} {:+13.6f}".format(oc_c[i].n, oc_c[i].s)
                l = "{} {} {} {}".format(l1, la, lb, lc)
                printlog(l, olog=olog)
                f.write(l + "\n")

        print("Plotting O-C...")
        file = np.genfromtxt(t0_file)

        epochs = file[:, 1]

        ocs = file[:, 14]
        ocs_err = file[:, 15]

        plt.errorbar(
            epochs,
            ocs,
            yerr=ocs_err,
            fmt="ko",
        )
        plt.axhline(0, color="firebrick", linestyle="--", linewidth=2)
        plt.xlabel("BJD_TDB - 2457000")
        plt.ylabel("O - C (s)")
        plt.savefig(os.path.join(main_folder, "OC_plot.pdf"))

        return result_lin, result_fit

    def ultranest(
        self,
        inpars=None,
        M=None,
        olog=None,
        new_params=None,
        T_0=None,
        T_ref=None,
        P_ref=None,
        log_omega0=None,
        log_S0=None,
        extra_priors=None,
    ):

        (
            visit_args,
            star_args,
            planet_args,
            emcee_args,
            ultranest_args,
            read_file_status,
        ) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.ultranest_args,
            inpars.read_file_status,
        )

        aperture = visit_args["aperture"]

        # =====================
        # name of the file in pycheops_data
        file_key = visit_args["file_key"]

        # visit_folder
        main_folder = visit_args["main_folder"]
        visit_number = visit_args["visit_number"]
        shape = visit_args["shape"]

        logs_folder = os.path.join(main_folder, "logs")

        star_name = star_args["star_name"]

        live_points = ultranest_args["live_points"]
        tolerance = ultranest_args["tol"]
        cluster_num_live_points = ultranest_args["cluster_num_live_points"]
        logdir = os.path.join(main_folder, "test_ultranest")
        resume = ultranest_args["resume"]
        adaptive_nsteps = ultranest_args["adaptive_nsteps"]

        # Run emcee from last best fit
        printlog("\n-Run Ultranest from last best fit with:", olog=olog)
        printlog(" live_points              = {}".format(live_points), olog=olog)
        printlog(" tolerance                = {}".format(tolerance), olog=olog)
        printlog(
            " cluster_num_live_points  = {}".format(cluster_num_live_points),
            olog=olog,
        )
        printlog(
            " logdir                   = {}".format(logdir),
            olog=olog,
        )
        printlog(
            " resume                   = {}".format(resume),
            olog=olog,
        )
        printlog("", olog=olog)

        sampler = M.fit_transit_ultranest(
            T_0=T_0,
            P=P_ref,
            # TTV yes or not, default: ttv=False
            ttv=False,
            # default for decorrelation:
            unroll=visit_args["unroll"],
            nroll=visit_args["nroll"],
            # if you want to check the effect of roll angle or how it does without
            # unroll=False,
            unwrap=visit_args["unwrap"],  # defaul False
            D=new_params["D"],
            W=new_params["W"],
            b=new_params["b"],
            f_c=new_params["f_c"],
            f_s=new_params["f_s"],
            h_1=new_params["h_1"],
            h_2=new_params["h_2"],
            log_omega0=log_omega0,
            log_S0=log_S0,
            extra_priors=extra_priors,
            live_points=live_points,
            tol=tolerance,
            cluster_num_live_points=cluster_num_live_points,
            logdir=logdir,
            resume=resume,
            adaptive_nsteps=adaptive_nsteps,
        )

        printlog("Plotting run", olog=olog)
        sampler.plot_run()

        printlog("Plotting traces", olog=olog)
        sampler.plot_trace()

        printlog("Generating corner plot", olog=olog)
        sampler.plot_corner()

        result_lin, result_fit = [], []

        return result_lin, result_fit
