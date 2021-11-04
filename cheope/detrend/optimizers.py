import numpy as np
import matplotlib.pyplot as plt
import cheope.pycheops_analysis as pyca
import os
import json
from pathlib import Path
from uncertainties import ufloat

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
        visit_name = "visit_{:02d}_{:s}_{:s}_shape_ap{:s}_BF".format(
            visit_number, file_key, shape.lower(), aperture.upper()
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
        visit_name = "visit_{:02d}_{:s}_{:s}_shape_ap{:s}_BF".format(
            visit_number, file_key, shape.lower(), aperture.upper()
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

        # TODO use params_lm_loop to update and create params_med and params_mle, the
        # update is going to use the info/result.json file]

        # print(sampler.paramnames)
        # print(type(sampler.paramnames))
        # print(sampler.derivedparamnames)
        # print(type(sampler.derivedparamnames))

        printlog("Plotting run", olog=olog)
        sampler.plot_run()

        printlog("Plotting traces", olog=olog)
        sampler.plot_trace()

        printlog("Generating corner plot", olog=olog)
        sampler.plot_corner()

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )

        result_json_path = os.path.join(logdir, "info/results.json")
        results = json.load(open(result_json_path))

        printlog("Saving MEDIAN PARAMETERS", olog=olog)
        pyca.quick_save_params_ultra(
            os.path.join(visit_folder.resolve(), "02_params_ultranest_median.dat"),
            planet_args,
            star_args,
            results,
            dataset.lc["bjd_ref"],
            mod="median",
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
