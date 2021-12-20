"""
Created on Mon Jul 5 27:20:55 2021

The main CHEOPE program

@author: Tiziano Zingales - Luca Borsato - INAF/Università di Padova
"""


def main():
    import argparse
    import datetime
    import time
    import numpy as np
    import glob
    import matplotlib
    import matplotlib.pyplot as plt
    from multiprocessing.pool import Pool
    from cheope.detrend import (
        SingleBayes,
        SingleBayesKeplerTess,
        SingleBayesASCII,
        SingleCheck,
        CheckEphemerids,
        MultivisitAnalysis,
    )
    from cheope.dace import DACESearch
    from cheope.tess import TESSSearch, ReadFits

    matplotlib.use("Agg")

    global check_gen_file

    parser = argparse.ArgumentParser(description="Cheope")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-ce",
        "--check-ephemerids",
        dest="check_ephemerids",
        default=False,
        help="Checks the observing time interval starting from the period and the Tref",
        action="store_true",
    )

    parser.add_argument(
        "-sb",
        "--single-bayes",
        dest="single_bayes",
        default=False,
        help="When set, runs single visit detrending to compute the Bayes Factor"
        + "as a function of the models and their parameters",
        action="store_true",
    )

    parser.add_argument(
        "-skt",
        "--single-kepler-tess",
        dest="single_kepler_tess",
        default=False,
        help="When set, runs single visit detrending to compute the Bayes Factor"
        + "as a function of the models and their parameters. Including Kepler/Tess points.",
        action="store_true",
    )

    parser.add_argument(
        "-sc",
        "--single-check",
        dest="single_check",
        default=False,
        help="When set, runs a check of the input dataset and provides a basic analysis",
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--multivisit",
        dest="multivisit",
        default=False,
        help="When set, runs a multivisit analysis for the datasets provided",
        action="store_true",
    )

    parser.add_argument(
        "-a",
        "--ascii",
        dest="ascii",
        default=False,
        help="When set, runs an analysis from a generic ascii files containing the flux",
        action="store_true",
    )

    parser.add_argument(
        "--selenium-dace",
        dest="selenium_dace",
        default=False,
        help="search automatically the latest information for the target from CHEOPS",
        action="store_true",
    )

    parser.add_argument(
        "--add-single-check",
        dest="add_sc",
        default=False,
        help="add Single Check for each of the new observations",
        action="store_true",
    )

    parser.add_argument(
        "--selenium-tess",
        dest="selenium_tess",
        default=False,
        help="search automatically the latest information for the target from TESS",
        action="store_true",
    )

    parser.add_argument(
        "--download",
        dest="download",
        default=False,
        help="if called, it navigate through the TESS viewing tool to download all the target lightcurves",
        action="store_true",
    )

    parser.add_argument(
        "-add-skt",
        "--add-single-kepler-tess",
        dest="add_skt",
        default=False,
        help="add Single Bayes analysis for each of the new observations",
        action="store_true",
    )

    parser.add_argument(
        "--read-fits",
        dest="read_fits",
        default=False,
        help="Read input TESS lightcurve",
        action="store_true",
    )

    args = parser.parse_args()

    print("Cheope PROGRAM STARTS AT %s" % datetime.datetime.now())

    if args.check_ephemerids:
        ce = CheckEphemerids(args.input_file)
        ce.plot_lightcurve()

    if args.single_check:
        sc = SingleCheck(args.input_file)
        sc.run()

    if args.single_bayes:
        sb = SingleBayes(args.input_file)
        sb.run()

    if args.single_kepler_tess:
        skt = SingleBayesKeplerTess(args.input_file)
        skt.run()

    if args.multivisit:
        multi = MultivisitAnalysis(args.input_file)
        multi.run()

    if args.ascii:
        multi = SingleBayesASCII(args.input_file)
        multi.run()

    if args.selenium_dace:
        search = DACESearch(args.input_file)
        keywords = search.get_observations()

        def check_gen_file(num, keyword):
            infile = search.substitute_file_key(keyword, num + 1)

            if args.add_sc:
                sb = SingleCheck(infile)
                sb.run()

        process_pool = Pool()
        data = list(enumerate(keywords))
        process_pool.starmap(check_gen_file, data)

    if args.selenium_tess and args.read_fits:
        fits = ReadFits(args.input_file)
        fits.plot_lightcurve()

    elif args.selenium_tess:
        search = TESSSearch(args.input_file)
        keywords = search.get_observations(download=args.download)

        def check_gen_file(num, keyword):
            infile = search.substitute_file_key(keyword, num + 1)

            if args.add_skt:
                sb = SingleBayesKeplerTess(infile)
                sb.run()

        process_pool = Pool()
        data = list(enumerate(keywords))
        process_pool.starmap(check_gen_file, data)

    print("Cheope PROGRAM FINISHES AT %s" % datetime.datetime.now())


if __name__ == "__main__":
    main()
