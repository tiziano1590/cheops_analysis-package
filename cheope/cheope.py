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
    import matplotlib.pyplot as plt
    from cheope.detrend import (
        SingleBayes,
        SingleBayesKeplerTess,
        SingleCheck,
        MultivisitAnalysis,
    )

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

    args = parser.parse_args()

    print("Cheope PROGRAM STARTS AT %s" % datetime.datetime.now())

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

    print("Cheope PROGRAM FINISHES AT %s" % datetime.datetime.now())


if __name__ == "__main__":
    main()
