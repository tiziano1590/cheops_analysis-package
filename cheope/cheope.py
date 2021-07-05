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
    from cheope.detrend import SingleBayes
        
    parser = argparse.ArgumentParser(description='Cheope')
    
    parser.add_argument("-i", "--input", dest='input_file', type=str,
                        required=True, help="Input par file to pass")
    
    parser.add_argument("-sb", "--single-bayes", dest="single_bayes", default=False,
                        help="When set, runs single visit detrending to compute the Bayes Factor" + 
                        "as a function of the models and their parameters",
                        action="store_true")



    args = parser.parse_args()
    
    print('Cheope PROGRAM STARTS AT %s' % datetime.datetime.now())

    if args.single_bayes:
        sb = SingleBayes(args.input_file)
        sb.run_analysis()


    print('Cheope PROGRAM FINISHES AT %s' % datetime.datetime.now())

if __name__ == "__main__":
    main()