import os, sys, shutil
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, datetimeify, save_dataframe, load_dataframe
from datetime import timedelta, datetime
from functools import partial
from multiprocessing import Pool
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
        
# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


def repair_WIZ():
    # repair window
    t0,t1 = [datetimeify(ti) for ti in ['2021-05-23','2021-05-24']]

    # constants
    td = TremorData()
    shutil.copyfile(td.file, td.file+'.bkp')    # make a backup!
    td.update(t0,t1)

if __name__ == "__main__":
    repair_WIZ()
    