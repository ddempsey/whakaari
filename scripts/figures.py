import numpy as np
from matplotlib import pyplot as plt

import os, sys, shutil, gc
sys.path.insert(0, os.path.abspath('..'))
from whakaari import ForecastModel
from datetime import timedelta, datetime

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
from tsfresh.utilities.dataframe_functions import impute
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

_MONTH = timedelta(days=365.25/12)

def fig_2016eruption():
    # setup forecast model
    n_jobs = 4  
    # i = 3
    for i in [0,1,2,4]:
        data_streams = ['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF']
        fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
            root='transformed_e{:d}'.format(i+1), savefile_type='pkl', station='WIZ',
            feature_dir=r'E:\whakaari\features')   
        #fm.data.update()

        # train-test split on five eruptions to compute model confidence of an eruption
            # remove duplicate linear features (because correlated), unhelpful fourier compoents
            # and fourier harmonics too close to Nyquist
        drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
            '*attr_"angle"*']  
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]

        # train a model with data from that eruption excluded
        te = fm.data.tes[i]
        exclude_dates = [[te-_MONTH, te+_MONTH]]
        fm.train(drop_features=drop_features, retrain=False, Ncl=500, n_jobs=n_jobs, exclude_dates=exclude_dates)        
        
        # test the model by constructing a hires forecast for excluded eruption
        ys = fm.hires_forecast(ti=te-_MONTH, tf=te+_MONTH, recalculate=True, 
            save=r'{:s}/hires.png'.format(fm.plotdir), 
            n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)

        # save the largest value of model consensus in the 48 hours prior
        y = ys['consensus']
        ci = fm._compute_CI(y)
        y0 = y-ci                   # take lower bound of 95% conf. int. for conservatism
        inds = (y.index<(te-fm.dt))&(y.index>(te-fm.dtf))
        conf, conf0 = y[inds].max(), y0[inds].max()
        with open(r'{:s}/confidence.txt'.format(fm.plotdir), 'w') as fp:
            fp.write('{:4.3f}, {:4.3f}'.format(conf, conf0))

if __name__ == "__main__":
    fig_2016eruption()