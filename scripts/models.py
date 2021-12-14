
import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import ForecastModel, TremorData, datetimeify
from datetime import timedelta

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

import numpy as np
from matplotlib import pyplot as plt

_MONTH = timedelta(days=365.25/12)
FEATURE_DIR = r'E:\whakaari\features'
FEATURE_DIR = r'/media/eruption_forecasting/eruptions/features'
DATA_DIR = r'/media/eruption_forecasting/eruptions/data'

def evaluation():
    # setup forecast model
    n_jobs = 8  
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams, 
        savefile_type='pkl', station='WIZ', feature_dir=FEATURE_DIR, data_dir=DATA_DIR)   

    # train a model with all eruptions
    drop_features = ['linear_trend_timewise','agg_linear_trend']  
    fm.train(tf='2020-01-01', drop_features=drop_features, retrain=False, Ncl=500, n_jobs=n_jobs)        
    
    # test the model by constructing a hires forecast for excluded eruption
    ys = fm.hires_forecast(ti=datetimeify('2020-02-01'), tf = fm.data.df.index[-1], recalculate=False, 
        save=r'{:s}/evaluation_hires.png'.format(fm.plotdir), 
        n_jobs=n_jobs, root=r'benchmark_e0_hires'.format(fm.root), threshold=0.8)

def reliability(root, data_streams, eruption, Ncl, eruption2=None):
    # setup forecast model
    n_jobs = 8 
    root = '{:s}_e{:d}'.format(root, eruption)
    if eruption2 is not None:
        root += '_p{:d}'.format(eruption2)
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root=root, savefile_type='pkl', station='WIZ',
        feature_dir=FEATURE_DIR, data_dir=DATA_DIR)   

    # train-test split on five eruptions to compute model confidence of an eruption
        # remove duplicate linear features (because correlated), unhelpful fourier compoents
        # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend']  
    if root is not 'benchmark':
        drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    
    # train a model with data from that eruption excluded
    te = fm.data.tes[eruption-1]
    exclude_dates = [[te-_MONTH, te+_MONTH]]
    if eruption2 is not None:
        te = fm.data.tes[eruption2-1]
        exclude_dates.append([te-_MONTH, te+_MONTH])
    fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for excluded eruption
    tf = te+_MONTH/28.
    if eruption == 3:
        tf = te+_MONTH/28.*15
    ys = fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=tf, recalculate=True, 
        save=r'{:s}/hires.png'.format(fm.plotdir), 
        n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)

def discriminability(root, data_streams, Ncl, eruption=None):
    # setup forecast model
    n_jobs =16 
    root = '{:s}_e0'.format(root)
    if eruption is not None:
        root += '{:s}_e{:d}_p0'.format(root, eruption)
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root=root, savefile_type='pkl', station='WIZ',
        feature_dir=FEATURE_DIR, data_dir=DATA_DIR, ti=datetimeify('2011-01-01'))   

    # remove duplicate linear features (because correlated), unhelpful fourier compoents
    # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend']  
    if root is not 'benchmark':
        drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    
    # construct hires model over entire dataset to compute false alarm rate
    exclude_dates = []
    if eruption is not None:
        te = fm.data.tes[eruption-1]
        exclude_dates = [[te-_MONTH, te+_MONTH]]
    fm.train(drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # forecast over whole dataset
    ys = fm.hires_forecast(ti=fm.ti_train, tf=fm.tf_train, recalculate=True, 
        n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)    

def model(root, data_streams, Ncl=100):
    # assess reliability by cross validation on five eruptions
    for eruption in range(1,6):
        reliability(root, data_streams, eruption, Ncl)

    # assess discriminability by high-resoultion simulation across dataset
    discriminability(root, data_streams, Ncl)

def calibration(root, data_streams, Ncl=100):
    # create sub-models for probability calibration
    for eruption in range(1,6)[::-1]:
        print(eruption)
        discriminability(root, data_streams, Ncl, eruption)
        continue
        for eruption2 in range(1,6):
            if eruption == eruption2:
                continue
            print(eruption,eruption2)
            reliability(root, data_streams, eruption, Ncl, eruption2)

def main():
    # # update data
    # td = TremorData()
    # td.update()

    # # model 0: evaluation of operational forecaster over 18 months
    #evaluation()
    # data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
    # model(root='transformed2',data_streams=data_streams, Ncl=500)

    # # model 1: benchmark from 2020 Nat Comms paper (100 classifiers)
    # data_streams = ['rsam','mf','hf','dsar']
    # model('benchmark', data_streams, Ncl=100)

    # # model 2: model 1 with regional earthquakes filtered (500 classifiers)
    # data_streams = ['rsamF','mfF','hfF','dsarF']
    # model(root='filtered',data_streams=data_streams, Ncl=500)
    
    # model 3: model 2 with data standardisation (500 classifiers)
    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
    # model(root='transformed2',data_streams=data_streams, Ncl=500)
    calibration(root='transformed2',data_streams=data_streams, Ncl=500)

if __name__ == "__main__":
    main()
