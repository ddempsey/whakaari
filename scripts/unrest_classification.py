import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import ForecastModel, TremorData, datetimeify
from datetime import timedelta
from inspect import getfile, currentframe

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
import pandas as pd

_MONTH = timedelta(days=365.25/12)

# Modified TremorData that also reads list of unrest periods (similar format to eruptions)
class TremorDataUnrest(TremorData):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        # get unrest periods
        with open(os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','{:s}_unrest_periods.txt'.format(self.station)]),'r') as fp:
            self.unrest = [[datetimeify(lni.rstrip()) for lni in ln.split(',')] for ln in fp.readlines()]

# Modified ForecastModel that assigns positive integer labels to unrest periods, -1 integer label
# to pre-eruption period
class ForecastModelUnrest(ForecastModel):
    ''' Derived class with modified training to incorporate unrest.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = TremorDataUnrest(parent=self)
        if any(['_' in ds for ds in self.data_streams]):
            self.data._compute_transforms()
    def _get_label(self, ts):
        """ Compute multi-class label vector.
            -1 = pre-eruption
            0 = non-eruptive
            i = unrest period i
        """
        label = -np.array(super()._get_label(ts))
        ts = pd.to_datetime(ts)
        for i,(u0,u1) in enumerate(self.data.unrest):
            label[np.where((ts>u0)&(ts<u1))] = i+1
        return label

def unrest_classification(u,e):
    # standard filtered, normalised model
    data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF','zsc2_dsarF']
    fm = ForecastModelUnrest(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='unrest'.format(u,e), savefile_type='pkl', station='WIZ',
            feature_dir=r'E:\whakaari\features')   

    # exclude selected eruption and unrest period
    n_jobs = 8
    te = fm.data.tes[e-1]
    u0,u1 = fm.data.unrest[u-1]
    exclude_dates = [[te-_MONTH,te+_MONTH],[u0-_MONTH,u1+_MONTH]]

    # exclude features
    drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
        '*attr_"angle"*']  
    freq_max = fm.dtw//fm.dt//4
    drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]    

    # train model with multi-label problem
    # samples 25% from: each two unrest periods in training set, all four pre-eruption in training set,
    # non-unrest/eruptive data
    fm.train(drop_features=drop_features, retrain=True, Ncl=500, n_jobs=n_jobs,
        exclude_dates=exclude_dates,  method='not minority')

    # forecast on unseen eruption
    fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=te+_MONTH/28., recalculate=True, 
        save=r'{:s}/eruption_u{:d}_e{:d}_hires.png'.format(fm.plotdir,u,e), 
        n_jobs=n_jobs, root=r'{:s}_u{:d}_e{:d}_hires'.format(fm.root,u,e), threshold=1.0)

    # forecast on unseen unrest period
    fm.hires_forecast(ti=u0-_MONTH/2., tf=u1+_MONTH/2., recalculate=True, 
        save=r'{:s}/unrest_u{:d}_e{:d}_hires.png'.format(fm.plotdir,u,e), 
        n_jobs=n_jobs, root=r'{:s}_u{:d}_hires'.format(fm.root,u), threshold=1.0)

def main():
    # train models to classify unrest and test on pre-eruption data

    # for each unrest period
    for u in range(1,4):
        # for each eruption
        for e in range(1,6):
            # generate prediction model
            unrest_classification(u=u, e=e)

if __name__ == "__main__":
    main()