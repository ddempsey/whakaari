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

class TremorDataUnrest(TremorData):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        # get unrest periods
        with open(os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','{:s}_unrest_periods.txt'.format(self.station)]),'r') as fp:
            self.unrest = [[datetimeify(lni.rstrip()) for lni in ln.split(',')] for ln in fp.readlines()]

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

def unrest_classification():
    data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF','zsc2_dsarF']
    fm = ForecastModelUnrest(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='unrest', savefile_type='pkl', station='WIZ')   

    # exclude final eruption and first unrest period
    n_jobs = 1
    te = fm.data.tes[4]
    exclude_dates = [[te-_MONTH,te+_MONTH],fm.data.unrest[0]]

    drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
        '*attr_"angle"*']  
    freq_max = fm.dtw//fm.dt//4
    drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]    

    fm.train(drop_features=drop_features, retrain=False, Ncl=500, n_jobs=n_jobs,
        exclude_dates=exclude_dates,  method='not minority')

    # fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=te+_MONTH/28., recalculate=True, 
    #     save=r'{:s}/hires.png'.format(fm.plotdir), 
    #     n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)

    u1 = fm.data.unrest[0][-1]
    fm.hires_forecast(ti=u1-_MONTH*0.75, tf=u1+_MONTH/4., recalculate=True, 
        save=r'{:s}/unrest_hires.png'.format(fm.plotdir), 
        n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)

def main():
    unrest_classification()

if __name__ == "__main__":
    main()