import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify
from datetime import timedelta, datetime

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

def datetimeify(t):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    from pandas._libs.tslibs.timestamps import Timestamp
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S',]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))
    
def forecast_dec2019():
    ''' forecast model for Dec 2019 eruption
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData()
        
    # construct model object
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75, 
        look_forward=2., data_streams=data_streams)
    
    # columns to manually drop from feature matrix because they are highly correlated to other 
    # linear regressors
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once 
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    te = td.tes[-1]
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=False, 
        exclude_dates=[[te-month,te+month],], n_jobs=n_jobs)      

    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys = fm.forecast(ti='2011-01-01', tf='2020-01-01', recalculate=True, n_jobs=n_jobs)    

    # plot forecast and quality metrics
    # plots will be saved to ../plots/*root*/
    fm.plot_forecast(ys, threshold=0.8, xlim = [te-month/4., te+month/15.], 
        save=r'{:s}/forecast.png'.format(fm.plotdir))
    fm.plot_accuracy(ys, save=r'{:s}/accuracy.png'.format(fm.plotdir))

    # construct a high resolution forecast (10 min updates) around the Dec 2019 eruption
    # note: building the feature matrix might take a while
    fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=te+month/30, recalculate=True, 
        save=r'{:s}/forecast_hires.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_test():
    ''' test scale forecast model
    '''
    # constants
    month = timedelta(days=365.25/12)
        
    # set up model
    #data_streams = ['log_zsc2_rsamF', 'zsc2_mfF']#, 'zsc2_hfF']#['log_zsc2_rsamF']#,'mf','hf','dsar']
    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF','diff_zsc2_rsamF','diff_zsc2_mfF','diff_zsc2_hfF','diff_zsc2_dsarF',
        'log_zsc2_rsamF','log_zsc2_mfF','log_zsc2_hfF','log_zsc2_dsarF']
    fm = ForecastModel(ti=None, tf=None, station='BELO', window=2., overlap=0.75, 
        look_forward=2., data_streams=data_streams, root='test', savefile_type='pkl')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6
    
    # train the model
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    #drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
    #    '*attr_"angle"*']  
    #freq_max = fm.dtw//fm.dt//4
    #drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]

    fm.train(ti=None, tf=None, drop_features=drop_features, retrain=True,
        n_jobs=n_jobs)      

    # plot a forecast for a future eruption
    # tf = te+month/30
    # fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=True, 
    #     save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

    te = fm.data.tes[1]
    #y = load_dataframe(r'D:\code\whakaari\predictions\test_hires\DecisionTreeClassifier_0000.pkl')
    #tf = y.index[-1] + month/30./10.
    #ti=te-fm.dtw-fm.dtf
    ti=te - month/2
    tf= ti + month
    fm.hires_forecast(ti=ti, tf=tf, recalculate=True, 
        save=r'{:s}/forecast_.png'.format(fm.plotdir), n_jobs=n_jobs)

def extract_all():
    ''' Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
        Overlap is set to 1.0 (max) 
    '''
    import time
    from functools import partial
    from multiprocessing import Pool
    
    ## data streams
    ds = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF','diff_zsc2_rsamF','diff_zsc2_mfF','diff_zsc2_hfF','diff_zsc2_dsarF',
        'log_zsc2_rsamF','log_zsc2_mfF','log_zsc2_hfF','log_zsc2_dsarF']
    ds = ['log_zsc2_rsamF', 'zsc2_hfF']
    ## stations
    ss = ['KRVZ','FWVZ','WIZ']
    ss = ['PV6']
    ## window sizes (days)
    ws = [2.] #, 14., 90., 365.]

    ## Run parallelization 
    ps = []
    for s in ss:
        for d in ds:
            for w in ws:
                ps.append([w,s,d])
    n_jobs = 5 # number of cores
    p = Pool(n_jobs)
    p.map(extract_one, ps)

def extract_one(p):
    ''' Load data from a certain station, window size, and datastream (auxiliary function for parallelization)
    p = [window size, station, datastream] '''
    w,s,d = p
    if w == 14.:
        o = 1.-6./(w*24*6)
    elif w == 90.:
        o = 1.-6.*6./(w*24*6)
    elif w == 365.:
        o = 1.-24.*6./(w*24*6)
    fm = ForecastModel(window=w, overlap=o, station = s,
        look_forward=2., data_streams=[d], feature_dir=r'U:\EruptionForecasting\eruptions\features\\', savefile_type='pkl') 
    fm._load_data(datetimeify(fm.ti_model), datetimeify(fm.tf_model), None)

def forecast_now():
    ''' forecast model for present day 
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
        
    # pull the latest data from GeoNet
    td = TremorData()
    td.update()

    # model from 2011 to present day (td.tf)
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75,  
        look_forward=2, data_streams=data_streams, root='online_forecaster')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6
    
    # The online forecaster is trained using all eruptions in the dataset. It only
    # needs to be trained once, or again after a new eruption.
    # (Hint: feature matrices can be copied from other models to avoid long recalculations
    # providing they have the same window length and data streams. Copy and rename 
    # to *root*_features.csv)
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, 
        retrain=True, n_jobs=n_jobs)      
    
    # forecast the last 7 days at high resolution
    fm.hires_forecast(ti=fm.data.tf - 7*day, tf=fm.data.tf, recalculate=True, 
        save='current_forecast.png', nztimezone=True, n_jobs=n_jobs)  

if __name__ == "__main__":
    #forecast_dec2019()
    forecast_test()
    #extract_all()
    #forecast_now()
    