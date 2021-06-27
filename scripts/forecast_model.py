import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe
from datetime import timedelta

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
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2012-07-15', tf='2012-08-16', window=2., overlap=0.75, 
        look_forward=2., data_streams=data_streams, root='test', savefile_type='pkl')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 1
    
    # train the model
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2012-07-15', tf='2012-08-16', drop_features=drop_features, retrain=False,
        n_jobs=n_jobs)      

    # plot a forecast for a future eruption
    te = fm.data.tes[1]
    y = load_dataframe(r'D:\code\whakaari\predictions\test_hires\DecisionTreeClassifier_0000.pkl')
    tf = y.index[-1] + month/30./10.
    fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=False, 
        save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

def download_tremor():
    td = TremorData(station='WIZ')
    td.update(n_jobs=6)

if __name__ == "__main__":
    # test problems to check your installation is working correctly
    # download_tremor()
    forecast_test()
    