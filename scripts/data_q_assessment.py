import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
import time
from functools import partial
from multiprocessing import Pool

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

def import_data():
    if False: # plot raw vel data
        from obspy.clients.fdsn import Client
        client = Client("IRIS")
        from obspy import UTCDateTime
        #t = UTCDateTime("2012-02-27T00:00:00.000")
        starttime = UTCDateTime("2014-01-28")
        endtime = UTCDateTime("2014-01-30")
        inventory = client.get_stations(network="AV", station="SSLW", starttime=starttime, endtime=endtime)
        st = client.get_waveforms(network = "AV", station = "SSLW", location = None, channel = "EHZ", starttime=starttime, endtime=endtime)
        st.plot()  
        asdf

    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    t0 = "2013-07-01"
    t1 = "2019-12-31"
    td = TremorData(station = 'SSLW')
    td.update(ti=t0, tf=t1)
    #td.update()

def data_Q_assesment():
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    station = 'SSLW'
    # read raw data
    td = TremorData(station = station)
    #t0 = "2007-08-22"
    #t1 = "2007-09-22"
    #td.update(ti=t0, tf=t1)

    # plot data 
    #td.plot( data_streams = ['rsamF'])#(ti=t0, tf=t1)
    t0 = "2009-03-10"
    t1 = "2009-03-16"
    #td.update(ti=t0, tf=t1)
    data_streams = ['rsamF','mfF','hfF']
    td.plot_zoom(data_streams = data_streams, range = [t0,t1])

    # interpolated data 
    #t0 = "2015-01-01"
    #t1 = "2015-02-01"
    td.plot_intp_data(range_dates = None)
    #td.plot_intp_data(range_dates = [t0,t1])

def calc_feature_pre_erup():
    ''' Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
        Overlap is set to 1.0 (max) 
    '''
    def extract_one(p):
        ''' Load data from a certain station, window size, and datastream (auxiliary function for parallelization)
        p = [window size, station, datastream] '''
        w,s,d = p
        #fm = ForecastModel(window=w, overlap=1., station = s,
        #    look_forward=2., data_streams=[d], feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl') 
        fm = ForecastModel(window=w, overlap=1., station = s,
            look_forward=2., data_streams=[d], savefile_type='csv') 
        fm._load_data(datetimeify(fm.ti_model), datetimeify(fm.tf_model), None)

    ## data streams
    #ds = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF','diff_zsc2_rsamF','diff_zsc2_mfF','diff_zsc2_hfF','diff_zsc2_dsarF',
    #    'log_zsc2_rsamF','log_zsc2_mfF','log_zsc2_hfF','log_zsc2_dsarF']
    ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
    ## stations
    ss = ['PVV','VNSS','SSLW','OKWR','REF','BELO','CRPO','VTUN','KRVZ','FWVZ','WIZ','AUR']
    ss = ['PVV','VNSS','SSLW','BELO','KRVZ','FWVZ','WIZ']
    ## window sizes (days)
    ws = [2.]
    ## Run parallelization 
    ps = []
    for s in ss:
        for d in ds:
            for w in ws:
                ps.append([w,s,d])
    n_jobs = 4 # number of cores
    p = Pool(n_jobs)
    p.map(extract_one, ps)


if __name__ == "__main__":
    import_data()
    #data_Q_assesment()
    #calc_feature_pre_erup()
