import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify
from datetime import timedelta, datetime
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
    ds + ['zsc2_vlfF','zsc2_lfF','zsc2_vlarF','zsc2_lrarF','zsc2_rmarF','diff_zsc2_vlfF','diff_zsc2_lfF','diff_zsc2_vlarF','diff_zsc2_lrarF',
        'diff_zsc2_rmarF','log_zsc2_vlfF','log_zsc2_lfF','log_zsc2_vlarF','log_zsc2_lrarF','log_zsc2_rmarF']
    #ds = ['log_zsc2_rsamF', 'zsc2_hfF']
    ## stations
    ss = ['KRVZ','FWVZ','WIZ','PV6','PVV','BELO','OKWR','VNSS','SSLW','CRPO','REF']
    #ss = ['PV6']
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
    if w==14.:
        o=1.-6./(w*24*6)
    elif w==90.:
        o=1.-6*6./(w*24*6)
    elif w==365.:
        o=1.-24.*6./(w*24*6)
    else:
        o=1.
    print(w, o, d, s)

    #fm = ForecastModel(window=w, overlap=1., station = s,
    #    look_forward=2., data_streams=[d], feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl') 
    fm = ForecastModel(window=w, overlap=o, station = s,
        look_forward=2., data_streams=[d], feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl')
    fm._load_data(datetimeify(fm.ti_model), datetimeify(fm.tf_model), None)

def download_all():
    from datetime import timedelta
    stations = ['PV6','PVV','BELO','SSLW','REF']
    for station in stations:
        td = TremorData(station=station)
        ti = td._probe_start()
        N
        # for i in range()
        #     td.update(ti, ti+timedelta(days=3), n_jobs=3)

def probe():
    # from obspy.clients.fdsn import Client as FDSNClient 
    # from obspy import UTCDateTime, read_inventory 
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #     from obspy.signal.filter import bandpass
    # from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
    # from obspy.clients.fdsn.header import FDSNNoDataException

    # station = 'PVV'

    # client = FDSNClient("IRIS")    
    # channel = 'HHZ'
    # if station in ['KRVZ','PV6','PVV','OKWR','VNSS','SSLW','REF']:
    #     channel = 'EHZ'
    # site = client.get_stations(station=station, level="response", channel=channel)
    # print(site)
    td=TremorData('PVV')
    td.update()

    # td=TremorData(station=station)
    # td._probe_start()
    from pickle import load
    with open('st.pkl','rb') as fp:
        st,site = load(fp)
    ax = plt.subplots(1,1)[1]
    st.remove_sensitivity(inventory=site)
    st.detrend('linear')
    st.interpolate(100).merge(fill_value='interpolate')
    st.taper(max_percentage=0.05, type="hann")
    st.filter('bandpass', freqmin=8, freqmax=16)
    i0=int(0.2*24*3600*100)
    i1=int(24*3600*100)
    ax.plot(st.traces[0].data[i0:i0+i1])
    plt.show()


if __name__ == "__main__":
    #forecast_dec2019()
    #forecast_test()
    download_all()
    # extract_all()
    # probe()
    #forecast_now()
    
