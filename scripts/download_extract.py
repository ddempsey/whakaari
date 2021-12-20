import os, sys, shutil, traceback
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
from glob import glob

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

from obspy import UTCDateTime

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
    ds += ['zsc2_vlfF','zsc2_lfF','zsc2_vlarF','zsc2_lrarF','zsc2_rmarF','diff_zsc2_vlfF','diff_zsc2_lfF','diff_zsc2_vlarF','diff_zsc2_lrarF',
        'diff_zsc2_rmarF','log_zsc2_vlfF','log_zsc2_lfF','log_zsc2_vlarF','log_zsc2_lrarF','log_zsc2_rmarF']
    #ds = ['log_zsc2_rsamF', 'zsc2_hfF']
    ## stations
    rs = ['rsam','mf','hf','vlf','lf','dsar','vlar','lrar','rmar']
    ds = ['zsc2_'+r+'F' for r in rs]
    ss = ['KRVZ','FWVZ','WIZ','PVV','BELO','OKWR','VNSS','SSLW','REF','VNSS','MEA01','GOD','TBTN','ONTA']
    ss = ss[:7]
    #ss = ['VNSS']
    #ss = ['PV6']
    ## window sizes (days)
    ws = [2.] #, 14., 90., 365.]

    ## Run parallelization 
    ps = []
    for s in ss:
        for d in ds:
            for w in ws:
                #extract_one([w,s,d])
                ps.append([w,s,d])
    for fl in glob('*.err'):
        os.remove(fl)
    n_jobs = 16 # number of cores
    p = Pool(n_jobs)
    p.map(extract_one, ps)

def extract_vulcano():
    fm = ForecastModel(window=2., overlap=1., station='AUS', look_forward=2., data_streams=['zsc2_dsarF'],
            feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl')
    fm.compute_only_features=['median']
    fm._load_data(fm.data.df.index[0], fm.data.df.index[-1],None)

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
        look_forward=2., data_streams=[d], feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl',data_dir='/media/eruption_forecasting/eruptions/data/')
    from datetime import timedelta
    month = timedelta(days=365.25/12.)
    fm.n_jobs = 1
    for yr in list(range(fm.data.ti.year, fm.data.tf.year+1)):
        try:
            ti = np.max([datetime(yr,1,1,0,0,0),fm.data.ti+fm.dtw])
            tf = np.min([datetime(yr+1,1,1,0,0,0)-fm.dt,fm.data.tf])
            fm._load_data(ti,tf)
        except:
            print('errored on: ', w, o, d, s)
            with open('{:s}_{:3.2f}_{:s}.err'.format(s,w,d),'w') as fp:
                fp.write(str(traceback.format_exc())+'\n')
                fp.write(str(sys.exc_info()[0]))

def download_all():
    from datetime import timedelta
    stations = ['WIZ','WSRZ','FWVZ','KRVZ','IVGP','BELO','REF','SSLW','VNSS','OKWR']
#    stations = ['FWVZ','BELO','REF','SSLW','VNSS','OKWR']
    stations=['PVV']
    dt = timedelta(days=64.)
    for station in stations:
        try:
            td = TremorData(station=station)
            ti = td._probe_start()
            if station == 'WIZ': ti = UTCDateTime(datetimeify('2008-01-01'))
            if station == 'OKWR': ti = UTCDateTime(datetimeify('2008-01-01'))
            if station == 'VNSS': ti = UTCDateTime(datetimeify('2013-01-01'))
            if station == 'AUS': ti = UTCDateTime(datetimeify('2005-11-01'))
            if station == 'IVGP': ti = UTCDateTime(datetimeify('2019-08-10'))
            if station == 'FWVZ': ti = UTCDateTime(datetimeify('2005-06-01'))
            if station == 'PVV': ti = UTCDateTime(datetimeify('2014-01-01'))
            if td.tf is not None:
                from copy import deepcopy
                ti = UTCDateTime(deepcopy(td.tf))
            N = int(np.ceil((datetime.today()-ti._get_datetime())/dt))
            for i in range(N):
                t0=ti+i*dt
                t1=ti+(i+1)*dt
                if t1>datetime.today():
                    t1 = datetime.today()
                td.update(t0, t1, n_jobs=32)
        except:
            with open('{:s}_download.err'.format(station),'w') as fp:
                fp.write(str(traceback.format_exc())+'\n')
                fp.write(str(sys.exc_info()[0]))
            try:
                shutil.rmtree('_tmp')
            except:
                pass
            pass

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

def check_ratios():
    fm = ForecastModel(window=2, overlap=1, station = 'WIZ',
        look_forward=2., data_streams=['zsc2_rmarF'], savefile_type='pkl',
        feature_dir=r'U:\Research\EruptionForecasting\eruptions\features')
    from datetime import timedelta
    month = timedelta(days=365.25/12)
    f, axs = plt.subplots(5,1)
    for te,ax in zip(fm.data.tes,axs):
        fM,ys = fm._load_data(te-month, te, None) 
        ax.plot(fM.index, fM['zsc2_rmarF__median'], 'k-')
        
    plt.show()


if __name__ == "__main__":
    #forecast_dec2019()
    #forecast_test()
    #download_all()
    #check_ratios()
    extract_all()
    
