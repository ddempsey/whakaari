import os, sys, shutil, traceback
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify, STATIONS
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
from datetime import timedelta
import pickle

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

def check():
    ''' Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
        Overlap is set to 1.0 (max) 
    '''
    month = timedelta(days=365.25/12.)
    ks = ['WIZ','WIZ','FWVZ','KRVZ','VNSS',]
    # ks = ['VNSS','BELO']
    all_data = []
    data = ['rsam']
    ds = ['zsc2_{:s}F'.format(d) for d in data]+['rsam']
    for k in ks:
        fm = ForecastModel(window=2., overlap=1., station=k,
            look_forward=2., data_streams=ds, 
            # feature_dir=r'U:\Research\EruptionForecasting\eruptions\features', 
            data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
        fm.compute_only_features=['median']
    
        j = fm.data.df.index
        for i,te in enumerate(fm.data.tes):
            te = datetimeify('2020-03-01')
            if k == 'WIZ' and i==2:
                te = datetimeify('2013-10-11 07:09:00')
            FM,_ = fm._load_data(te-month,te,None)
            
            f,ax = plt.subplots(1,1)
            ax.plot(FM.index, FM['zsc2_{:s}F__median'.format(data[0])], 'k-')
            df = fm.data.df[(j>(te-month))&(j<te)]
            ax.plot(df.index, df['zsc2_{:s}F'.format(data[0])].rolling(2*6*24).median(), 'r--')
            plt.show()


def get_all():
    ''' Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
        Overlap is set to 1.0 (max) 
    '''
    month = timedelta(days=365.25/12.)
    ks = ['WIZ','FWVZ','KRVZ','VNSS','BELO']
    # ks = ['VNSS','BELO']
    all_data = []
    data = ['rmar','dsar','rsam','hf','mf']
    ds = ['zsc2_{:s}F'.format(d) for d in data]+['diff_{:s}F'.format(d) for d in data]
    for k in ks:
        print(k)
        fm = ForecastModel(window=2., overlap=1., station=k,
            look_forward=2., data_streams=ds, 
            feature_dir=r'U:\Research\EruptionForecasting\eruptions\features', 
            data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
        fm.compute_only_features=['median']
    
        for i,te in enumerate(fm.data.tes):
            if k == 'WIZ' and i==2:
                te = datetimeify('2013-10-11 07:09:00')
            print(te)
            FM,_ = fm._load_data(te-month,te,None)
            all_data.append([
                k,
                te,
                FM[['{:s}__median'.format(d) for d in ds]], 
                fm.data.get_data(te-month,te)
            ])

    with open('all_data.pkl','wb') as fp:
        pickle.dump(all_data, fp)

def study_all():
    
    data = ['rmar','dsar','rsam','hf','mf']
    ds = ['zsc2_{:s}F'.format(d) for d in data]+['diff_{:s}F'.format(d) for d in data[-3:]]

    with open('all_data.pkl','rb') as fp:
        all_data = pickle.load(fp)
        
    # with open('all_data2.pkl','rb') as fp:
    #     all_data += pickle.load(fp)

    f,axs = plt.subplots(5,3,figsize=(24,12))

    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    pattern = all_data[4][3]['zsc2_dsarF'].rolling(2*6*24).median().values[2*6*24:]

    for i,e1,e2,e3,ax1,ax2,ax3 in zip(range(5),all_data[:5], all_data[5:10], all_data[10:], axs.T[0], axs.T[1], axs.T[2]):
        for e,ax in zip([e1,e2,e3],[ax1,ax2,ax3]):

            # for d,c in zip(ds[:5], ['m', 'g','k','b','r']):
            #     ax.plot(e[2].index, e[2][d+'__median'], c+'-', lw=1, label=d)
            # ax.plot(e[3].index, e[3]['zsc2_dsarF'], 'k-', label='dsar')
            ax_ = ax.twinx()
            # ax_.plot(e[3].index, e[3]['diff_dsarF'], 'b-')
            # ax.plot(e[3].index, e[2]['zsc2_dsarF__median'], 'b-')
            ax.plot(e[3].index, e[3]['zsc2_dsarF'].rolling(2*6*24).median(), 'b-')
            # ax_.plot(e[3].index, e[3]['zsc2_dsarF'].rolling(2*6*24).apply(chqv), 'g-')
            # ax.plot([],[],'b-','dsar_rate')
            test_pattern = e[3]['zsc2_dsarF'].rolling(2*6*24).median().values[2*6*24:]
            # ax.plot(e[3].index, test_pattern, 'r--')
            # d,p=fastdtw(pattern/np.sum(pattern), test_pattern/np.sum(test_pattern), dist=euclidean)            
            # ax.text(0.03,0.95,e[0]+', {:s}, {:3.2f}'.format(e[1].strftime('%d-%m-%Y'), d), transform=ax.transAxes,ha='left', va='top')

    ax.legend()

    plt.tight_layout()
    plt.savefig('all_eruptions.png',dpi=300)

def chqv2(y):
    y0,y1 = np.percentile(y, [15,85])
    return (y1-y0)**2

def chqv(y):
    y0,y1 = np.percentile(y, [40,60])
    # return y1-y0
    inds = np.where((y>y0)&(y<y1))
    return np.var(np.diff(y, prepend=0)[inds])

if __name__ == "__main__":
    # get_all()
    # study_all()
    check()
    
