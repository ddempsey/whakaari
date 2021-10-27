import os, sys, shutil, traceback
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify, STATIONS
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
from datetime import timedelta
import pickle
from scipy.signal import convolve

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

from functools import partial
from obspy import UTCDateTime
from cross_correlation import correlate_template

def convolution():
    ''' Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
        Overlap is set to 1.0 (max) 
    '''
    dt = 'zsc2_dsarF'
    day = timedelta(days=1)
    fm = ForecastModel(window=2., overlap=1., station='WIZ',
        look_forward=2., data_streams=[dt], 
        #data_dir=r'U:\Research\EruptionForecasting\eruptions\data',
        )
    te = fm.data.tes[-1]

    # rolling median and signature length window
    N, M = [2,28]
    # time
    j = fm.data.df.index
    # construct signature
    df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
    archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
    # convolve over the data
    df = fm.data.df[(j<te)]
    test = df[dt].rolling(N*24*6).median()[N*24*6:]
    out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))

    cc_te = []
    for te in fm.data.tes[:-1]:
        if te == fm.data.tes[2]:
            te = datetimeify('2013-10-11 07:09:00')
        cc_te.append(out[out.index.get_loc(te, method='nearest')])
    cc_te = np.array(cc_te)
    print(cc_te)

    # plot results
    f,(ax,ax2) = plt.subplots(1,2)
    ax.plot(test.index, test.values, 'k-', label='zsc_DSAR')
    ax.plot(archtype.index, archtype.values, 'r:', label='archtype')
    ax_ = ax.twinx()
    ax_.plot(test.index[-out.shape[-1]:], out, 'm--', label='CC')
    ax.plot([],[],'m--',label='CC')
    ax.legend()

    # distribution
    h,e = np.histogram(out, bins=np.linspace(-1,1,41))
    ax2.bar(e[:-1], h/(np.sum(h)*(e[1]-e[0])), e[1]-e[0], align='edge', label='all CC')
    y0 = 1.6*np.mean(ax2.get_ylim())

    # 2-sample Kolmogorov Smirnov test for difference in underlying distributions
    from scipy.stats import kstest
    pv = kstest(cc_te, out.iloc[archtype.shape[0]::24*6].values).pvalue

    # show eruptions on CDF
    ax2_=ax2.twinx()
    ov = np.sort(out.values[archtype.shape[0]::])
    cdf = 1.*np.arange(len(ov))/(len(ov)-1)
    ax2_.plot(ov, cdf, 'k-')
    cdf_te = np.array([cdf[np.argmin(abs(ov-cci))] for cci in cc_te])
    ax2_.plot(cc_te, cdf_te, 'b*', label='eruptions')
    ax2.plot([],[],'k-',label='CDF')
    ax2.plot([],[],'b*',label='eruptions')
    ax2.set_title('KS test p-value = '+str(pv))
    ax2.legend()

    plt.show()

def conv(at, x):
    y = ((x-np.mean(x))/np.std(x)*at.values).mean()
    return y

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
            data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
        # fm.compute_only_features=['median']
    
        for i,te in enumerate(fm.data.tes):
            if k == 'WIZ' and i==2:
                te = datetimeify('2013-10-11 07:09:00')
            print(te)
            # FM,_ = fm._load_data(te-month,te,None)
            all_data.append([
                k,
                te,
                None, 
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

    N = 2
    for i,e1,e2,e3,ax1,ax2,ax3 in zip(range(5),all_data[:5], all_data[5:10], all_data[10:], axs.T[0], axs.T[1], axs.T[2]):
        for e,ax in zip([e1,e2,e3],[ax1,ax2,ax3]):

            # for d,c in zip(ds[:5], ['m', 'g','k','b','r']):
            #     ax.plot(e[2].index, e[2][d+'__median'], c+'-', lw=1, label=d)
            # ax.plot(e[3].index, e[3]['zsc2_dsarF'], 'k-', label='dsar')
            # ax_.plot(e[3].index, e[3]['diff_dsarF'], 'b-')
            # ax.plot(e[3].index, e[2]['zsc2_dsarF__median'], 'b-')

            f = 0.9
            i,j = [int(f*N*6*24), N*6*24-int(f*N*6*24)]

            t = e[3].index
            dsari = e[3]['zsc2_dsarF']
            dsari = e[3]['zsc2_mfF']/e[3]['zsc2_hfF']
            dsar = dsari.rolling(N*6*24).median()[N*6*24:]
            mf = e[3]['zsc2_mfF'].rolling(N*6*24).median()[N*6*24:]
            hf = e[3]['zsc2_hfF'].rolling(N*6*24).median()[N*6*24:]
            dsar = mf/hf
            ax.plot(t[N*6*24:], dsar, 'g-')            
            ax_ = ax.twinx()
            ddsardt=dsari.rolling(i).median().diff().rolling(j).median()[N*6*24+1:]
            ddsardt=dsar.diff()[1:]
            # ax_.plot(t[N*6*24+1:], ddsardt, 'r-', lw=0.5)
            dmfdt=mf.diff()[1:]
            dhfdt=hf.diff()[1:]
            # ax_.plot(t[N*6*24+1:], dmfdt, 'g--')
            ddsardt2 = (hf[1:]*dmfdt-mf[1:]*dhfdt)/(hf[1:])**2
            # ax_.plot(t[N*6*24+1:], ddsardt2/np.max(abs(ddsardt2))*np.max(abs(ddsardt)), 'g--', lw=0.5)
            ax_.plot(t[8*N*6*24+1:], (hf[1:]*dmfdt).rolling(12).mean()[7*N*6*24:], 'b-', lw=0.5)
            ax_.plot(t[8*N*6*24+1:],(-mf[1:]*dhfdt).rolling(12).mean()[7*N*6*24:], 'r-', lw=0.5)
            ax_.set_yscale('symlog')

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
    get_all()
    study_all()
    convolution()
    
