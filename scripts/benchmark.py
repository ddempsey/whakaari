
import os, sys, shutil, gc
sys.path.insert(0, os.path.abspath('..'))
from whakaari import ForecastModel
from datetime import timedelta, datetime

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

_MONTH = timedelta(days=365.25/12)

def benchmark_accuracy(i):
    # setup forecast model
    n_jobs = 4  
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='benchmark_e{:d}'.format(i+1), savefile_type='pkl', station='WIZ',
        feature_dir=r'E:\whakaari\features')   
    #fm.data.update()

    # train-test split on five eruptions to compute model confidence of an eruption
        # remove duplicate linear features (because correlated), unhelpful fourier compoents
        # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
        '*attr_"angle"*']  
    freq_max = fm.dtw//fm.dt//4
    drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]

    # train a model with data from that eruption excluded
    te = fm.data.tes[i]
    exclude_dates = [[te-_MONTH, te+_MONTH]]
    fm.train(drop_features=drop_features, retrain=True, Ncl=500, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for excluded eruption
    ys = fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=te+_MONTH/28., recalculate=True, 
        save=r'{:s}/hires.png'.format(fm.plotdir), 
        n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)

    # save the largest value of model consensus in the 48 hours prior
    y = ys['consensus']
    ci = fm._compute_CI(y)
    y0 = y-ci                   # take lower bound of 95% conf. int. for conservatism
    inds = (y.index<(te-fm.dt))&(y.index>(te-fm.dtf))
    conf = y0[inds].max()
    with open(r'{:s}/confidence.txt'.format(fm.plotdir), 'w') as fp:
        fp.write('{:4.3f}'.format(conf))

def benchmark_falsealarms():
    # setup forecast model
    n_jobs = 8
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='benchmark_e0', savefile_type='pkl', station='WIZ',
        feature_dir=r'E:\whakaari\features')   
    #fm.data.update()

    # remove duplicate linear features (because correlated), unhelpful fourier compoents
    # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
        '*attr_"angle"*']  
    freq_max = fm.dtw//fm.dt//4
    drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]    

    # construct hires model over entire dataset to compute false alarm rate
    fm.train(drop_features=drop_features, retrain=False, Ncl=500, n_jobs=n_jobs)        
    
    # forecast over whole dataset
    ys = fm.hires_forecast(ti=fm.ti_train, tf=fm.tf_train, recalculate=False, 
        n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)    

    import numpy as np
    from matplotlib import pyplot as plt
    y = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
    dy = fm._compute_CI(y)

    thresholds = np.linspace(0,1.0,101)
    ialert = int(fm.look_forward/(fm.dt.total_seconds()/(24*3600)))
    FP, FN, TP, TN, dur, MCC = fm.get_performance(ys.index, y-dy, thresholds, ialert, fm.dt)
    f,ax1 = plt.subplots(1,1)
    ax1.plot(thresholds, dur, 'k-', label='fractional alert duration')
    ax1.plot(thresholds, TP/(FP+TP), 'k--', label='probability of eruption\nduring alert')
    ax1_ = ax1.twinx()
    ax1_.plot(thresholds, dur*ys.shape[0]/6/24/(FP+TP), 'b-')
    ax1.plot([], [], 'b-', label='average alert length')

    conf = []
    for i in [1,2,3,5]:
        with open(fm.plotdir+'/../benchmark_e{:d}/confidence.txt'.format(i)) as fp:
            conf.append(float(fp.readline().strip()))
    conf = min(conf)
    j = np.argmin(abs(thresholds-conf))
    th = thresholds[j]
    th0 = thresholds[0]
    th1 = thresholds[-1]
    ax1.axvline(th, color = 'k', linestyle=':', alpha=0.5, 
        label='model conf. {:3.2f}'.format(th))
    ax1.plot([th0,th], [dur[j],dur[j]], 'k-', alpha=0.5, lw=0.5)
    tp = TP[j]/(FP[j]+TP[j])
    ax1.plot([th0,th], [tp,tp], 'k--', alpha=0.5, lw=0.5)
    ax1.text(-.01*th1, tp, '{:3.2f}'.format(tp), ha='right', va='center')
    ax1.text(-.01*th1, dur[j], '{:3.2f}'.format(dur[j]), ha='right', va='center')
    
    dj = dur[j]*ys.shape[0]/6/24/(FP[j]+TP[j])
    ax1_.plot([th,th1], [dj, dj], 'b-', alpha=0.5, lw=0.5)
    ax1_.text(1.01*th1, dj, '{:2.1f} days'.format(dj), ha='left', va='center')
    ax1.set_xlim([th0,th1])
    ax1_.set_xlim([th0,th1])
    
    ax1.legend()
    ax1.set_xlabel('alert threshold')
    ax1_.set_ylabel('days')
    plt.savefig('benchmark.png',dpi=400)

def model():
    # test()
    benchmark_falsealarms()    
    # return

    # build and run benchmark forecast model
    # for i in range(5):
    #     benchmark_accuracy(i)    
    

if __name__ == "__main__":
    model()