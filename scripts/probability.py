
import os, sys, shutil, gc
sys.path.insert(0, os.path.abspath('..'))
from whakaari import ForecastModel, load_dataframe, save_dataframe
from datetime import timedelta, datetime
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import check_array, indexable, column_or_1d
from scipy.optimize import fmin_bfgs
from scipy.special import expit, xlogy
from math import log

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
_DAY = timedelta(days=1)

def transformed_accuracy(i):
    # setup forecast model
    n_jobs = 4  
    data_streams = ['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF']
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='transformed_e{:d}'.format(i+1), savefile_type='pkl', station='WIZ',
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
    fm.train(drop_features=drop_features, retrain=False, Ncl=500, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for excluded eruption
    ys = fm.hires_forecast(ti=te-_MONTH, tf=te+_MONTH, recalculate=False, 
        save=r'{:s}/hires.png'.format(fm.plotdir), 
        n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)

    # save the largest value of model consensus in the 48 hours prior
    y = ys['consensus']
    ci = fm._compute_CI(y)
    y0 = y-ci                   # take lower bound of 95% conf. int. for conservatism
    inds = (y.index<(te-fm.dt))&(y.index>(te-fm.dtf))
    conf, conf0 = y[inds].max(), y0[inds].max()
    with open(r'{:s}/confidence.txt'.format(fm.plotdir), 'w') as fp:
        fp.write('{:4.3f}, {:4.3f}'.format(conf, conf0))

def transformed_falsealarms():
    # setup forecast model
    n_jobs = 8
    data_streams = ['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF']
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='transformed_e0', savefile_type='pkl', station='WIZ',
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
        with open(fm.plotdir+'/../transformed_e{:d}/confidence.txt'.format(i)) as fp:
            c,c0 = [float(ci) for ci in fp.readline().strip().split(',')]
            conf.append(c0)
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
    plt.savefig('transformed.png',dpi=400)

def calibrate_probability():

    plot = False
    data_streams = ['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF']
    fm = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='transformed_e0', savefile_type='pkl', station='WIZ',
        feature_dir=r'E:\whakaari\features')   

    fls = glob('../predictions/transformed_e0_hires/consensus_*.pkl')
    ys = []
    for fl in fls:
        ys.append(load_dataframe(fl))
    y0 = [pd.concat(ys).sort_index(),]

    if plot:
        f,ax1 = plt.subplots(1,1)
        ax1.plot(y0[0].index, y0[0]['consensus'], 'k-', label='trained including eruption')

    for i,te in enumerate(fm.data.tes):
        fls = glob('../predictions/transformed_e{:d}_hires/consensus_*.pkl'.format(i+1))
        ys = []   
        for fl in fls:
            ys.append(load_dataframe(fl))     
        y0.insert(0, pd.concat(ys).sort_index())
        y0[-1] = y0[-1][(y0[-1].index<y0[0].index[0])|(y0[-1].index>y0[0].index[-1])]
    y0 = pd.concat(y0)
    y0 = y0[~y0.index.duplicated(keep='first')].sort_index()

    if plot:
        ax1.plot(y0.index, y0['consensus'], 'r-', label='trained excluding eruption')
        ax1.plot([],[],'b-',label='normalized rsam')
        ax1_ = ax1.twinx()
        ax1_.plot(fm.data.df.index, fm.data.df['zsc_rsamF'], 'b-')

        for i,te, in enumerate(fm.data.tes):
            ax1.axvline(te, color = 'm', linestyle = '--')
            ax1.axvline(te-4*_DAY, color = 'm', linestyle = ':')
        ax1.plot([],[],'m--', label='eruption')
        ax1_.set_ylim([0,100])

        ax1.legend()
        plt.show()

    label_fl = '../predictions/transformed_e0_hires/labels.pkl'
    if not os.path.isfile(label_fl):
        yl = np.array([fm.data._is_eruption_in(days=fm.look_forward, from_time=ti) for ti in pd.to_datetime(y0.index)])
        save_dataframe(yl, label_fl)
    yl = load_dataframe(label_fl)

    a,b = _sigmoid_calibration(y0['consensus'].values, yl)
    
    f,ax = plt.subplots(1,1)
    lk = expit(-(a*y0['consensus'].values+b))/expit(-b)
    ax.plot(y0.index, lk, 'k-', label='relative likelihood')
    for i,te, in enumerate(fm.data.tes):
        ax.axvline(te, color = 'm', linestyle = '--')
    ax.plot([],[],'m--', label='eruption')
    ax.set_ylabel('relative likelihood of eruption in next 48 hours')
    ax.set_xlabel('date')
    ax.plot([],[],'b-',label='normalized rsam')
    ax_ = ax.twinx()
    ax_.plot(fm.data.df.index, fm.data.df['zsc_rsamF'], 'b-')
    ax_.set_ylim([0,100])
    ax.legend()
    plt.show()

    return a,b

def _sigmoid_calibration(df, y, sample_weight=None):
    """Probability Calibration with sigmoid method (Platt 2000)
    See sklearn.calibration module L392-452
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/calibration.py#L392
    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.
    y : ndarray, shape (n_samples,)
        The targets.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.
    Returns
    -------
    a : float
        The slope.
    b : float
        The intercept.
    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        P = expit(-(AB[0] * F + AB[1]))
        loss = -(xlogy(T, P) + xlogy(T1, 1. - P))
        if sample_weight is not None:
            return (sample_weight * loss).sum()
        else:
            return loss.sum()

    def grad(AB):
        # gradient of the objective function
        P = expit(-(AB[0] * F + AB[1]))
        TEP_minus_T1P = T - P
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]

def model():
    # build and run transformed forecast model
    # for i in range(5):
    #     transformed_accuracy(i)    

    # transformed_falsealarms()    

    calibrate_probability()

if __name__ == "__main__":
    model()