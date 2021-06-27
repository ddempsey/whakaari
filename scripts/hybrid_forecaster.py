import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import *
from datetime import timedelta
from functools import partial
from multiprocessing import Pool
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew
import GPy
        
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

class ForecastModelHybrid(ForecastModel):
    ''' Derived class with modified training to incorporate interpolation.
    '''
    def __init__(self, *args, **kwargs):
        super(ForecastModelHybrid, self).__init__(*args, **kwargs)
        self._t_c = None
        self._interpolators = None
    def train(self, ti=None, tf=None, Nfts=20, Ncl=500, retrain=False, classifier="DT", random_seed=0,
            drop_features=[], n_jobs=6, exclude_dates=[], use_only_features=[], recalculate_posteriors=True):
        """ Construct classifier models.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of training period (default is beginning model analysis period).
            tf : str, datetime.datetime
                End of training period (default is end of model analysis period).
            Nfts : int
                Number of most-significant features to use in classifier.
            Ncl : int
                Number of classifier models to train.
            retrain : boolean
                Use saved models (False) or train new ones.
            classifier : str, list
                String or list of strings denoting which classifiers to train (see options below.)
            random_seed : int
                Set the seed for the undersampler, for repeatability.
            drop_features : list
                Names of tsfresh features to be dropped prior to training (for manual elimination of 
                feature correlation.)
            n_jobs : int
                CPUs to use when training classifiers in parallel.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Classifier options:
            -------------------
            SVM - Support Vector Machine.
            KNN - k-Nearest Neighbors
            DT - Decision Tree
            RF - Random Forest
            NN - Neural Network
            NB - Naive Bayes
            LR - Logistic Regression
        """
        self.classifier = classifier
        self.exclude_dates = exclude_dates
        self.use_only_features = use_only_features
        self.n_jobs = n_jobs
        self.Ncl = Ncl
        makedir(self.modeldir)

        # initialise training interval
        self.ti_train = self.ti_model if ti is None else datetimeify(ti)
        self.tf_train = self.tf_model if tf is None else datetimeify(tf)
        if self.ti_train - self.dtw < self.data.ti:
            self.ti_train = self.data.ti+self.dtw
        
        # check if any model training required
        if not retrain:
            run_models = False
            pref = type(get_classifier(self.classifier)[0]).__name__ 
            for i in range(Ncl):         
                if not os.path.isfile('{:s}/{:s}_{:04d}.pkl'.format(self.modeldir, pref, i)):
                    run_models = True
            if not run_models:
                return # not training required
        else:
            # delete old model files
            _ = [os.remove(fl) for fl in  glob('{:s}/*'.format(self.modeldir))]

        # get feature matrix and label vector
        fM, ys = self._load_data(self.ti_train, self.tf_train)

        # manually drop features (columns)
        fM = self._drop_features(fM, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            use_only_features = [df for df in self.use_only_features if df in fM.columns]
            fM = fM[use_only_features]
            Nfts = len(use_only_features)+1

        # drop windows not present in both WIZ and WSRZ
        if self._t_c is not None:
            fM = fM.loc[self._t_c]
            ys = ys.loc[self._t_c]
            # add labels for additional interpolating windows
            if self._interpolators is not None:
                for i in self._interpolators[-1].index:
                    ys.loc[i] = 1

        # manually drop windows (rows)
        fM, ys = self._exclude_dates(fM, ys, exclude_dates)
        
        # select training subset
        inds = (ys.index>=self.ti_train)&(ys.index<self.tf_train)
        fM = fM.loc[inds]
        ys = ys['label'].loc[inds]
        
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")

        if self._interpolators is not None:
            # create posteriors if not already generated
            if recalculate_posteriors:
                np.random.seed(random_seed)
                interpolate_feature_matrix(fM, *self._interpolators, Ncl)

            fi = train_one_model_posterior
        else:
            fi = train_one_model
        
        f = partial(fi, fM, ys, Nfts, self.modeldir, self.classifier, retrain, random_seed)

        # set up model training
        if self.n_jobs > 1:
            p = Pool(self.n_jobs)
            mapper = p.imap
        else:
            mapper = map
        # train models with glorious progress bar
        f(0)
        for i, _ in enumerate(mapper(f, range(Ncl))):
            cf = (i+1)/Ncl
            print(f'building models: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
        if self.n_jobs > 1:
            p.close()
            p.join()
        
        # free memory
        del fM
        gc.collect()
        self._collect_features()

def hybrid_falsealarms():    
    # load feature matrices for WIZ and WSRZ
    data_streams = ['zsc_rsamF','zsc_hfF','zsc_mfF','zsc_dsarF']
    fm0 = ForecastModelHybrid(window=2., overlap=0.75, look_forward=2., data_streams=data_streams, 
        root='original_forecaster', savefile_type='pkl', station='WIZ', feature_dir=r'E:\whakaari\features')
    fM0,_ = fm0._load_data(fm0.data.ti, fm0.data.tf)
    
    fm = ForecastModelHybrid(window=2., overlap=0.75, look_forward=2., data_streams=data_streams, 
        root='hybrid_e0', savefile_type='pkl', station='WSRZ', feature_dir=r'E:\whakaari\features')
    fM,_ = fm._load_data(fm.data.ti, fm.data.tf)

    t_c, t_m, t_p = get_window_overlap(None, None, None, None, None, None, None, False)

    # for each feature, train an interpolating function
    interpolators_file = 'zsc_gprs_e0.pkl'
    if not os.path.isfile(interpolators_file):
        gprs = batch_interpolation(fMs=[fM0, fM], t = [t_c, t_m, t_p], plot=True, recalculate=True)
        save_dataframe(gprs, interpolators_file)
    gprs = load_dataframe(interpolators_file)

    # pass interpolating function information to ForecastModel object for training
    fm._t_c = sorted(np.array(list(t_c)+list(t_m)))
    fm._interpolators = [gprs, fM0.loc[t_m]]
    
    # remove duplicate linear features (because correlated), unhelpful fourier compoents
    # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
        '*attr_"angle"*']  
    freq_max = fm.dtw//fm.dt//4
    drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]    

    n_jobs = 8

    fm.train(drop_features=drop_features, retrain=True, Ncl=500, n_jobs=n_jobs,
        recalculate_posteriors=True)        
    
    ys = fm.hires_forecast(ti=fm.ti_train, tf=fm.tf_train, recalculate=True, n_jobs=n_jobs,
            root=r'{:s}_hires'.format(fm.root), threshold=1.0)    

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
    for i in [2,5]:
        with open(fm.plotdir+'/../hybrid_e{:d}/confidence.txt'.format(i)) as fp:
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
    plt.savefig('hybrid.png',dpi=400)
def hybrid_forecaster(i):
    """ This function constructs a hybrid forecaster for WSRZ using data from WIZ to fill any outage gaps. 
    
        Notes:
        ------
        Gaussian Process regression is used to interpolate data gaps in feature space.
    """
    # get high resolution outage data
    to_WIZ, to_WSRZ = get_outages()
    
    # load feature matrices for WIZ and WSRZ
    data_streams = ['zsc_rsamF','zsc_hfF','zsc_mfF','zsc_dsarF']
    fm0 = ForecastModelHybrid(window=2., overlap=0.75, look_forward=2., data_streams=data_streams, 
        root='original_forecaster', savefile_type='pkl', station='WIZ', feature_dir=r'E:\whakaari\features')
    fM0,_ = fm0._load_data(fm0.data.ti, fm0.data.tf)
    te = fm0.data.tes[i]

    fm = ForecastModelHybrid(window=2., overlap=0.75, look_forward=2., data_streams=data_streams, 
        root='hybrid_e{:d}'.format(i+1), savefile_type='pkl', station='WSRZ', feature_dir=r'E:\whakaari\features')
    fM,_ = fm._load_data(fm.data.ti, fm.data.tf)

    # locate "common" windows (containing sufficient data for both WIZ and WSRZ), "missing" and "present"
    # eruption windows for WSRZ
    # t_c, t_m, t_p = get_window_overlap(tos=[to_WIZ, to_WSRZ], tis=[fM0.index, fM.index], tes=fm0.data.tes, 
    #     window=fm0.window, look_forward=fm0.look_forward, dt=fm0.dt, tolerable_outage_fraction=0.125, recalculate=True)
    t_c, t_m, t_p = get_window_overlap(None, None, None, None, None, None, None, False)

    # for each feature, train an interpolating function
    interpolators_file = 'zsc_gprs_e{:d}.pkl'.format(i+1)
    if not os.path.isfile(interpolators_file):
        # if they exist, drop "test" eruption windows from "present" (test data cannot be used to train
        # the interpolators)
        t_p = t_p[np.where((t_p<(te-_MONTH))|(t_p>(te+_MONTH)))]
        gprs = batch_interpolation(fMs=[fM0, fM], t = [t_c, t_m, t_p], plot=True, recalculate=True)
        save_dataframe(gprs, interpolators_file)
    gprs = load_dataframe(interpolators_file)

    # pass interpolating function information to ForecastModel object for training
    fm._t_c = sorted(np.array(list(t_c)+list(t_m)))
    fm._interpolators = [gprs, fM0.loc[t_m]]
    
    # remove duplicate linear features (because correlated), unhelpful fourier compoents
    # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
        '*attr_"angle"*']  
    freq_max = fm.dtw//fm.dt//4
    drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]    

    n_jobs = 8
    
    exclude_dates = [[te-_MONTH,te+_MONTH],]

    fm.train(drop_features=drop_features, retrain=True, Ncl=500, n_jobs=n_jobs,
        exclude_dates=exclude_dates, recalculate_posteriors=True)        
    
    ys = fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=te+_MONTH/28., recalculate=True, 
            save=r'{:s}/hires.png'.format(fm.plotdir), n_jobs=n_jobs,
            root=r'{:s}_hires'.format(fm.root), threshold=1.0)

    y = ys['consensus']
    ci = fm._compute_CI(y)
    y0 = y-ci
    inds = (y.index<(te-fm.dt))&(y.index>(te-fm.dtf))
    conf = y0[inds].max()
    with open(r'{:s}/confidence.txt'.format(fm.plotdir), 'w') as fp:
        fp.write('{:4.3f}'.format(conf))
        # break
def train_one_model_posterior(fM, ys, Nfts, modeldir, classifier, retrain, random_seed, random_state):
    
    # undersample data
    rus = RandomUnderSampler(0.75, random_state=random_state+random_seed)
    fMi = get_hybrid_feature_matrix(fM, random_state)
    fMt,yst = rus.fit_resample(fMi,ys)
    yst = pd.Series(yst, index=range(len(yst)))
    fMt.index = yst.index

    # find significant features
    select = FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt,yst)
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    fMt = fMt[fts]
    with open('{:s}/{:04d}.fts'.format(modeldir, random_state),'w') as fp:
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    # get sklearn training objects
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state+random_seed)
    model, grid = get_classifier(classifier)            
        
    # check if model has already been trained
    pref = type(model).__name__
    fl = '{:s}/{:s}_{:04d}.pkl'.format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return
    
    # train and save classifier
    model_cv = GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy",error_score=np.nan)
    model_cv.fit(fMt,yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)
def get_outages():
    return _get_outages(station='WIZ'), _get_outages(station='WSRZ')
def _get_outages(station):
    """ Returns time stamps of outages

        Parameters:
        -----------
        station : str
            Station to assess for data gaps.

        Returns:
        --------
        inds : array-like
            List of time indices of missing data.

        Notes:
        ------
        When originally processed, data was imputed by linear interpolation. Thus, if a datapoint is the average of the 
        two either side of it, it is assumed to be an outage.

    """
    # load data at station
    td = TremorData(station=station)
    # td.update()

    t = []
    for ti,v0,vm,v1 in zip(td.df.index[1:-1],td.df['rsam'].values[:-2],td.df['rsam'].values[1:-1],td.df['rsam'].values[2:]):
        # check if linear interpolation has been used
        if abs(vm-(v0+v1)/2.)<1.e-3:
            t.append(ti)
    return np.array(t)
def get_window_overlap(tos, tis, tes, window, look_forward, dt, tolerable_outage_fraction=0.125, recalculate=True):
    """

        Parameters:
        -----------
        tos : array-like
            List of outage time indices at WIZ and WSRZ.
        tis : array-like
            List of window time indices for WIZ and WSRZ data.
        tes : array-like
            List of eruption time indices.
        window : float
            Window length in days.
        look_forward : float
            Look forward period in days.
        dt : timedelta
            Length of sampling period (high resolution).
        tolerable_outage_fraction : float
            Maximum fraction of window data points that can be outages without rejecting the window.
        recalculate : bool
            This step takes a while, so there is an option to pickle and load output from previous.

        Returns:
        --------
        t_c : array-like
            Time indices of windows where both stations have sufficient data ('c'ommon).
        t_m : array-like
            Time indices of pre-eruption windows where WSRZ has insufficient data ('m'issing).
        t_p : array-like
            Time indices of pre-eruption windows where WSRZ has sufficient data ('p'resent).
    """
    # option to load from previous
    fl = 'common_indices.pkl'
    if not recalculate and os.path.isfile(fl):
        ts = load_dataframe(fl)
        return ts[0], ts[1], ts[2]

    day = timedelta(days=1)
    to_WIZ, to_WSRZ = tos     # high res outage indices
    ti_WIZ, ti_WSRZ = tis     # lo res window indices

    # compute integer lengths
    wL = int(window*day/dt)                 # window
    iL = int(tolerable_outage_fraction*wL)  # permissible missing entries in window

    # for each window in each station, check if it contains too many outage indices
    ts = []
    for ti, to in zip(tis, tos):    # for each station
        t = []
        for tii in ti:                  # for each window
            if np.sum((to>(tii-window*day))&(to<tii)) < iL:      # keep if outages less than max
                t.append(tii)
        ts.append(t)
        
    # find intersection of outage free window indices across two stations
    t_c = np.array(sorted(list(set(ts[0]).intersection(set(ts[-1])))))
    
    # find missing eruptions indices
    t_e = []
    for te in tes:
        t_e += list(ti_WIZ[(ti_WIZ>(te-look_forward*day))&(ti_WIZ<te)])

    t_m = np.array(sorted(set(t_e) - set(t_c)))     # missing indices
    t_p = np.array(sorted(set(t_e) - set(t_m)))     # present indices

    save_dataframe([t_c, t_m, t_p], fl)
    return t_c, t_m, t_p
def batch_interpolation(fMs, t, plot=False, recalculate=True):
    """ Construct interpolators via multiprocessing.

        Parameters:
        -----------
        fMs : array-like
            Feature matrices for WIZ and WSRZ.
        t : array_like
            Time index vectors, output of get_window_overlap().
        plot : bool
            Flag to create plots of interpolators.
        recalculate : bool
            Flag to overwrite calculations or use existing.

        Returns:
        --------
        gprs : dict
            Trained regression objects using feature names as dictionary keys.

        Notes:
        ------
        Interpolators for each feature are saved to disc (pickle) with location format
        ./posteriors/*feature_name*.pkl
    """
    # hard code variables
    f = partial(_interpolate_one, fMs, t, plot, recalculate)
    
    # serial option, for debugging
    # gprs = [f(feature) for feature in fMs[0].columns]

    # multiprocessing
    p = Pool(6)
    gprs = p.map(f, fMs[0].columns)
    return dict(gprs)
def _interpolate_one(fMs, t, plot, recalculate, feature):
    ''' Create one feature interpolator.

        Parameters:
        -----------
        fMs : array-like
            Feature matrices for WIZ and WSRZ.
        t : array_like
            Time index vectors, output of get_window_overlap().
        plot : bool
            Flag to create plots of interpolators.
        recalculate : bool
            Flag to overwrite calculations or use existing.
        feature : str
            Name of feature to interpolate.

        Returns:
        --------
        gpr : GPy.models.GPRegression
            Interpolator object.

        Notes:
        ------
        Interpolation method is Gaussian Process Regression with RBF kernel. Other kernel
        options may yield improved performance.
    '''
    # option to load existing interpolator
    fl = 'posteriors/{:s}.pkl'.format(feature).replace('"','')
    if os.path.isfile(fl) and not recalculate:
        return (feature, load_dataframe(fl))
    print(feature)

    # training data
    x_n = fMs[0][feature][t[2]].values.reshape(-1, 1)
    x_n2 = fMs[0][feature][np.array(list(t[1])+list(t[2]))].values.reshape(-1, 1)
    y_n = fMs[1][feature][t[2]].values.reshape(-1, 1)
    
    # transformation option - some features work better in log space
    if np.min(x_n2)<=0 or np.min(y_n)<=0 or skew(fMs[0][feature])<1:
        f1 = lambda x: x
        f2 = lambda x: x
        logscale = False
    else:
        f1 = lambda x: np.log10(x)
        f2 = lambda x: 10**x
        logscale = True

    # kernel selection and interpolator training
    kern = GPy.kern.RBF(input_dim=1, variance=5.0, lengthscale=5.0)
    # kern = GPy.kern.Poly(input_dim=1., order=3)
    # kern = GPy.kern.Spline(input_dim=1., variance = 1.0, c = 5)
    try:
        gpr = GPy.models.GPRegression(f1(x_n), f1(y_n), kern)
        gpr.optimize()
    except:
        save_dataframe(None, fl)
        return (feature, None)

    # optional plotting
    if plot:
        f = plt.figure(figsize=(6,6))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        ax.plot(fMs[0][feature][t[0]], fMs[1][feature][t[0]], 'k.')
        ax.plot(x_n, y_n, 'bo')
    
        if logscale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.plot(xlim,ylim, 'k--', lw=0.5)

        np.random.seed(0)
        X_test = np.linspace(*f1(xlim), 100).reshape(-1, 1)
        
        try:
            y, y_var = gpr.predict(X_test)
            y = y.ravel()
            std = 2*np.sqrt(y_var.ravel())
            ax.plot(f2(X_test), f2(y),'y-', zorder=4)
            ax.fill_between(f2(X_test.ravel()), f2(y + std), f2(y - std), color='y', alpha = 0.5, zorder=3)
            
            x_n = fMs[0][feature][t[1]].values.reshape(-1, 1)
            y = gpr.posterior_samples(f1(x_n), size=1).ravel()
            ax.plot(x_n, f2(y), 'o', mec='c', mfc='w', mew=1.5, ms=8, zorder=10)
        except:
            pass

        ax.set_xlabel('WIZ feature value')
        ax.set_ylabel('WSRZ feature value')
        plt.savefig(fl.replace('.pkl', '.png'), dpi=400)
        plt.close()

    if logscale:
        gpr._logscale=True
    else:
        gpr._logscale=False
    save_dataframe(gpr, fl)

    return (feature, gpr)
def interpolate_feature_matrix(fM, gprs, fM_int, M):
    ''' Fill missing feature matrix values with interpolation.

        Parameters:
        -----------
        fM : DataFrame
            Feature matrix with missing values.
        gprs : dict
            Interpolator objects, keyed by feature name.
        fM_int : DataFrame
            Feature matrix with values to interpolate from.
        M : int 
            Number of replications.
        
        Returns:
        --------
        fM : Dataframe
            Modified feature matrix with interpolated values.
    '''
    ks = []
    ys = []
    N = len(fM_int.index)
    # loop over all feature values and associated interpolators
    for k,gpr in gprs.items():          
        ks.append(k)
        if gpr is None:
            # interpolator couldn't be constructed, fill with NaNs
            ys.append(np.array(N*[M*[np.nan]]))
        else:
            x_n = fM_int[k].values.reshape(-1,1)
            if gpr._logscale:
                # a logscale interpolator was used
                inds = np.where(x_n<0)
                x_n[inds] = 1.
                y_n = 10**gpr.posterior_samples(np.log10(x_n), size=M)
                for i in inds[0]:
                    y_n[i,:,:] = -np.inf
            else:
                y_n = gpr.posterior_samples(x_n, size=M)
            ys.append(np.array(y_n[:,0,:]))
    # save posteriors as dataframes for quick loading later
    for i in range(M): 
        df = pd.DataFrame(np.array([y[:,i] for y in ys]).T, columns=ks, index=fM_int.index)
        save_dataframe(df, 'posteriors/df_post{:04d}.pkl'.format(i))
def get_hybrid_feature_matrix(fM, i):
    # create a merged data frame
    df = load_dataframe('posteriors/df_post{:04d}.pkl'.format(i))
    df = pd.concat([df,fM])
    df = df[~df.index.duplicated(keep='first')]
    df = impute(df)
    df = df.loc[fM.index]
    return df

def main():
    # build and run hybrid forecast model
    for i in [1, 3, 4]:
        hybrid_forecaster(i)   

    hybrid_falsealarms()

if __name__ == "__main__":
    main()
    