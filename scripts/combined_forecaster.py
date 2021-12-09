import os, sys, gc
sys.path.insert(0, os.path.abspath('..'))
from whakaari import (TremorData, ForecastModel, 
    datetimeify, makedir, get_classifier, train_one_model, get_data_for_day)
from datetime import timedelta, datetime
from inspect import getfile, currentframe
from functools import partial
from multiprocessing import Pool
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
        
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

class TremorDataCombined(TremorData):
    def __init__(self, stations, parent=None):
        self.stations = stations
        self.station = '_'.join(sorted(self.stations))
        self._datas = []
        self.tes = []
        self.df = []
        for station in stations:
            self._datas.append(TremorData(station, parent))
            #fl = os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','{:s}_eruptive_periods.txt'.format(station)])
            fl = '..\\data\\'+'{:s}_eruptive_periods.txt'.format(station)
            with open(fl,'r') as fp:
                self._datas[-1].tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            self.tes += self._datas[-1].tes
            self.df.append(self._datas[-1].df)
        self.df = pd.concat(self.df)
        #self.tes = sorted(list(set(self.tes))) # ## testing: need to be checked
        self.ti = np.min([station.ti for station in self._datas])
        self.tf = np.max([station.tf for station in self._datas])
    def _compute_transforms(self):
        [station._compute_transforms() for station in self._datas]
        self.df = pd.concat([station.df for station in self._datas])
    def update(self, ti=None, tf=None, n_jobs = None):
        [station.update(ti,tf,n_jobs) for station in self._datas]
    def get_data(self, ti=None, tf=None):
        return pd.concat([station.get_data(ti,tf) for station in self._datas])
    def plot(self):
        raise NotImplementedError('method not implemented')

class ForecastModelCombined(ForecastModel):
    ''' Derived class with modified training to incorporate interpolation.
    '''
    def __init__(self, stations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stations = stations
        self._data = TremorDataCombined(stations=stations, parent=self)
        self.data = self._data
        try:
            dss = kwargs['data_streams']
        except KeyError:
            dss = self.data_streams
        if any(['_' in ds for ds in dss]):
            self.data._compute_transforms()
            
        try:
            ti = kwargs['ti']
        except KeyError:
            ti = self.data.ti
        self.ti_model = datetimeify(ti)
        
        try:
            tf = kwargs['tf']
        except KeyError:
            tf = self.data.tf
        self.tf_model = datetimeify(tf)

        #self._featfile = lambda st,ds: (r'{:s}/{:3.2f}w_{:3.2f}o_{:s}'.format(self.featdir,self.window, self.overlap, st)+'_{:s}.'+self.savefile_type).format(ds)
        self._featfile = lambda aux, st, ds: (r'{:s}/fm_{:3.2f}w_{:s}_{:s}'.format(self.featdir,self.window, ds, self.station)+'{:s}.'+self.savefile_type).format(st)

    def _exclude_dates(self, X, y, exclude_dates):
        """ Drop rows from feature matrix and label vector.

            Parameters:
            -----------
            X : pd.DataFrame
                Matrix to drop columns.
            y : pd.DataFrame
                Label vector.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Returns:
            --------
            Xr : pd.DataFrame
                Reduced matrix.
            yr : pd.DataFrame
                Reduced label vector.
        """
        self.exclude_dates = exclude_dates
        if len(self.exclude_dates) != 0:
            for edr in self.exclude_dates:
                if len(edr) == 3:
                    t0,t1 = [datetimeify(dt) for dt in edr[:2]]
                else:
                    t0,t1 = [datetimeify(dt) for dt in edr]
                inds = (y.index<t0)|(y.index>=t1)
                X = X.loc[inds]
                y = y.loc[inds]
        return X,y
    def _load_data(self, ti, tf):
        fM = []
        ys = []
        for i, station in enumerate(self._data._datas):
            self.data = station
            self.featfile = partial(self._featfile, station.station)
            tii = np.max([ti, station.ti])
            tfi = np.min([tf, station.tf])
            fMi,ysi = super()._load_data(tii,tfi)
            # fMi,ysi = self._load_data(tii,tfi)
            fM.append(fMi)
            ys.append(ysi)
        del fMi, ysi
        self.data = self._data
        #fM = pd.concat(fM, axis=1, sort=False)
        #ys = pd.concat(ys, axis=1, sort=False)

        return fM, ys
    #def train(self, ti=None, tf=None, Nfts=20, Ncl=500, retrain=False, classifier="DT", random_seed=0,
    #   drop_features=[], n_jobs=6, exclude_dates=[], use_only_features=[], method=0.75):
    def train(self, ti=None, tf=None, Nfts=20, Ncl=500, retrain=False, classifier="DT", random_seed=0,
            drop_features=[], n_jobs=6, exclude_dates=[], use_only_features=[], method=0.75):
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
        fMs, yss = self._load_data(self.ti_train, self.tf_train)
        
        # manually drop windows (rows) by station
        fMa = []
        ysa = []
        for fM, ys, i in zip(fMs, yss, range(len(yss))):
            station = self.data._datas[i].station
            ed = []
            for edi in exclude_dates:
                if len(edi) == 2:
                    ed.append(edi)
                elif len(edi) == 3:
                    if edi[-1] == station:
                        ed.append(edi)
            fM, ys = self._exclude_dates(fM, ys, ed)
            fMa.append(fM)
            ysa.append(ys)

        # merge feature matrices
        fM = pd.concat(fMa, sort=False)
        ys = pd.concat(ysa, sort=False)

        # manually drop features (columns)
        fM = self._drop_features(fM, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            use_only_features = [df for df in self.use_only_features if df in fM.columns]
            fM = fM[use_only_features]
            Nfts = len(use_only_features)+1

        # select training subset
        inds = (ys.index>=self.ti_train)&(ys.index<self.tf_train)
        fM = fM.loc[inds]
        ys = ys['label'].loc[inds]

        # check dimensionality has been preserved
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")

        # set up model training
        if self.n_jobs > 1:
            p = Pool(self.n_jobs)
            mapper = p.imap
        else:
            mapper = map
        f = partial(train_one_model, fM, ys, Nfts, self.modeldir, self.classifier, retrain, random_seed, method)

        # train models with glorious progress bar
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
    def hires_forecast(self, ti, tf, station, recalculate=True, save=None, root=None, nztimezone=False, 
        n_jobs=None, threshold=0.8, alt_rsam=None, xlim=None):
        """ Construct forecast at resolution of data.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of forecast period.
            tf : str, datetime.datetime
                End of forecast period.
            recalculate : bool
                Flag indicating forecast should be recalculated, otherwise forecast will be
                loaded from previous save file (if it exists).
            save : None or str
                If given, plot forecast and save to filename.
            root : None or str
                Naming convention for saving feature matrix.
            nztimezone : bool
                Flag to plot forecast using NZ time zone instead of UTC.            
            n_jobs : int
                CPUs to use when forecasting in parallel.
            Notes:
            ------
            Requires model to have already been trained.
        """
        # error checking
        try:
            _ = self.ti_train
        except AttributeError:
            raise ValueError('Train model before constructing hires forecast.')
        
        if save == '':
            save = '{:s}/hires_forecast.png'.format(self.plotdir)
            makedir(self.plotdir)
        
        if n_jobs is not None: self.n_jobs = n_jobs
 
        # calculate hires feature matrix
        if root is None:
            root = self.root+'_hires'
        _fm = ForecastModel(self.window, 1., self.look_forward, station=station, ti=ti, tf=tf, 
            data_streams=self.data_streams, root=root, savefile_type=self.savefile_type, feature_root=root)
        _fm.Ncl = self.Ncl            
        _fm.compute_only_features = list(set([ft.split('__')[1] for ft in self._collect_features()[0]]))
        for ds in self.data_streams:
            _fm._extract_features(ti, tf, ds)

        # predict on hires features
        ys = _fm.forecast(ti, tf, recalculate, use_model=self.modeldir, n_jobs=n_jobs)
        si = self.data.stations.index(station)
        _fm.data.tes = self.data._datas[si].tes
        
        if save is not None:
            _fm._plot_hires_forecast(ys, save, threshold, station, nztimezone=nztimezone, alt_rsam=alt_rsam, xlim=xlim)

        return ys
    
def combined_forecaster():
    """ This function creates a combined forecasting model using data from multiple volcanoes.
    """
    month = timedelta(days=365.25/12)
    n_jobs = 6
    drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
        '*attr_"angle"*']
    data_streams = ['zsc_rsam','zsc_mf','zsc_hf','zsc_dsar']
    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']#['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF', 'log_zsc2_rsamF', 'diff_zsc2_rsamF']
    
    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF','diff_zsc2_rsamF','diff_zsc2_mfF','diff_zsc2_hfF','diff_zsc2_dsarF',
        'log_zsc2_rsamF','log_zsc2_mfF','log_zsc2_hfF','log_zsc2_dsarF']
    #data_streams = ['zsc2_hfF']
    data_streams = ['log_zsc2_rsamF', 'zsc2_hfF']
    #data_streams = ['zsc2_hfF']

    # load feature matrices for WIZ and FWVZ
    fm0 = ForecastModelCombined(window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root='combined_forecaster', savefile_type='pkl', stations=['PV6'])#'WIZ','FWVZ'])#,'KRVZ']) 'FWVZ', 
    
    # drop all Fourier coefficients sampling at higher than 1/4 Nyquist
    freq_max = fm0.dtw//fm0.dt//4
    drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]

    #for d in fm0.data._datas:
    #    d.update()

    for j, station in enumerate(fm0.data._datas):
        print('Training from station: '+station.station+' ('+str(j+1)+'/'+str(len(fm0.data._datas))+')')
        for i,te in enumerate(station.tes):
            
            fm = ForecastModelCombined(window=2., overlap=0.5, look_forward=2., data_streams=data_streams,
                root='combined_forecaster_{:s}_e{:d}'.format(station.station,i+1), savefile_type='pkl', 
                stations=fm0.stations)

            exclude_dates = [[te-month, te+month, station.station]]

            print('Eruption: '+str(te.year)+' ('+str(i+1)+'/'+str(len(station.tes))+')')

            fm.train(drop_features=drop_features, retrain=False, Ncl=500, n_jobs=n_jobs, 
                exclude_dates=exclude_dates)        
        
            ys = fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=te+month/28., station=station.station,
                    recalculate=True, save=r'{:s}/forecast_hires.png'.format(fm.plotdir), 
                    n_jobs=n_jobs, root=r'{:s}'.format(fm.root), threshold=.8) # root=r'{:s}_hires'.

            y = ys['consensus']
            ci = fm._compute_CI(y)
            y0 = y-ci
            inds = (y.index<(te-fm.dt))&(y.index>(te-fm.dtf))
            conf = y0[inds].max()
            with open(r'{:s}/forecast_confidence.txt'.format(fm.plotdir), 'w') as fp:
                fp.write('{:4.3f}'.format(conf))

def model():
    # build and run combined forecast model
    combined_forecaster()    

if __name__ == "__main__":
    model()
    