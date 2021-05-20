import os, sys, shutil
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, save_dataframe, datetimeify
from datetime import timedelta, datetime
from functools import partial
from multiprocessing import Pool
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import GPy
        
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
    # tf = te+month/30
    # fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=True, 
    #     save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

    te = fm.data.tes[1]
    y = load_dataframe(r'D:\code\whakaari\predictions\test_hires\DecisionTreeClassifier_0000.pkl')
    tf = y.index[-1] + month/30./10.
    fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=False, 
        save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_now():
    ''' forecast model for present day 
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
        
    # pull the latest data from GeoNet
    td = TremorData()
    td.update()

    # model from 2011 to present day (td.tf)
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75,  
        look_forward=2, data_streams=data_streams, root='online_forecaster')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6
    
    # The online forecaster is trained using all eruptions in the dataset. It only
    # needs to be trained once, or again after a new eruption.
    # (Hint: feature matrices can be copied from other models to avoid long recalculations
    # providing they have the same window length and data streams. Copy and rename 
    # to *root*_features.csv)
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, 
        retrain=True, n_jobs=n_jobs)      
    
    # forecast the last 7 days at high resolution
    fm.hires_forecast(ti=fm.data.tf - 7*day, tf=fm.data.tf, recalculate=True, 
        save='current_forecast.png', nztimezone=True, n_jobs=n_jobs)  

def forecast_scratch():
    ''' test scale forecast model
    '''
    # constants
    month = timedelta(days=365.25/12)
        
    # set up model
    ti = '2011-01-01'
    tf = '2021-01-01'
    # data_streams = ['rsam','mf','hf','dsar','rsamF','mfF','hfF','dsarF']
    # data_streams = ['zsc_'+ds for ds in data_streams]
    fm = ForecastModel(ti=ti, tf=tf, window=2., overlap=0.75, 
        look_forward=2., data_streams=data_streams, root='test', savefile_type='pkl')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 16
    
    # train the model
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti=ti, tf=tf, drop_features=drop_features, retrain=False, n_jobs=n_jobs)      
    return

    # plot a forecast for a future eruption
    # tf = te+month/30
    # fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=True, 
    #     save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

    te = fm.data.tes[1]
    y = load_dataframe(r'D:\code\whakaari\predictions\test_hires\DecisionTreeClassifier_0000.pkl')
    tf = y.index[-1] + month/30./10.
    ys = fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=False, 
        save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

def build_models():
    # constants
    month = timedelta(days=365.25/12)
        
    # set up model
    ti = '2011-01-01'
    tf = '2021-01-01'
    groups = [
        [1,['rsam','mf','hf','dsar'],],
        [2,['rsamF','mfF','hfF','dsarF'],],
        [3,['rsam','mf','hf'],],
        [4,['rsamF','mfF','hfF',],],
        [5,['zsc_rsam','zsc_mf','zsc_hf','zsc_dsar'],],
        [6,['zsc_rsam','zsc_mf','zsc_hf'],],
        [7,['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF'],],
        [8,['zsc_rsamF','zsc_mfF','zsc_hfF'],],
    ]
    groups = [
        # [5,['zsc_rsam','zsc_mf','zsc_hf','zsc_dsar'],],
        [7,['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF'],],
    ]

    # set the available CPUs higher or lower as appropriate
    n_jobs = 7
    td = TremorData()

    # for each combination of input data
    for i, ds in groups:
        
        # for each eruption to exlude (0 = exclude no eruptions)
        for j in range(6):
            # if j!=5:continue
            exclude_dates = []
            if j > 0:
                te = td.tes[j-1]
                exclude_dates.append([te-month,te+month])
            else:
                continue
            # copy in feature matrices
                # find raw zsc matrices but not hires model ones
            fls = list((set(glob(r'C:\Users\rccuser\code\david\whakaari\features\*zsc*.pkl'))
                - set(glob(r'C:\Users\rccuser\code\david\whakaari\features\*hires.pkl'))))
            for fl in fls: 
                os.remove(fl)

            for fl in glob(r'C:\Users\rccuser\code\david\whakaari\features\zsc_excl_{:d}\*zsc*.pkl'.format(j)): 
                fn = fl.split(os.sep)[-1]
                shutil.copyfile(fl, r'C:\Users\rccuser\code\david\whakaari\features'+os.sep+fn)

            fm = ForecastModel(ti=ti, tf=tf, window=2., overlap=0.75, look_forward=2., 
                exclude_dates = exclude_dates, data_streams=ds, root='model{:02d}_e{:d}'.format(i,j), savefile_type='pkl')
    
            drop_features = ['linear_trend_timewise','agg_linear_trend']
            fm.train(ti=ti, tf=tf, exclude_dates=exclude_dates, drop_features=drop_features, retrain=True, n_jobs=n_jobs)      

            # check out of sample performance
            # ys = fm.forecast(recalculate=True, n_jobs=n_jobs)
            # fm.plot_accuracy(ys, save=r'{:s}/model{:02d}_e{:d}_accuracy.png'.format(fm.plotdir, i, j))
            ys = fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=te+month/28., recalculate=True, 
                save=r'{:s}/model{:02d}_e{:d}_hires.png'.format(fm.plotdir, i, j), n_jobs=n_jobs,
                root=r'model{:02d}_e{:d}_hires'.format(i, j), threshold=1.0)
            # continue
            y = ys['consensus']
            ci = fm._compute_CI(y)
            y0 = y-ci
            inds = (y.index<(te-fm.dt))&(y.index>(te-fm.dtf))
            conf = y0[inds].max()
            with open(r'{:s}/model{:02d}_e{:d}_confidence.txt'.format(fm.plotdir, i, j), 'w') as fp:
                fp.write('{:4.3f}'.format(conf))
            # return
    return
    # for each combination of input data
    with open('confs.csv','w') as fpi:
        for i, ds in groups:
            # for each eruption to exlude (0 = exclude no eruptions)
            confs = []
            for j in range(1,6):
                with open(r'C:\Users\rccuser\code\david\whakaari\plots/model{:02d}_e{:d}/model{:02d}_e{:d}_confidence.txt'.format(i,j,i,j)) as fp:
                    conf = np.float(fp.readline().strip())
                confs.append(conf)
                fpi.write('{:4.3f},'.format(conf))
            cmin = np.min([*confs[:3], confs[-1]])
            fpi.write('{:4.3f}'.format(cmin))
            fpi.write('\n')

    

    # plot a forecast for a future eruption
    # tf = te+month/30
    # fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=True, 
    #     save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

    # te = fm.data.tes[1]
    # y = load_dataframe(r'D:\code\whakaari\predictions\test_hires\DecisionTreeClassifier_0000.pkl')
    # tf = y.index[-1] + month/30./10.
    # ys = fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=False, 
    #     save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_parallel(i, ds):
    # constants
    month = timedelta(days=365.25/12)
        
    # set up model
    ti = datetimeify('2011-01-01')
    tf = datetimeify('2021-01-01')
    td = TremorData()
    te = td.tes[i]
    fm = ForecastModel(ti=ti, tf=tf, window=2., overlap=0.75, 
        look_forward=2., exclude_dates=[[te-month,te+month]], data_streams=[ds], root='test', savefile_type='pkl')
    
    # set the available CPUs higher or lower as appropriate
    fm.n_jobs = 0
    fm._load_data(ti,tf)

    fdir = fm.featdir+'/zsc_excl_{:d}'.format(i+1)
    os.makedirs(fdir, exist_ok=True)
    fl = fm.featfile(ds)
    shutil.move(fl, fdir+'/'+fl.split('/')[-1])

def forecast_all():
    data_streams = ['rsam','mf','hf','dsar','rsamF','mfF','hfF','dsarF']
    data_streams = ['zsc_'+ds for ds in data_streams]
    for i in range(2,5):
        fp = partial(forecast_parallel, i)
        p = Pool(8)
        p.map(fp, data_streams)
        p.close()
        p.join()

def download_tremor():
    td = TremorData(station='WIZ')
    td.update(n_jobs=16)

def WIZ_WSRZ_correlation():

    to0,to1 = get_outage_times()

    fm0 = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=['rsam'], 
        root='test', savefile_type='pkl', station='WIZ')

    fM0,_ = fm0._load_data(fm0.data.ti, fm0.data.tf)

    fm1 = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=['rsam'], 
        root='test', savefile_type='pkl', station='WSRZ')

    fM1,_ = fm1._load_data(fm1.data.ti, fm1.data.tf)

    r4fft_0 = fM0['rsam__fft_coefficient__coeff_12__attr_"abs"']
    r4fft_1 = fM1['rsam__fft_coefficient__coeff_12__attr_"abs"']

    # for each time index, compute the number of outage indices in its prior window
    day = timedelta(days=1)
    t0 = []
    for ti in r4fft_0.index:
        if np.sum((to0>(ti-2*day))&(to0<ti)) < 6*6:
            t0.append(ti)
    t1 = []
    for ti in r4fft_1.index:
        if np.sum((to1>(ti-2*day))&(to1<ti)) < 6*6:
            t1.append(ti)

    inds = np.array(sorted(list(set(t0).intersection(set(t1)))))
    rs0, rs1 = [r4fft_0[inds],], [r4fft_1[inds],]

    rs2 = []
    for te in fm0.data.tes:
        ie = inds[(inds>(te-2*day))&(inds<te)]
        rs0.append(r4fft_0[ie])
        rs1.append(r4fft_1[ie])
        rs2.append(r4fft_0[(r4fft_0.index>(te-2*day))&(r4fft_0.index<te)])
    
    return rs0, rs1, rs2

def plot_WIZ_WSRZ_correlation():

    fl = 'WIZ_WSRZ_correlation.pkl'
    if not os.path.isfile(fl):
        rs0, rs1, rs2 = WIZ_WSRZ_correlation()
        save_dataframe([rs0, rs1, rs2], fl)
    rs0,rs1,rs2 = load_dataframe(fl)

    f = plt.figure(figsize=(6,6))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.plot(rs0[0], rs1[0], 'k.')

    X_n = []
    t_n = []
    for c,r0,r1 in zip(['b','g','c','m','r'], rs0[1:], rs1[1:]):
        # continue
        ax.plot(r0, r1, c+'o', ms=8)
        X_n += list(r0)
        t_n += list(r1)
    X_n = np.array(X_n).reshape(-1, 1)
    t_n = np.array(t_n).reshape(-1, 1)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.plot(xlim,ylim, 'k--', lw=0.5)

    # Gaussian process regression
    interpolates = []
    np.random.seed(0)
    if True:
        X_test = np.linspace(*np.log10(xlim), 100).reshape(-1, 1)
        kern = GPy.kern.RBF(input_dim=1, variance=5.0, lengthscale=5.0)
        kern = GPy.kern.Poly(input_dim=1., order=3)
        kern = GPy.kern.Spline(input_dim=1., variance = 1.0, c = 5)
        gpr = GPy.models.GPRegression(np.log10(X_n), np.log10(t_n), kern)
        # gpr = GPy.models.GPHeteroscedasticRegression(np.log10(X_n), np.log10(t_n), kern)
        gpr.optimize()
        
        y, y_var = gpr.predict(X_test)
        y = y.ravel()
        std = 2*np.sqrt(y_var.ravel())
        ax.plot(10**X_test, 10**y,'y-', zorder=4)
        ax.fill_between(10**(X_test.ravel()), 10**(y + std), 10**(y - std), color='y', alpha = 0.5, zorder=3)

    if False:
        for rsi in rs2[0]:
            ax.axvline(rsi, color = 'b')
        for rsi in rs2[2]:
            ax.axvline(rsi, color = 'c')
           
    np.random.seed(2) 
    if True:
        # y, y_var = gpr.predict()
        y = gpr.posterior_samples(np.log10(rs2[0].values).reshape(-1,1), size=1).ravel()
        ax.plot(rs2[0].values, 10**y, 'o', mec='b', mfc='w', mew=1.5, ms=8, zorder=10)
        
        y = gpr.posterior_samples(np.log10(rs2[2].values).reshape(-1,1), size=1).ravel()
        ax.plot(rs2[2].values, 10**y, 'o', mec='c', mfc='w', mew=1.5, ms=8, zorder=10)
        
        # ax.plot(rs2[2].values, 10**gpr.predict(np.log10(rs2[2].values).reshape(-1,1))[0][:,0], mec='c', mfc='w', mew=1.5, ms=8)
        
        #     cols = ['b','g','c','m','r']
        #     x,y = [fp1.loc[tp], fp2.loc[tp]]
        #     day = timedelta(days=1)
        #     for r in range(n_reps):
        #         tmp = []
        #         for i,te in enumerate(tes):
        #             ti = [t for t in tp if ((te-t) < 4*day)&((te-t) > 0*day)]
        #             if plot: ax.plot(x[ti], y[ti], cols[i]+'.', ms=5,zorder=5)

        #             try:
        #                 _ = gpr
        #                 ti = [t for t in t_interpolate if ((te-t) < 4*day)&((te-t)>0*day)]
        #                 X = np.array(fp1.loc[ti]).reshape(-1, 1)
        #                 # print(X)
        #                 if X.shape[0] > 0:
        #                     WSRZ_ft = gpr.posterior_samples(X, size=1)
        #                     for tii, Xi in zip(ti,WSRZ_ft):
        #                         tmp.append([tii,np.float64(pow(10,Xi))])
        #                     if plot: ax.plot(X.ravel(), WSRZ_ft.ravel(), cols[i]+'.', mec = cols[i], mfc = 'w', mew=1., ms=5, zorder=5)
        #             except NameError:
        #                 pass
        #             except ValueError:
        #                 pass
        #         interpolates.append(tmp)
        # except np.linalg.LinAlgError:
        #     interpolates = [[] for i in range(n_reps)]
        #     drop = True
        #     pass
    

    ax.set_xlabel('WIZ feature value')
    ax.set_ylabel('WSRZ feature value')
    plt.savefig('correlation5.png', dpi=400)

    pass

def rename():
    from shutil import move
    for i in range(6):
        fls = glob(r'C:\Users\rccuser\code\david\whakaari\features\zsc_excl_{:d}\*.pkl'.format(i))
        for fl in fls:
            fli = 'o_WIZ_'.join(fl.split('o_'))
            move(fl, fli)

def get_outage_times():
    td1 = TremorData(station='WIZ')
    td2 = TremorData(station='WSRZ')

    ts = []
    for td in [td1, td2]:
        t = []
        for ti,v0,vm,v1 in zip(td.df.index[1:-1],td.df['rsam'].values[:-2],td.df['rsam'].values[1:-1],td.df['rsam'].values[2:]):
            if abs(vm-(v0+v1)/2.)<1.e-3:
                t.append(ti)
        ts.append(np.array(t))
    
    return ts

def plot_WIZ_WSRZ_raw():
    fm0 = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=['rsam', 'zsc_rsam'], 
        root='test', savefile_type='pkl', station='WIZ')
    fm1 = ForecastModel(window=2., overlap=0.75, look_forward=2., data_streams=['rsam', 'zsc_rsam'], 
        root='test', savefile_type='pkl', station='WSRZ')

    from datetime import timedelta
    day = timedelta(days=1)

    if True:
        f = plt.figure(figsize=(12,4))
        ax1 = plt.axes([0.08, 0.10, 0.4, 0.8])
        ax2 = plt.axes([0.58, 0.10, 0.4, 0.8])

        for i,ax in zip([1,4],[ax1,ax2]):
            te = fm0.data.tes[i]

            t = [te-2*day, te+day]
            rsam0 = fm0.data.get_data(t[0], t[-1])['zsc_rsam']
            rsam1 = fm1.data.get_data(t[0], t[-1])['zsc_rsam']
            
            ax.plot(rsam0.index, rsam0.values*1.e-3, 'k-', label='WIZ')
            ax.plot(rsam1.index, rsam1.values*1.e-3, 'b-', label='WSRZ')
            ax.axvline(te, color = 'r', linestyle='--', linewidth=2, label='eruption')
            ax.legend()

            tf = t[-1]
            t0 = tf.replace(hour=0, minute=0, second=0)
            xts = [t0 - timedelta(days=i) for i in range(7)][::-1]
            lxts = [xt.strftime('%d %b') for xt in xts]
            ax.set_xticks(xts)
            ax.set_xticklabels(lxts)
            ax.set_xlim(t)
            ax.text(0.98, 0.96, rsam0.index[-1].strftime('%Y'), ha='right', va='top', transform=ax.transAxes)
            ax.set_ylabel('normalized RSAM')# [$\mu$m s$^{-1}$]')
            ax.set_ylim([0,5])
        
        plt.savefig('WIZ_WSRZ_zsc.png', dpi=400)

    if False:
        f = plt.figure(figsize=(8,4))
        ax1 = plt.axes([0.10, 0.13, 0.4, 0.8])
        ax2 = plt.axes([0.58, 0.13, 0.4, 0.8])

        for fm,c in zip([fm0,fm1],['k','b']):
            
            r = fm.data.df['rsam'].values
            lr = np.log10(r)
            lr = lr[np.where(lr==lr)]
            bins = 10**np.linspace(1, 4, int(np.sqrt(len(lr)/2)))
            h,e = np.histogram(r, bins)
            ax1.fill_between(0.5*(e[1:]+e[:-1]), 0*h, h, color=c, alpha=0.3, label=fm.station)
            ax1.set_xlim([bins[0], bins[-1]])
            
            r = fm.data.df['zsc_rsam'].values
            lr = np.log10(r)
            lr = lr[np.where(lr==lr)]
            bins = 10**np.linspace(-3, 3, int(np.sqrt(len(lr)/2)))
            h,e = np.histogram(r, bins)
            ax2.fill_between(0.5*(e[1:]+e[:-1]), 0*h, h, color=c, alpha=0.3, label=fm.station)
            ax2.set_xlim([bins[0], bins[-1]])
            
        for ax in [ax1,ax2]:
            ax.legend()
            ax.set_xscale('log')
            ax.set_yticks([])
            ax.set_ylim([0,None])
        ax1.set_ylabel('relative frequency')
        ax1.set_xlabel('RSAM [$\mu$m s$^{-1}$]')
        ax2.set_xlabel('log-normalised RSAM')

        plt.savefig('WIZ_WSRZ_dist.png', dpi=400)

if __name__ == "__main__":
    # forecast_dec2019()
    # forecast_test()
    # forecast_now()
    # forecast_scratch()
    # download_tremor()
    # forecast_all()
    build_models()
    # rename()
    # plot_WIZ_WSRZ_correlation()
    # plot_WIZ_WSRZ_raw()
    