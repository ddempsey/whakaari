import os, sys, traceback, smtplib, ssl, yagmail, shutil, argparse
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, datetimeify
from datetime import timedelta, datetime
from subprocess import Popen, PIPE
from pathlib import Path
from dateutil import tz
from time import sleep, time
import pandas as pd
import numpy as np
import twitter

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

class Alert(object):
    def __init__(self, mail_from, monitor_mail_to, alert_mail_to, keyfile):
        # notification settings
        self.mail_from = mail_from
        self.monitor_mail_to = monitor_mail_to
        self.alert_mail_to = alert_mail_to
        self.key = Key(keyfile)
        # alert settings
        self.reset()
        # time settings
        self.time_utc = datetime.utcnow()
        self.time_local = self.time_utc.replace(tzinfo=tz.gettz('UTC')).astimezone(tz.gettz('Pacific/Auckland'))
    def __repr__(self):
        pass
    def reset(self):
        self.new_alert = False
        self.debug_problem = False
        self.new_system_down = False
        self.heartbeat = False
        self.errors = []
    def set_alert_level(self):
        # create alert object to pass info
        self.reset()

        if not os.path.isfile('alert.csv'):
            return 

        # check alert condition
        with open('alert.csv','r') as fp:
            self.in_alert = int(fp.readline().strip())

        # categorize response
        if self.in_alert == -1 and not os.path.isfile('system_down'):
            # GeoNet system is not responding - broacast an alert that the Forecaster is not in operation
            _ = [os.remove(fl) for fl in ['off_alert','on_alert'] if os.path.isfile(fl)]
            Path('system_down').touch()
            self.new_system_down = True
        elif self.in_alert == 0:
            # Forecaster is not in alert
            if os.path.isfile('on_alert'):
                # If previous on alert, delete that file 
                _ = [os.remove(fl) for fl in ['system_down','on_alert'] if os.path.isfile(fl)]
                Path('off_alert').touch()
                self.post_close_alert()
            elif not os.path.isfile('off_alert'):
                # create 'no alert' file if not already existing
                Path('off_alert').touch()
        elif self.in_alert == 1:
            # Forecaster is in alert
            if os.path.isfile('off_alert'):
                # This is a new alert - broadcast
                _ = [os.remove(fl) for fl in ['system_down','off_alert'] if os.path.isfile(fl)]
            
            if not os.path.isfile('on_alert'):    
                Path('on_alert').touch()
                self.new_alert = True
                self.post_open_alert()
    # Tweeting
    def post(self, msg, media):
        try:
            api = twitter.Api(consumer_key=self.key['API_key'],
                        consumer_secret=self.key['API_secret_key'],
                        access_token_key=self.key['token'],
                        access_token_secret=self.key['token_secret'])
            status = api.PostUpdate(msg, media=media)
        except:
            pass
    def post_no_alert(self, msg=-1, media=-1):
        if msg ==-1: msg = 'No alert.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    def post_startupdate(self, msg=-1, media=-1):
        if msg ==-1: msg = 'Waking up.'
        if media ==-1: media = None
        self.post(msg, media)
    def post_open_alert(self, msg=-1, media=-1):
        if msg ==-1: 
            msg = '🚨🚨 ERUPTION ALERT TRIGGERED 🚨🚨 There is a 1 in 12 chance for an eruption to occur. Alerts last on average 5 days.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    def post_continue_alert(self, msg=-1, media=-1):
        if msg ==-1: 
            msg = '🚨🚨 YESTERDAY\'S ALERT CONTINUES 🚨🚨 There is a 1 in 12 chance for an eruption to occur during an alert. Alerts last on average 5 days.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    def post_close_alert(self, msg=-1, media=-1):
        if msg ==-1: 
            msg = 'The alert has closed.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    # Emailing
    def send_startup_email(self):
        yag = yagmail.SMTP(self.mail_from)
        yag.send(to=self.monitor_mail_to,subject='starting up',contents='whakaari forecaster is starting up',attachments=None)
    def send_email_alerts(self):
        """ sends email alerts if appropriate
        """

        if not any([self.new_alert, self.new_system_down, self.debug_problem, self.heartbeat]):
            return
        
        if self.new_alert:
            subject = "Whakaari Eruption Forecast: New alert"
            message = [
                'Whakaari Eruption Forecast model issued a new alert for {:s} local time.'.format(self.time_local.strftime('%d %b %Y at %H:%M')),
                'Alerts last 48-hours and will auto-extend if elevated activity persists.',
                'Based on historical activity, the probability of an eruption occuring during an alert is about 10%.',
                {yagmail.inline('./current_forecast.png'):'Forecast_{:s}.png'.format(self.time_local.strftime('%Y%m%d-%H%M%S'))},]
            attachment = None
            mail_to = self.alert_mail_to
        elif self.new_system_down:
            subject = "Whakaari Eruption Forecast: Not receiving data"
            message = "Whakaari Eruption Forecast model is not receiving new data. Forecasting is suspended. Latest forecast is attached."
            fcstfile = 'WhakaariForecast_latest.png'
            shutil.copyfile('current_forecast.png',fcstfile)
            attachment = fcstfile
            mail_to = self.monitor_mail_to
        elif self.debug_problem:
            subject = "Whakaari Eruption Forecast: System Raising Errors"
            message = "Whakaari Eruption Forecast model is raising errors. Summary below."
            for err in self.errors:
                with open(err,'r') as fp:
                    lns = fp.readlines()
                message += '\n{:s}\n{:s}\n'.format(err, ''.join(lns))
            attachment = self.errors
            mail_to = self.monitor_mail_to
        elif self.heartbeat:
            subject = "Whakaari Eruption Forecast is working hard for YOU!"
            message = "All is well here. Have a great day."
            attachment = None
            mail_to = self.monitor_mail_to

        yag = yagmail.SMTP(self.mail_from)
        yag.send(
            to=mail_to,
            subject=subject,
            contents=message, 
            attachments=attachment,
        )
            
class Controller(object):
    def __init__(self, mail_from, monitor_mail_to, alert_mail_to, keyfile, test=False):
        self.test = test
        self.alert = Alert(mail_from,monitor_mail_to,alert_mail_to,keyfile)
    def run(self):
        """ Top-level function to update forecast.
        """
        # check notifications working
        self.alert.send_startup_email()
        self.alert.post_startupdate()

        update_geonet_err_count = 0                     # counter to allow Nmax geonet update errors
        current_forecast_err_count = 0
        geonet_err_max = 24*6 
        if not self.test:
            heartbeat_update = 24*3600
        else:
            heartbeat_update = 5
        tstart = time()
        while True:
            # take start time of look
            t0 = time()
            print('running...')
            # prepare directory
            if os.path.isfile('lock'):
                print('lock file detected, aborting')
                break
            clean()
            errors = []

            # update forecast
            command=['python','controller.py','-m']
            if not self.test: 
                command.append('update_forecast')
            else:
                command.append('update_forecast_test')
            # spawn subprocess
            with open("whakaari_stdout.txt","wb") as out, open("whakaari_stderr.txt","wb") as err:
                p=Popen(command, stdout=out, stderr=err)
                p.communicate()
                
            if os.path.isfile('run_forecast.err'):
                errors.append('run_forecast.err')
                
            if os.path.isfile('update_geonet.err'):
                if update_geonet_err_count > geonet_err_max:
                    # too many errs
                    with open('update_geonet.err','w') as fp:
                        fp.write('geonet has not been responding for 24 hours')
                    errors.append('update_geonet.err')
                else:
                    # update count
                    update_geonet_err_count += 1
                    wait = 600 - (time() - t0)
                    if wait > 0: sleep(wait)
                    continue
            else:
                # reset counter
                update_geonet_err_count = 0
            
            # set alert, check up to date
            self.alert.set_alert_level()
            if len(errors) > 0:
                self.alert.errors += errors
                self.alert.debug_problem = True

            # copy files
            if not os.path.isfile('current_forecast.png'):
                if current_forecast_err_count < 5:
                    current_forecast_err_count += 1
                else:
                    with open('controller.err','w') as fp:
                        fp.write("'current_forecast.png' was not generated")
                    self.alert.debug_problem = True
                    errors.append('controller.err')
            else:
                current_forecast_err_count = 0
                shutil.copyfile('current_forecast.png','/var/www/html/current_forecast.png')

            # send alerts
            self.alert.send_email_alerts()

            # create controller terminate file to prevent endless error email
            if self.alert.debug_problem:
                Path('lock').touch()
                break

            # wait out the rest of 10 mins
            if not self.test:
                wait = 600 - (time() - t0)
                if wait > 0:
                    sleep(wait)

            # check to send heartbeat email - every 24 hrs
            if (time() - tstart) > heartbeat_update:
                self.alert.reset()
                self.alert.heartbeat = True
                tstart = time()
                self.alert.send_email_alerts()
                if self.alert.in_alert == 1:
                    self.alert.post_continue_alert()
                else:
                    self.alert.post_no_alert()

def rebuild_hires_features():
    ''' Call this once if the feature matrix file has been damaged and needs to be rebuilt without murdering memory
    '''
    # model from 2011 to present day (td.tf)
    station = 'WSRZ'
    root = 'online_forecaster_'+station
    td = TremorData(station=station) 
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75, station=station,  
        look_forward=2, data_streams=data_streams, root=root, savefile_type='pkl')
        
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2012-01-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
        retrain=False, n_jobs=1)    
        
    fl = '../features/{:s}_hires_features.csv'.format(root)
    if os.path.isfile(fl):
        t = pd.to_datetime(pd.read_csv(fl, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
        ti0,tf0 = t[0],t[-1]
    else:
        tf0 = datetimeify('2020-08-01')
    Ndays = int((fm.data.tf - tf0).total_seconds()/(24*3600))
    day = timedelta(days=1)
    for i in range(1,Ndays-1):
        # forecast from beginning of training period at high resolution
        fm.hires_forecast(ti=datetimeify('2020-08-01'), tf=tf0+i*day, recalculate=False, n_jobs=1) 
        
def update_forecast():
    ''' Update model forecast.
    '''
    try:
        # constants
        month = timedelta(days=365.25/12)
        day = timedelta(days=1)
            
        # pull the latest data from GeoNet
        td = TremorData(station='WIZ')
        td.update()

        td0 = TremorData(station='WSRZ')
        td0.update()
    except Exception:
        fp = open('update_geonet.err','w')
        fp.write('{:s}\n'.format(traceback.format_exc()))
        fp.close()
        return

    try:
        # model from 2011 to present day (td.tf)
        data_streams = ['rsam','mf','hf','dsar']
        fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75,  
            look_forward=2, data_streams=data_streams, root='online_forecaster_WIZ',savefile_type='pkl')
        fm0 = ForecastModel(ti='2012-01-01', tf=td0.tf, window=2, overlap=0.75,  
            look_forward=2, data_streams=data_streams, root='online_forecaster_WSRZ',savefile_type='pkl')
        
        # The online forecaster is trained using all eruptions in the dataset. It only
        # needs to be trained once, or again after a new eruption.
        # (Hint: feature matrices can be copied from other models to avoid long recalculations
        # providing they have the same window length and data streams. Copy and rename 
        # to *root*_features.csv)
        drop_features = ['linear_trend_timewise','agg_linear_trend']
        fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
            retrain=False, n_jobs=1)      
        fm0.train(ti='2012-01-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
            retrain=False, n_jobs=1)      
        
        # forecast from beginning of training period at high resolution
        tf = datetime.utcnow()
        ys = fm.hires_forecast(ti=datetimeify('2020-01-01'), tf=fm.data.tf, recalculate=False, n_jobs=1) 
        ys0 = fm0.hires_forecast(ti=datetimeify('2020-01-01'), tf=fm0.data.tf, recalculate=False, n_jobs=1) 

        plot_dashboard(ys,ys0,fm,fm0,'current_forecast.png')

        al = (ys['consensus'].values[ys.index>(tf-fm.dtf)] > 0.8)*1.
        al0 = (ys0['consensus'].values[ys0.index>(tf-fm0.dtf)] > 0.8)*1.
        if len(al) == 0 and len(al0) == 0:
            in_alert = -1
        else:
            in_alert = int(np.max([np.max(al0), np.max(al)]))
        with open('alert.csv', 'w') as fp:                
            fp.write('{:d}\n'.format(in_alert))
            
    except Exception:
        fp = open('run_forecast.err','w')
        fp.write('{:s}\n'.format(traceback.format_exc()))
        fp.close()
        return

def plot_dashboard(ys,ys0,fm,fm0,save):
    # parameters
    threshold = 0.8
    
    # set up figures and axes
    f = plt.figure(figsize=(16,8))
    ax1 = plt.axes([0.08, 0.55, 0.4, 0.4])
    ax2 = plt.axes([0.08, 0.08, 0.4, 0.4])
    ax3 = plt.axes([0.56, 0.55, 0.4, 0.4])
    ax4 = plt.axes([0.56, 0.08, 0.4, 0.4])
    
    t = pd.to_datetime(ys.index.values)
    t0 = pd.to_datetime(ys0.index.values)
    rsam = fm.data.get_data(t[0], t[-1])['rsam']
    trsam = rsam.index
    rsam0 = fm0.data.get_data(t0[0], t0[-1])['rsam']
    trsam0 = rsam0.index
    
    t = to_nztimezone(t)
    trsam = to_nztimezone(trsam)
    t0 = to_nztimezone(t0)
    trsam0 = to_nztimezone(trsam0)
    
    ts = [t[-1], trsam[-1]]
    tmax = np.max(ts)
    ts0 = [t0[-1], trsam0[-1]]
    tmax0 = np.max(ts0)
    
    ax2.set_xlabel('Local time')
    ax3.set_xlabel('Local time')
    y = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
    y0 = np.mean(np.array([ys0[col] for col in ys0.columns]), axis=0)
    
    ax2.set_xlim([tmax-timedelta(days=7), tmax])
    ax3.set_xlim([tmax0-timedelta(days=7), tmax0])
    ax1.set_xlim([t[0], np.max([tmax, tmax0])])
    ax1.set_title('Whakaari Eruption Forecast')

    for ax in [ax1,ax2]:
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0,0.25,0.50,0.75,1.00])
        ax.set_ylabel('ensemble mean')
    
        # consensus threshold
        ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

        # modelled alert
        ax.plot(t, y, 'c-', label='ensemble mean', zorder=4, lw=0.75)
        ax_ = ax.twinx()
        ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
        ax_.set_ylim([0,5])
        ax_.set_xlim(ax.get_xlim())
        ax_.plot(trsam, rsam.values*1.e-3, 'k-', lw=0.75)

        for tii,yi in zip(t, y):
            if yi > threshold:
                ax.fill_between([tii, tii+fm.dtf], [0,0], [100,100], color='y', zorder=3)
                
        ax.fill_between([], [], [], color='y', label='eruption forecast')
        ax.plot([],[],'k-', lw=0.75, label='RSAM')
    
    ax1.legend(loc=1, ncol=2)

    for ax in [ax3]:
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0,0.25,0.50,0.75,1.00])
        ax.set_ylabel('ensemble mean')
    
        # consensus threshold
        ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

        # modelled alert
        ax.plot(t0, y0, 'c-', label='ensemble mean', zorder=4, lw=0.75)
        ax_ = ax.twinx()
        ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
        ax_.set_ylim([0,5])
        ax_.set_xlim(ax.get_xlim())
        ax_.plot(trsam0, rsam0.values*1.e-3, 'k-', lw=0.75)

        for tii,yi in zip(t0, y0):
            if yi > threshold:
                ax.fill_between([tii, tii+fm0.dtf], [0,0], [100,100], color='y', zorder=3)
                
        ax.fill_between([], [], [], color='y', label='eruption forecast')
        ax.plot([],[],'k-', lw=0.75, label='RSAM')
        ax.set_title('WSRZ forecast')
    
    if xlim is not None: 
        ax2.set_xlim(xlim)
        ax3.set_xlim(xlim)
        tmax = xlim[-1] 
        tmax0 = xlim[-1] 

    tf = tmax 
    tf0 = tmax0
    ta = tf.replace(hour=0, minute=0, second=0)
    xts = [ta - timedelta(days=i) for i in range(7)][::-1]
    lxts = [xt.strftime('%d %b') for xt in xts]
    ax2.set_xticks(xts)
    ax2.set_xticklabels(lxts)
    tfi  = fm.data.tf
    tfi = to_nztimezone([tfi])[0]
    ax2.text(0.025, 0.95, 'model last updated {:s}'.format(tfi.strftime('%H:%M, %d %b %Y')), size = 12, ha = 'left', 
        va = 'top', transform=ax2.transAxes)
    
    ta = datetimeify('2020-01-01')
    xts = [ta.replace(month=i) for i in range(1, tf.month+1)]
    lxts = [xt.strftime('%d %b') for xt in xts]
    ax1.set_xticks(xts)
    ax1.set_xticklabels(lxts)
    ax1.text(0.025, 0.95, ta.strftime('%Y'), size = 12, ha = 'left', 
        va = 'top', transform=ax1.transAxes)

    plt.savefig(save, dpi=300)
    plt.close(f)

def update_forecast_bkp():
    ''' Update model forecast.
    '''
    try:
        # constants
        month = timedelta(days=365.25/12)
        day = timedelta(days=1)
            
        # pull the latest data from GeoNet
        td = TremorData()
        td.update()

        td0 = TremorData(station='WSRZ')
        td0.update()
    except Exception:
        fp = open('update_geonet.err','w')
        fp.write('{:s}\n'.format(traceback.format_exc()))
        fp.close()
        return

    try:
        # model from 2011 to present day (td.tf)
        data_streams = ['rsam','mf','hf','dsar']
        fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75,  
            look_forward=2, data_streams=data_streams, root='online_forecaster',savefile_type='pkl')
        
        # The online forecaster is trained using all eruptions in the dataset. It only
        # needs to be trained once, or again after a new eruption.
        # (Hint: feature matrices can be copied from other models to avoid long recalculations
        # providing they have the same window length and data streams. Copy and rename 
        # to *root*_features.csv)
        drop_features = ['linear_trend_timewise','agg_linear_trend']
        fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
            retrain=False, n_jobs=1)      
        
        # forecast from beginning of training period at high resolution
        tf = datetime.utcnow()
        if (tf - td.df.index[-1])>(day/6):
            data = np.log10(td.get_data()['rsam'].values)
            mean0,std0 = np.mean(data), np.std(data)
            # scale WSRZ
            data = td0.df['rsam'].dropna()
            data = data[data>1.e-5]
            data = np.log10(data.values)
            mean,std = np.mean(data), np.std(data)
            alt_rsam = 10**((np.log10(td0.df['rsam'])-mean)/std*std0+mean0)
        else:
            alt_rsam = None
        ys = fm.hires_forecast(ti=datetimeify('2020-01-01'), tf=fm.data.tf, recalculate=False, 
            save='current_forecast.png', nztimezone=True, n_jobs=1, alt_rsam=alt_rsam) 

        al = (ys['consensus'].values[ys.index>(tf-fm.dtf)] > 0.8)*1.
        if len(al) == 0:
            in_alert = -1
        else:
            in_alert = int(np.max(al))
        with open('alert.csv', 'w') as fp:                
            fp.write('{:d}\n'.format(in_alert))
            
    except Exception:
        fp = open('run_forecast.err','w')
        fp.write('{:s}\n'.format(traceback.format_exc()))
        fp.close()
        return

def update_forecast_test():
    with open('alert.csv', 'w') as fp:    
        in_alert = int(np.random.rand()>0.5)            
        fp.write('{:d}\n'.format(in_alert))
    shutil.copyfile('../current_forecast.png','current_forecast.png')

def plot_date(dt):
    td = TremorData()
    dt = datetimeify(dt)
    day = timedelta(days=1) 
    # model from 2011 to present day (td.tf)
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75,
            look_forward=2, data_streams=data_streams, root='online_forecaster', savefile_type='pkl')

    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
        retrain=False, n_jobs=1)

    # forecast from beginning of training period at high resolution
    fm.hires_forecast(ti=datetimeify('2020-01-01'), tf=dt, recalculate=False,
        save='forecast_{:s}.png'.format(dt.strftime('%Y%m%d')), nztimezone=True, n_jobs=1, xlim=[dt-7*day,dt])

def clean():
    # remove files
    fls = ['current_forecast.png', 'run_forecast.err', 'update_geonet.err','controller.err']
    _ = [os.remove(fl) for fl in fls if os.path.isfile(fl)]

def Key(keyfile):
    with open(keyfile, 'r') as fp:
        lns = fp.readlines()
    return dict([ln.strip().split(':') for ln in lns]) 

if __name__ == "__main__":  
  # set parameters
    keyfile = r'/home/ubuntu/twitter_keys.txt'
    mail_from = 'noreply.whakaariforecaster@gmail.com'
    
    # heartbeat and error raising emails
    monitor_mail_to_file = r'/home/ubuntu/whakaari_monitor_mail_to.txt'
    with open(monitor_mail_to_file, 'r') as fp:
        lns = fp.readlines()
    monitor_mail_to = [ln.strip() for ln in lns]
    
    # forecast alert emails
    alert_mail_to_file = r'/home/ubuntu/whakaari_alert_mail_to.txt'
    with open(alert_mail_to_file, 'r') as fp:
        lns = fp.readlines()
    alert_mail_to = [ln.strip() for ln in lns]

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", 
        type=str,
        default='controller',
        help="flag indicating how controller is to run")
    args = parser.parse_args()
    if args.m == 'controller':
        controller = Controller(mail_from, monitor_mail_to, alert_mail_to, keyfile)
        controller.run()
    elif args.m == 'controller-test':
        controller = Controller(mail_from, monitor_mail_to, alert_mail_to, keyfile, test=True)
        controller.run()
    elif args.m == 'update_forecast':
        update_forecast()
    elif args.m == 'update_forecast_test':
        update_forecast_test()
    elif args.m == 'plot_date':
        plot_date('2020-09-18')
        plot_date('2020-09-03') 
    elif args.m == 'rebuild_hires_features':
        rebuild_hires_features()
    
