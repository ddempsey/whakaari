import os, sys, traceback, yagmail, shutil, argparse
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, to_nztimezone, datetimeify, load_dataframe
from datetime import timedelta, datetime
from subprocess import Popen, PIPE
from pathlib import Path
from dateutil import tz
from matplotlib import pyplot as plt
from time import sleep, time
import pandas as pd
import numpy as np
import twitter
from functools import partial
from dateutil.relativedelta import relativedelta

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

# CHANGE THESE GLOBAL VARIABLES 
THR = 0.85          # threshold to issue alarm
TI = datetimeify('2021-05-15')      # date to start forecasting from

class Alert(object):
    def __init__(self, mail_from, monitor_mail_to_file, alert_mail_to_file, keyfile):
        # notification settings
        self.mail_from = mail_from
        self.monitor_mail_to_file = monitor_mail_to_file
        self.alert_mail_to_file = alert_mail_to_file
        self.monitor_mail_to = get_emails(monitor_mail_to_file)
        self.alert_mail_to = get_emails(alert_mail_to_file)
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
        if self.key is None:
            return
        try:
            api = twitter.Api(consumer_key=self.key['API_key'],
                        consumer_secret=self.key['API_secret_key'],
                        access_token_key=self.key['token'],
                        access_token_secret=self.key['token_secret'])
            api.PostUpdate(msg, media=media)
        except:
            pass
    def post_no_alert(self, msg=-1, media=-1):
        if msg ==-1: msg = 'No alarm. The probability of an eruption is 0.06% (about 1 in 2000) over the next 48 hours.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    def post_startupdate(self, msg=-1, media=-1):
        if msg ==-1: msg = 'Waking up.'
        if media ==-1: media = None
        self.post(msg, media)
    def post_open_alert(self, msg=-1, media=-1):
        if msg ==-1: 
            msg = '🚨🚨 ERUPTION ALARM TRIGGERED 🚨🚨 There is a 1 in 4 chance for an eruption to occur during the average 6.5 day alarm. The likelihood of eruption is 134 times higher than normal.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    def post_continue_alert(self, msg=-1, media=-1):
        if msg ==-1: 
            msg = '🚨🚨 YESTERDAY\'S ALARM CONTINUES 🚨🚨 There is a 1 in 4 chance for an eruption to occur during the average 6.5 day alarm. The likelihood of eruption is 134 times higher than normal.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    def post_close_alert(self, msg=-1, media=-1):
        if msg ==-1: 
            msg = 'The alert has closed.'
        if media ==-1: media = 'current_forecast.png'
        self.post(msg, media)
    # Emailing
    def send_startup_email(self):
        if any([var is None for var in [self.mail_from,self.monitor_mail_to]]):
            return
        yag = yagmail.SMTP(self.mail_from)
        yag.send(to=self.monitor_mail_to,subject='starting up',contents='whakaari forecaster is starting up',attachments=None)
    def send_email_alerts(self):
        """ sends email alerts if appropriate
        """

        if not any([self.new_alert, self.new_system_down, self.debug_problem, self.heartbeat]):
            return
        
        if self.new_alert:
            subject = "Whakaari Eruption Forecast: New alarm"
            message = [
                'Whakaari Eruption Forecast model issued a new alarm for {:s} local time.'.format(self.time_local.strftime('%d %b %Y at %H:%M')),
                'Alerts last 48 hours and will auto-extend if elevated activity persists.',
                'Based on historic performance, the probability of an eruption occuring during an alarm is 8.2% in 48 hours.']
            attachment = './current_forecast.png'
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

        if any([var is None for var in [self.mail_from,mail_to]]):
            return

        yag = yagmail.SMTP(self.mail_from)
        yag.send(
            to=mail_to,
            subject=subject,
            contents=message, 
            attachments=attachment,
        )
            
class Controller(object):
    def __init__(self, mail_from, monitor_mail_to_file, alert_mail_to_file, keyfile, test=False):
        self.test = test
        self.alert = Alert(mail_from,monitor_mail_to_file,alert_mail_to_file,keyfile)
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
            # update email addresses
            self.alert.monitor_mail_to = get_emails(self.alert.monitor_mail_to_file, self.alert.monitor_mail_to)
            self.alert.alert_mail_to = get_emails(self.alert.alert_mail_to_file, self.alert.alert_mail_to)
            if self.test:
                self.alert.alert_mail_to = self.alert.monitor_mail_to
        
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
    fm = ForecastModel(ti='2013-05-01', tf=td.tf, window=2, overlap=0.75, station=station,  
        look_forward=2, data_streams=data_streams, root=root, savefile_type='pkl')
        
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2013-05-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
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
        
def update_forecast_v1():
    ''' Update model forecast.
    '''
    try:
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
        fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75, station='WIZ', 
            look_forward=2, data_streams=data_streams, root='online_forecaster_WIZ',savefile_type='pkl')
        fm0 = ForecastModel(ti='2013-05-01', tf=td0.tf, window=2, overlap=0.75, station='WSRZ',
            look_forward=2, data_streams=data_streams, root='online_forecaster_WSRZ',savefile_type='pkl')
        
        # The online forecaster is trained using all eruptions in the dataset. It only
        # needs to be trained once, or again after a new eruption.
        # (Hint: feature matrices can be copied from other models to avoid long recalculations
        # providing they have the same window length and data streams. Copy and rename 
        # to *root*_features.csv)
        drop_features = ['linear_trend_timewise','agg_linear_trend']
        fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
            retrain=False, n_jobs=1)      
        fm0.train(ti='2013-05-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
            retrain=False, n_jobs=1)      
        
        # forecast from beginning of training period at high resolution
        tf = datetime.utcnow()
        ys = fm.hires_forecast(ti=datetimeify('2020-08-01'), tf=fm.data.tf, recalculate=True, n_jobs=1)
        ys0 = fm0.hires_forecast(ti=datetimeify('2020-08-01'), tf=fm0.data.tf, recalculate=True, n_jobs=1) 

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

def update_forecast_v2():
    ''' Update model forecast.
    '''
    try:
        # pull the latest data from GeoNet
        td = TremorData(station='WIZ')
        td.update()
    except Exception:
        fp = open('update_geonet.err','w')
        fp.write('{:s}\n'.format(traceback.format_exc()))
        fp.close()
        return

    try:

        # model from 2011 to present day (td.tf)
        data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
        fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75, station='WIZ', 
            look_forward=2, data_streams=data_streams, root='online_forecaster_WIZ',savefile_type='pkl')
        
        # The online forecaster is trained using all eruptions in the dataset. It only
        # needs to be trained once, or again after a new eruption.
        # (Hint: feature matrices can be copied from other models to avoid long recalculations
        # providing they have the same window length and data streams. Copy and rename 
        # to *root*_features.csv)
        drop_features = ['linear_trend_timewise','agg_linear_trend']  
        drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
        
        fm.train(ti='2011-01-01', tf='2021-01-01', drop_features=drop_features, Ncl=500,
            retrain=False, n_jobs=1)      
        
        # forecast from beginning of training period at high resolution
        tf = datetime.utcnow()
        ys = fm.hires_forecast(ti=TI, tf=fm.data.tf, recalculate=False, n_jobs=1)
        
        dashboard_v2(ys['consensus'], fm, 'current_forecast.png')

        al = (ys['consensus'].values[ys.index>(tf-fm.dtf)] > THR)*1.
        if len(al) == 0:
            in_alert = -1
        else:
            in_alert = 1
        with open('alert.csv', 'w') as fp:                
            fp.write('{:d}\n'.format(in_alert))
            
    except Exception:
        fp = open('run_forecast.err','w')
        fp.write('{:s}\n'.format(traceback.format_exc()))
        fp.close()
        return

def dashboard_v2(ys,fm,save):
    # parameters
    threshold = 0.85
    
    # set up figures and axes
    f = plt.figure(figsize=(16,8))
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.36])
    ax2 = plt.axes([0.54, 0.55, 0.4, 0.36])
    ax3 = plt.axes([0.05, 0.08, 0.4, 0.36])
    ax4 = plt.axes([0.63, 0.08, 0.3, 0.36])
    axs = [ax1,ax2,ax3]
    ax_s = [ax.twinx() for ax in axs]
    
    t = pd.to_datetime(ys.index.values)
    rsam = fm.data.get_data(t[0], t[-1])['rsam']
    t = np.array(to_nztimezone(t))
    trsam = to_nztimezone(rsam.index)

    cal = load_dataframe('online_WIZ_isotonic_calibrator.pkl')
    pf = partial(lambda c,x : c.predict([x])[0] if not is_iterable(x) else c.predict(x), cal)
    
    ts = [t[-1], trsam[-1]]
    tmax = np.max(ts)

    # ax2.set_xlabel('Local time')
    # ax3.set_xlabel('Local time')
    y = ys.values       # model output
    dy = fm._compute_CI(y)
    p = pf(y)           # eruption probability (in 48-hours)
    plo = pf(y-dy)           # confidence interval
    phi = pf(y+dy)           
    pc = integrate_probability(p, 48*6)
    p0 = pf(0.)         # 'normal' probability
        
    lims = [[t[0], t[-1]],[tmax-timedelta(days=7), tmax]]
    for ax, ax_, lim in zip([ax1,ax2], ax_s, lims):
        # plot RSAM
        ax_.plot(trsam, rsam.values*1.e-3, 'k-', lw=0.5)
        ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
        ax_.set_ylim([0,5])
        ax_.yaxis.set_label_position("left")
        ax_.yaxis.tick_left()

        # plot model output
        ax.fill_between(t, y-dy, y+dy, color='c', zorder=0, linewidth=0., alpha=0.5)
        ax.plot(t, y, 'c-', label='model', zorder=4, lw=0.5)
        ax.set_ylim([0, 1.05])
        ax.set_yticks([0,0.25,0.50,0.75,1.00])
        ax.set_ylabel('model output')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        tks, tls = get_ticks(*lim)
        ax.set_xticks(tks)
        ax.set_xticklabels(tls)
    
        # consensus threshold
        ax.axhline(threshold, color='k', linestyle=':', label='alarm trigger', zorder=4)

        # modelled alert
        for tii,yi in zip(t, y):
            if yi > threshold:
                ax.fill_between([tii, tii+fm.dtf], [0,0], [100,100], color='y', zorder=3)
                
        ax.fill_between([], [], [], color='y', label='alarm period')
        ax.plot([],[],'k-', lw=0.5, label='RSAM')
        ax.plot([],[],'b-', lw=0.5, label='instantaneous\nprobability')
        ax.plot([],[],'b-', lw=2, label='time-averaged\nprobability')
        ax.set_xlim(lim)
        
    # plot probability trace
    ax = ax3
    ax_2 = ax.twinx()
    ax_2.spines['right'].set_position(('outward', 60))
    ax_2.set_frame_on(True)
    ax_2.patch.set_visible(False)
    ax_2.yaxis.set_label_position("right")
    ax_2.set_ylabel('likelihood relative to \'normal\'\n(probability gain)')
    
    inds = np.where((t>=lim[0])&(t<=lim[-1]))
    ti = t[inds]
    pi = p[inds]
    pci = pc[inds]
    ploi = plo[inds]
    phii = phi[inds]
    ax.fill_between(ti, ploi*100, phii*100, color='b', zorder=0, linewidth=0., alpha=0.5)
    ax.plot(ti, pi*100, 'b-', lw=0.5)
    ax.plot(ti, pci*100, 'b-', lw=2)
    ax_2.set_ylim(ax_2.get_ylim()/(100*p0))

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax_.set_xlim(lim)
    ax.set_xlim(lim)
    ax_2.set_xlim(lim)
    ax.set_xticks(tks)
    ax.set_xticklabels(tls)
    ax.set_xlim(lim)
    
    ax1.legend(loc=1, ncol=3)

    # plot time-averaged probability
    ax_s[2].plot(trsam, rsam.values*1.e-3, 'k-', lw=0.5)
    ax_s[2].set_ylabel('RSAM [$\mu$m s$^{-1}$]')
    ax_s[2].set_ylim([0,5])
    ax_s[2].yaxis.set_label_position("left")
    ax_s[2].yaxis.tick_left()
    ax_s[2].set_xlim(lim)

    ax.set_ylabel('48-hr eruption probability [%]')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xticks(tks)
    ax.set_xticklabels(tls)
    ax.set_xlim(lim)

    ax4.set_xlim([0,1]); ax4.set_xticks([])
    ax4.set_ylim([0,1]); ax4.set_yticks([])

    ax4_str = [
        'ALARMS',
        'alarm threshold = 0.85',
        '48-hr eruption probability INSIDE alarm = 7.4%',
        '48-hr eruption probability OUTSIDE alarm = 0.06%',
        'alarm probability gain = x134',
        'average alarm length = 6.5 days',
        'probability of eruption inside alarm = 24%',
        'long-term alarm duration = 4.3%',
        '',
        'PROBABILITY',
        'instantaneous 48-hr eruption probability = {:3.2f}%'.format(p[-1]*100),
        'time to exceed 10$^{-4}$ eruption risk'+' = {:2.1f} hours'.format(48*np.log(1.-1.e-4)/np.log(1.-p[-1])),
        'time-averaged 48-hr eruption probability = {:3.2f}%'.format(pc[-1]*100),
    ]
    
    ax4.annotate('\n'.join(ax4_str),(0.1,0.5), ha='left', va='center')
    ax4.axis('off')
   
    ax1.set_title('Historic Forecast')
    tfi = to_nztimezone([fm.data.df.index[-1]])[0]
    ax2.set_title('7-day Forecast - last updated {:s}'.format(tfi.strftime('%H:%M, %d %b %Y')))
    ax3.set_title('Probability')
    # ax4.set_title('Risk calculations')

    plt.savefig(save, dpi=400)
    plt.close(f)

def plot_dashboard(ys,ys0,fm,fm0,save):
    # parameters
    threshold = 0.8
    
    # set up figures and axes
    f = plt.figure(figsize=(16,8))
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.36])
    ax2 = plt.axes([0.05, 0.08, 0.4, 0.36])
    ax3 = plt.axes([0.56, 0.55, 0.4, 0.36])
    ax4 = plt.axes([0.56, 0.08, 0.4, 0.36])
    
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

    for ax in [ax1,ax2]:
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0,0.25,0.50,0.75,1.00])
        ax.set_ylabel('ensemble mean')
    
        # consensus threshold
        ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

        # modelled alert
        ax.plot(t, y, 'c-', label='WIZ ensemble mean', zorder=4, lw=0.75)
        ax_ = ax.twinx()
        ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
        ax_.set_ylim([0,5])
        ax_.set_xlim(ax.get_xlim())
        ax_.plot(trsam, rsam.values*1.e-3, 'k-', lw=0.75)

        for tii,yi in zip(t, y):
            if yi > threshold:
                ax.fill_between([tii, tii+fm.dtf], [0,0], [100,100], color='y', zorder=3)
                
        ax.fill_between([], [], [], color='y', label='WIZ alert')
        ax.plot([],[],'k-', lw=0.75, label='RSAM (WIZ)')
    
    # th,time = np.genfromtxt('risk.txt', delimiter=',', skip_header=1).T
    # risk  = []
    # for yi in y:
    #     i = np.argmin(abs(th-yi))+1
    #     risk.append(np.min(time[:i]))
    # ax4.plot(t, risk, 'k-', lw=0.75)
    # ax4.set_yscale('log')
    # ax4.set_yticks([1./60, 10/60., 1., 6., 24., 24*7])
    # ax4.set_yticklabels(['1 min','10 mins','1 hr','6 hrs','1 day','1 week'])
    for tii,yi in zip(t0, y0):
        if yi > threshold:
            ax1.fill_between([tii, tii+fm0.dtf], [0,0], [100,100], color=[0.5,0.5,0.5], zorder=3)
    ax1.plot(t0, y0, 'm-', label='WSRZ ensemble mean', zorder=4, lw=0.75)
    ax1.fill_between([], [], [], color=[0.5,0.5,0.5], label='WSRZ alert')
    ax4.set_xlim([0,1]); ax4.set_xticks([])
    ax4.set_ylim([0,1]); ax4.set_yticks([])
    ax4.text(0.5,0.5,'Under Construction',fontstyle='italic',size=12,ha='center',va='center')
    ax4.set_title('Time to exceed 10$^{-4}$ annual risk (experimental)')

    ax1.legend(loc=1, ncol=3)

    ax3.set_ylim([-0.05, 1.05])
    ax3.set_yticks([0,0.25,0.50,0.75,1.00])
    ax3.set_ylabel('ensemble mean')

    # consensus threshold
    ax3.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

    # modelled alert
    ax3.plot(t0, y0, 'm-', label='ensemble mean', zorder=4, lw=0.75)
    ax_ = ax3.twinx()
    ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
    ax_.set_ylim([0,5])
    ax_.set_xlim(ax3.get_xlim())
    ax_.plot(trsam0, rsam0.values*1.e-3, 'k-', lw=0.75)

    for tii,yi in zip(t0, y0):
        if yi > threshold:
            ax3.fill_between([tii, tii+fm0.dtf], [0,0], [100,100], color='y', zorder=3)
            
    ax3.fill_between([], [], [], color='y', label='eruption forecast')
    ax3.plot([],[],'k-', lw=0.75, label='RSAM')
    
    ax1.set_title('Whakaari Eruption Forecast (Historic)')
    ax2.set_title('WIZ forecast')
    ax3.set_title('WSRZ forecast (experimental)')
    ax4.set_title('Risk calculation (experimental)')

    tf = tmax 
    ta = tf.replace(hour=0, minute=0, second=0)
    xts = [ta - timedelta(days=i) for i in range(7)][::-1]
    lxts = [xt.strftime('%d %b') for xt in xts]
    ax2.set_xticks(xts)
    ax2.set_xticklabels(lxts)
    tfi  = fm.data.tf
    tfi = to_nztimezone([tfi])[0]
    ax2.text(0.025, 0.95, 'model last updated {:s}'.format(tfi.strftime('%H:%M, %d %b %Y')), size = 12, ha = 'left', 
        va = 'top', transform=ax2.transAxes)

    tf0 = tmax0
    ta0 = tf0.replace(hour=0, minute=0, second=0)
    xts = [ta0 - timedelta(days=i) for i in range(7)][::-1]
    lxts = [xt.strftime('%d %b') for xt in xts]
    ax3.set_xticks(xts)
    ax3.set_xticklabels(lxts)
    tfi  = fm0.data.tf
    tfi = to_nztimezone([tfi])[0]
    ax3.text(0.025, 0.95, 'model last updated {:s}'.format(tfi.strftime('%H:%M, %d %b %Y')), size = 12, ha = 'left', 
        va = 'top', transform=ax3.transAxes)
    
    ta = datetimeify('2020-08-01')
    xts = [ta.replace(month=i) for i in range(1, tf.month+1)]
    lxts = [xt.strftime('%d %b') for xt in xts]
    ax1.set_xticks(xts)
    ax1.set_xticklabels(lxts)
    ax1.text(0.025, 0.95, ta.strftime('%Y'), size = 12, ha = 'left', 
        va = 'top', transform=ax1.transAxes)
    ax1.set_xlim([datetimeify('2020-08-01'), np.max([tmax, tmax0])])

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
    if keyfile is None:
        return None
    with open(keyfile, 'r') as fp:
        lns = fp.readlines()
    return dict([ln.strip().split(':') for ln in lns]) 

def get_emails(from_file, prev=None):
    ''' Parse email addresses from file.

        Parameters:
        -----------
        from_file : str
            path to file with addresses

        Returns:
        --------
        monitor_mail_to : list
            email addresses (strings) in file
    '''
    if from_file is None: 
        return None
    try:
        with open(from_file, 'r') as fp:
            lns = fp.readlines()
        monitor_mail_to = [ln.strip() for ln in lns]
        return monitor_mail_to
    except Exception as e:
        if prev is not None:
            fp = open('.'.join(from_file.split('.')[:-1])+'.err','w')
            fp.write('{:s}\n'.format(traceback.format_exc()))
            fp.close()
            return prev
        else:
            raise e

def integrate_probability(p, N):
    pi = np.concatenate([np.zeros(N)+p[0], p])
    Np = len(pi)
    return np.mean([(1.-i/N)*pi[N-i:Np-i] for i in range(N)], axis=0)*2.

def get_ticks(tmin, tmax):
    
    dt = (tmax-tmin).total_seconds()
    if dt < 10.*24*3600:
        ndays = int(np.ceil(dt/(24*3600)))+1
        t0 = tmax.replace(hour=0, minute=0, second=0)
        xts = [t0 - timedelta(days=i) for i in range(ndays)][::-1]
        lxts = [xt.strftime('%d %b').lstrip('0') for xt in xts]
    elif dt < 20.*24*3600:
        ndays = int(np.ceil(dt/(24*3600))/2)+1
        t0 = tmax.replace(hour=0, minute=0, second=0)
        xts = [t0 - timedelta(days=2*i) for i in range(ndays)][::-1]
        lxts = [xt.strftime('%d %b').lstrip('0') for xt in xts]
    elif dt < 70.*24*3600:
        ndays = int(np.ceil(dt/(24*3600))/7)
        t0 = tmax.replace(hour=0, minute=0, second=0)
        xts = [t0 - timedelta(days=7*i) for i in range(ndays)][::-1]
        lxts = [xt.strftime('%d %b').lstrip('0') for xt in xts]
    elif dt < 365.25*24*3600:
        t0 = tmax.replace(day=1, hour=0, minute=0, second=0)
        du = relativedelta(months=2)
        nmonths = int(np.ceil(dt/(24*3600*365.25/12)))
        xts = [t0 - i*du for i in range(nmonths)][::-1]
        lxts = [xt.strftime('%b') for xt in xts]
    elif dt < 2*365.25*24*3600:
        t0 = tmax.replace(day=1, hour=0, minute=0, second=0)
        du = relativedelta(months=2)
        nmonths = int(np.ceil(dt/(24*3600*365.25/12))/2)+1
        xts = [t0 - i*du for i in range(nmonths)][::-1]
        lxts = [xt.strftime('%b %Y') if xt.month == 1 else xt.strftime('%b') for xt in xts]
    return xts, lxts

def is_iterable(x):
    try:
        [_ for _ in x]
        return True
    except TypeError:
        return False

if __name__ == "__main__":  
    ''' For real-time operation, enter at the command line:

        > python controller.py

        Other options for experts.
    '''
    # set parameters (set to None to turn of emailing)
    keyfile = r'/home/rccuser/twitter_keys.txt'
    mail_from = 'noreply.whakaariforecaster@gmail.com'
    
    # heartbeat and error raising emails (set to None to turn of emailing)
    monitor_mail_to_file = r'/home/rccuser/whakaari_monitor_mail_to.txt'
    
    # forecast alert emails (set to None to turn of emailing)
    alert_mail_to_file = r'/home/rccuser/whakaari_alert_mail_to.txt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", 
        type=str,
        default='controller',
        help="flag indicating how controller is to run")
    args = parser.parse_args()
    if args.m == 'controller':
        controller = Controller(None, None, None, keyfile, test=False)
        controller = Controller(mail_from, monitor_mail_to_file, alert_mail_to_file, keyfile, test=False)
        controller.run()
    elif args.m == 'controller-test':
        controller = Controller(None, None, None, None, test=True)
        controller.run()
    elif args.m == 'update_forecast':
        update_forecast_v2()
    elif args.m == 'update_forecast_test':
        update_forecast_test()
    elif args.m == 'plot_date':
        plot_date('2020-09-18')
        plot_date('2020-09-03') 
    elif args.m == 'rebuild_hires_features':
        rebuild_hires_features()
    
