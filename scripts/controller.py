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
    def __init__(self, mail_from, mail_to, keyfile):
        # notification settings
        self.mail_from = mail_from
        self.mail_to = mail_to
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
        yag.send(to=self.mail_to,subject='starting up',contents='whakaari forecaster is starting up',attachments=None)
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
        elif self.new_system_down:
            subject = "Whakaari Eruption Forecast: Not receiving data"
            message = "Whakaari Eruption Forecast model is not receiving new data. Forecasting is suspended. Latest forecast is attached."
            fcstfile = 'WhakaariForecast_latest.png'
            shutil.copyfile('current_forecast.png',fcstfile)
            attachment = fcstfile
        elif self.debug_problem:
            subject = "Whakaari Eruption Forecast: System Raising Errors"
            message = "Whakaari Eruption Forecast model is raising errors. Summary below."
            for err in self.errors:
                with open(err,'r') as fp:
                    lns = fp.readlines()
                message += '\n{:s}\n{:s}\n'.format(err, ''.join(lns))
            attachment = self.errors
        elif self.heartbeat:
            subject = "Whakaari Eruption Forecast is working hard for YOU!"
            message = "All is well here. Have a great day."
            attachment = None

        yag = yagmail.SMTP(self.mail_from)
        yag.send(
            to=self.mail_to,
            subject=subject,
            contents=message, 
            attachments=attachment,
        )
            
class Controller(object):
    def __init__(self, mail_from, mail_to, keyfile, test=False):
        self.test = test
        self.alert = Alert(mail_from,mail_to,keyfile)
    def run(self):
        """ Top-level function to update forecast.
        """
        # check notifications working
        self.alert.send_startup_email()
        self.alert.post_startupdate()

        update_geonet_err_count = 0                     # counter to allow Nmax geonet update errors
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
                with open('controller.err','w') as fp:
                    fp.write("'current_forecast.png' was not generated")
                self.alert.debug_problem = True
                errors.append('controller.err')
            else:
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
    td = TremorData() 
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=2, overlap=0.75,  
        look_forward=2, data_streams=data_streams, root='online_forecaster',savefile_type='pkl')
        
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, Ncl=500,
        retrain=False, n_jobs=1)    
        
    t = pd.to_datetime(pd.read_csv('../features/online_forecaster_hires_features.csv', index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
    ti0,tf0 = t[0],t[-1]
    Ndays = int((fm.data.tf - tf0).total_seconds()/(24*3600))
    day = timedelta(days=1)
    for i in range(1,Ndays-1):
        # forecast from beginning of training period at high resolution
        fm.hires_forecast(ti=datetimeify('2020-03-01'), tf=tf0+i*day, recalculate=False, 
            save='current_forecast.png', nztimezone=True, save_alerts='alert.csv', n_jobs=1) 
        
def update_forecast():
    ''' Update model forecast.
    '''
    try:
        # constants
        month = timedelta(days=365.25/12)
        day = timedelta(days=1)
            
        # pull the latest data from GeoNet
        td = TremorData()
        td.update()
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
        ys = fm.hires_forecast(ti=datetimeify('2020-01-01'), tf=fm.data.tf, recalculate=False, 
            save='current_forecast.png', nztimezone=True, n_jobs=1) 

        tf = datetime.utcnow()
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
    mail_to = ['d.dempsey@auckland.ac.nz']
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", 
        type=str,
        default='controller',
        help="flag indicating how controller is to run")
    args = parser.parse_args()
    if args.m == 'controller':
        controller = Controller(mail_from, mail_to, keyfile)
        controller.run()
    elif args.m == 'controller-test':
        controller = Controller(mail_from, mail_to, keyfile, test=True)
        controller.run()
    elif args.m == 'update_forecast':
        update_forecast()
    elif args.m == 'update_forecast_test':
        update_forecast_test()
    elif args.m == 'rebuild_hires_features':
        rebuild_hires_features()
    
