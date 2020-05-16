#!/usr/bin/env python
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, os.path.abspath('..'))

from forecast_model import prepare_dec2019_forecast

def plot(ys, focus='all'):
    events = {'2012': lambda t: np.logical_and(t.index > '2012-07-01 00:00:00', t.index < '2012-10-01 00:00:00'),
              '2013': lambda t: np.logical_and(t.index > '2013-08-01 00:00:00', t.index < '2013-11-01 00:00:00'),
              '2016': lambda t: np.logical_and(t.index > '2016-03-01 00:00:00', t.index < '2016-06-01 00:00:00'),
              '2019': lambda t: np.logical_and(t.index > '2019-12-01 00:00:00', t.index < '2019-12-15 00:00:00'),
              'all': lambda t: t.index
         }
    window = events[focus](ys)
    calibrated = pd.Series(cccv.predict_proba(ys)[:,1],
                       index=ys.index)
    calibrated.loc[window].plot()
    target.loc[events[focus](target)].plot(linestyle=':')
    plt.ylim(0,0.25)
    plt.ylabel('probability')


rerun = False
if rerun:
    fm, month, te, ys = prepare_dec2019_forecast(retrain=False)
    import joblib
    joblib.dump((fm, month, te, ys), 'DS20eEb_fitted_consensus_model.joblib', compress=9)
else:
    fm, month, te, ys = joblib.load('DS20eEb_fitted_consensus_model.joblib')


fm.get_forecast_target(ys)
target = pd.Series(fm._ys, index=ys.index)
calibration_window = ys.index[ys.index < '2019-12-01 00:00:00']

clf = LogisticRegression()

cccv = CalibratedClassifierCV(clf)
cccv.fit(ys.loc[calibration_window], target.loc[calibration_window])

plot(ys)
plt.savefig('DS20eEb_a__calibrated.pdf', bbox_inches='tight')
plt.close()

plot(ys, '2012')
plt.axvline(x=pd.datetime.fromisoformat('2012-08-04 16:52:00'),
           color='red')
plt.savefig('DS20eEb_a__calibrated_2012.pdf', bbox_inches='tight')
plt.close()

plot(ys, '2013')
plt.axvline(x=pd.datetime.fromisoformat('2013-08-19 22:23:00'),
           color='red')
plt.axvline(x=pd.datetime.fromisoformat('2013-10-03 12:35:00'),
           color='red')
plt.savefig('DS20eEb_a__calibrated_2013.pdf', bbox_inches='tight')
plt.close()

plot(ys, '2016')
plt.axvline(x=pd.datetime.fromisoformat('2016-04-27 09:37:00'),
           color='red')
plt.savefig('DS20eEb_a__calibrated_2016.pdf', bbox_inches='tight')
plt.close()

out = fm.hires_forecast(ti=pd.datetime.fromisoformat('2019-12-02 00:00:00'),
                  tf=pd.datetime.fromisoformat('2019-12-16 00:00:00'),
                  recalculate=True, 
        save='Dec2019_forecast.png', nztimezone=True)  
plt.close()

plot(out, '2019')
plt.axvline(x=pd.datetime.fromisoformat('2019-12-09 01:11:00'),
           color='red')
plt.axhline(y=0.2,linestyle='--', color='k')
plt.savefig('DS20eEb_a__calibrated_2019.pdf', bbox_inches='tight')
plt.close()




