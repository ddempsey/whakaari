#!/usr/bin/env python
# coding: utf-8

# The aim of this notebook is to calibrate the output of David's White Island/whakaari forecasts in order to have probabilities instead of "consensus".

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# In[2]:


from pathlib import Path
from zipfile import ZipFile

project_path = Path('/Users/akem134/04-Research/WhiteIsland/calibration')

with ZipFile(project_path / 'DS20eEa_WhiteIsland.zip', 'r') as zipped_data:
    df = pd.read_csv(zipped_data.open('two_features.csv'), index_col=0)
    
df.shape


# In[3]:


df.head()


# In[5]:


import sys
sys.path.append('/Users/akem134/04-Research/WhiteIsland/whakaari/scripts')
sys.path.append('/Users/akem134/04-Research/WhiteIsland/whakaari')


# In[6]:


from forecast_model import prepare_dec2019_forecast


# In[8]:


import joblib

rerun = False
if rerun:
    fm, month, te, ys = prepare_dec2019_forecast(retrain=False)
    import joblib
    joblib.dump((fm, month, te, ys), 'DS20eEb_fitted_consensus_model.joblib', compress=9)
else:
    fm, month, te, ys = joblib.load('DS20eEb_fitted_consensus_model.joblib')


# In[9]:


fm.get_forecast_target(ys)


# In[10]:


target = pd.Series(fm._ys, index=ys.index)


# In[12]:


calibration_window = ys.index[ys.index < '2019-12-01 00:00:00']


# In[13]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()


# In[14]:


from sklearn.calibration import CalibratedClassifierCV
cccv = CalibratedClassifierCV(clf)


# In[15]:


cccv = CalibratedClassifierCV(clf)
cccv.fit(ys.loc[calibration_window], target.loc[calibration_window])


# In[49]:


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

plot(ys)
plt.savefig('DS20eEb_a__calibrated.pdf', bbox_inches='tight')
plt.close()


# In[50]:


plot(ys, '2012')
plt.axvline(x=pd.datetime.fromisoformat('2012-08-04 16:52:00'),
           color='red')
plt.savefig('DS20eEb_a__calibrated_2012.pdf', bbox_inches='tight')
plt.close()


# In[51]:


plot(ys, '2013')
plt.axvline(x=pd.datetime.fromisoformat('2013-08-19 22:23:00'),
           color='red')
plt.axvline(x=pd.datetime.fromisoformat('2013-10-03 12:35:00'),
           color='red')
plt.savefig('DS20eEb_a__calibrated_2013.pdf', bbox_inches='tight')
plt.close()


# In[52]:


plot(ys, '2016')
plt.axvline(x=pd.datetime.fromisoformat('2016-04-27 09:37:00'),
           color='red')
plt.savefig('DS20eEb_a__calibrated_2016.pdf', bbox_inches='tight')
plt.close()


# In[43]:


out = fm.hires_forecast(ti=pd.datetime.fromisoformat('2019-12-02 00:00:00'),
                  tf=pd.datetime.fromisoformat('2019-12-16 00:00:00'),
                  recalculate=True, 
        save='Dec2019_forecast.png', nztimezone=True)  
plt.close()


# In[53]:


plot(out, '2019')
plt.axvline(x=pd.datetime.fromisoformat('2019-12-09 01:11:00'),
           color='red')
plt.axhline(y=0.2,linestyle='--', color='k')
plt.savefig('DS20eEb_a__calibrated_2019.pdf', bbox_inches='tight')
plt.close()


# In[ ]:




