import os, sys, traceback, smtplib, ssl, yagmail, shutil, argparse
sys.path.insert(0, os.path.abspath('..'))
# from whakaari import *

import obspy
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader

makedir = lambda name: os.makedirs(name, exist_ok=True)

def main(clean=False, station='WIZ'):
    # pull raw geonet data
    makedir('_tmp')

    client = FDSNClient("GEONET")
    client_nrt = FDSNClient('https://service-nrt.geonet.org.nz')
    try:
        site = client.get_stations(station=station, level="response", channel="HHZ")
    except FDSNNoDataException:
        pass


    # default data range if not given 
    ti = ti or datetime(self.tf.year,self.tf.month,self.tf.day,0,0,0)
    tf = tf or datetime.today() + _DAY
    
    ti = datetimeify(ti)
    tf = datetimeify(tf)

    ndays = (tf-ti).days

    # parallel data collection - creates temporary files in ./_tmp
    pars = [[i,ti,self.station] for i in range(ndays)]
    p = Pool(6)
    p.starmap(get_data_for_day, pars)
    p.close()
    p.join()

    # special case of no file to update - create new file
    if not self.exists:
        shutil.copyfile('_tmp/_tmp_fl_00000.dat',self.file)
        self.exists = True
        shutil.rmtree('_tmp')
        return

    # read temporary files in as dataframes for concatenation with existing data
    dfs = [self.df[datas]]
    for i in range(ndays):
        fl = '_tmp/_tmp_fl_{:05d}.csv'.format(i)
        if not os.path.isfile(fl): 
            continue
        # dfs.append(pd.read_csv(fl, index_col=0, parse_dates=[0,], infer_datetime_format=True))
        dfs.append(load_dataframe(fl, index_col=0, parse_dates=[0,], infer_datetime_format=True))
    shutil.rmtree('_tmp')
    self.df = pd.concat(dfs)

    # impute missing data using linear interpolation and save file
    self.df = self.df.loc[~self.df.index.duplicated(keep='last')]
    self.df = self.df.resample('10T').interpolate('linear')

    # remove artefact in computing dsar
    for i in range(1,int(np.floor(self.df.shape[0]/(24*6)))): 
        ind = i*24*6
        self.df['dsar'][ind] = 0.5*(self.df['dsar'][ind-1]+self.df['dsar'][ind+1])

    # self.df.to_csv(self.file, index=True)
    save_dataframe(self.df, self.file, index=True)
    self.ti = self.df.index[0]
    self.tf = self.df.index[-1]

def get_data_for_day(i,t0,station):
    """ Download WIZ data for given 24 hour period, writing data to temporary file.

        Parameters:
        -----------
        i : integer
            Number of days that 24 hour download period is offset from initial date.
        t0 : datetime.datetime
            Initial date of data download period.
        
    """
    t0 = UTCDateTime(t0)

    # open clients
    client = FDSNClient("GEONET")
    client_nrt = FDSNClient('https://service-nrt.geonet.org.nz')
    
    daysec = 24*3600
    data_streams = [[2, 5], [4.5, 8], [8,16]]
    names = ['rsam','mf','hf']

    # download data
    datas = []
    try:
        site = client.get_stations(starttime=t0+i*daysec, endtime=t0 + (i+1)*daysec, station=station, level="response", channel="HHZ")
    except FDSNNoDataException:
        pass

    try:
        WIZ = client.get_waveforms('NZ',station, "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
        
        # if less than 1 day of data, try different client
        if len(WIZ.traces[0].data) < 600*100:
            raise FDSNNoDataException('')
    except (ObsPyMSEEDFilesizeTooSmallError,FDSNNoDataException) as e:
        try:
            WIZ = client_nrt.get_waveforms('NZ',station, "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
        except FDSNNoDataException:
            return

    # process frequency bands
    WIZ.remove_sensitivity(inventory=site)
    data = WIZ.traces[0].data
    ti = WIZ.traces[0].meta['starttime']
        # round start time to nearest 10 min increment
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))
    ti = tiday+int(np.round((ti-tiday)/600))*600
    N = 600*100                             # 10 minute windows in seconds
    Nm = int(N*np.floor(len(data)/N))
    for data_stream, name in zip(data_streams, names):
        filtered_data = bandpass(data, data_stream[0], data_stream[1], 100)
        filtered_data = abs(filtered_data[:Nm])
        datas.append(filtered_data.reshape(-1,N).mean(axis=-1)*1.e9)

    # compute dsar
    data = cumtrapz(data, dx=1./100, initial=0)
    data -= np.mean(data)
    j = names.index('mf')
    mfd = bandpass(data, data_streams[j][0], data_streams[j][1], 100)
    mfd = abs(mfd[:Nm])
    mfd = mfd.reshape(-1,N).mean(axis=-1)
    j = names.index('hf')
    hfd = bandpass(data, data_streams[j][0], data_streams[j][1], 100)
    hfd = abs(hfd[:Nm])
    hfd = hfd.reshape(-1,N).mean(axis=-1)
    dsar = mfd/hfd
    datas.append(dsar)
    names.append('dsar')

    # write out temporary file
    datas = np.array(datas)
    time = [(ti+j*600).datetime for j in range(datas.shape[1])]
    df = pd.DataFrame(zip(*datas), columns=names, index=pd.Series(time))
    save_dataframe(df, '_tmp/_tmp_fl_{:05d}.csv'.format(i), index=True, index_label='time')

def main2():

    # Rectangular domain containing parts of southern Germany.
    domain = RectangularDomain(minlatitude=-37.8, maxlatitude=-37.3,
                            minlongitude=177, maxlongitude=177.4)

    restrictions = Restrictions(
        # Get data for a whole year.
        starttime=obspy.UTCDateTime(1976, 12, 12),
        endtime=obspy.UTCDateTime(2020, 10, 1),
        # Chunk it to have one file per day.
        chunklength_in_sec=86400,
        # Considering the enormous amount of data associated with continuous
        # requests, you might want to limit the data based on SEED identifiers.
        # If the location code is specified, the location priority list is not
        # used; the same is true for the channel argument and priority list.
        network="NZ", station="WIZ", location="10", channel="HHZ",
        # The typical use case for such a data set are noise correlations where
        # gaps are dealt with at a later stage.
        reject_channels_with_gaps=False,
        # Same is true with the minimum length. All data might be useful.
        minimum_length=0.0,
        # Guard against the same station having different names.
        minimum_interstation_distance_in_m=100.0)

    # Restrict the number of providers if you know which serve the desired
    # data. If in doubt just don't specify - then all providers will be
    # queried.
    mdl = MassDownloader(providers=["GEONET"])
    mdl.download(domain, restrictions, mseed_storage="waveforms",
                stationxml_storage="stations")

if __name__ == "__main__":
    main2()