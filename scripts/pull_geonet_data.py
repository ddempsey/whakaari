import os, sys, traceback, smtplib, ssl, yagmail, shutil, argparse
sys.path.insert(0, os.path.abspath('..'))
from whakaari import *

def main(clean=False, station='WIZ'):
    # pull raw geonet data
    makedir('_tmp_'+station)
    if clean:
        fls = glob('_tmp_{:s}/*.pkl'.format(station))
        _ = [os.remove(fl) for fl in fls]

    # default data range if not given 
    ti = datetime(2008,5,22,0,0,0)    # first reliable date for WIZ
    tf = datetime.today()

    ndays = (tf-ti).days+1

    # parallel data collection - creates temporary files in ./_tmp
    # pars = [[i,ti,station] for i in range(ndays)]
    f = partial(get_data_for_day, ti, station)
    p = Pool(7)
    for i, _ in enumerate(p.imap(f, range(ndays))):
        cf = (i+1)/ndays
        print(f'grabbing geonet data: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
    
    p.close()
    p.join()

def get_data_for_day(t0,station,i):
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
            client_nrt = FDSNClient('https://service-nrt.geonet.org.nz')
            WIZ = client_nrt.get_waveforms('NZ',station, "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
        except FDSNNoDataException:
            return

    # process frequency bands
    WIZ.remove_sensitivity(inventory=site)
    # data = WIZ.traces[0].data
    # WIZ.traces[0].filter('lowpass',freq=20.0)
    WIZ.traces[0].decimate(5)
    ti = WIZ.traces[0].meta['starttime']
    # WIZ.traces[0].write("_tmp_{:s}/{:d}-{:02d}-{:02d}.mseed".format(station,ti.year, ti.month, ti.day), format='MSEED')
    save_dataframe(WIZ.traces[0], "_tmp_{:s}/{:d}-{:02d}-{:02d}.pkl".format(station,ti.year, ti.month, ti.day))
    

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
                stationxml_storage="stations", threads_per_client=6)

if __name__ == "__main__":
    main()